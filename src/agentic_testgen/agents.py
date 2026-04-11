from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

try:
    import dspy
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]

from agentic_testgen.checkpointing import CheckpointStore
from agentic_testgen.config import AppConfig
from agentic_testgen.coverage import CoverageAnalyzer, summarize_tree
from agentic_testgen.gitlab import GitLabRepositoryManager
from agentic_testgen.logging import RunLogger
from agentic_testgen.memory import MemoryManager
from agentic_testgen.models import (
    AttemptRecord,
    CoverageComparison,
    CoverageRecord,
    FileWorkItem,
    GlobalCoverageSummary,
    IntegrationDecision,
    ModelDefinition,
    RepoContext,
    RunCheckpoint,
    SubagentResult,
    WorkflowRunResult,
)
from agentic_testgen.reporting import ReportWriter
from agentic_testgen.tools import SafeToolset, ToolContext
from agentic_testgen.tracing import MlflowTracer
from agentic_testgen.utils import new_run_id, prompt_hash, slugify, utc_timestamp, write_json
from agentic_testgen.workspace import RunWorkspace, WorkspaceManager


PROMPT_VERSION = "daddy_subagents_reflective_v1"


class DSPyRuntime:
    def __init__(self, config: AppConfig, logger: RunLogger, model_override: ModelDefinition | None = None):
        self.config = config
        self.logger = logger
        self.model_override = model_override
        self.enabled = False
        self.model_id = "unconfigured"
        self._configure()

    def _configure(self) -> None:
        if dspy is None:
            self.logger.log_event("dspy.configure", "skipped", summary="DSPy not installed")
            return
        model_name = self.model_override.model_name if self.model_override else self.config.model.model_name
        if not model_name:
            self.logger.log_event("dspy.configure", "skipped", summary="No model configured")
            return
        api_key = ""
        api_base = ""
        if self.model_override:
            import os

            api_key = os.getenv(self.model_override.api_key_env, "")
            api_base = self.model_override.api_base or ""
            self.model_id = self.model_override.model_id
        else:
            api_key = self.config.model.api_key
            api_base = self.config.model.api_base
            self.model_id = model_name

        final_model = model_name
        if "/" not in final_model and self.config.model.provider:
            final_model = f"{self.config.model.provider}/{final_model}"
        try:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["api_base"] = api_base
            lm = dspy.LM(final_model, **kwargs)
            dspy.configure(lm=lm)
            self.enabled = True
            self.logger.log_event("dspy.configure", "completed", summary=f"Configured {self.model_id}")
        except Exception as exc:
            self.logger.log_event("dspy.configure", "failed", summary=str(exc))

    def overview(
        self,
        repo_tree: str,
        module_paths: list[str],
    ) -> str:
        if not self.enabled:
            return (
                "# Repository Overview\n\n"
                f"- Modules: {', '.join(module_paths) if module_paths else '(none detected)'}\n\n"
                "## Tree\n\n```text\n"
                f"{repo_tree}\n```"
            )
        try:
            program = dspy.ChainOfThought("repo_tree, module_paths -> overview_markdown")
            result = program(
                repo_tree=repo_tree,
                module_paths=", ".join(module_paths),
            )
            return getattr(result, "overview_markdown", str(result))
        except Exception as exc:
            self.logger.log_event("dspy.overview", "failed", summary=str(exc))
            return (
                "# Repository Overview\n\n"
                f"- Modules: {', '.join(module_paths) if module_paths else '(none detected)'}\n"
            )

    def reflect(self, objective: str, latest_output: str, prior_failures: str) -> str:
        if not self.enabled:
            return latest_output or prior_failures or "No reflection available."
        try:
            program = dspy.Predict("objective, latest_output, prior_failures -> summary")
            result = program(
                objective=objective,
                latest_output=latest_output[:8000],
                prior_failures=prior_failures[:8000],
            )
            return getattr(result, "summary", str(result))
        except Exception as exc:
            self.logger.log_event("dspy.reflect", "failed", summary=str(exc))
            return latest_output or prior_failures or str(exc)


class DaddySubagentsReflectiveWorkflow:
    def __init__(self, config: AppConfig):
        self.config = config
        self.workspace_manager = WorkspaceManager(config.workspace_root)
        self.coverage = CoverageAnalyzer(config)
        self.memory = MemoryManager(config.workspace_root)

    def run_from_gitlab(
        self,
        repo_url: str,
        run_id: str | None = None,
        *,
        max_files: int | None = None,
    ) -> WorkflowRunResult:
        self.config.validate_for_run()
        run_id = run_id or new_run_id()
        workspace = self.workspace_manager.create(run_id)
        repo_name = self._repo_name(repo_url)
        clone_target = workspace.clone_dir / repo_name
        logger = self._build_logger(run_id, workspace)
        tracer = MlflowTracer(self.config.mlflow, logger)
        tracer.validate()
        tracer.configure()
        with tracer.run(
            f"workflow-{run_id}",
            tags={"workflow": "daddy_subagents_reflective", "source_type": "gitlab", "run_id": run_id},
        ):
            tracer.log_params({"repo_url": repo_url, "repo_name": repo_name, "run_id": run_id})
            with logger.step("gitlab.clone", details={"repo": repo_url}) as step:
                manager = GitLabRepositoryManager(self.config, logger)
                result = manager.clone(repo_url, clone_target)
                if not result.ok:
                    raise RuntimeError(result.stderr or result.stdout)
                step["summary"] = f"Cloned {repo_name}"
            return self._execute(
                run_id=run_id,
                repo_url=repo_url,
                source_type="gitlab",
                repo_name=repo_name,
                repo_root=clone_target,
                workspace=workspace,
                logger=logger,
                tracer=tracer,
                model_override=None,
                max_files=max_files,
            )

    def run_from_local_path(
        self,
        repo_path: Path,
        *,
        run_id: str | None = None,
        source_name: str | None = None,
        selected_files: list[str] | None = None,
        model_override: ModelDefinition | None = None,
        max_files: int | None = None,
    ) -> WorkflowRunResult:
        run_id = run_id or new_run_id("eval")
        workspace = self.workspace_manager.create(run_id)
        repo_name = source_name or repo_path.name
        clone_target = workspace.clone_dir / repo_name
        logger = self._build_logger(run_id, workspace)
        tracer = MlflowTracer(self.config.mlflow, logger)
        tracer.validate()
        tracer.configure()
        with tracer.run(
            f"workflow-{run_id}",
            tags={"workflow": "daddy_subagents_reflective", "source_type": "fixture", "run_id": run_id},
        ):
            tracer.log_params({"repo_source": str(repo_path), "repo_name": repo_name, "run_id": run_id})
            with logger.step("fixture.copy", details={"source": str(repo_path)}) as step:
                self.workspace_manager.copy_local_repo(repo_path, clone_target)
                self._ensure_git_repository(clone_target, logger)
                step["summary"] = f"Prepared fixture {repo_name}"
            return self._execute(
                run_id=run_id,
                repo_url=str(repo_path),
                source_type="fixture",
                repo_name=repo_name,
                repo_root=clone_target,
                workspace=workspace,
                logger=logger,
                tracer=tracer,
                model_override=model_override,
                selected_files=selected_files,
                max_files=max_files,
            )

    def resume(self, run_id: str) -> WorkflowRunResult:
        workspace = self.workspace_manager.create(run_id)
        logger = self._build_logger(run_id, workspace)
        checkpoint_store = CheckpointStore(workspace.checkpoints_dir)
        checkpoint = checkpoint_store.load()
        if not checkpoint:
            raise ValueError(f"No checkpoint found for {run_id}")
        clone_path = Path(checkpoint.metadata["clone_path"])
        runtime = DSPyRuntime(self.config, logger)
        tracer = MlflowTracer(self.config.mlflow, logger)
        tracer.validate()
        tracer.configure()
        repo_context = self._make_repo_context(
            run_id=run_id,
            repo_url=checkpoint.repo_url,
            repo_name=checkpoint.repo_name,
            clone_path=clone_path,
            workspace=workspace,
            source_type=checkpoint.metadata.get("source_type", "gitlab"),
            module_paths=checkpoint.metadata.get("module_paths", []),
        )
        results = checkpoint.completed_results[:]
        work_items = checkpoint.pending_work_items[:]
        attempts = self._attempts_from_results(results)
        coverage_comparison = self._coverage_comparison_from_metadata(checkpoint.metadata)
        baseline_payload = checkpoint.metadata.get("baseline_coverage")
        baseline_summary = (
            GlobalCoverageSummary(**baseline_payload)
            if baseline_payload
            else GlobalCoverageSummary(covered_lines=0, missed_lines=0, coverage_percent=0.0, report_count=0)
        )
        coverage_context_path = Path(
            checkpoint.metadata.get("coverage_context_markdown_path", str(workspace.artifacts_dir / "coverage-context.md"))
        )
        self.memory.initialize_run_memory(self._run_memory_path(workspace), repo_context)
        with tracer.run(
            f"workflow-resume-{run_id}",
            tags={"workflow": "daddy_subagents_reflective", "source_type": repo_context.source_type, "run_id": run_id},
        ):
            tracer.log_params({"run_id": run_id, "resume": True})
            new_results = self._dispatch_subagents(
                repo_context,
                workspace,
                logger,
                runtime,
                work_items,
                results,
                checkpoint_store,
                baseline_summary,
                coverage_context_path,
            )
            results.extend(new_results)
            attempts = self._attempts_from_results(results)
            pending = self._read_pending_integrations(workspace)
            self._save_checkpoint(
                checkpoint_store,
                repo_context,
                phase="resumed_completed",
                pending_work_items=[],
                completed_results=results,
                pending_integrations=pending,
                paused=False,
            )
            overview_path = workspace.artifacts_dir / "overview.md"
            workbook_path, summary_path = self._write_reports(
                repo_context=repo_context,
                workspace=workspace,
                work_items=work_items,
                results=results,
                coverage_comparison=coverage_comparison,
            )
            self._log_run_artifacts(
                tracer=tracer,
                workspace=workspace,
                workbook_path=workbook_path,
                summary_path=summary_path,
            )
            tracer.log_metrics(
                {
                    "attempt_count": len(attempts),
                    "completed_results": len(results),
                    "pending_integrations": len(pending),
                }
            )
            return WorkflowRunResult(
                run_id=run_id,
                repo_context=repo_context,
                work_items=[],
                subagent_results=results,
                attempts=attempts,
                overview_path=str(overview_path),
                workbook_path=str(workbook_path),
                summary_path=str(summary_path),
                coverage_comparison_path=checkpoint.metadata.get("coverage_comparison_path"),
            )

    def _execute(
        self,
        *,
        run_id: str,
        repo_url: str,
        source_type: str,
        repo_name: str,
        repo_root: Path,
        workspace: RunWorkspace,
        logger: RunLogger,
        tracer: MlflowTracer,
        model_override: ModelDefinition | None,
        selected_files: list[str] | None = None,
        max_files: int | None = None,
    ) -> WorkflowRunResult:
        checkpoint_store = CheckpointStore(workspace.checkpoints_dir)
        runtime = DSPyRuntime(self.config, logger, model_override=model_override)
        report_writer = ReportWriter(workspace.artifacts_dir)
        repo_context = self._make_repo_context(
            run_id=run_id,
            repo_url=repo_url,
            repo_name=repo_name,
            clone_path=repo_root,
            workspace=workspace,
            source_type=source_type,
        )
        self._ensure_git_repository(repo_root, logger)
        repo_tree = summarize_tree(repo_root)
        modules = self.coverage.discover_modules(repo_root)
        repo_context.module_paths = modules
        self.memory.initialize_run_memory(self._run_memory_path(workspace), repo_context)
        with logger.step("repo.analyze", details={"modules": modules}) as step:
            overview = runtime.overview(repo_tree, modules)
            overview_path = report_writer.write_overview(overview)
            step["summary"] = f"Overview written to {overview_path.name}"
        tracer.log_params(
            {
                "repo_name": repo_name,
                "source_type": source_type,
                "module_count": len(modules),
                "model_id": runtime.model_id,
            }
        )
        tracer.log_text(overview, "overview.md")
        self._save_checkpoint(
            checkpoint_store,
            repo_context,
            phase="analysis_completed",
            pending_work_items=[],
            completed_results=[],
            pending_integrations=[],
            paused=False,
        )
        with logger.step("coverage.run") as step:
            coverage_result, coverage_records, maven_log_paths = self.coverage.run_tests_with_coverage(
                repo_root,
                maven_logs_dir=workspace.logs_dir / "maven",
                log_prefix="daddy-project-coverage",
            )
            baseline_summary = self.coverage.summarize_global_coverage(coverage_records)
            work_items = self.coverage.build_work_items(coverage_records)
            if selected_files:
                selected = set(selected_files)
                work_items = [item for item in work_items if item.file_path in selected]
            work_items = self._apply_work_item_limit(work_items, max_files)
            step["summary"] = f"Coverage records: {len(coverage_records)}, work items: {len(work_items)}"
            step["exit_code"] = getattr(coverage_result, "exit_code", None)
            step["maven_log_paths"] = maven_log_paths
            step["baseline_coverage_percent"] = baseline_summary.coverage_percent
        coverage_context_markdown_path, coverage_context_json_path = self._write_coverage_context_artifacts(
            workspace,
            baseline_summary,
            coverage_records,
            work_items,
        )
        if runtime.enabled:
            self._run_daddy_react(repo_context, workspace, logger, runtime, work_items)
        self._save_checkpoint(
            checkpoint_store,
            repo_context,
            phase="coverage_completed",
            pending_work_items=work_items,
            completed_results=[],
            pending_integrations=[],
            paused=False,
            extra_metadata={
                "baseline_coverage": baseline_summary.to_json(),
                "baseline_coverage_records": [item.to_json() for item in coverage_records],
                "all_work_items": [item.to_json() for item in work_items],
                "coverage_context_markdown_path": str(coverage_context_markdown_path),
                "coverage_context_json_path": str(coverage_context_json_path),
            },
        )
        if self._pause_requested(workspace):
            self._save_checkpoint(
                checkpoint_store,
                repo_context,
                phase="paused",
                pending_work_items=work_items,
                completed_results=[],
                pending_integrations=[],
                paused=True,
            )
            workbook_path, summary_path = self._write_reports(
                repo_context=repo_context,
                workspace=workspace,
                work_items=work_items,
                results=[],
            )
            return WorkflowRunResult(
                run_id=run_id,
                repo_context=repo_context,
                work_items=work_items,
                subagent_results=[],
                attempts=[],
                overview_path=str(workspace.artifacts_dir / "overview.md"),
                workbook_path=str(workbook_path),
                summary_path=str(summary_path),
                coverage_comparison_path=None,
            )
        subagent_results = self._dispatch_subagents(
            repo_context,
            workspace,
            logger,
            runtime,
            work_items,
            [],
            checkpoint_store,
            baseline_summary,
            coverage_context_markdown_path,
        )
        attempts = self._attempts_from_results(subagent_results)
        pending_integrations = self._read_pending_integrations(workspace)
        self._save_checkpoint(
            checkpoint_store,
            repo_context,
            phase="completed",
            pending_work_items=[],
            completed_results=subagent_results,
            pending_integrations=pending_integrations,
            paused=False,
            extra_metadata={
                "baseline_coverage": baseline_summary.to_json(),
                "baseline_coverage_records": [item.to_json() for item in coverage_records],
                "all_work_items": [item.to_json() for item in work_items],
            },
        )
        coverage_comparison = None
        coverage_comparison_path = None
        if self.config.auto_integrate_successful_worktrees and any(
            item.status == "integrated" for item in pending_integrations
        ):
            coverage_comparison = self._rerun_post_merge_coverage(
                checkpoint_store=checkpoint_store,
                repo_context=repo_context,
                workspace=workspace,
                logger=logger,
                baseline_summary=baseline_summary,
                baseline_records=coverage_records,
            )
            if coverage_comparison:
                coverage_comparison_path = str(workspace.artifacts_dir / "coverage-comparison.md")
        workbook_path, summary_path = self._write_reports(
            repo_context=repo_context,
            workspace=workspace,
            work_items=work_items,
            results=subagent_results,
            coverage_comparison=coverage_comparison,
        )
        tracer.log_metrics(
            {
                "work_item_count": len(work_items),
                "subagent_count": len(subagent_results),
                "attempt_count": len(attempts),
                "passed_subagents": sum(1 for item in subagent_results if item.status == "passed"),
                "pending_integrations": len(pending_integrations),
                "baseline_coverage_percent": baseline_summary.coverage_percent,
                "after_merge_coverage_percent": coverage_comparison.after.coverage_percent if coverage_comparison else baseline_summary.coverage_percent,
                "coverage_percentage_increase": coverage_comparison.percentage_increase if coverage_comparison else 0.0,
            }
        )
        self._log_run_artifacts(
            tracer=tracer,
            workspace=workspace,
            workbook_path=workbook_path,
            summary_path=summary_path,
            coverage_comparison_path=coverage_comparison_path,
        )
        return WorkflowRunResult(
            run_id=run_id,
            repo_context=repo_context,
            work_items=work_items,
            subagent_results=subagent_results,
            attempts=attempts,
            overview_path=str(workspace.artifacts_dir / "overview.md"),
            workbook_path=str(workbook_path),
            summary_path=str(summary_path),
            coverage_comparison_path=coverage_comparison_path,
        )

    def _dispatch_subagents(
        self,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        runtime: DSPyRuntime,
        work_items: list[FileWorkItem],
        existing_results: list[SubagentResult],
        checkpoint_store: CheckpointStore,
        baseline_summary: GlobalCoverageSummary,
        coverage_context_path: Path,
    ) -> list[SubagentResult]:
        results: list[SubagentResult] = []
        pending_queue = work_items[:]
        in_flight: dict[Any, FileWorkItem] = {}
        run_memory_path = self._run_memory_path(workspace)
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_parallel_subagents)) as pool:
            while pending_queue and len(in_flight) < max(1, self.config.max_parallel_subagents):
                item = pending_queue.pop(0)
                if self._pause_requested(workspace):
                    break
                item.assigned_subagent_id = item.assigned_subagent_id or f"subagent_{item.priority_rank:03d}"
                future = pool.submit(
                    self._run_subagent,
                    repo_context,
                    workspace,
                    logger,
                    runtime,
                    item,
                    checkpoint_store,
                    baseline_summary,
                    coverage_context_path,
                )
                in_flight[future] = item
            while in_flight:
                done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    item = in_flight.pop(future)
                    result = future.result()
                    self.memory.record_result(run_memory_path, repo_context, item, result)
                    results.append(result)
                    while pending_queue and len(in_flight) < max(1, self.config.max_parallel_subagents):
                        next_item = pending_queue.pop(0)
                        if self._pause_requested(workspace):
                            pending_queue.insert(0, next_item)
                            break
                        next_item.assigned_subagent_id = next_item.assigned_subagent_id or f"subagent_{next_item.priority_rank:03d}"
                        next_future = pool.submit(
                            self._run_subagent,
                            repo_context,
                            workspace,
                            logger,
                            runtime,
                            next_item,
                            checkpoint_store,
                            baseline_summary,
                            coverage_context_path,
                        )
                        in_flight[next_future] = next_item
                    pending_items = pending_queue + list(in_flight.values())
                    self._save_checkpoint(
                        checkpoint_store,
                        repo_context,
                        phase="subagents_running",
                        pending_work_items=pending_items,
                        completed_results=existing_results + results,
                        pending_integrations=self._read_pending_integrations(workspace),
                        paused=self._pause_requested(workspace),
                    )
        return sorted(results, key=lambda item: item.subagent_id)

    def _make_repo_context(
        self,
        *,
        run_id: str,
        repo_url: str,
        repo_name: str,
        clone_path: Path,
        workspace: RunWorkspace,
        source_type: str,
        module_paths: list[str] | None = None,
    ) -> RepoContext:
        return RepoContext(
            run_id=run_id,
            repo_url=repo_url,
            repo_name=repo_name,
            clone_path=clone_path,
            workspace_root=workspace.root,
            source_type=source_type,
            module_paths=module_paths or [],
        )

    def _attempts_from_results(self, results: list[SubagentResult]) -> list[AttemptRecord]:
        return [attempt for result in results for attempt in result.attempts]

    def _write_reports(
        self,
        *,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        work_items: list[FileWorkItem],
        results: list[SubagentResult],
        coverage_comparison: CoverageComparison | None = None,
    ) -> tuple[Path, Path]:
        report_writer = ReportWriter(workspace.artifacts_dir)
        attempts = self._attempts_from_results(results)
        workbook_path = report_writer.write_workbook(
            repo_context,
            work_items,
            attempts,
            [],
            coverage_comparison=coverage_comparison,
        )
        summary_path = report_writer.write_json_summary(
            repo_context,
            work_items,
            results,
            [],
            coverage_comparison=coverage_comparison,
        )
        return workbook_path, summary_path

    def _write_coverage_context_artifacts(
        self,
        workspace: RunWorkspace,
        baseline_summary: GlobalCoverageSummary,
        coverage_records: list[CoverageRecord],
        work_items: list[FileWorkItem],
    ) -> tuple[Path, Path]:
        markdown_path = workspace.artifacts_dir / "coverage-context.md"
        json_path = workspace.artifacts_dir / "coverage-context.json"
        top_records = sorted(
            coverage_records,
            key=lambda item: (item.coverage_percent, -item.missed_lines, item.file_path),
        )[:100]
        top_work_items = sorted(work_items, key=lambda item: item.priority_rank)[:100]
        markdown_lines = [
            "# Coverage Context",
            "",
            "## Global Coverage",
            "",
            f"- Coverage percent: {baseline_summary.coverage_percent}%",
            f"- Covered lines: {baseline_summary.covered_lines}",
            f"- Missed lines: {baseline_summary.missed_lines}",
            f"- File reports: {baseline_summary.report_count}",
            "",
            "## Candidate Files",
            "",
            "| rank | file | module | coverage % | covered | missed |",
            "|---|---|---|---:|---:|---:|",
        ]
        for item in top_work_items:
            markdown_lines.append(
                f"| {item.priority_rank} | {item.file_path} | {item.module} | {item.coverage_percent} | {item.covered_lines} | {item.missed_lines} |"
            )
        markdown_lines.extend(
            [
                "",
                "## Raw File Coverage (lowest coverage first)",
                "",
                "| file | module | coverage % | covered | missed |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for record in top_records:
            markdown_lines.append(
                f"| {record.file_path} | {record.module} | {record.coverage_percent} | {record.covered_lines} | {record.missed_lines} |"
            )
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        write_json(
            json_path,
            {
                "generated_at": utc_timestamp(),
                "global_coverage": baseline_summary.to_json(),
                "candidate_files": [item.to_json() for item in top_work_items],
                "file_coverage": [item.to_json() for item in top_records],
            },
        )
        return markdown_path, json_path

    def _log_run_artifacts(
        self,
        *,
        tracer: MlflowTracer,
        workspace: RunWorkspace,
        workbook_path: Path,
        summary_path: Path,
        coverage_comparison_path: str | None = None,
    ) -> None:
        tracer.log_artifact(workbook_path)
        tracer.log_artifact(summary_path)
        if coverage_comparison_path:
            tracer.log_artifact(coverage_comparison_path)
        tracer.log_artifact(workspace.artifacts_dir / "memory.json")
        tracer.log_artifact(workspace.artifacts_dir / "coverage-context.md")
        tracer.log_artifact(workspace.artifacts_dir / "coverage-context.json")
        tracer.log_artifact(workspace.artifacts_dir / "file-coverage-comparison.json")
        tracer.log_artifact(workspace.artifacts_dir / "file-coverage-comparison.md")
        tracer.log_artifact(workspace.artifacts_dir / "file-coverage-comparison.csv")
        tracer.log_artifact(workspace.logs_dir / "run.log")
        tracer.log_artifact(workspace.logs_dir / "events.jsonl")
        tracer.log_artifact(workspace.logs_dir / "dspy_traces.jsonl")

    def _compare_file_coverage(
        self,
        before: list[CoverageRecord],
        after: list[CoverageRecord],
    ) -> list[dict[str, Any]]:
        before_map = {item.file_path: item for item in before}
        after_map = {item.file_path: item for item in after}
        rows: list[dict[str, Any]] = []
        for file_path in sorted(set(before_map.keys()) | set(after_map.keys())):
            before_record = before_map.get(file_path)
            after_record = after_map.get(file_path)
            before_percent = before_record.coverage_percent if before_record else 0.0
            after_percent = after_record.coverage_percent if after_record else 0.0
            before_missed = before_record.missed_lines if before_record else 0
            after_missed = after_record.missed_lines if after_record else 0
            if before_record and after_record:
                status = "changed"
            elif before_record and not after_record:
                status = "removed"
            else:
                status = "new"
            rows.append(
                {
                    "file_path": file_path,
                    "module": (after_record.module if after_record else before_record.module if before_record else ""),
                    "before_coverage_percent": before_percent,
                    "after_coverage_percent": after_percent,
                    "coverage_delta": round(after_percent - before_percent, 2),
                    "before_missed_lines": before_missed,
                    "after_missed_lines": after_missed,
                    "missed_line_delta": before_missed - after_missed,
                    "status": status,
                }
            )
        return sorted(rows, key=lambda item: (item["coverage_delta"], item["missed_line_delta"], item["file_path"]))

    def _run_subagent(
        self,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        runtime: DSPyRuntime,
        item: FileWorkItem,
        checkpoint_store: CheckpointStore,
        baseline_summary: GlobalCoverageSummary,
        coverage_context_path: Path,
    ) -> SubagentResult:
        subagent_id = item.assigned_subagent_id or new_run_id("subagent")
        branch_name = f"subagent/{subagent_id}"
        worktree_path = workspace.worktrees_dir / subagent_id
        tool_context = ToolContext(
            run_id=repo_context.run_id,
            repo_root=repo_context.clone_path,
            clone_root=repo_context.clone_path,
            worktrees_root=workspace.worktrees_dir,
            config=self.config,
            logger=logger,
            subagent_id=subagent_id,
        )
        toolset = SafeToolset(tool_context)
        attempts: list[AttemptRecord] = []
        final_summary = ""
        commit_hash = ""
        integration_status = "pending_review"
        coverage_after: float | None = None
        coverage_delta = 0.0
        missed_line_reduction = 0
        with logger.step("subagent.prepare", subagent_id=subagent_id, file_path=item.file_path) as step:
            toolset.create_worktree(subagent_id, branch_name)
            tool_context.active_worktree = worktree_path
            step["summary"] = f"Worktree {worktree_path.name}"
        if not runtime.enabled:
            final_summary = "No DSPy model configured. Subagent cannot generate tests."
            result = SubagentResult(
                subagent_id=subagent_id,
                file_path=item.file_path,
                status="failed",
                worktree_path=worktree_path,
                branch_name=branch_name,
                attempts=attempts,
                final_summary=final_summary,
                integration_status="skipped",
                error_message=final_summary,
            )
            return result

        prior_failures: list[str] = []
        memory_insights = self.memory.lessons_for_item(self._run_memory_path(workspace), repo_context, item)
        for iteration in range(1, self.config.max_subagent_iterations + 1):
            suggested_test_path = self._suggest_test_path(worktree_path, item.file_path, iteration)
            objective = self._subagent_objective(
                repo_context,
                item,
                suggested_test_path,
                iteration,
                prior_failures,
                memory_insights,
                baseline_summary,
                coverage_context_path,
            )
            generated_file = ""
            tool_summary = ""
            validation_output = ""
            status = "failed"
            with logger.step(
                "subagent.iteration",
                subagent_id=subagent_id,
                file_path=item.file_path,
                iteration=iteration,
                details={"suggested_test_path": suggested_test_path},
            ) as step:
                react = dspy.ReAct(
                    "objective, file_path, suggested_test_path -> answer",
                    tools=toolset.build_dspy_tools(),
                    max_iters=6,
                )
                prediction = react(
                    objective=objective,
                    file_path=item.file_path,
                    suggested_test_path=suggested_test_path,
                )
                trajectory = getattr(prediction, "trajectory", {})
                logger.log_trace(
                    {
                        "run_id": repo_context.run_id,
                        "subagent_id": subagent_id,
                        "iteration": iteration,
                        "file_path": item.file_path,
                        "objective": objective,
                        "trajectory": trajectory,
                        "answer": getattr(prediction, "answer", str(prediction)),
                    }
                )
                if tool_context.written_files:
                    generated_file = tool_context.written_files[-1]
                    validation_output = toolset.run_single_test(generated_file)
                    status = "passed" if tool_context.last_single_test_exit_code == 0 else "failed"
                else:
                    validation_output = "No test file was generated."
                tool_summary = json.dumps(trajectory, default=str)[:4000]
                reflective_summary = runtime.reflect(objective, validation_output, "\n".join(prior_failures))
                attempt = AttemptRecord(
                    run_id=repo_context.run_id,
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                    prompt_version=PROMPT_VERSION,
                    prompt_hash=prompt_hash(objective),
                    tool_call_summary=tool_summary,
                    generated_test_file=generated_file or None,
                    single_test_command=" ".join(
                        self.config.maven_command(
                            "-q",
                            f"-Dtest={Path(generated_file).stem if generated_file else '<none>'}",
                            "test",
                        )
                    ),
                    status=status,
                    failure_summary="" if status == "passed" else validation_output[:4000],
                    reflective_summary=reflective_summary[:4000],
                )
                attempts.append(attempt)
                item.status = status
                step["summary"] = f"Iteration {iteration} {status}"
                step["attempt_status"] = status
                step["generated_test_file"] = generated_file or ""
                step["failure_feedback"] = attempt.failure_summary[:800]
                step["reflective_summary"] = attempt.reflective_summary[:800]
                logger.log_event(
                    "subagent.feedback",
                    "completed",
                    summary=f"{subagent_id} iteration {iteration} {status}",
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                    details={
                        "generated_test_file": generated_file or "",
                        "failure_feedback": attempt.failure_summary[:1500],
                        "reflective_summary": attempt.reflective_summary[:1500],
                        "single_test_command": attempt.single_test_command,
                    },
                )
                write_json(
                    workspace.checkpoints_dir / f"{subagent_id}_iter_{iteration}.json",
                    {"attempt": attempt.to_json(), "file_path": item.file_path},
                )
                if status == "passed":
                    toolset.run_project_tests_with_coverage()
                    refreshed = self.coverage.collect_reports(worktree_path)
                    for record in refreshed:
                        if record.file_path == item.file_path:
                            coverage_after = record.coverage_percent
                            coverage_delta = round(record.coverage_percent - item.coverage_percent, 2)
                            missed_line_reduction = max(0, item.missed_lines - record.missed_lines)
                            break
                    commit_hash = toolset.commit_worktree_change(
                        f"Add generated tests for {Path(item.file_path).name}"
                    )
                    final_summary = reflective_summary
                    break
                prior_failures.append(reflective_summary)
                if self._pause_requested(workspace):
                    break
        if not commit_hash:
            logger.log_event(
                "subagent.feedback.summary",
                "completed",
                summary=f"{subagent_id} finished without passing test",
                subagent_id=subagent_id,
                file_path=item.file_path,
                details={
                    "attempt_count": len(attempts),
                    "last_failure_feedback": (attempts[-1].failure_summary[:1500] if attempts else ""),
                    "last_reflective_summary": (attempts[-1].reflective_summary[:1500] if attempts else ""),
                },
            )
        if commit_hash:
            decision = IntegrationDecision(
                subagent_id=subagent_id,
                branch_name=branch_name,
                commit_hash=commit_hash,
                status="pending_review",
                file_path=item.file_path,
                reason="Generated tests passed single-test validation",
                priority_rank=item.priority_rank,
            )
            if self.config.auto_integrate_successful_worktrees:
                try:
                    toolset.integrate_worktree_result(commit_hash)
                    decision.status = "integrated"
                    integration_status = "integrated"
                except Exception as exc:
                    decision.status = "integration_failed"
                    decision.reason = str(exc)
                    integration_status = "integration_failed"
            self._append_integration(workspace, decision)
        if not final_summary:
            latest_output = attempts[-1].failure_summary if attempts else "No attempts executed."
            final_summary = runtime.reflect("final summary", latest_output, "\n".join(prior_failures))
        return SubagentResult(
            subagent_id=subagent_id,
            file_path=item.file_path,
            status="passed" if commit_hash else "failed",
            worktree_path=worktree_path,
            branch_name=branch_name,
            commit_hash=commit_hash or None,
            generated_test_files=[attempt.generated_test_file for attempt in attempts if attempt.generated_test_file],
            attempts=attempts,
            final_summary=final_summary,
            integration_status=integration_status,
            error_message="" if commit_hash else (attempts[-1].failure_summary if attempts else "No attempts executed."),
            coverage_after=coverage_after,
            coverage_delta=coverage_delta,
            missed_line_reduction=missed_line_reduction,
        )

    def _build_logger(self, run_id: str, workspace: RunWorkspace) -> RunLogger:
        return RunLogger(run_id, workspace.logs_dir, secrets=[self.config.gitlab_token, self.config.model.api_key])

    def _repo_name(self, repo_url: str) -> str:
        tail = repo_url.rstrip("/").split("/")[-1]
        name = tail[:-4] if tail.endswith(".git") else tail
        return slugify(name)[:40] or "repo"

    def _suggest_test_path(self, worktree_root: Path, source_file_path: str, iteration: int) -> str:
        source = Path(source_file_path)
        parts = list(source.parts)
        if "main" in parts:
            parts[parts.index("main")] = "test"
        elif "src" in parts:
            parts = ["src", "test", "java", *parts[1:]]
        if parts and parts[-1].endswith(".java"):
            parts[-1] = f"{Path(parts[-1]).stem}GeneratedTestIter{iteration}.java"
        else:
            parts.append(f"GeneratedTestIter{iteration}.java")
        return str((worktree_root / Path(*parts)).resolve())

    def _subagent_objective(
        self,
        repo_context: RepoContext,
        item: FileWorkItem,
        suggested_test_path: str,
        iteration: int,
        prior_failures: list[str],
        memory_insights: list[str],
        baseline_summary: GlobalCoverageSummary,
        coverage_context_path: Path,
    ) -> str:
        failure_text = "\n".join(f"- {failure}" for failure in prior_failures[-3:])
        memory_text = "\n".join(f"- {insight}" for insight in memory_insights[:4])
        return (
            "You are a Java unit-test generation subagent working inside a Git worktree.\n"
            "Write meaningful tests only for the assigned source file.\n"
            "Rules:\n"
            "- Create only a new test file.\n"
            "- Do not modify production code.\n"
            "- Do not modify existing tests.\n"
            "- Use folder/file search and file reads before writing.\n"
            "- Pass the full absolute file path to write_new_test_file.\n"
            "- Run the single test after writing.\n"
            f"Assigned file: {item.file_path}\n"
            f"Coverage context artifact: {coverage_context_path}\n"
            f"Global coverage baseline: {baseline_summary.coverage_percent}% "
            f"(covered={baseline_summary.covered_lines}, missed={baseline_summary.missed_lines})\n"
            f"Coverage: {item.coverage_percent}% with missed lines {item.missed_line_numbers}\n"
            f"Shared memory:\n{memory_text or '- none yet'}\n"
            f"Suggested test path: {suggested_test_path}\n"
            f"Iteration: {iteration}\n"
            f"Prior failures:\n{failure_text or '- none'}"
        )

    def _run_memory_path(self, workspace: RunWorkspace) -> Path:
        return workspace.artifacts_dir / "memory.json"

    def _pause_requested(self, workspace: RunWorkspace) -> bool:
        return (workspace.control_dir / "pause.requested").exists()

    def _save_checkpoint(
        self,
        store: CheckpointStore,
        repo_context: RepoContext,
        *,
        phase: str,
        pending_work_items: list[FileWorkItem],
        completed_results: list[SubagentResult],
        pending_integrations: list[IntegrationDecision],
        paused: bool,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        existing = store.load()
        checkpoint = RunCheckpoint(
            run_id=repo_context.run_id,
            phase=phase,
            repo_url=repo_context.repo_url,
            repo_name=repo_context.repo_name,
            paused=paused,
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
            pending_work_items=pending_work_items,
            completed_results=completed_results,
            pending_integrations=pending_integrations,
            metadata={
                **(existing.metadata if existing else {}),
                "clone_path": str(repo_context.clone_path),
                "source_type": repo_context.source_type,
                "module_paths": repo_context.module_paths,
                **(extra_metadata or {}),
            },
        )
        store.save(checkpoint)

    def _append_integration(self, workspace: RunWorkspace, decision: IntegrationDecision) -> None:
        current = self._read_pending_integrations(workspace)
        current = [item for item in current if item.commit_hash != decision.commit_hash]
        current.append(decision)
        write_json(workspace.integrations_path, [item.to_json() for item in self._sort_integrations(current)])

    def _read_pending_integrations(self, workspace: RunWorkspace) -> list[IntegrationDecision]:
        from agentic_testgen.utils import read_json

        payload = read_json(workspace.integrations_path, default=[])
        return self._sort_integrations([IntegrationDecision(**item) for item in payload])

    def _apply_work_item_limit(self, work_items: list[FileWorkItem], max_files: int | None) -> list[FileWorkItem]:
        effective_max = self.config.max_files_per_run if max_files is None else max_files
        if effective_max is None or effective_max <= 0:
            return work_items
        return work_items[:effective_max]

    def _sort_integrations(self, decisions: list[IntegrationDecision]) -> list[IntegrationDecision]:
        return sorted(decisions, key=lambda item: (item.priority_rank or 0, item.file_path, item.commit_hash))

    def rerun_after_merge_coverage(self, run_id: str) -> CoverageComparison | None:
        workspace = self.workspace_manager.create(run_id)
        checkpoint_store = CheckpointStore(workspace.checkpoints_dir)
        checkpoint = checkpoint_store.load()
        if not checkpoint:
            raise ValueError(f"No checkpoint found for {run_id}")
        baseline_payload = checkpoint.metadata.get("baseline_coverage")
        if not baseline_payload:
            return None
        baseline_summary = GlobalCoverageSummary(**baseline_payload)
        baseline_records = [
            CoverageRecord(**item) for item in checkpoint.metadata.get("baseline_coverage_records", [])
        ]
        repo_context = self._make_repo_context(
            run_id=run_id,
            repo_url=checkpoint.repo_url,
            repo_name=checkpoint.repo_name,
            clone_path=Path(checkpoint.metadata["clone_path"]),
            workspace=workspace,
            source_type=checkpoint.metadata.get("source_type", "gitlab"),
            module_paths=checkpoint.metadata.get("module_paths", []),
        )
        logger = self._build_logger(run_id, workspace)
        comparison = self._rerun_post_merge_coverage(
            checkpoint_store=checkpoint_store,
            repo_context=repo_context,
            workspace=workspace,
            logger=logger,
            baseline_summary=baseline_summary,
            baseline_records=baseline_records,
        )
        if comparison:
            self._refresh_reports_from_checkpoint(
                checkpoint=checkpoint_store.load() or checkpoint,
                repo_context=repo_context,
                workspace=workspace,
                coverage_comparison=comparison,
            )
        return comparison

    def _finalize_after_merge(
        self,
        *,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        baseline_summary: GlobalCoverageSummary,
        baseline_records: list[CoverageRecord],
    ) -> CoverageComparison | None:
        report_writer = ReportWriter(workspace.artifacts_dir)
        with logger.step("coverage.after_merge", details={"repo_root": str(repo_context.clone_path)}) as step:
            result, records, maven_log_paths = self.coverage.run_tests_with_coverage(
                repo_context.clone_path,
                maven_logs_dir=workspace.logs_dir / "maven",
                log_prefix="after-merge-project-coverage",
            )
            after_summary = self.coverage.summarize_global_coverage(records)
            comparison = self.coverage.compare_global_coverage(baseline_summary, after_summary)
            comparison_path = report_writer.write_coverage_comparison(comparison)
            file_rows = self._compare_file_coverage(baseline_records, records)
            file_json_path, file_md_path, file_csv_path = report_writer.write_file_coverage_comparison(file_rows)
            step["summary"] = f"Coverage increased by {comparison.percentage_increase}%"
            step["exit_code"] = result.exit_code
            step["maven_log_paths"] = maven_log_paths
            step["comparison_path"] = str(comparison_path)
            step["file_coverage_comparison_json"] = str(file_json_path)
            step["file_coverage_comparison_md"] = str(file_md_path)
            step["file_coverage_comparison_csv"] = str(file_csv_path)
            return comparison

    def _rerun_post_merge_coverage(
        self,
        *,
        checkpoint_store: CheckpointStore,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        baseline_summary: GlobalCoverageSummary,
        baseline_records: list[CoverageRecord],
    ) -> CoverageComparison | None:
        comparison = self._finalize_after_merge(
            repo_context=repo_context,
            workspace=workspace,
            logger=logger,
            baseline_summary=baseline_summary,
            baseline_records=baseline_records,
        )
        if comparison:
            self._persist_coverage_comparison_metadata(
                checkpoint_store,
                comparison,
                str(workspace.artifacts_dir / "coverage-comparison.md"),
                str(workspace.artifacts_dir / "file-coverage-comparison.json"),
                str(workspace.artifacts_dir / "file-coverage-comparison.md"),
                str(workspace.artifacts_dir / "file-coverage-comparison.csv"),
            )
        return comparison

    def _persist_coverage_comparison_metadata(
        self,
        checkpoint_store: CheckpointStore,
        comparison: CoverageComparison,
        comparison_path: str,
        file_comparison_json_path: str,
        file_comparison_markdown_path: str,
        file_comparison_csv_path: str,
    ) -> None:
        checkpoint = checkpoint_store.load()
        if not checkpoint:
            return
        checkpoint.metadata["after_merge_coverage"] = comparison.after.to_json()
        checkpoint.metadata["coverage_percentage_increase"] = comparison.percentage_increase
        checkpoint.metadata["coverage_comparison_path"] = comparison_path
        checkpoint.metadata["file_coverage_comparison_json_path"] = file_comparison_json_path
        checkpoint.metadata["file_coverage_comparison_markdown_path"] = file_comparison_markdown_path
        checkpoint.metadata["file_coverage_comparison_csv_path"] = file_comparison_csv_path
        checkpoint_store.save(checkpoint)

    def _refresh_reports_from_checkpoint(
        self,
        *,
        checkpoint: RunCheckpoint,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        coverage_comparison: CoverageComparison | None = None,
    ) -> tuple[Path, Path]:
        work_items = self._work_items_from_checkpoint(checkpoint)
        return self._write_reports(
            repo_context=repo_context,
            workspace=workspace,
            work_items=work_items,
            results=checkpoint.completed_results,
            coverage_comparison=coverage_comparison,
        )

    def _coverage_comparison_from_metadata(self, metadata: dict[str, Any]) -> CoverageComparison | None:
        before_payload = metadata.get("baseline_coverage")
        after_payload = metadata.get("after_merge_coverage")
        if not before_payload or not after_payload:
            return None
        return CoverageComparison(
            before=GlobalCoverageSummary(**before_payload),
            after=GlobalCoverageSummary(**after_payload),
            percentage_increase=float(metadata.get("coverage_percentage_increase", 0.0)),
            covered_line_increase=after_payload.get("covered_lines", 0) - before_payload.get("covered_lines", 0),
            missed_line_reduction=before_payload.get("missed_lines", 0) - after_payload.get("missed_lines", 0),
        )

    def _work_items_from_checkpoint(self, checkpoint: RunCheckpoint) -> list[FileWorkItem]:
        payload = checkpoint.metadata.get("all_work_items")
        if payload:
            return [FileWorkItem(**item) for item in payload]
        return checkpoint.pending_work_items[:]

    def _run_daddy_react(
        self,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        runtime: DSPyRuntime,
        work_items: list[FileWorkItem],
    ) -> None:
        toolset = SafeToolset(
            ToolContext(
                run_id=repo_context.run_id,
                repo_root=repo_context.clone_path,
                clone_root=repo_context.clone_path,
                worktrees_root=workspace.worktrees_dir,
                config=self.config,
                logger=logger,
                subagent_id="daddy",
            )
        )
        objective = (
            "Analyze the repository and confirm the highest-value source files for new test generation.\n"
            f"Top candidates: {[item.file_path for item in work_items[:5]]}\n"
            "Use the read/search tools as needed and summarize the best testing opportunities."
        )
        try:
            react = dspy.ReAct("objective -> answer", tools=toolset.build_repo_dspy_tools(), max_iters=4)
            prediction = react(objective=objective)
            logger.log_trace(
                {
                    "run_id": repo_context.run_id,
                    "subagent_id": "daddy",
                    "objective": objective,
                    "trajectory": getattr(prediction, "trajectory", {}),
                    "answer": getattr(prediction, "answer", str(prediction)),
                }
            )
        except Exception as exc:
            logger.log_event("daddy.react", "failed", summary=str(exc))

    def _ensure_git_repository(self, repo_root: Path, logger: RunLogger) -> None:
        if not (repo_root / ".git").exists():
            from agentic_testgen.utils import run_command

            run_command(["git", "init"], cwd=repo_root)
            run_command(["git", "config", "user.email", "agentic-testgen@example.com"], cwd=repo_root)
            run_command(["git", "config", "user.name", "Agentic Testgen"], cwd=repo_root)
            run_command(["git", "add", "."], cwd=repo_root)
            run_command(["git", "commit", "-m", "Initial fixture snapshot"], cwd=repo_root)
            logger.log_event("git.init", "completed", summary=f"Initialized fixture repo at {repo_root}")
        else:
            from agentic_testgen.utils import run_command

            run_command(["git", "config", "user.email", "agentic-testgen@example.com"], cwd=repo_root)
            run_command(["git", "config", "user.name", "Agentic Testgen"], cwd=repo_root)
