from __future__ import annotations

import shutil
from pathlib import Path

from agentic_testgen.execution.checkpointing import CheckpointStore
from agentic_testgen.core.config import AppConfig
from agentic_testgen.analysis.coverage import CoverageAnalyzer, summarize_tree
from agentic_testgen.analysis.coverage_comparison import CoverageComparator
from agentic_testgen.agents.dspy_runtime import DSPyRuntime
from agentic_testgen.integrations.gitlab import GitLabRepositoryManager
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.execution.memory import MemoryManager
from agentic_testgen.core.models import (
    AttemptRecord,
    CoverageComparison,
    CoverageRecord,
    FileWorkItem,
    GlobalCoverageSummary,
    ModelDefinition,
    RepoContext,
    RunCheckpoint,
    SubagentResult,
    WorkflowRunResult,
)
from agentic_testgen.analysis.reporting import ReportWriter
from agentic_testgen.agents.subagent_dispatcher import SubagentDispatcher
from agentic_testgen.integrations.gitlab import sanitize_repo_url
from agentic_testgen.integrations.tracing import MlflowTracer
from agentic_testgen.core.utils import ensure_dir, new_run_id, prompt_hash, run_command, slugify, utc_timestamp, write_json
from agentic_testgen.execution.workspace import RunWorkspace, WorkspaceManager


class OrchestratorWorkflow:
    def __init__(self, config: AppConfig):
        self.config = config
        self.workspace_manager = WorkspaceManager(config.workspace_root)
        self.coverage = CoverageAnalyzer(config)
        self.memory = MemoryManager(config.workspace_root)
        self._dispatcher = SubagentDispatcher(config, self.memory, self.coverage)
        self._comparator = CoverageComparator(self.coverage)

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
            tags={"workflow": "orchestrator_reflective", "source_type": "gitlab", "run_id": run_id},
        ):
            tracer.log_params({"repo_url": repo_url, "repo_name": repo_name, "run_id": run_id})
            manager = GitLabRepositoryManager(self.config, logger)
            cached_repo = self._prepare_cached_repo(repo_url, repo_name, manager, logger)
            with logger.step("gitlab.copy_cached_repo", details={"source": str(cached_repo), "destination": str(clone_target)}) as step:
                self.workspace_manager.copy_local_repo(cached_repo, clone_target)
                step["summary"] = f"Copied cached repository into run workspace for {repo_name}"
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
            tags={"workflow": "orchestrator_reflective", "source_type": "fixture", "run_id": run_id},
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
        coverage_comparison = CoverageComparator.from_metadata(checkpoint.metadata)
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
            tags={"workflow": "orchestrator_reflective", "source_type": repo_context.source_type, "run_id": run_id},
        ):
            tracer.log_params({"run_id": run_id, "resume": True})
            new_results = self._dispatcher.dispatch(
                repo_context,
                workspace,
                logger,
                runtime,
                tracer,
                work_items,
                results,
                checkpoint_store,
                coverage_context_path,
            )
            results.extend(new_results)
            attempts = self._attempts_from_results(results)
            pending = self._dispatcher.read_pending_integrations(workspace)
            checkpoint_store.build_and_save(
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
            token_budget_path = self._write_token_budget_artifact(tracer, workspace, logger)
            tracer.log_artifact(token_budget_path)
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
            tracer.tag_last_trace(
                {
                    "run_id": run_id,
                    "workflow": "orchestrator_reflective",
                    "dspy_call": "orchestrator.overview",
                    "source_type": source_type,
                    "repo_name": repo_name,
                }
            )
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
        checkpoint_store.build_and_save(
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
                log_prefix="orchestrator-project-coverage",
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
        checkpoint_store.build_and_save(
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
            checkpoint_store.build_and_save(
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
        subagent_results = self._dispatcher.dispatch(
            repo_context,
            workspace,
            logger,
            runtime,
            tracer,
            work_items,
            [],
            checkpoint_store,
            coverage_context_markdown_path,
        )
        attempts = self._attempts_from_results(subagent_results)
        pending_integrations = self._dispatcher.read_pending_integrations(workspace)
        checkpoint_store.build_and_save(
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
            coverage_comparison = self._comparator.rerun_post_merge(
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
        token_budget_path = self._write_token_budget_artifact(tracer, workspace, logger)
        tracer.log_artifact(token_budget_path)
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
        comparison = self._comparator.rerun_post_merge(
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

    def _write_token_budget_artifact(
        self,
        tracer: MlflowTracer,
        workspace: RunWorkspace,
        logger: RunLogger,
    ) -> Path:
        token_budget = tracer.token_usage_summary()
        token_budget_path = workspace.artifacts_dir / "token-budget.json"
        write_json(token_budget_path, token_budget)
        logger.log_event(
            "tokens.summary",
            "completed",
            summary=f"total_tokens={token_budget.get('total_tokens', 0)}",
            details=token_budget,
        )
        tracer.log_metrics(
            {
                "token_input": token_budget.get("input_tokens", 0),
                "token_output": token_budget.get("output_tokens", 0),
                "token_total": token_budget.get("total_tokens", 0),
                "token_traces": token_budget.get("trace_count", 0),
            }
        )
        return token_budget_path

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

    def _work_items_from_checkpoint(self, checkpoint: RunCheckpoint) -> list[FileWorkItem]:
        payload = checkpoint.metadata.get("all_work_items")
        if payload:
            return [FileWorkItem(**item) for item in payload]
        return checkpoint.pending_work_items[:]

    def _apply_work_item_limit(self, work_items: list[FileWorkItem], max_files: int | None) -> list[FileWorkItem]:
        effective_max = self.config.max_files_per_run if max_files is None else max_files
        if effective_max is None or effective_max <= 0:
            return work_items
        return work_items[:effective_max]

    def _pause_requested(self, workspace: RunWorkspace) -> bool:
        return (workspace.control_dir / "pause.requested").exists()

    def _build_logger(self, run_id: str, workspace: RunWorkspace) -> RunLogger:
        return RunLogger(run_id, workspace.logs_dir, secrets=[self.config.gitlab_token, self.config.model.api_key])

    def _repo_name(self, repo_url: str) -> str:
        tail = repo_url.rstrip("/").split("/")[-1]
        name = tail[:-4] if tail.endswith(".git") else tail
        return slugify(name)[:40] or "repo"

    def _run_memory_path(self, workspace: RunWorkspace) -> Path:
        return workspace.artifacts_dir / "memory.json"

    def _cached_repo_path(self, repo_url: str, repo_name: str) -> Path:
        repo_key = f"{repo_name}-{prompt_hash(sanitize_repo_url(repo_url))}"
        return ensure_dir(self.config.workspace_root / "repo-cache") / repo_key

    def _prepare_cached_repo(
        self,
        repo_url: str,
        repo_name: str,
        manager: GitLabRepositoryManager,
        logger: RunLogger,
    ) -> Path:
        cached_repo = self._cached_repo_path(repo_url, repo_name)
        with logger.step("gitlab.clone_or_reuse_cache", details={"repo": repo_url, "cache_path": str(cached_repo)}) as step:
            if cached_repo.exists() and (cached_repo / ".git").exists():
                step["summary"] = f"Using cached clone for {repo_name}"
                return cached_repo
            if cached_repo.exists():
                shutil.rmtree(cached_repo)
            result = manager.clone(repo_url, cached_repo)
            if not result.ok:
                raise RuntimeError(result.stderr or result.stdout)
            step["summary"] = f"Cloned {repo_name} into local cache"
            self._install_maven_dependencies(cached_repo, logger)
            return cached_repo

    def _install_maven_dependencies(self, repo_root: Path, logger: RunLogger) -> None:
        with logger.step("repo.maven_install", details={"repo_root": str(repo_root)}) as step:
            if not (repo_root / "pom.xml").exists():
                step["summary"] = "Skipped Maven install because pom.xml was not found at repository root"
                step["status"] = "skipped"
                return
            env: dict[str, str] = {}
            if self.config.java_home:
                env["JAVA_HOME"] = self.config.java_home
            if self.config.maven_home:
                env["MAVEN_HOME"] = self.config.maven_home
            command = self.config.maven_command("install")
            result = run_command(command, cwd=repo_root, env=env or None)
            step["command"] = " ".join(command)
            step["exit_code"] = result.exit_code
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            if result.ok:
                step["summary"] = "Installed Maven dependencies in local repository"
                return
            step["summary"] = "Maven install failed; continuing workflow"
            step["status"] = "failed"

    def _ensure_git_repository(self, repo_root: Path, logger: RunLogger) -> None:
        if not (repo_root / ".git").exists():
            run_command(["git", "init"], cwd=repo_root)
            run_command(["git", "config", "user.email", "agentic-testgen@example.com"], cwd=repo_root)
            run_command(["git", "config", "user.name", "Agentic Testgen"], cwd=repo_root)
            run_command(["git", "add", "."], cwd=repo_root)
            run_command(["git", "commit", "-m", "Initial fixture snapshot"], cwd=repo_root)
            logger.log_event("git.init", "completed", summary=f"Initialized fixture repo at {repo_root}")
        else:
            run_command(["git", "config", "user.email", "agentic-testgen@example.com"], cwd=repo_root)
            run_command(["git", "config", "user.name", "Agentic Testgen"], cwd=repo_root)
