from __future__ import annotations

import json
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from agentic_testgen.execution.checkpointing import CheckpointStore
from agentic_testgen.core.config import AppConfig
from agentic_testgen.analysis.coverage import CoverageAnalyzer
from agentic_testgen.agents.custom_react import CustomReAct
from agentic_testgen.agents.dspy_runtime import DSPyRuntime
from agentic_testgen.agents.repo_analysis_agent import RepoAnalysisAgent
from agentic_testgen.agents.test_writing_agent import TestWritingAgent
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.execution.memory import MemoryManager
from agentic_testgen.core.models import (
    AttemptRecord,
    FileWorkItem,
    IntegrationDecision,
    RepoContext,
    SubagentResult,
)
from agentic_testgen.execution.tools import SafeToolset, ToolContext
from agentic_testgen.integrations.tracing import MlflowTracer
from agentic_testgen.core.utils import new_run_id, prompt_hash, read_json, utc_timestamp, write_json
from agentic_testgen.execution.workspace import RunWorkspace

PROMPT_VERSION = "orchestrator_reflective_v1"


class SubagentDispatcher:
    def __init__(self, config: AppConfig, memory: MemoryManager, coverage: CoverageAnalyzer):
        self.config = config
        self.memory = memory
        self.coverage = coverage
        self._clone_lock = threading.Lock()

    def dispatch(
        self,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        runtime: DSPyRuntime,
        tracer: MlflowTracer,
        work_items: list[FileWorkItem],
        existing_results: list[SubagentResult],
        checkpoint_store: CheckpointStore,
        coverage_context_path: Path,
    ) -> list[SubagentResult]:
        results: list[SubagentResult] = []
        completed_files = {item.file_path for item in existing_results}
        pending_queue = self._dedupe_work_items(work_items, exclude_files=completed_files)
        in_flight: dict = {}
        in_flight_files: set[str] = set()
        run_memory_path = self._run_memory_path(workspace)
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_parallel_subagents)) as pool:
            while pending_queue and len(in_flight) < max(1, self.config.max_parallel_subagents):
                item = pending_queue.pop(0)
                if item.file_path in in_flight_files:
                    continue
                if self._pause_requested(workspace):
                    break
                item.assigned_subagent_id = item.assigned_subagent_id or f"subagent_{item.priority_rank:03d}"
                future = pool.submit(
                    self._run_subagent,
                    repo_context,
                    workspace,
                    logger,
                    runtime,
                    tracer,
                    item,
                    checkpoint_store,
                    coverage_context_path,
                )
                in_flight[future] = item
                in_flight_files.add(item.file_path)
            while in_flight:
                done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    item = in_flight.pop(future)
                    in_flight_files.discard(item.file_path)
                    result = future.result()
                    self.memory.record_result(run_memory_path, repo_context, item, result)
                    results.append(result)
                    completed_files.add(result.file_path)
                    while pending_queue and len(in_flight) < max(1, self.config.max_parallel_subagents):
                        next_item = pending_queue.pop(0)
                        if next_item.file_path in completed_files or next_item.file_path in in_flight_files:
                            continue
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
                            tracer,
                            next_item,
                            checkpoint_store,
                            coverage_context_path,
                        )
                        in_flight[next_future] = next_item
                        in_flight_files.add(next_item.file_path)
                    pending_items = pending_queue + list(in_flight.values())
                    checkpoint_store.build_and_save(
                        repo_context,
                        phase="subagents_running",
                        pending_work_items=pending_items,
                        completed_results=existing_results + results,
                        pending_integrations=self.read_pending_integrations(workspace),
                        paused=self._pause_requested(workspace),
                    )
        return sorted(results, key=lambda item: item.subagent_id)

    def _run_subagent(
        self,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        runtime: DSPyRuntime,
        tracer: MlflowTracer,
        item: FileWorkItem,
        checkpoint_store: CheckpointStore,
        coverage_context_path: Path,
    ) -> SubagentResult:
        subagent_id = item.assigned_subagent_id or new_run_id("subagent")
        branch_name = f"subagent/{subagent_id}"
        worktree_path = workspace.worktrees_dir / subagent_id
        tool_context = ToolContext(
            run_id=repo_context.run_id,
            repo_root=worktree_path,
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
            return SubagentResult(
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

        prior_failures: list[str] = []
        # failure_context carries the structured FailureAnalysisMessage JSON
        # from TestWritingAgent back to RepoAnalysisAgent on each retry.
        failure_context = ""
        memory_insights = self.memory.lessons_for_item(self._run_memory_path(workspace), repo_context, item)
        missed_code_snippets = self._missed_code_snippets(worktree_path, item)

        # Build specialised agents once; they share the same toolset / ToolContext.
        analysis_pv = _resolve_prompt_version(self.config, "analysis")
        writing_pv = _resolve_prompt_version(self.config, "writing")
        analysis_agent = RepoAnalysisAgent(
            toolset=toolset,
            repo_root=worktree_path,
            max_iters=self.config.max_react_iters_analysis,
            prompt_version=analysis_pv,
        )
        writing_agent = TestWritingAgent(
            toolset=toolset,
            repo_root=worktree_path,
            max_iters=self.config.max_react_iters_subagent,
            prompt_version=writing_pv,
        )

        for iteration in range(1, self.config.max_subagent_iterations + 1):
            suggested_test_path = self._suggest_test_path(worktree_path, item.file_path, item.module, iteration)
            generated_file = ""
            status = "failed"
            failure_analysis = ""

            with logger.step(
                "subagent.iteration",
                subagent_id=subagent_id,
                file_path=item.file_path,
                iteration=iteration,
                details={"suggested_test_path": suggested_test_path},
            ) as step:
                # ── Phase 1: RepoAnalysisAgent ─────────────────────────────
                # Read-only exploration; on retry, guided by failure_context.
                with logger.step(
                    "subagent.analysis",
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                ) as analysis_step:
                    analysis_summary = analysis_agent.run(
                        file_path=item.file_path,
                        failure_context=failure_context,
                        memory_insights=memory_insights,
                        missed_code_snippets=missed_code_snippets,
                    )
                    tracer.tag_last_trace(
                        {
                            "run_id": repo_context.run_id,
                            "workflow": "orchestrator_reflective",
                            "dspy_call": "subagent.analysis",
                            "subagent_id": subagent_id,
                            "iteration": str(iteration),
                            "file_path": item.file_path,
                        }
                    )
                    analysis_step["summary"] = (
                        f"Analysis complete — gaps: {bool(analysis_summary.coverage_gaps)}"
                    )
                    analysis_step["has_few_shot_examples"] = bool(analysis_summary.few_shot_examples)

                # ── Phase 2: TestWritingAgent ──────────────────────────────
                # Writes one test file and runs it; returns structured result.
                with logger.step(
                    "subagent.writing",
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                ) as writing_step:
                    writing_result = writing_agent.run(
                        analysis_summary=analysis_summary,
                        file_path=item.file_path,
                        suggested_test_path=suggested_test_path,
                        iteration=iteration,
                    )
                    tracer.tag_last_trace(
                        {
                            "run_id": repo_context.run_id,
                            "workflow": "orchestrator_reflective",
                            "dspy_call": "subagent.writing",
                            "subagent_id": subagent_id,
                            "iteration": str(iteration),
                            "file_path": item.file_path,
                        }
                    )
                    writing_step["summary"] = f"Writing {writing_result.status}"
                    writing_step["generated_file"] = writing_result.generated_file

                generated_file = writing_result.generated_file
                status = writing_result.status
                validation_output = writing_result.validation_output
                iteration_created_test_count = writing_result.created_test_count
                iteration_successful_test_count = writing_result.successful_test_count

                # ── Structured handoff: TestWritingAgent → RepoAnalysisAgent
                if writing_result.failure_message is not None:
                    failure_context = json.dumps(
                        writing_result.failure_message.to_json(), default=str
                    )
                    failure_analysis = writing_result.failure_message.error_message
                    logger.log_trace(
                        {
                            "run_id": repo_context.run_id,
                            "subagent_id": subagent_id,
                            "iteration": iteration,
                            "file_path": item.file_path,
                            "handoff": "TestWritingAgent -> RepoAnalysisAgent",
                            "failure_message": writing_result.failure_message.to_json(),
                        }
                    )
                else:
                    failure_context = ""
                    failure_analysis = ""

                tool_summary = json.dumps(
                    analysis_summary.to_json(), default=str
                )[:4000]
                reflective_summary = (validation_output or failure_analysis or "")[:4000]

                failure_memory = failure_analysis or (
                    validation_output.splitlines()[0] if validation_output else "Failure cause unavailable."
                )
                attempt = AttemptRecord(
                    run_id=repo_context.run_id,
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                    prompt_version=PROMPT_VERSION,
                    prompt_hash=prompt_hash(analysis_summary.to_context()),
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
                    failure_summary="" if status == "passed" else failure_memory,
                    reflective_summary=reflective_summary[:4000],
                    failure_analysis=failure_analysis,
                    created_test_count=iteration_created_test_count,
                    successful_test_count=iteration_successful_test_count,
                    candidate_count=max(1, len([f for f in tool_context.written_files if f == generated_file or not generated_file])),
                )
                attempts.append(attempt)
                item.status = status
                step["summary"] = f"Iteration {iteration} {status}"
                step["attempt_status"] = status
                step["generated_test_file"] = generated_file or ""
                step["candidate_count"] = attempt.candidate_count
                step["created_test_count"] = attempt.created_test_count
                step["successful_test_count"] = attempt.successful_test_count
                step["failure_feedback"] = attempt.failure_summary[:800]
                step["reflective_summary"] = attempt.reflective_summary[:800]
                step["failure_analysis"] = attempt.failure_analysis[:800]
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
                        "failure_analysis": attempt.failure_analysis[:1500],
                        "single_test_command": attempt.single_test_command,
                    },
                )
                logger.log_event(
                    "subagent.iteration.test_counts",
                    "completed",
                    summary=(
                        f"{subagent_id} iteration {iteration} created={attempt.created_test_count} "
                        f"successful={attempt.successful_test_count}"
                    ),
                    subagent_id=subagent_id,
                    file_path=item.file_path,
                    iteration=iteration,
                    details={
                        "candidate_count": attempt.candidate_count,
                        "created_test_count": attempt.created_test_count,
                        "successful_test_count": attempt.successful_test_count,
                        "test_success_ratio": (
                            round(attempt.successful_test_count / attempt.created_test_count, 4)
                            if attempt.created_test_count
                            else 0.0
                        ),
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
                    tool_context.written_files = [generated_file]
                    commit_hash = toolset.commit_worktree_change(
                        f"Add generated tests for {Path(item.file_path).name}"
                    )
                    final_summary = reflective_summary
                    break
                prior_failures.append(failure_memory)
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
                    "last_failure_analysis": (attempts[-1].failure_analysis[:1500] if attempts else ""),
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
            with self._clone_lock:
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
            latest_output = attempts[-1].failure_analysis if attempts and attempts[-1].failure_analysis else (
                attempts[-1].failure_summary if attempts else "No attempts executed."
            )
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

    def _suggest_test_path(self, worktree_root: Path, source_file_path: str, module: str, iteration: int) -> str:
        source = Path(source_file_path)
        module_root = Path(module) if module and module != "." else Path(".")
        try:
            relative_to_module = source.relative_to(module_root)
        except ValueError:
            relative_to_module = source
        parts = list(relative_to_module.parts)
        if "main" in parts:
            parts[parts.index("main")] = "test"
        else:
            parts = ["src", "test", "java", *parts]
        if parts and parts[-1].endswith(".java"):
            parts[-1] = f"{Path(parts[-1]).stem}GeneratedTestIter{iteration}.java"
        else:
            parts.append(f"GeneratedTestIter{iteration}.java")
        return str((worktree_root / module_root / Path(*parts)).resolve())

    def _subagent_objective(
        self,
        repo_context: RepoContext,
        item: FileWorkItem,
        suggested_test_path: str,
        iteration: int,
        prior_failures: list[str],
        memory_insights: list[str],
        coverage_context_path: Path,
        missed_code_snippets: list[str],
    ) -> str:
        failure_text = "\n".join(f"- {failure}" for failure in prior_failures[-3:])
        memory_text = "\n".join(f"- {insight}" for insight in memory_insights[:4])
        missed_code_text = "\n".join(f"- {snippet}" for snippet in missed_code_snippets[:10])
        return (
            "You are a Java unit-test generation subagent working inside a Git worktree.\n"
            "Write fresh, meaningful behavior-driven tests only for the assigned source file.\n"
            "Rules:\n"
            "- Create only a new test file.\n"
            "- Do not modify production code.\n"
            "- Do not modify existing tests.\n"
            "- Do not depend on existing tests; create fresh test cases irrespective of existing tests.\n"
            "- Use folder/file search and file reads before writing.\n"
            "- Pass the full absolute file path to write_new_test_file.\n"
            "- Run the single test after writing.\n"
            f"Assigned file: {item.file_path}\n"
            f"Coverage context artifact: {coverage_context_path}\n"
            f"Current file coverage: {item.coverage_percent}%\n"
            f"Uncovered code snippets in assigned file:\n{missed_code_text or '- none available'}\n"
            "Target methods and branches represented by these uncovered snippets first.\n"
            f"Shared memory:\n{memory_text or '- none yet'}\n"
            f"Suggested test path: {suggested_test_path}\n"
            f"Iteration: {iteration}\n"
            f"Prior failure context (why):\n{failure_text or '- none'}"
        )

    def _missed_code_snippets(self, repo_root: Path, item: FileWorkItem, *, limit: int = 20) -> list[str]:
        source_path = (repo_root / item.file_path).resolve()
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        snippets: list[str] = []
        seen: set[str] = set()
        for line_number in item.missed_line_numbers:
            if line_number < 1 or line_number > len(lines):
                continue
            snippet = lines[line_number - 1].strip()
            if not snippet:
                continue
            if snippet not in seen:
                snippets.append(snippet[:300])
                seen.add(snippet)
            if len(snippets) >= limit:
                break
        return snippets

    def _dedupe_work_items(
        self,
        work_items: list[FileWorkItem],
        *,
        exclude_files: set[str] | None = None,
    ) -> list[FileWorkItem]:
        seen: set[str] = set(exclude_files or set())
        deduped: list[FileWorkItem] = []
        for item in sorted(work_items, key=lambda value: (value.priority_rank or 0, value.file_path)):
            if item.file_path in seen:
                continue
            seen.add(item.file_path)
            deduped.append(item)
        return deduped

    def _pause_requested(self, workspace: RunWorkspace) -> bool:
        return (workspace.control_dir / "pause.requested").exists()

    def _append_integration(self, workspace: RunWorkspace, decision: IntegrationDecision) -> None:
        current = self.read_pending_integrations(workspace)
        current = [
            item
            for item in current
            if item.commit_hash != decision.commit_hash and item.file_path != decision.file_path
        ]
        current.append(decision)
        write_json(workspace.integrations_path, [item.to_json() for item in self._sort_integrations(current)])

    def read_pending_integrations(self, workspace: RunWorkspace) -> list[IntegrationDecision]:
        payload = read_json(workspace.integrations_path, default=[])
        return self._sort_integrations([IntegrationDecision(**item) for item in payload])

    def _sort_integrations(self, decisions: list[IntegrationDecision]) -> list[IntegrationDecision]:
        return sorted(decisions, key=lambda item: (item.priority_rank or 0, item.file_path, item.commit_hash))

    def _run_memory_path(self, workspace: RunWorkspace) -> Path:
        return workspace.artifacts_dir / "memory.json"

    @staticmethod
    def _aggregate_iteration_test_counts(candidate_metrics: list[tuple[int, int]]) -> tuple[int, int]:
        created_total = 0
        successful_total = 0
        for created_count, passing_count in candidate_metrics:
            normalized_created = max(0, int(created_count))
            normalized_passing = max(0, int(passing_count))
            created_total += normalized_created
            successful_total += min(normalized_created, normalized_passing)
        return created_total, successful_total


def _resolve_prompt_version(config, agent: str):
    """Look up the requested prompt version for *agent* from PromptRegistry.

    Resolution order matches AppConfig.prompt_version_*:
        ""        → none (use in-source default Signature)
        "latest"  → registry.latest(agent)
        "pinned"  → registry.pinned(agent)
        otherwise → registry.load(agent, version)
    Returns None on miss so agents fall back to the default Signature.
    """
    from agentic_testgen.core.prompt_registry import PromptRegistry

    requested = (
        config.prompt_version_analysis if agent == "analysis" else config.prompt_version_writing
    )
    if not requested:
        return None
    registry = PromptRegistry(config.workspace_root / "prompts")
    if requested == "latest":
        return registry.latest(agent)
    if requested == "pinned":
        return registry.pinned(agent)
    try:
        return registry.load(agent, requested)
    except (FileNotFoundError, OSError):
        return None
