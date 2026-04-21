from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    import dspy
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]

from agentic_testgen.execution.checkpointing import CheckpointStore
from agentic_testgen.core.config import AppConfig
from agentic_testgen.analysis.coverage import CoverageAnalyzer
from agentic_testgen.agents.dspy_runtime import DSPyRuntime
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

PROMPT_VERSION = "daddy_subagents_reflective_v1"


class SubagentDispatcher:
    def __init__(self, config: AppConfig, memory: MemoryManager, coverage: CoverageAnalyzer):
        self.config = config
        self.memory = memory
        self.coverage = coverage

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
        memory_insights = self.memory.lessons_for_item(self._run_memory_path(workspace), repo_context, item)
        missed_code_snippets = self._missed_code_snippets(repo_context.clone_path, item)
        for iteration in range(1, self.config.max_subagent_iterations + 1):
            files_before_iteration = list(tool_context.written_files)
            suggested_test_path = self._suggest_test_path(worktree_path, item.file_path, iteration)
            objective = self._subagent_objective(
                repo_context,
                item,
                suggested_test_path,
                iteration,
                prior_failures,
                memory_insights,
                coverage_context_path,
                missed_code_snippets,
            )
            generated_file = ""
            tool_summary = ""
            validation_output = ""
            status = "failed"
            failure_analysis = ""
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
                tracer.tag_last_trace(
                    {
                        "run_id": repo_context.run_id,
                        "workflow": "daddy_subagents_reflective",
                        "dspy_call": "subagent.react",
                        "subagent_id": subagent_id,
                        "iteration": str(iteration),
                        "file_path": item.file_path,
                    }
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
                iteration_files = [f for f in tool_context.written_files if f not in files_before_iteration]
                if iteration_files:
                    best_file = iteration_files[0]
                    best_passing = -1
                    best_exit_code: int | None = None
                    validation_output = ""
                    for candidate in iteration_files:
                        candidate_output = toolset.run_single_test(candidate)
                        candidate_passing = tool_context.last_single_test_passing_count
                        candidate_exit_code = tool_context.last_single_test_exit_code
                        if candidate_passing > best_passing:
                            best_file = candidate
                            best_passing = candidate_passing
                            best_exit_code = candidate_exit_code
                            validation_output = candidate_output
                    generated_file = best_file
                    status = "passed" if best_exit_code == 0 else "failed"
                else:
                    generated_file = ""
                    validation_output = "No test file was generated."
                    status = "failed"
                tool_summary = json.dumps(trajectory, default=str)[:4000]
                reflective_summary = runtime.reflect(objective, validation_output, "\n".join(prior_failures))
                tracer.tag_last_trace(
                    {
                        "run_id": repo_context.run_id,
                        "workflow": "daddy_subagents_reflective",
                        "dspy_call": "subagent.reflect",
                        "subagent_id": subagent_id,
                        "iteration": str(iteration),
                        "file_path": item.file_path,
                    }
                )
                if status != "passed":
                    failure_analysis = runtime.analyze_failure(item.file_path, iteration, validation_output)
                    tracer.tag_last_trace(
                        {
                            "run_id": repo_context.run_id,
                            "workflow": "daddy_subagents_reflective",
                            "dspy_call": "subagent.failure_analyst",
                            "subagent_id": subagent_id,
                            "iteration": str(iteration),
                            "file_path": item.file_path,
                        }
                    )
                else:
                    failure_analysis = ""
                failure_memory = failure_analysis or (
                    validation_output.splitlines()[0] if validation_output else "Failure cause unavailable."
                )
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
                    failure_summary="" if status == "passed" else failure_memory,
                    reflective_summary=reflective_summary[:4000],
                    failure_analysis=failure_analysis,
                )
                attempts.append(attempt)
                item.status = status
                step["summary"] = f"Iteration {iteration} {status}"
                step["attempt_status"] = status
                step["generated_test_file"] = generated_file or ""
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
