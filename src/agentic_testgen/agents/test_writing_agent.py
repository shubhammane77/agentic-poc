"""TestWritingAgent — write-and-run second stage of the two-agent pipeline.

Responsibilities:
  1. Receive an AnalysisSummary from RepoAnalysisAgent.
  2. Write one new test file at suggested_test_path using write_new_test_file.
  3. Run the test via run_single_test.
  4. If tests fail, call the LLM to produce a structured FailureAnalysisMessage
     and return it so TwoAgentPipeline can hand it back to RepoAnalysisAgent.
  5. Repeat until the test passes or the caller's retry limit is reached.

The system prompt now lives in `TestWritingSignature` (agents/signatures.py)
so GEPA can mutate it during self-improvement. Per-call context (analysis
summary, iteration index) flows through the `analysis_context` InputField.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from agentic_testgen.agents.custom_react import CustomReAct
from agentic_testgen.agents.signatures import TestWritingSignature
from agentic_testgen.core.models import AnalysisSummary, FailureAnalysisMessage
from agentic_testgen.core.prompt_registry import PromptVersion
from agentic_testgen.execution.tools import SafeToolset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type returned to TwoAgentPipeline
# ---------------------------------------------------------------------------

@dataclass
class WritingResult:
    """Outcome of a single TestWritingAgent.run() call."""

    status: str                              # "passed" | "failed"
    generated_file: str                      # relative path recorded by SafeToolset
    validation_output: str                   # stdout+stderr from the test run
    created_test_count: int                  # @Test methods declared in the file
    successful_test_count: int               # tests that passed
    failure_message: FailureAnalysisMessage | None  # structured handoff on failure


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestWritingAgent:
    """Write-and-run agent: produces a test file and reports success or failure.

    On failure it uses a DSPy Predict call to generate a structured
    FailureAnalysisMessage that TwoAgentPipeline forwards to RepoAnalysisAgent
    for targeted re-analysis.
    """

    def __init__(
        self,
        toolset: SafeToolset,
        repo_root: Path,
        max_iters: int = 6,
        signature: type | None = None,
        prompt_version: PromptVersion | None = None,
    ) -> None:
        self.toolset = toolset
        self.repo_root = repo_root
        self.max_iters = max_iters
        # Signature (mutable instructions) is injectable so GEPA-optimized
        # variants can be loaded at runtime via the PromptRegistry.
        self.signature = signature or TestWritingSignature
        self.prompt_version = prompt_version

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_summary: AnalysisSummary,
        file_path: str,
        suggested_test_path: str,
        iteration: int,
    ) -> WritingResult:
        """Write and run tests for *file_path* guided by *analysis_summary*.

        Args:
            analysis_summary: Structured analysis from RepoAnalysisAgent.
            file_path: Absolute source file path to generate tests for.
            suggested_test_path: Absolute path where the test file should be
                written (pre-computed by SubagentDispatcher / TwoAgentPipeline).
            iteration: Current retry index (1-based), used in the prompt.
        """
        if dspy is None or self.signature is None:
            return WritingResult(
                status="failed",
                generated_file="",
                validation_output="DSPy not installed.",
                created_test_count=0,
                successful_test_count=0,
                failure_message=FailureAnalysisMessage(
                    failed_test_name="<none>",
                    error_message="DSPy not installed.",
                    suspected_cause="Runtime dependency missing.",
                ),
            )

        files_before = list(self.toolset.context.written_files)
        analysis_context = self._build_analysis_context(analysis_summary, iteration)

        try:
            react = CustomReAct(
                self.signature,
                tools=self.toolset.build_writing_dspy_tools(),
                max_iters=self.max_iters,
            )
            if self.prompt_version is not None:
                _apply_prompt_version(self.prompt_version, react)
            react(
                file_path=file_path,
                suggested_test_path=suggested_test_path,
                analysis_context=analysis_context,
            )
        except Exception as exc:
            logger.warning("TestWritingAgent react loop failed: %s", exc)
            return WritingResult(
                status="failed",
                generated_file="",
                validation_output=str(exc),
                created_test_count=0,
                successful_test_count=0,
                failure_message=FailureAnalysisMessage(
                    failed_test_name="<unknown>",
                    error_message=str(exc),
                    suspected_cause="Agent raised an unexpected exception.",
                ),
            )

        # Determine which files this iteration produced.
        new_files = [f for f in self.toolset.context.written_files if f not in files_before]

        if not new_files:
            failure_msg = FailureAnalysisMessage(
                failed_test_name="<none>",
                error_message="No test file was generated.",
                suspected_cause=(
                    "Agent did not call write_new_test_file.  "
                    "The analysis summary may be missing critical import or class information."
                ),
                requested_reanalysis=(
                    "Re-check the correct package declaration, imports, and test framework "
                    "being used in the project.  Verify the suggested_test_path is correct."
                ),
            )
            return WritingResult(
                status="failed",
                generated_file="",
                validation_output="No test file was generated.",
                created_test_count=0,
                successful_test_count=0,
                failure_message=failure_msg,
            )

        # Select the file with the highest passing-test count (robustness for
        # edge cases where the agent writes more than one candidate).
        best_file = new_files[0]
        best_passing = -1
        best_exit_code: int | None = None
        best_output = ""
        best_created = 0

        for candidate in new_files:
            # run_single_test updates tool_context counters as a side-effect.
            candidate_output = self.toolset.run_single_test(candidate)
            passing = self.toolset.context.last_single_test_passing_count
            exit_code = self.toolset.context.last_single_test_exit_code
            created = self.toolset.context.written_file_test_counts.get(candidate, 0)
            if passing > best_passing:
                best_file = candidate
                best_passing = passing
                best_exit_code = exit_code
                best_output = candidate_output
                best_created = created

        status = "passed" if best_exit_code == 0 else "failed"

        if status == "passed":
            return WritingResult(
                status="passed",
                generated_file=best_file,
                validation_output=best_output,
                created_test_count=best_created,
                successful_test_count=best_passing,
                failure_message=None,
            )

        # Build structured failure message for handoff back to RepoAnalysisAgent.
        failure_msg = self._analyze_failure(
            file_path=file_path,
            iteration=iteration,
            test_file=best_file,
            validation_output=best_output,
        )
        return WritingResult(
            status="failed",
            generated_file=best_file,
            validation_output=best_output,
            created_test_count=best_created,
            successful_test_count=best_passing,
            failure_message=failure_msg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_analysis_context(
        self,
        analysis_summary: AnalysisSummary,
        iteration: int,
    ) -> str:
        return (
            f"Repository root (prefix for all paths): {self.repo_root}\n"
            f"Iteration: {iteration}\n\n"
            f"## Analysis Summary\n\n{analysis_summary.to_context()}"
        )

    def _analyze_failure(
        self,
        file_path: str,
        iteration: int,
        test_file: str,
        validation_output: str,
    ) -> FailureAnalysisMessage:
        """Use a DSPy Predict call to produce a structured FailureAnalysisMessage."""
        test_name = Path(test_file).stem if test_file else "<none>"

        if dspy is None:
            return FailureAnalysisMessage(
                failed_test_name=test_name,
                error_message=validation_output[:500],
                suspected_cause="DSPy not available for failure analysis.",
            )

        try:
            program = dspy.Predict(
                "file_path, iteration, test_file, failure_output -> "
                "failed_test_name, error_message, suspected_cause, requested_reanalysis"
            )
            result = program(
                file_path=file_path,
                iteration=str(iteration),
                test_file=test_file,
                failure_output=validation_output[:8000],
            )
            return FailureAnalysisMessage(
                failed_test_name=getattr(result, "failed_test_name", test_name),
                error_message=getattr(result, "error_message", validation_output[:500]),
                suspected_cause=getattr(result, "suspected_cause", "Unknown"),
                requested_reanalysis=getattr(result, "requested_reanalysis", ""),
            )
        except Exception as exc:
            logger.warning("TestWritingAgent failure analysis LLM call failed: %s", exc)
            return FailureAnalysisMessage(
                failed_test_name=test_name,
                error_message=validation_output[:500],
                suspected_cause=str(exc),
            )


def _apply_prompt_version(version: PromptVersion, module) -> None:
    """Copy GEPA-evolved instructions onto a freshly built CustomReAct."""
    named = dict(module.named_predictors())
    for name, instructions in version.predictors.items():
        pred = named.get(name)
        if pred is None:
            continue
        try:
            pred.signature = pred.signature.with_instructions(instructions)
        except AttributeError:
            pred.signature.instructions = instructions
