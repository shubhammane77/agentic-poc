"""RepoAnalysisAgent — read-only first stage of the two-agent pipeline.

Responsibilities:
  1. Explore the repository with read-only tools.
  2. Extract class signatures, existing test patterns, coverage gaps, and
     few-shot examples from real test files.
  3. Return a structured AnalysisSummary JSON that TestWritingAgent consumes.
  4. On retry, accept a FailureAnalysisMessage from TestWritingAgent and focus
     the re-analysis on the suspected root cause (missing imports, incorrect
     assumptions, dependency issues, etc.).

The system prompt now lives in `RepoAnalysisSignature` (agents/signatures.py)
so GEPA can mutate it during self-improvement. Per-call context (failure
message, memory insights, missed snippets) flows through the
`runtime_context` InputField and stays out of GEPA's mutation surface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from agentic_testgen.agents.custom_react import CustomReAct
from agentic_testgen.agents.signatures import RepoAnalysisSignature
from agentic_testgen.core.models import AnalysisSummary
from agentic_testgen.core.prompt_registry import PromptVersion
from agentic_testgen.execution.tools import SafeToolset

logger = logging.getLogger(__name__)


class RepoAnalysisAgent:
    """Read-only agent: explores the repository and returns a structured AnalysisSummary.

    Called by TwoAgentPipeline before each test-writing attempt. On retry the
    caller supplies a FailureAnalysisMessage (as a JSON string) so the agent can
    re-focus the analysis on the suspected failure cause.
    """

    def __init__(
        self,
        toolset: SafeToolset,
        repo_root: Path,
        max_iters: int = 8,
        signature: type | None = None,
        prompt_version: "PromptVersion | None" = None,
    ) -> None:
        self.toolset = toolset
        self.repo_root = repo_root
        self.max_iters = max_iters
        # Signature (mutable instructions) is injectable so GEPA-optimized
        # variants can be loaded at runtime via the PromptRegistry.
        self.signature = signature or RepoAnalysisSignature
        self.prompt_version = prompt_version

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        file_path: str,
        failure_context: str = "",
        memory_insights: list[str] | None = None,
        missed_code_snippets: list[str] | None = None,
    ) -> AnalysisSummary:
        """Analyze the repository for *file_path* and return an AnalysisSummary.

        Args:
            file_path: Absolute path to the source file to analyze.
            failure_context: Serialized FailureAnalysisMessage JSON from a
                previous TestWritingAgent run.  Empty string on first attempt.
            memory_insights: Lessons learned from prior subagent runs in this
                workflow session (passed through from MemoryManager).
            missed_code_snippets: Lines of source code not yet covered by tests,
                extracted from coverage data.
        """
        if dspy is None or self.signature is None:
            return AnalysisSummary()

        runtime_context = self._build_runtime_context(
            failure_context, memory_insights or [], missed_code_snippets or []
        )

        try:
            react = CustomReAct(
                self.signature,
                tools=self.toolset.build_analysis_dspy_tools(),
                max_iters=self.max_iters,
            )
            if self.prompt_version is not None:
                _apply_prompt_version(self.prompt_version, react)
            prediction = react(
                file_path=file_path,
                repo_root=str(self.repo_root),
                runtime_context=runtime_context,
            )
            raw_json: str = getattr(prediction, "analysis_summary_json", "")
            return self._parse_analysis_summary(raw_json)
        except Exception as exc:
            logger.warning("RepoAnalysisAgent failed: %s", exc)
            return AnalysisSummary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runtime_context(
        self,
        failure_context: str,
        memory_insights: list[str],
        missed_code_snippets: list[str],
    ) -> str:
        parts: list[str] = []

        if missed_code_snippets:
            snippets_text = "\n".join(f"  - {s}" for s in missed_code_snippets[:10])
            parts.append(f"Uncovered code snippets (prioritise these):\n{snippets_text}")

        if memory_insights:
            insights_text = "\n".join(f"  - {i}" for i in memory_insights[:4])
            parts.append(f"Shared memory from prior runs:\n{insights_text}")

        if failure_context:
            parts.append(
                "FAILURE CONTEXT FROM PREVIOUS TEST ATTEMPT — re-analyse with this in mind:\n"
                + failure_context
                + "\nFocus on missing imports, incorrect assumptions, or dependency issues."
            )

        return "\n\n".join(parts) if parts else "(no additional runtime context)"

    def _parse_analysis_summary(self, raw: str) -> AnalysisSummary:  # noqa: D401
        return _parse_analysis_summary_impl(raw)


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


def _parse_analysis_summary_impl(raw: str) -> AnalysisSummary:
    """Parse the LLM's JSON string into an AnalysisSummary; fall back gracefully."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[1:end])
        data: dict = json.loads(cleaned)
        return AnalysisSummary(
            class_signatures=str(data.get("class_signatures", "")),
            dependencies=str(data.get("dependencies", "")),
            existing_test_patterns=str(data.get("existing_test_patterns", "")),
            coverage_gaps=str(data.get("coverage_gaps", "")),
            few_shot_examples=str(data.get("few_shot_examples", "")),
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        return AnalysisSummary(class_signatures=raw or "")
