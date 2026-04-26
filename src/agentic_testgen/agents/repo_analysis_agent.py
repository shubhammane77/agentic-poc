"""RepoAnalysisAgent — read-only first stage of the two-agent pipeline.

Responsibilities:
  1. Explore the repository with read-only tools.
  2. Extract class signatures, existing test patterns, coverage gaps, and
     few-shot examples from real test files.
  3. Return a structured AnalysisSummary JSON that TestWritingAgent consumes.
  4. On retry, accept a FailureAnalysisMessage from TestWritingAgent and focus
     the re-analysis on the suspected root cause (missing imports, incorrect
     assumptions, dependency issues, etc.).

Absolute-path contract (enforced in system prompt and tool docstrings):
  All file paths passed to tools MUST be absolute paths.
  Unix: starts with /   (e.g. /home/user/project/src/main/java/Foo.java)
  Windows: starts with a drive letter  (e.g. C:\\Users\\user\\project\\src\\...)
  Relative paths are accepted by the underlying tools as a fallback, but the
  agent is instructed to always supply absolute paths.
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
from agentic_testgen.core.models import AnalysisSummary
from agentic_testgen.execution.tools import SafeToolset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — injected as the DSPy Signature instructions so it appears
# in every LLM call made by CustomReAct.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a read-only repository analysis agent. Your only job is to gather
information — you MUST NOT write or modify any files.

ABSOLUTE PATH REQUIREMENT:
  All file paths passed to tools must be absolute paths.
  Unix  : starts with /   e.g. /home/user/project/src/main/java/Foo.java
  Windows: starts with a drive letter  e.g. C:\\Users\\user\\project\\...
  Before every tool call, verify the path is absolute.  If it is not absolute,
  prepend the repository root:  REPO_ROOT + '/' + relative_path.

Analysis steps (in order):
  1. Call read_folder_structure with the repository root to understand layout.
  2. Call read_file on the target source file to read its full content.
  3. Call search_occurrences to locate existing test files for the class.
  4. Call read_file on 2-3 representative test files to extract few-shot examples.
  5. Search for dependency/import patterns used in existing tests.
  6. Finish and produce the structured analysis_summary_json output.

Output format for analysis_summary_json — valid JSON with these exact keys:
  {
    "class_signatures": "...",        // all classes/methods/params in source file
    "existing_test_patterns": "...",  // test file structure, naming, setup/teardown
    "coverage_gaps": "...",           // untested methods, branches, edge cases
    "few_shot_examples": "...",       // 2-3 representative test code snippets verbatim
    "dependencies": "..."             // required imports/frameworks found in test files
  }
"""


class RepoAnalysisAgent:
    """Read-only agent: explores the repository and returns a structured AnalysisSummary.

    Called by TwoAgentPipeline before each test-writing attempt.  On retry the
    caller supplies a FailureAnalysisMessage (as a JSON string) so the agent can
    re-focus the analysis on the suspected failure cause.
    """

    def __init__(
        self,
        toolset: SafeToolset,
        repo_root: Path,
        max_iters: int = 8,
    ) -> None:
        self.toolset = toolset
        self.repo_root = repo_root
        self.max_iters = max_iters

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
        if dspy is None:
            return AnalysisSummary()

        objective = self._build_objective(
            file_path, failure_context, memory_insights or [], missed_code_snippets or []
        )

        try:
            react = CustomReAct(
                "objective, file_path, repo_root -> analysis_summary_json",
                tools=self.toolset.build_analysis_dspy_tools(),
                max_iters=self.max_iters,
            )
            prediction = react(
                objective=objective,
                file_path=file_path,
                repo_root=str(self.repo_root),
            )
            raw_json: str = getattr(prediction, "analysis_summary_json", "")
            return self._parse_analysis_summary(raw_json)
        except Exception as exc:
            logger.warning("RepoAnalysisAgent failed: %s", exc)
            return AnalysisSummary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        file_path: str,
        failure_context: str,
        memory_insights: list[str],
        missed_code_snippets: list[str],
    ) -> str:
        parts = [
            _SYSTEM_PROMPT,
            f"Repository root (prefix for all paths): {self.repo_root}",
            f"Target source file: {file_path}",
        ]

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

        return "\n\n".join(parts)

    def _parse_analysis_summary(self, raw: str) -> AnalysisSummary:
        """Parse the LLM's JSON string into an AnalysisSummary; fall back gracefully."""
        try:
            cleaned = raw.strip()
            # Strip markdown code fences produced by some models.
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
            # The model produced non-JSON; surface whatever it said as signatures.
            return AnalysisSummary(class_signatures=raw or "")
