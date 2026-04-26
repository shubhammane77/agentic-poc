"""DSPy Signatures whose docstrings hold the agent system prompts.

GEPA mutates `signature.instructions` on the predictors inside CustomReAct. By
moving the system prompts here (as class docstrings), they become the
optimization target. Runtime context (file path, repo root, failure message,
memory insights, missed snippets) stays as InputFields and is therefore *not*
mutated by GEPA.

The default docstrings preserve the prompts that previously lived as
`_SYSTEM_PROMPT` constants in repo_analysis_agent.py and test_writing_agent.py
verbatim — Phase 1 must not change agent behaviour.
"""

from __future__ import annotations

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class RepoAnalysisSignature(dspy.Signature):
        """You are a read-only repository analysis agent. Your only job is to gather
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

        file_path: str = dspy.InputField(desc="Absolute path to the source file to analyse.")
        repo_root: str = dspy.InputField(desc="Absolute path to the repository root.")
        runtime_context: str = dspy.InputField(
            desc=(
                "Per-call context such as uncovered code snippets, memory insights, "
                "and any FailureAnalysisMessage handed back from a previous test "
                "writing attempt. Empty on the first call."
            )
        )
        analysis_summary_json: str = dspy.OutputField(
            desc="A JSON object with keys class_signatures, existing_test_patterns, coverage_gaps, few_shot_examples, dependencies."
        )

    class TestWritingSignature(dspy.Signature):
        """You are a Java unit-test writing agent working inside a Git worktree.
You have been given an analysis summary produced by RepoAnalysisAgent.

ABSOLUTE PATH REQUIREMENT:
  All file paths passed to tools must be absolute paths.
  Unix  : starts with /   e.g. /home/user/project/src/test/java/FooTest.java
  Windows: starts with a drive letter  e.g. C:\\Users\\user\\project\\...
  Before every tool call, verify the path is absolute.  If it is not absolute,
  prepend the repository root:  REPO_ROOT + '/' + relative_path.

Rules:
  - Create only ONE new test file at the exact suggested_test_path (absolute).
  - Do NOT modify any production source files.
  - Do NOT modify any existing test files.
  - Follow the test patterns and few-shot examples in the analysis summary.
  - Target the coverage gaps listed in the analysis summary.
  - After writing, call run_single_test with the absolute path of the test file.
  - Do not call finish until after run_single_test has been called.
"""

        file_path: str = dspy.InputField(desc="Absolute path to the source file to test.")
        suggested_test_path: str = dspy.InputField(
            desc="Absolute path where the test file should be written. Use exactly."
        )
        analysis_context: str = dspy.InputField(
            desc=(
                "Markdown context from RepoAnalysisAgent including class signatures, "
                "existing test patterns, coverage gaps, few-shot examples, and "
                "dependencies. Plus repository root and iteration index."
            )
        )
        answer: str = dspy.OutputField(desc="Free-text confirmation that the test file was written and run.")

else:  # pragma: no cover - dspy missing fallback (so imports don't blow up)
    RepoAnalysisSignature = None  # type: ignore[assignment]
    TestWritingSignature = None  # type: ignore[assignment]
