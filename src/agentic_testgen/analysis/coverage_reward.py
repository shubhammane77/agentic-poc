"""Reward function used by GEPA to score per-rollout test-generation quality.

Contract (matches dspy.GEPA's metric protocol):
    metric(gold, pred, trace=None, pred_name=None, pred_trace=None) ->
        ScoreWithFeedback(score: float, feedback: str)

Where:
    gold     : dspy.Example built by the dataset loader. Required keys in
               .toDict():
                 - file_path             (relative to repo_root)
                 - repo_root             (absolute path of the cloned fixture)
                 - env                   (RolloutEnvironment)
    pred     : dspy.Prediction returned by AnalysisProgram or WritingProgram.
    trace    : DSPy trace (unused here; coverage is the ground truth).
    pred_name: Predictor name when GEPA is asking for predictor-level
               feedback (e.g. "_template.react"). We currently emit the same
               feedback string at every level — refinement is future work.

Score in [0.0, 1.0]:
    * 0.0 if no test file was written, or compilation/test execution failed.
    * Otherwise: linear interpolation of "missed-line reduction" against the
      baseline missed_lines for the file. A test that closes 100% of the gap
      scores 1.0; closing half the gap scores 0.5.
    * Anti-gameability dampener: if the generated test file contains no
      `assert*` / `verify(` / `expect*` calls, the score is multiplied by 0.25.
      Coverage gain that comes from importing/executing code without checking
      behaviour is worth less than coverage gain from a real assertion.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

try:
    import dspy
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]
    ScoreWithFeedback = None  # type: ignore[assignment]

from agentic_testgen.agents.programs import RolloutEnvironment
from agentic_testgen.analysis.coverage import CoverageAnalyzer
from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.models import CoverageRecord

logger = logging.getLogger(__name__)


# A test file with no calls to one of these markers is treated as
# "executes code but does not assert" → score is dampened.
_ASSERT_PATTERN = re.compile(
    r"\b("
    r"assert[A-Z_]\w*"     # assertEquals, assertTrue, assertThat, ...
    r"|verify\s*\("        # Mockito.verify(...)
    r"|expect[A-Z]\w*"     # AssertJ expect*
    r"|fail\s*\("          # explicit fail()
    r")"
)
_GAMEABILITY_DAMPENER = 0.25


def make_coverage_reward(
    config: AppConfig,
    *,
    failure_score: float = 0.0,
    perfect_score: float = 1.0,
):
    """Return a metric closure bound to the given AppConfig.

    The closure is what GEPA calls. We bind config so the metric can rerun
    Maven+JaCoCo on the right environment.
    """
    coverage = CoverageAnalyzer(config)

    def metric(
        gold: "dspy.Example",
        pred: "dspy.Prediction",
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ):
        env: RolloutEnvironment | None = getattr(gold, "env", None)
        if env is None:
            return _score(failure_score, "No RolloutEnvironment attached to gold example.")

        # 1. The agent must have produced a test file. The gold example carries
        # `generated_test_path` only after a successful WritingProgram rollout
        # — when AnalysisProgram is the optimization target, we still need a
        # test file to compute coverage, which is why the dataset loader wires
        # an inner WritingProgram pass on top.
        test_path = _resolve_test_path(env, pred)
        if not test_path or not Path(test_path).exists():
            return _score(failure_score, "Agent did not produce a test file.")

        # 2. Test must compile and pass — read the exit code recorded by the
        # ToolContext during the rollout. We expose this via pred.toolset_exit
        # if available; otherwise we can re-run the test file standalone
        # through Maven, but that path is expensive — prefer the recorded
        # value.
        exit_code = _last_test_exit_code(pred)
        if exit_code is None:
            return _score(failure_score, "No test exit code recorded — assumed failure.")
        if exit_code != 0:
            return _score(
                failure_score,
                f"Generated test failed to compile or pass (exit_code={exit_code}). "
                "Coverage gain only counts when the test actually runs green.",
            )

        # 3. Re-run JaCoCo against the worktree and read the *after* coverage
        # for the specific target file.
        try:
            _, records, _ = coverage.run_tests_with_coverage(env.repo_root)
        except Exception as exc:  # noqa: BLE001 — Maven can raise anything
            return _score(failure_score, f"JaCoCo execution failed: {exc}")

        target = _find_record_for_file(records, env.file_path)
        if target is None:
            return _score(
                failure_score,
                f"No JaCoCo record found for target file {env.file_path}. "
                "The test may not exercise the file at all.",
            )

        # 4. Score: fraction of baseline missed_lines that became covered.
        baseline_missed = max(env.baseline_missed_lines, 1)
        gap_closed = max(0, env.baseline_missed_lines - target.missed_lines)
        raw = min(1.0, gap_closed / baseline_missed)

        # 5. Gameability dampener: shrink reward if the test has no
        # assertions even if coverage moved.
        try:
            test_source = Path(test_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            test_source = ""
        if not _ASSERT_PATTERN.search(test_source):
            raw *= _GAMEABILITY_DAMPENER
            assertion_note = (
                " Reward dampened: no assertions detected in generated test "
                "(coverage without verification is partial credit only)."
            )
        else:
            assertion_note = ""

        score = max(failure_score, min(perfect_score, raw))
        feedback = (
            f"Closed {gap_closed} of {env.baseline_missed_lines} baseline missed lines on "
            f"{env.file_path} (after_missed={target.missed_lines}, "
            f"after_pct={target.coverage_percent}%, before_pct={env.baseline_coverage_percent}%)."
            f"{assertion_note}"
        )
        return _score(score, feedback)

    return metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(value: float, feedback: str):
    if ScoreWithFeedback is None:  # pragma: no cover
        return float(value)
    return ScoreWithFeedback(score=float(value), feedback=feedback)


def _resolve_test_path(env: RolloutEnvironment, pred: "dspy.Prediction") -> str:
    """Best-effort: prefer pred.generated_test_path, fall back to env."""
    explicit = getattr(pred, "generated_test_path", None)
    if explicit:
        return str(explicit)
    return env.suggested_test_path


def _last_test_exit_code(pred: "dspy.Prediction") -> int | None:
    """The ToolContext records last_single_test_exit_code; programs surface it
    on the Prediction so the metric doesn't have to re-run anything.
    """
    return getattr(pred, "last_single_test_exit_code", None)


def _find_record_for_file(records: list[CoverageRecord], file_path: str) -> CoverageRecord | None:
    norm = file_path.replace("\\", "/")
    for rec in records:
        if rec.file_path.replace("\\", "/").endswith(norm) or norm.endswith(rec.file_path.replace("\\", "/")):
            return rec
    return None
