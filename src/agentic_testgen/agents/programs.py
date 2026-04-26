"""dspy.Module wrappers used as GEPA students.

GEPA's `compile(student, trainset, valset)` needs a single `dspy.Module` whose
`forward()` produces a `dspy.Prediction` from one training example. We therefore
expose two thin programs:

  * AnalysisProgram — wraps RepoAnalysisAgent. Useful for optimizing the
    analysis prompt in isolation.
  * WritingProgram  — wraps TestWritingAgent against a *fixed* baseline analysis
    summary. Useful for optimizing the writing prompt in isolation.

Both programs hold the relevant CustomReAct (constructed from the agent's
Signature) as a child module, which is what makes their predictors visible to
`student.named_predictors()` — GEPA's mutation surface.

Per-rollout state (worktree, ToolContext) is rebuilt by a `RolloutEnvironment`
factory that the metric uses to compute coverage delta. We deliberately keep
that environment OUT of the dspy.Module so each rollout is isolated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import dspy
    from dspy.primitives.module import Module as DspyModule
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]
    DspyModule = object  # type: ignore[assignment,misc]

from agentic_testgen.agents.custom_react import CustomReAct
from agentic_testgen.agents.signatures import RepoAnalysisSignature, TestWritingSignature
from agentic_testgen.core.models import AnalysisSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout environment
# ---------------------------------------------------------------------------

@dataclass
class RolloutEnvironment:
    """Bundle of per-example execution state passed to programs/metrics.

    Built by the dataset loader for each dspy.Example. The RolloutEnvironment
    is *not* part of the dspy.Module — it lives in the example metadata so the
    metric can rebuild a worktree, run JaCoCo, and compute coverage delta.
    """

    repo_root: Path
    file_path: str
    suggested_test_path: str
    analysis_tools_factory: Callable[[], list]  # build_analysis_dspy_tools()-like
    writing_tools_factory: Callable[[], list]   # build_writing_dspy_tools()-like
    baseline_coverage_percent: float
    baseline_missed_lines: int
    missed_code_snippets: list[str]
    # The active SafeToolset.context. Populated lazily by the dataset loader
    # so the metric can read last_single_test_exit_code / written_files
    # without re-executing the rollout.
    tool_context: object | None = None


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

class AnalysisProgram(DspyModule):
    """dspy.Module wrapping RepoAnalysisAgent for GEPA compilation."""

    def __init__(
        self,
        signature=None,
        max_iters: int = 8,
    ) -> None:
        if dspy is None:
            raise RuntimeError("DSPy is required to construct AnalysisProgram.")
        super().__init__()
        self.max_iters = max_iters
        self.signature = signature or RepoAnalysisSignature
        # NOTE: tool list is empty here so the predictors register with the
        # right signature; the *real* CustomReAct is rebuilt per-call with the
        # rollout-specific tool factory. GEPA mutates `self.react` predictors,
        # and we copy mutated instructions into the per-call CustomReAct below.
        self._template = CustomReAct(self.signature, tools=[_noop_tool], max_iters=max_iters)

    @property
    def react(self):
        return self._template.react

    @property
    def extract(self):
        return self._template.extract

    def forward(self, file_path: str, repo_root: str, runtime_context: str, env: RolloutEnvironment | None = None) -> "dspy.Prediction":
        """Run one analysis pass.

        env carries the tool factory; when env is None we degrade to a
        no-op tool list (useful for unit tests / dry runs).
        """
        tools = env.analysis_tools_factory() if env is not None else [_noop_tool]
        react = CustomReAct(self.signature, tools=tools, max_iters=self.max_iters)
        # Sync mutable instructions from the GEPA-tracked template predictors
        # into the freshly built per-call CustomReAct so optimization carries.
        _copy_predictor_instructions(self._template, react)
        return react(file_path=file_path, repo_root=repo_root, runtime_context=runtime_context)


class WritingProgram(DspyModule):
    """dspy.Module wrapping TestWritingAgent for GEPA compilation."""

    def __init__(
        self,
        signature=None,
        max_iters: int = 6,
    ) -> None:
        if dspy is None:
            raise RuntimeError("DSPy is required to construct WritingProgram.")
        super().__init__()
        self.max_iters = max_iters
        self.signature = signature or TestWritingSignature
        self._template = CustomReAct(self.signature, tools=[_noop_tool], max_iters=max_iters)

    @property
    def react(self):
        return self._template.react

    @property
    def extract(self):
        return self._template.extract

    def forward(
        self,
        file_path: str,
        suggested_test_path: str,
        analysis_context: str,
        env: RolloutEnvironment | None = None,
    ) -> "dspy.Prediction":
        tools = env.writing_tools_factory() if env is not None else [_noop_tool]
        react = CustomReAct(self.signature, tools=tools, max_iters=self.max_iters)
        _copy_predictor_instructions(self._template, react)
        prediction = react(
            file_path=file_path,
            suggested_test_path=suggested_test_path,
            analysis_context=analysis_context,
        )
        # Decorate the Prediction with rollout side-effects so the
        # coverage_reward metric can score it without re-executing anything.
        if env is not None:
            ctx = getattr(env, "tool_context", None)
            generated = ""
            if ctx is not None and ctx.written_files:
                # SafeToolset records relative paths; resolve against repo_root.
                generated = str(env.repo_root / ctx.written_files[-1])
            exit_code = getattr(ctx, "last_single_test_exit_code", None) if ctx else None
            prediction.generated_test_path = generated or suggested_test_path
            prediction.last_single_test_exit_code = exit_code
        return prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_tool(value: str = "") -> str:
    """Placeholder tool so CustomReAct can be constructed without env state."""
    return value


def _copy_predictor_instructions(src_module: "DspyModule", dst_module: "DspyModule") -> None:
    """Copy `signature.instructions` from src.named_predictors() onto dst.

    GEPA mutates predictors in place on the student passed to compile(). Since
    we rebuild a fresh CustomReAct per call (with real tools), we need to
    propagate the GEPA-evolved instructions onto the new instance.
    """
    src = dict(src_module.named_predictors())
    dst = dict(dst_module.named_predictors())
    for name, src_pred in src.items():
        dst_pred = dst.get(name)
        if dst_pred is None:
            continue
        try:
            dst_pred.signature = dst_pred.signature.with_instructions(src_pred.signature.instructions)
        except AttributeError:
            # Fallback: assign directly. dspy.Signature objects expose
            # `.with_instructions(...)` in 3.x; older builds may not.
            dst_pred.signature.instructions = src_pred.signature.instructions
