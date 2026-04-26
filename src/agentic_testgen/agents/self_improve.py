"""SelfImprovementOrchestrator — drives one GEPA optimization round.

Usage:
    orch = SelfImprovementOrchestrator(config, project_root=Path('.'))
    summary = orch.improve(
        fixtures_path=Path('examples/model_matrix.toml'),
        agent='writing',                      # 'analysis' | 'writing'
        auto='light',                         # GEPA budget preset
        reflection_model=None,                # falls back to AppConfig.model
    )
    print(summary.version, summary.train_score, summary.val_score)

Side effects:
    - Materializes fixture sandboxes under workspace_root/self-improve/...
    - Runs Maven + JaCoCo (cached baseline) per fixture.
    - Writes the optimized prompt to workspace_root/prompts/{agent}/{version}.json.
    - Appends an entry to workspace_root/prompts/index.jsonl.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from agentic_testgen.agents.dspy_runtime import DSPyRuntime
from agentic_testgen.agents.programs import AnalysisProgram, WritingProgram
from agentic_testgen.agents.self_improve_dataset import (
    SelfImprovementDatasetBuilder,
    load_fixtures,
)
from agentic_testgen.analysis.coverage_reward import make_coverage_reward
from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.prompt_registry import PromptRegistry, PromptVersion
from agentic_testgen.core.utils import ensure_dir, new_run_id, utc_timestamp

logger = logging.getLogger(__name__)

AgentTarget = Literal["analysis", "writing"]


@dataclass
class ImprovementSummary:
    agent: AgentTarget
    version: str
    train_score: float | None
    val_score: float | None
    test_score: float | None
    artifact_path: str


class SelfImprovementOrchestrator:
    def __init__(self, config: AppConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.registry = PromptRegistry(config.workspace_root / "prompts")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def improve(
        self,
        fixtures_path: Path,
        *,
        agent: AgentTarget = "writing",
        auto: Literal["light", "medium", "heavy"] = "light",
        reflection_model: str | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 0,
    ) -> ImprovementSummary:
        if dspy is None:
            raise RuntimeError("DSPy is required for self-improvement.")

        run_id = new_run_id("gepa")
        logs_dir = ensure_dir(self.config.workspace_root / "self-improve" / run_id / "logs")
        run_logger = RunLogger(run_id=run_id, logs_dir=logs_dir, secrets=[self.config.model.api_key])
        runtime = DSPyRuntime(self.config, run_logger)
        if not runtime.enabled:
            raise RuntimeError("DSPy runtime is not configured — set MODEL_NAME / MODEL_API_KEY.")

        # 1. Build dataset
        fixtures = load_fixtures(fixtures_path)
        if not fixtures:
            raise ValueError(f"No fixtures found in {fixtures_path}")
        builder = SelfImprovementDatasetBuilder(self.config, self.project_root)
        examples = builder.build(fixtures, target=agent)
        if not examples:
            raise ValueError("Dataset is empty — fixtures had no missed-line work.")
        train, val, test = _split(examples, train_ratio, val_ratio, seed=seed)
        logger.info("GEPA dataset: train=%d val=%d test=%d", len(train), len(val), len(test))

        # 2. Build student
        student = AnalysisProgram() if agent == "analysis" else WritingProgram()

        # 3. Reflection LM — defaults to the same model as the student. Real
        # GEPA papers use a *stronger* model for reflection; we let the user
        # override via reflection_model.
        reflection_lm = _build_reflection_lm(self.config, reflection_model)

        # 4. Compile via GEPA
        metric = make_coverage_reward(self.config)
        optimizer = dspy.GEPA(
            metric=metric,
            auto=auto,
            reflection_lm=reflection_lm,
            num_threads=max(1, self.config.max_parallel_subagents or 1),
            track_stats=True,
            seed=seed,
        )
        compiled = optimizer.compile(student, trainset=train, valset=val or train)

        # 5. Score on held-out test split (best-effort)
        test_score: float | None = None
        if test:
            test_score = _evaluate(compiled, metric, test)

        # 6. Persist
        predictors = {
            name: pred.signature.instructions
            for name, pred in compiled.named_predictors()
        }
        version = utc_timestamp().replace(":", "").replace("-", "")
        scores = {
            "train": _evaluate(compiled, metric, train) if train else None,
            "val": _evaluate(compiled, metric, val) if val else None,
            "test": test_score,
        }
        prompt_version = PromptVersion(
            version=version,
            agent=agent,
            predictors=predictors,
            scores=scores,
            parent_version=None,
            model_id=runtime.model_id,
        )
        artifact_path = self.registry.save(prompt_version)

        return ImprovementSummary(
            agent=agent,
            version=version,
            train_score=scores["train"],
            val_score=scores["val"],
            test_score=scores["test"],
            artifact_path=str(artifact_path),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(
    examples: list,
    train_ratio: float,
    val_ratio: float,
    *,
    seed: int = 0,
) -> tuple[list, list, list]:
    if not examples:
        return [], [], []
    import random

    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _evaluate(program, metric, dataset) -> float:
    """Mean score across dataset; missing/exception cases score 0.0."""
    if not dataset:
        return 0.0
    total = 0.0
    for ex in dataset:
        try:
            pred = program(**ex.inputs(), env=getattr(ex, "env", None))
            out = metric(ex, pred)
            total += float(getattr(out, "score", out))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation rollout raised %s — counted as 0.", exc)
    return total / len(dataset)


def _build_reflection_lm(config: AppConfig, override: str | None):
    """Return a dspy.LM for GEPA reflection.

    GEPA can reuse the student's LM (the dspy.configure default) but a stronger
    model often yields better mutations. When override is set we build a new
    LM bound to that model name.
    """
    if dspy is None:
        return None
    if not override:
        # Reuse the configured student LM. dspy.GEPA accepts None and falls
        # back to dspy.settings.lm.
        return None
    kwargs: dict = {}
    if config.model.api_key:
        kwargs["api_key"] = config.model.api_key
    if config.model.api_base:
        kwargs["api_base"] = config.model.api_base
    return dspy.LM(override, **kwargs)
