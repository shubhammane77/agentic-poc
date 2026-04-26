"""On-disk registry of GEPA-optimized prompts.

Layout (under workspace_root/prompts/):

    prompts/
      analysis/
        20260426T120000Z.json
        20260426T140000Z.json
        ...
      writing/
        20260426T120000Z.json
        ...
      index.jsonl                # append-only event log

Each version file holds the optimized predictor instructions for one agent:

    {
      "version": "20260426T120000Z",
      "agent": "analysis",
      "predictors": {
          "_template.react": "...full GEPA-evolved instructions...",
          "_template.extract.predict": "..."
      },
      "scores": {"train": 0.41, "val": 0.39, "test": null},
      "parent_version": null,
      "model_id": "openai/gpt-4o-mini",
      "created_at": "2026-04-26T12:00:00Z"
    }

`latest()` returns the version with the highest val score (ties → newest).
`pin(version)` rewrites a `pinned.json` symlink-equivalent the runtime reads
first.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from agentic_testgen.core.utils import ensure_dir, utc_timestamp

AgentName = Literal["analysis", "writing"]


@dataclass
class PromptVersion:
    version: str
    agent: AgentName
    predictors: dict[str, str]
    scores: dict[str, float | None] = field(default_factory=dict)
    parent_version: str | None = None
    model_id: str = ""
    created_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class PromptRegistry:
    def __init__(self, root: Path):
        self.root = ensure_dir(root)
        self.index_path = self.root / "index.jsonl"

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, version: PromptVersion) -> Path:
        agent_dir = ensure_dir(self.root / version.agent)
        if not version.created_at:
            version.created_at = utc_timestamp()
        path = agent_dir / f"{version.version}.json"
        path.write_text(json.dumps(version.to_json(), indent=2), encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "save", **version.to_json()}) + "\n")
        return path

    def load(self, agent: AgentName, version: str) -> PromptVersion:
        path = self.root / agent / f"{version}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return PromptVersion(**data)

    def list(self, agent: AgentName) -> list[PromptVersion]:
        agent_dir = self.root / agent
        if not agent_dir.exists():
            return []
        out: list[PromptVersion] = []
        for path in sorted(agent_dir.glob("*.json")):
            try:
                out.append(PromptVersion(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return out

    def latest(self, agent: AgentName) -> PromptVersion | None:
        versions = self.list(agent)
        if not versions:
            return None
        # Highest val score wins; fall back to created_at lexically (UTC ts is sortable).
        return max(
            versions,
            key=lambda v: (v.scores.get("val") if v.scores.get("val") is not None else -1.0, v.created_at),
        )

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin(self, agent: AgentName, version: str) -> Path:
        agent_dir = ensure_dir(self.root / agent)
        pinned_path = agent_dir / "pinned.json"
        loaded = self.load(agent, version)
        pinned_path.write_text(json.dumps(loaded.to_json(), indent=2), encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "pin", "agent": agent, "version": version}) + "\n")
        return pinned_path

    def pinned(self, agent: AgentName) -> PromptVersion | None:
        path = self.root / agent / "pinned.json"
        if not path.exists():
            return None
        return PromptVersion(**json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Apply to a dspy.Module
    # ------------------------------------------------------------------

    def apply(self, version: PromptVersion, module) -> int:
        """Copy `version.predictors` onto `module.named_predictors()`.

        Returns the number of predictors successfully updated.
        """
        applied = 0
        named = dict(module.named_predictors())
        for name, instructions in version.predictors.items():
            pred = named.get(name)
            if pred is None:
                continue
            try:
                pred.signature = pred.signature.with_instructions(instructions)
            except AttributeError:
                pred.signature.instructions = instructions
            applied += 1
        return applied
