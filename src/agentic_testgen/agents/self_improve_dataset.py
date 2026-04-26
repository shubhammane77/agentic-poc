"""Build a GEPA training dataset from fixture repos.

Each example is a `dspy.Example` with:
    inputs   : file_path, repo_root, runtime_context (analysis) or
               file_path, suggested_test_path, analysis_context (writing)
    metadata : env (RolloutEnvironment) — used by coverage_reward.

Workflow per fixture:
    1. Copy the fixture into a per-example sandbox under workspace_root.
    2. Run `mvn test jacoco:report` once to record baseline coverage.
       Cached by (repo_path, head_sha) under workspace_root/baseline-cache/.
    3. Build a SafeToolset for that sandbox and bind its ToolContext into
       the RolloutEnvironment so the program rollout's tool side effects
       are visible to the metric.
    4. Compute suggested_test_path from the project layout (mirror src/main →
       src/test, append "Test").
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tomllib
from pathlib import Path
from typing import Any

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from agentic_testgen.agents.programs import RolloutEnvironment
from agentic_testgen.analysis.coverage import CoverageAnalyzer
from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.models import CoverageRecord
from agentic_testgen.core.utils import ensure_dir, new_run_id
from agentic_testgen.execution.tools import SafeToolset, ToolContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixture spec
# ---------------------------------------------------------------------------

class FixtureSpec(dict):
    """Thin dict subclass for typing: {'name', 'repo_path', 'target_files'}."""


def load_fixtures(config_path: Path) -> list[FixtureSpec]:
    """Read examples/model_matrix.toml and return its `[[fixtures]]` table."""
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return [FixtureSpec(item) for item in data.get("fixtures", [])]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class SelfImprovementDatasetBuilder:
    """Materialize fixture repos into GEPA-shaped dspy.Examples."""

    def __init__(self, config: AppConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.coverage = CoverageAnalyzer(config)
        self._sandbox_root = ensure_dir(config.workspace_root / "self-improve")
        self._baseline_cache = ensure_dir(self._sandbox_root / "baseline-cache")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        fixtures: list[FixtureSpec],
        *,
        target: str = "writing",
    ) -> list["dspy.Example"]:
        """Return a list of dspy.Example items.

        target: "analysis" | "writing"
            Determines which input fields the example exposes — must match the
            program being compiled so dspy.Example.with_inputs(...) lines up.
        """
        if dspy is None:
            raise RuntimeError("DSPy is required to build the GEPA dataset.")

        examples: list[dspy.Example] = []
        for fixture in fixtures:
            for relative_file in fixture.get("target_files", []):
                ex = self._build_example(fixture, relative_file, target=target)
                if ex is not None:
                    examples.append(ex)
        return examples

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_example(
        self,
        fixture: FixtureSpec,
        relative_file: str,
        *,
        target: str,
    ) -> "dspy.Example | None":
        repo_src = (self.project_root / fixture["repo_path"]).resolve()
        if not repo_src.exists():
            logger.warning("Fixture repo missing: %s", repo_src)
            return None

        sandbox = self._materialize_sandbox(fixture, repo_src)
        baseline_records = self._baseline_for(repo_src, sandbox)
        target_record = _find_target_record(baseline_records, relative_file)
        if target_record is None or target_record.missed_lines == 0:
            logger.info(
                "Skipping %s/%s (no missed lines, nothing for the agent to improve).",
                fixture.get("name"),
                relative_file,
            )
            return None

        suggested_test_path = _suggest_test_path(sandbox, relative_file)
        absolute_file_path = str((sandbox / relative_file).resolve())

        # Build a ToolContext + SafeToolset bound to this sandbox so rollouts
        # mutate the right repo. Each example owns its own ToolContext so
        # written_files / exit codes stay isolated across rollouts.
        tool_context = ToolContext(
            run_id=new_run_id("gepa"),
            repo_root=sandbox,
            clone_root=sandbox,
            worktrees_root=sandbox.parent / "worktrees",
            config=self.config,
            logger=self._stub_logger(sandbox),
            subagent_id="gepa-rollout",
        )
        toolset = SafeToolset(tool_context)

        env = RolloutEnvironment(
            repo_root=sandbox,
            file_path=relative_file,
            suggested_test_path=str(suggested_test_path),
            analysis_tools_factory=toolset.build_analysis_dspy_tools,
            writing_tools_factory=toolset.build_writing_dspy_tools,
            baseline_coverage_percent=target_record.coverage_percent,
            baseline_missed_lines=target_record.missed_lines,
            missed_code_snippets=_slice_missed_snippets(sandbox / relative_file, target_record.missed_line_numbers),
            tool_context=tool_context,
        )

        if target == "analysis":
            example = dspy.Example(
                file_path=absolute_file_path,
                repo_root=str(sandbox),
                runtime_context=_initial_runtime_context(env),
            ).with_inputs("file_path", "repo_root", "runtime_context")
        elif target == "writing":
            example = dspy.Example(
                file_path=absolute_file_path,
                suggested_test_path=str(suggested_test_path),
                analysis_context=_seed_analysis_context(env),
            ).with_inputs("file_path", "suggested_test_path", "analysis_context")
        else:
            raise ValueError(f"Unknown GEPA target: {target!r}")

        # Attach metadata. dspy.Example supports arbitrary attribute access.
        example.env = env
        example.fixture_name = fixture.get("name", "")
        return example

    # ------------------------------------------------------------------
    # Sandbox + baseline
    # ------------------------------------------------------------------

    def _materialize_sandbox(self, fixture: FixtureSpec, repo_src: Path) -> Path:
        """Copy the fixture into a per-fixture sandbox under workspace_root.

        We DO NOT pin to git head here because fixtures may not be git repos.
        We hash (path, mtime tree) to invalidate when source changes.
        """
        digest = _hash_tree(repo_src)
        target = self._sandbox_root / f"{fixture['name']}-{digest[:10]}"
        if not target.exists():
            shutil.copytree(repo_src, target, ignore=shutil.ignore_patterns("target", "build", "out"))
        return target

    def _baseline_for(self, repo_src: Path, sandbox: Path) -> list[CoverageRecord]:
        digest = _hash_tree(repo_src)
        cache_path = self._baseline_cache / f"{digest}.json"
        if cache_path.exists():
            import json
            return [CoverageRecord(**rec) for rec in json.loads(cache_path.read_text(encoding="utf-8"))]
        _, records, _ = self.coverage.run_tests_with_coverage(sandbox)
        import json
        cache_path.write_text(
            json.dumps([rec.to_json() for rec in records], indent=2),
            encoding="utf-8",
        )
        return records

    def _stub_logger(self, sandbox: Path) -> RunLogger:
        # ToolContext requires a RunLogger; we want one whose log files live
        # under the sandbox so logs don't leak into the user's run history.
        logs_dir = ensure_dir(sandbox.parent / "logs" / sandbox.name)
        return RunLogger(run_id=f"gepa-{sandbox.name}", logs_dir=logs_dir, secrets=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_tree(root: Path) -> str:
    """Stable hash of relative paths + mtimes — invalidates baseline cache when
    the fixture changes without requiring git.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if any(part in {"target", "build", "out", ".git"} for part in path.parts):
            continue
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    return h.hexdigest()


def _find_target_record(records: list[CoverageRecord], relative_file: str) -> CoverageRecord | None:
    norm = relative_file.replace("\\", "/")
    for rec in records:
        if rec.file_path.replace("\\", "/").endswith(norm):
            return rec
    return None


def _suggest_test_path(repo_root: Path, relative_file: str) -> Path:
    """Mirror src/main → src/test and add a Test suffix to the class name."""
    rel = Path(relative_file)
    parts = list(rel.parts)
    # Replace the first 'main' with 'test' (Maven layout).
    for idx, part in enumerate(parts):
        if part == "main":
            parts[idx] = "test"
            break
    test_class_name = rel.stem + "GepaTest" + rel.suffix
    parts[-1] = test_class_name
    return repo_root / Path(*parts)


def _slice_missed_snippets(file_path: Path, line_numbers: list[int], window: int = 1) -> list[str]:
    """Return small source slices for the missed lines so the analysis agent
    has concrete uncovered code to reason about.
    """
    if not file_path.exists() or not line_numbers:
        return []
    text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    snippets: list[str] = []
    for ln in line_numbers[:10]:
        idx = max(0, ln - 1 - window)
        end = min(len(text), ln + window)
        block = "\n".join(text[idx:end])
        snippets.append(f"L{ln}: {block}")
    return snippets


def _initial_runtime_context(env: RolloutEnvironment) -> str:
    if not env.missed_code_snippets:
        return "(no additional runtime context)"
    body = "\n".join(f"  - {s}" for s in env.missed_code_snippets[:8])
    return f"Uncovered code snippets (prioritise these):\n{body}"


def _seed_analysis_context(env: RolloutEnvironment) -> str:
    """Used as the WritingProgram's analysis_context input when we're optimizing
    the writing prompt only and don't want to recompute analysis per rollout.
    Filled in lazily by SelfImprovementOrchestrator if a richer summary is
    pre-computed; otherwise this stub reminds the agent which lines are
    uncovered so it can make progress without an analysis pass.
    """
    if not env.missed_code_snippets:
        return f"Repository root: {env.repo_root}\nIteration: 1\n## Analysis Summary\n(empty — agent must explore)"
    snippets = "\n".join(f"- {s}" for s in env.missed_code_snippets[:8])
    return (
        f"Repository root: {env.repo_root}\n"
        f"Iteration: 1\n\n"
        f"## Analysis Summary\n"
        f"### Uncovered Lines (target file: {env.file_path})\n{snippets}"
    )
