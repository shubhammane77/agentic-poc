from __future__ import annotations

from pathlib import Path

import typer

from agentic_testgen.agents.agents import OrchestratorWorkflow
from agentic_testgen.execution.tools import remove_merge_blockers
from agentic_testgen.execution.checkpointing import CheckpointStore
from agentic_testgen.core.config import AppConfig
from agentic_testgen.analysis.evaluation import ModelMatrixEvaluator
from agentic_testgen.core.models import IntegrationDecision
from agentic_testgen.core.utils import read_json, run_command, tail_lines, write_json
from agentic_testgen.execution.workspace import WorkspaceManager

app = typer.Typer(help="DSPy-based test generation platform for GitLab Maven repos.")


def _config() -> AppConfig:
    return AppConfig.load()


def main() -> None:
    app()


@app.command()
def run(
    repo_url: str | None = typer.Option(None, "--repo-url"),
    repo_path: Path | None = typer.Option(None, "--repo-path"),
    run_id: str | None = None,
    max_files: int | None = None,
) -> None:
    """Run the orchestrator_reflective workflow against a GitLab repo."""
    config = _config()
    effective_repo_path = repo_path or (Path(config.repo_path).expanduser() if config.repo_path else None)
    if effective_repo_path and repo_url:
        raise typer.BadParameter("Provide only one of --repo-url or --repo-path")
    if effective_repo_path:
        if not effective_repo_path.exists() or not effective_repo_path.is_dir():
            raise typer.BadParameter(f"Invalid --repo-path: {effective_repo_path}")
        workflow = OrchestratorWorkflow(config)
        result = workflow.run_from_local_path(
            effective_repo_path.resolve(),
            run_id=run_id,
            source_name=effective_repo_path.name,
            max_files=max_files,
        )
        typer.echo(f"run_id={result.run_id}")
        typer.echo(f"overview={result.overview_path}")
        typer.echo(f"workbook={result.workbook_path}")
        typer.echo(f"subagent_results={len(result.subagent_results)}")
        typer.echo(f"work_items={len(result.work_items)}")
        token_budget = read_json(Path(result.summary_path).parent / "token-budget.json", default={})
        if token_budget:
            typer.echo(f"token_total={token_budget.get('total_tokens', 0)}")
            typer.echo(f"token_input={token_budget.get('input_tokens', 0)}")
            typer.echo(f"token_output={token_budget.get('output_tokens', 0)}")
        return
    effective_repo_url = repo_url or config.repo_url
    if not effective_repo_url:
        raise typer.BadParameter("Provide --repo-path/--repo-url or set REPO_PATH/REPO_URL in .env")
    workflow = OrchestratorWorkflow(config)
    result = workflow.run_from_gitlab(effective_repo_url, run_id=run_id, max_files=max_files)
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"overview={result.overview_path}")
    typer.echo(f"workbook={result.workbook_path}")
    typer.echo(f"subagent_results={len(result.subagent_results)}")
    typer.echo(f"work_items={len(result.work_items)}")
    token_budget = read_json(Path(result.summary_path).parent / "token-budget.json", default={})
    if token_budget:
        typer.echo(f"token_total={token_budget.get('total_tokens', 0)}")
        typer.echo(f"token_input={token_budget.get('input_tokens', 0)}")
        typer.echo(f"token_output={token_budget.get('output_tokens', 0)}")


@app.command()
def status(run_id: str) -> None:
    """Show the latest checkpoint for a run."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    checkpoint = CheckpointStore(workspace.checkpoints_dir).load()
    if not checkpoint:
        raise typer.Exit(code=1)
    typer.echo(f"run_id={checkpoint.run_id}")
    typer.echo(f"phase={checkpoint.phase}")
    typer.echo(f"paused={checkpoint.paused}")
    typer.echo(f"pending_work_items={len(checkpoint.pending_work_items)}")
    typer.echo(f"completed_results={len(checkpoint.completed_results)}")
    typer.echo(f"pending_integrations={len(checkpoint.pending_integrations)}")
    token_budget = read_json(workspace.artifacts_dir / "token-budget.json", default={})
    if token_budget:
        typer.echo(f"token_total={token_budget.get('total_tokens', 0)}")
        typer.echo(f"token_input={token_budget.get('input_tokens', 0)}")
        typer.echo(f"token_output={token_budget.get('output_tokens', 0)}")


@app.command()
def logs(run_id: str, lines: int = 50) -> None:
    """Tail the human-readable run log."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    for line in tail_lines(workspace.logs_dir / "run.log", limit=lines):
        typer.echo(line)


@app.command()
def pause(run_id: str) -> None:
    """Request a safe pause for a running workflow."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    pause_flag = workspace.control_dir / "pause.requested"
    pause_flag.write_text("pause", encoding="utf-8")
    typer.echo(f"pause requested for {run_id}")


@app.command()
def resume(run_id: str) -> None:
    """Resume a previously paused workflow."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    pause_flag = workspace.control_dir / "pause.requested"
    if pause_flag.exists():
        pause_flag.unlink()
    workflow = OrchestratorWorkflow(config)
    result = workflow.resume(run_id)
    typer.echo(f"resumed={result.run_id}")
    typer.echo(f"subagent_results={len(result.subagent_results)}")
    token_budget = read_json(Path(result.summary_path).parent / "token-budget.json", default={})
    if token_budget:
        typer.echo(f"token_total={token_budget.get('total_tokens', 0)}")
        typer.echo(f"token_input={token_budget.get('input_tokens', 0)}")
        typer.echo(f"token_output={token_budget.get('output_tokens', 0)}")


@app.command()
def review(run_id: str) -> None:
    """List pending integration decisions."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    pending = read_json(workspace.integrations_path, default=[])
    for item in pending:
        typer.echo(
            f"rank={item.get('priority_rank', 0)} {item['status']} subagent={item['subagent_id']} commit={item['commit_hash'][:7]} file={item['file_path']}"
        )
    if not pending:
        typer.echo("No pending integrations.")


@app.command()
def integrate(run_id: str, commit_hash: str | None = None) -> None:
    """Apply queued integrations into the cloned parent checkout."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    checkpoint_store = CheckpointStore(workspace.checkpoints_dir)
    checkpoint = checkpoint_store.load()
    if not checkpoint:
        raise typer.Exit(code=1)
    clone_path = Path(checkpoint.metadata["clone_path"])
    pending = [IntegrationDecision(**item) for item in read_json(workspace.integrations_path, default=[])]
    pending = sorted(pending, key=lambda item: (item.priority_rank or 0, item.file_path, item.commit_hash))
    remaining: list[IntegrationDecision] = []
    integrated = 0
    for decision in pending:
        if commit_hash and decision.commit_hash != commit_hash:
            remaining.append(decision)
            continue
        remove_merge_blockers(clone_path)
        result = run_command(["git", "cherry-pick", decision.commit_hash], cwd=clone_path)
        if result.ok:
            integrated += 1
            for completed in checkpoint.completed_results:
                if completed.commit_hash == decision.commit_hash:
                    completed.integration_status = "integrated"
        else:
            run_command(["git", "cherry-pick", "--abort"], cwd=clone_path)
            decision.status = "integration_failed"
            decision.reason = result.stderr or result.stdout
            remaining.append(decision)
            for completed in checkpoint.completed_results:
                if completed.commit_hash == decision.commit_hash:
                    completed.integration_status = "integration_failed"
                    completed.error_message = decision.reason
    write_json(workspace.integrations_path, [item.to_json() for item in remaining])
    checkpoint.pending_integrations = remaining
    checkpoint_store.save(checkpoint)
    typer.echo(f"integrated={integrated}")
    typer.echo(f"remaining={len(remaining)}")
    if integrated > 0:
        workflow = OrchestratorWorkflow(config)
        comparison = workflow.rerun_after_merge_coverage(run_id)
        if comparison:
            typer.echo(f"coverage_before={comparison.before.coverage_percent}")
            typer.echo(f"coverage_after={comparison.after.coverage_percent}")
            typer.echo(f"coverage_increase={comparison.percentage_increase}")
            typer.echo(f"coverage_report={workspace.artifacts_dir / 'coverage-comparison.md'}")


@app.command(name="eval")
def evaluate(config_path: Path) -> None:
    """Run the model-only evaluation harness over synthetic fixtures."""
    config = _config()
    evaluator = ModelMatrixEvaluator(config)
    results = evaluator.run(config_path)
    typer.echo(f"cases={len(results)}")
    typer.echo(f"completed={sum(1 for item in results if item.status == 'completed')}")
    typer.echo(f"failed={sum(1 for item in results if item.status == 'failed')}")


# ---------------------------------------------------------------------------
# Self-improvement (GEPA) commands
# ---------------------------------------------------------------------------

@app.command(name="self-improve")
def self_improve(
    fixtures: Path = typer.Option(
        Path("examples/model_matrix.toml"),
        "--fixtures",
        help="Path to a TOML file containing [[fixtures]] entries.",
    ),
    agent: str = typer.Option(
        "writing",
        "--agent",
        help="Which agent prompt to optimize: analysis | writing.",
    ),
    auto: str = typer.Option(
        "light",
        "--auto",
        help="GEPA budget preset: light | medium | heavy.",
    ),
    reflection_model: str | None = typer.Option(
        None,
        "--reflection-model",
        help="Optional stronger model name for GEPA reflection (e.g. openai/gpt-4o).",
    ),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run one round of GEPA prompt optimization and persist the result."""
    from agentic_testgen.agents.self_improve import SelfImprovementOrchestrator

    if agent not in {"analysis", "writing"}:
        raise typer.BadParameter("--agent must be 'analysis' or 'writing'")
    if auto not in {"light", "medium", "heavy"}:
        raise typer.BadParameter("--auto must be 'light' | 'medium' | 'heavy'")

    config = _config()
    project_root = Path.cwd()
    orch = SelfImprovementOrchestrator(config, project_root)
    summary = orch.improve(
        fixtures_path=fixtures,
        agent=agent,  # type: ignore[arg-type]
        auto=auto,  # type: ignore[arg-type]
        reflection_model=reflection_model,
        seed=seed,
    )
    typer.echo(f"agent={summary.agent}")
    typer.echo(f"version={summary.version}")
    typer.echo(f"train_score={summary.train_score}")
    typer.echo(f"val_score={summary.val_score}")
    typer.echo(f"test_score={summary.test_score}")
    typer.echo(f"artifact={summary.artifact_path}")


prompts_app = typer.Typer(help="Inspect and pin GEPA-optimized prompts.")
app.add_typer(prompts_app, name="prompts")


def _registry() -> "PromptRegistry":  # type: ignore[name-defined]
    from agentic_testgen.core.prompt_registry import PromptRegistry

    config = _config()
    return PromptRegistry(config.workspace_root / "prompts")


@prompts_app.command("list")
def prompts_list(agent: str = typer.Argument(...)) -> None:
    """List saved prompt versions for an agent (analysis | writing)."""
    if agent not in {"analysis", "writing"}:
        raise typer.BadParameter("agent must be 'analysis' or 'writing'")
    registry = _registry()
    versions = registry.list(agent)  # type: ignore[arg-type]
    if not versions:
        typer.echo("(no versions)")
        return
    for v in versions:
        scores = " ".join(f"{k}={v.scores.get(k)}" for k in ("train", "val", "test"))
        typer.echo(f"{v.version}  {scores}  model={v.model_id}")


@prompts_app.command("show")
def prompts_show(agent: str, version: str) -> None:
    """Show the full instructions stored under one version."""
    if agent not in {"analysis", "writing"}:
        raise typer.BadParameter("agent must be 'analysis' or 'writing'")
    pv = _registry().load(agent, version)  # type: ignore[arg-type]
    typer.echo(f"version={pv.version}")
    typer.echo(f"agent={pv.agent}")
    typer.echo(f"created_at={pv.created_at}")
    typer.echo(f"scores={pv.scores}")
    for name, instr in pv.predictors.items():
        typer.echo(f"\n--- {name} ---\n{instr}")


@prompts_app.command("pin")
def prompts_pin(agent: str, version: str) -> None:
    """Pin a specific version so runtime always loads it (set PROMPT_VERSION_*=pinned)."""
    if agent not in {"analysis", "writing"}:
        raise typer.BadParameter("agent must be 'analysis' or 'writing'")
    path = _registry().pin(agent, version)  # type: ignore[arg-type]
    typer.echo(f"pinned={path}")


@prompts_app.command("diff")
def prompts_diff(agent: str, a: str, b: str) -> None:
    """Show a unified diff of predictor instructions between two versions."""
    import difflib

    if agent not in {"analysis", "writing"}:
        raise typer.BadParameter("agent must be 'analysis' or 'writing'")
    registry = _registry()
    va = registry.load(agent, a)  # type: ignore[arg-type]
    vb = registry.load(agent, b)  # type: ignore[arg-type]
    keys = sorted(set(va.predictors) | set(vb.predictors))
    for k in keys:
        left = (va.predictors.get(k) or "").splitlines(keepends=False)
        right = (vb.predictors.get(k) or "").splitlines(keepends=False)
        diff = list(difflib.unified_diff(left, right, fromfile=f"{a}:{k}", tofile=f"{b}:{k}", lineterm=""))
        if diff:
            typer.echo("\n".join(diff))
            typer.echo("")
