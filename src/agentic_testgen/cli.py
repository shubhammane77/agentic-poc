from __future__ import annotations

from pathlib import Path

import typer

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.checkpointing import CheckpointStore
from agentic_testgen.config import AppConfig
from agentic_testgen.evaluation import ModelMatrixEvaluator
from agentic_testgen.models import IntegrationDecision
from agentic_testgen.utils import read_json, run_command, tail_lines, write_json
from agentic_testgen.workspace import WorkspaceManager

app = typer.Typer(help="DSPy-based test generation platform for GitLab Maven repos.")


def _config() -> AppConfig:
    return AppConfig.load()


def main() -> None:
    app()


@app.command()
def run(repo_url: str, run_id: str | None = None) -> None:
    """Run the daddy_subagents_reflective workflow against a GitLab repo."""
    config = _config()
    workflow = DaddySubagentsReflectiveWorkflow(config)
    result = workflow.run_from_gitlab(repo_url, run_id=run_id)
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"overview={result.overview_path}")
    typer.echo(f"workbook={result.workbook_path}")
    typer.echo(f"subagent_results={len(result.subagent_results)}")


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
    workflow = DaddySubagentsReflectiveWorkflow(config)
    result = workflow.resume(run_id)
    typer.echo(f"resumed={result.run_id}")
    typer.echo(f"subagent_results={len(result.subagent_results)}")


@app.command()
def review(run_id: str) -> None:
    """List pending integration decisions."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    pending = read_json(workspace.integrations_path, default=[])
    for item in pending:
        typer.echo(
            f"{item['status']} subagent={item['subagent_id']} commit={item['commit_hash'][:7]} file={item['file_path']}"
        )
    if not pending:
        typer.echo("No pending integrations.")


@app.command()
def integrate(run_id: str, commit_hash: str | None = None) -> None:
    """Apply queued integrations into the cloned parent checkout."""
    config = _config()
    workspace = WorkspaceManager(config.workspace_root).create(run_id)
    checkpoint = CheckpointStore(workspace.checkpoints_dir).load()
    if not checkpoint:
        raise typer.Exit(code=1)
    clone_path = Path(checkpoint.metadata["clone_path"])
    pending = [IntegrationDecision(**item) for item in read_json(workspace.integrations_path, default=[])]
    remaining: list[IntegrationDecision] = []
    integrated = 0
    for decision in pending:
        if commit_hash and decision.commit_hash != commit_hash:
            remaining.append(decision)
            continue
        result = run_command(["git", "cherry-pick", decision.commit_hash], cwd=clone_path)
        if result.ok:
            integrated += 1
        else:
            decision.status = "integration_failed"
            decision.reason = result.stderr or result.stdout
            remaining.append(decision)
    write_json(workspace.integrations_path, [item.to_json() for item in remaining])
    typer.echo(f"integrated={integrated}")
    typer.echo(f"remaining={len(remaining)}")


@app.command(name="eval")
def evaluate(config_path: Path) -> None:
    """Run the model-only evaluation harness over synthetic fixtures."""
    config = _config()
    evaluator = ModelMatrixEvaluator(config)
    results = evaluator.run(config_path)
    typer.echo(f"cases={len(results)}")
    typer.echo(f"completed={sum(1 for item in results if item.status == 'completed')}")
    typer.echo(f"failed={sum(1 for item in results if item.status == 'failed')}")
