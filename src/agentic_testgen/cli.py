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


def _remove_merge_blockers(clone_path: Path) -> None:
    blocker = clone_path / "coverage.xml"
    if blocker.exists() and blocker.is_file():
        blocker.unlink()


def main() -> None:
    app()


@app.command()
def run(
    repo_url: str | None = typer.Option(None, "--repo-url"),
    repo_path: Path | None = typer.Option(None, "--repo-path"),
    run_id: str | None = None,
    max_files: int | None = None,
) -> None:
    """Run the daddy_subagents_reflective workflow against a GitLab repo."""
    config = _config()
    effective_repo_path = repo_path or (Path(config.repo_path).expanduser() if config.repo_path else None)
    if effective_repo_path and repo_url:
        raise typer.BadParameter("Provide only one of --repo-url or --repo-path")
    if effective_repo_path:
        if not effective_repo_path.exists() or not effective_repo_path.is_dir():
            raise typer.BadParameter(f"Invalid --repo-path: {effective_repo_path}")
        workflow = DaddySubagentsReflectiveWorkflow(config)
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
    workflow = DaddySubagentsReflectiveWorkflow(config)
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
    workflow = DaddySubagentsReflectiveWorkflow(config)
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
        _remove_merge_blockers(clone_path)
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
        workflow = DaddySubagentsReflectiveWorkflow(config)
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
