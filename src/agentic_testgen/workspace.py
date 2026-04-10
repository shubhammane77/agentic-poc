from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from agentic_testgen.utils import ensure_dir


@dataclass
class RunWorkspace:
    root: Path
    clone_dir: Path
    worktrees_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    control_dir: Path
    integrations_path: Path


class WorkspaceManager:
    def __init__(self, root: Path):
        self.root = root

    def create(self, run_id: str) -> RunWorkspace:
        run_root = ensure_dir(self.root / "runs" / run_id)
        clone_dir = ensure_dir(run_root / "clone")
        worktrees_dir = ensure_dir(run_root / "worktrees")
        artifacts_dir = ensure_dir(run_root / "artifacts")
        logs_dir = ensure_dir(run_root / "logs")
        checkpoints_dir = ensure_dir(run_root / "checkpoints")
        control_dir = ensure_dir(run_root / "control")
        return RunWorkspace(
            root=run_root,
            clone_dir=clone_dir,
            worktrees_dir=worktrees_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            control_dir=control_dir,
            integrations_path=run_root / "pending_integrations.json",
        )

    def copy_local_repo(self, source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
