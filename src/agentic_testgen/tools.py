from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import dspy

from agentic_testgen.config import AppConfig
from agentic_testgen.coverage import CoverageAnalyzer
from agentic_testgen.logging import RunLogger
from agentic_testgen.utils import run_command, sanitize_command


@dataclass
class ToolContext:
    run_id: str
    repo_root: Path
    clone_root: Path
    worktrees_root: Path
    config: AppConfig
    logger: RunLogger
    subagent_id: str | None = None
    active_worktree: Path | None = None
    written_files: list[str] = field(default_factory=list)
    last_single_test_exit_code: int | None = None
    last_project_test_exit_code: int | None = None


class SafeToolset:
    def __init__(self, context: ToolContext):
        self.context = context
        self.coverage = CoverageAnalyzer(context.config)

    @property
    def active_root(self) -> Path:
        return self.context.active_worktree or self.context.repo_root

    def _resolve_active_path(self, path_value: str) -> Path:
        path = Path(path_value)
        resolved = path if path.is_absolute() else (self.active_root / path)
        resolved = resolved.resolve()
        if not str(resolved).startswith(str(self.active_root.resolve())):
            raise ValueError(f"Path outside allowed root: {path_value}")
        return resolved

    def _test_root(self) -> Path:
        for candidate in self.active_root.rglob("src/test/java"):
            if candidate.is_dir():
                return candidate
        return self.active_root / "src" / "test" / "java"

    def read_file(self, file_path: str) -> str:
        target = self._resolve_active_path(file_path)
        self.context.logger.log_event(
            "tool.read_file",
            "completed",
            summary=f"Read {target.relative_to(self.active_root)}",
            subagent_id=self.context.subagent_id,
            file_path=str(target.relative_to(self.active_root)),
        )
        return target.read_text(encoding="utf-8")

    def search_file(self, file_path: str) -> str:
        target = self._resolve_active_path(file_path)
        return str(target.relative_to(self.active_root))

    def read_folder_structure(self, folder_path: str = ".", max_depth: int = 3) -> str:
        root = self._resolve_active_path(folder_path)
        lines: list[str] = []
        for path in sorted(root.rglob("*")):
            depth = len(path.relative_to(root).parts)
            if depth > max_depth:
                continue
            lines.append(str(path.relative_to(root)))
        summary = "\n".join(lines[:500])
        self.context.logger.log_event(
            "tool.read_folder_structure",
            "completed",
            summary=f"Listed {root.relative_to(self.active_root)}",
            subagent_id=self.context.subagent_id,
        )
        return summary

    def search_occurrences(self, query: str, folder_path: str = ".") -> str:
        root = self._resolve_active_path(folder_path)
        try:
            result = run_command(["rg", "-n", query, str(root)])
        except FileNotFoundError:
            result = run_command(["grep", "-R", "-n", query, str(root)])
        output = result.stdout or result.stderr
        self.context.logger.log_event(
            "tool.search_occurrences",
            "completed" if result.ok else "failed",
            summary=f"Searched for '{query}'",
            subagent_id=self.context.subagent_id,
            details={"exit_code": result.exit_code},
        )
        return output[:5000]

    def write_new_test_file(self, relative_path: str, content: str) -> str:
        test_root = self._test_root().resolve()
        target = (self.active_root / relative_path).resolve()
        if not str(target).startswith(str(test_root)):
            raise ValueError("write_new_test_file may only write inside src/test/java")
        if target.exists():
            raise ValueError(f"Refusing to overwrite existing file: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        relative = str(target.relative_to(self.active_root))
        self.context.written_files.append(relative)
        self.context.logger.log_event(
            "tool.write_new_test_file",
            "completed",
            summary=f"Wrote {relative}",
            subagent_id=self.context.subagent_id,
            file_path=relative,
        )
        return relative

    def create_worktree(self, worktree_path: str, branch_name: str) -> str:
        target = (self.context.worktrees_root / worktree_path).resolve()
        result = run_command(["git", "worktree", "add", "-b", branch_name, str(target), "HEAD"], cwd=self.context.clone_root)
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        self.context.logger.log_event(
            "tool.create_worktree",
            "completed",
            summary=f"Created worktree {target.name}",
            subagent_id=self.context.subagent_id,
        )
        return str(target)

    def commit_worktree_change(self, message: str) -> str:
        if not self.context.active_worktree:
            raise ValueError("No active worktree configured")
        run_command(["git", "add", "."], cwd=self.context.active_worktree)
        status = run_command(["git", "status", "--short"], cwd=self.context.active_worktree)
        if not status.stdout.strip():
            return ""
        commit = run_command(["git", "commit", "-m", message], cwd=self.context.active_worktree)
        if not commit.ok:
            raise RuntimeError(commit.stderr or commit.stdout)
        rev = run_command(["git", "rev-parse", "HEAD"], cwd=self.context.active_worktree)
        commit_hash = rev.stdout.strip()
        self.context.logger.log_event(
            "tool.commit_worktree_change",
            "completed",
            summary=f"Committed {commit_hash[:7]}",
            subagent_id=self.context.subagent_id,
        )
        return commit_hash

    def integrate_worktree_result(self, commit_hash: str) -> str:
        result = run_command(["git", "cherry-pick", commit_hash], cwd=self.context.clone_root)
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        self.context.logger.log_event(
            "tool.integrate_worktree_result",
            "completed",
            summary=f"Integrated {commit_hash[:7]}",
            subagent_id=self.context.subagent_id,
        )
        return commit_hash

    def run_single_test(self, test_file_path: str) -> str:
        target = self._resolve_active_path(test_file_path)
        class_name = target.stem
        module_root = target.parent
        while module_root != self.active_root and not (module_root / "pom.xml").exists():
            module_root = module_root.parent
        cmd = [
            self.context.config.maven_executable(),
            "-q",
            f"-Dtest={class_name}",
            "test",
        ]
        result = run_command(cmd, cwd=module_root)
        self.context.last_single_test_exit_code = result.exit_code
        self.context.logger.log_event(
            "tool.run_single_test",
            "completed" if result.ok else "failed",
            summary=f"Ran {class_name}",
            subagent_id=self.context.subagent_id,
            file_path=str(target.relative_to(self.active_root)),
            details={"command": sanitize_command(cmd), "exit_code": result.exit_code},
        )
        return (result.stdout + "\n" + result.stderr).strip()

    def run_project_tests_with_coverage(self) -> str:
        result, reports = self.coverage.run_tests_with_coverage(self.active_root)
        self.context.last_project_test_exit_code = result.exit_code
        self.context.logger.log_event(
            "tool.run_project_tests_with_coverage",
            "completed" if result.ok else "failed",
            summary=f"Reports: {len(reports)}",
            subagent_id=self.context.subagent_id,
            details={"exit_code": result.exit_code},
        )
        return f"{result.stdout}\n{result.stderr}".strip()

    def cleanup_worktree(self) -> None:
        if self.context.active_worktree and self.context.active_worktree.exists():
            run_command(["git", "worktree", "remove", str(self.context.active_worktree), "--force"], cwd=self.context.clone_root)
            shutil.rmtree(self.context.active_worktree, ignore_errors=True)

    def build_dspy_tools(self) -> list[dspy.Tool]:
        return [
            dspy.Tool(self.read_file),
            dspy.Tool(self.write_new_test_file),
            dspy.Tool(self.read_folder_structure),
            dspy.Tool(self.search_occurrences),
            dspy.Tool(self.search_file),
            dspy.Tool(self.run_single_test),
            dspy.Tool(self.run_project_tests_with_coverage),
        ]

    def build_repo_dspy_tools(self) -> list[dspy.Tool]:
        return [
            dspy.Tool(self.read_file),
            dspy.Tool(self.read_folder_structure),
            dspy.Tool(self.search_occurrences),
            dspy.Tool(self.search_file),
            dspy.Tool(self.run_project_tests_with_coverage),
        ]
