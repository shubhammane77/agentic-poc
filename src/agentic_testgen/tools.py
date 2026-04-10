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
        with self.context.logger.step(
            "tool.read_file",
            subagent_id=self.context.subagent_id,
            file_path=file_path,
            details={"requested_path": file_path, "active_root": str(self.active_root)},
        ) as step:
            target = self._resolve_active_path(file_path)
            content = target.read_text(encoding="utf-8")
            step["summary"] = f"Read {target.relative_to(self.active_root)}"
            step["resolved_path"] = str(target)
            step["chars"] = len(content)
            return content

    def search_file(self, file_path: str) -> str:
        with self.context.logger.step(
            "tool.search_file",
            subagent_id=self.context.subagent_id,
            file_path=file_path,
            details={"requested_path": file_path},
        ) as step:
            target = self._resolve_active_path(file_path)
            relative = str(target.relative_to(self.active_root))
            step["summary"] = f"Resolved {relative}"
            step["resolved_path"] = str(target)
            return relative

    def read_folder_structure(self, folder_path: str = ".", max_depth: int = 3) -> str:
        with self.context.logger.step(
            "tool.read_folder_structure",
            subagent_id=self.context.subagent_id,
            file_path=folder_path,
            details={"folder_path": folder_path, "max_depth": max_depth},
        ) as step:
            root = self._resolve_active_path(folder_path)
            lines: list[str] = []
            for path in sorted(root.rglob("*")):
                depth = len(path.relative_to(root).parts)
                if depth > max_depth:
                    continue
                lines.append(str(path.relative_to(root)))
            summary = "\n".join(lines[:500])
            step["summary"] = f"Listed {root.relative_to(self.active_root)}"
            step["resolved_path"] = str(root)
            step["entry_count"] = len(lines)
            step["preview"] = lines[:20]
            return summary

    def search_occurrences(self, query: str, folder_path: str = ".") -> str:
        with self.context.logger.step(
            "tool.search_occurrences",
            subagent_id=self.context.subagent_id,
            file_path=folder_path,
            details={"query": query, "folder_path": folder_path},
        ) as step:
            root = self._resolve_active_path(folder_path)
            command = ["rg", "-n", query, str(root)]
            result = run_command(command)
            if result.exit_code == 127:
                command = ["grep", "-R", "-n", query, str(root)]
                result = run_command(command)
            output = result.stdout or result.stderr
            step["summary"] = f"Searched for '{query}'"
            step["resolved_path"] = str(root)
            step["command"] = sanitize_command(command)
            step["exit_code"] = result.exit_code
            step["output_preview"] = output[:500]
            return output[:5000]

    def write_new_test_file(self, relative_path: str, content: str) -> str:
        with self.context.logger.step(
            "tool.write_new_test_file",
            subagent_id=self.context.subagent_id,
            file_path=relative_path,
            details={"requested_path": relative_path, "content_chars": len(content)},
        ) as step:
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
            step["summary"] = f"Wrote {relative}"
            step["resolved_path"] = str(target)
            step["test_root"] = str(test_root)
            return relative

    def create_worktree(self, worktree_path: str, branch_name: str) -> str:
        with self.context.logger.step(
            "tool.create_worktree",
            subagent_id=self.context.subagent_id,
            details={"worktree_path": worktree_path, "branch_name": branch_name},
        ) as step:
            target = (self.context.worktrees_root / worktree_path).resolve()
            command = ["git", "worktree", "add", "-b", branch_name, str(target), "HEAD"]
            result = run_command(command, cwd=self.context.clone_root)
            if not result.ok:
                step["command"] = sanitize_command(command)
                step["stderr"] = result.stderr[:800]
                raise RuntimeError(result.stderr or result.stdout)
            step["summary"] = f"Created worktree {target.name}"
            step["resolved_path"] = str(target)
            step["command"] = sanitize_command(command)
            return str(target)

    def commit_worktree_change(self, message: str) -> str:
        if not self.context.active_worktree:
            raise ValueError("No active worktree configured")
        with self.context.logger.step(
            "tool.commit_worktree_change",
            subagent_id=self.context.subagent_id,
            details={"message": message, "worktree": str(self.context.active_worktree)},
        ) as step:
            run_command(["git", "add", "."], cwd=self.context.active_worktree)
            status = run_command(["git", "status", "--short"], cwd=self.context.active_worktree)
            if not status.stdout.strip():
                step["summary"] = "No changes to commit"
                step["git_status"] = status.stdout
                return ""
            commit = run_command(["git", "commit", "-m", message], cwd=self.context.active_worktree)
            if not commit.ok:
                step["stderr"] = commit.stderr[:800]
                raise RuntimeError(commit.stderr or commit.stdout)
            rev = run_command(["git", "rev-parse", "HEAD"], cwd=self.context.active_worktree)
            commit_hash = rev.stdout.strip()
            step["summary"] = f"Committed {commit_hash[:7]}"
            step["git_status"] = status.stdout[:800]
            return commit_hash

    def integrate_worktree_result(self, commit_hash: str) -> str:
        with self.context.logger.step(
            "tool.integrate_worktree_result",
            subagent_id=self.context.subagent_id,
            details={"commit_hash": commit_hash},
        ) as step:
            command = ["git", "cherry-pick", commit_hash]
            result = run_command(command, cwd=self.context.clone_root)
            if not result.ok:
                step["command"] = sanitize_command(command)
                step["stderr"] = result.stderr[:800]
                raise RuntimeError(result.stderr or result.stdout)
            step["summary"] = f"Integrated {commit_hash[:7]}"
            step["command"] = sanitize_command(command)
            return commit_hash

    def run_single_test(self, test_file_path: str) -> str:
        with self.context.logger.step(
            "tool.run_single_test",
            subagent_id=self.context.subagent_id,
            file_path=test_file_path,
            details={"test_file_path": test_file_path},
        ) as step:
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
            output = (result.stdout + "\n" + result.stderr).strip()
            step["summary"] = f"Ran {class_name}"
            step["resolved_path"] = str(target)
            step["module_root"] = str(module_root)
            step["command"] = sanitize_command(cmd)
            step["exit_code"] = result.exit_code
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            return output

    def run_project_tests_with_coverage(self) -> str:
        with self.context.logger.step(
            "tool.run_project_tests_with_coverage",
            subagent_id=self.context.subagent_id,
            details={"repo_root": str(self.active_root)},
        ) as step:
            result, reports = self.coverage.run_tests_with_coverage(self.active_root)
            self.context.last_project_test_exit_code = result.exit_code
            step["summary"] = f"Reports: {len(reports)}"
            step["exit_code"] = result.exit_code
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            step["report_paths"] = [item.report_path for item in reports[:10]]
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
