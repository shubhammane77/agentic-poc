from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

try:
    import dspy
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]

from agentic_testgen.core.config import AppConfig
from agentic_testgen.analysis.coverage import CoverageAnalyzer
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.utils import run_command, sanitize_command, write_command_logs


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
    last_single_test_passing_count: int = 0
    last_project_test_exit_code: int | None = None


_TESTS_RUN_RE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
)


def _count_passing_tests(output: str, module_root: Path, class_name: str) -> int:
    """Return number of passing test cases for class_name.

    Tries stdout/stderr first (available when tests fail or -q is absent),
    then falls back to the surefire XML report (always written on disk).
    """
    for m in _TESTS_RUN_RE.finditer(output):
        run, failures, errors, skipped = (int(m.group(i)) for i in range(1, 5))
        return max(0, run - failures - errors - skipped)

    # -q suppresses the summary line when all tests pass; parse the XML instead
    for xml_path in (module_root / "target" / "surefire-reports").glob(f"TEST-*{class_name}*.xml"):
        try:
            text = xml_path.read_text(encoding="utf-8")
            m = re.search(r'<testsuite[^>]+tests="(\d+)"[^>]+failures="(\d+)"[^>]+errors="(\d+)"[^>]+skipped="(\d+)"', text)
            if not m:
                m = re.search(r'tests="(\d+)".*?failures="(\d+)".*?errors="(\d+)".*?skipped="(\d+)"', text)
            if m:
                run, failures, errors, skipped = (int(m.group(i)) for i in range(1, 5))
                return max(0, run - failures - errors - skipped)
        except OSError:
            continue
    return 0


def remove_merge_blockers(clone_path: Path) -> None:
    blocker = clone_path / "coverage.xml"
    if blocker.exists() and blocker.is_file():
        blocker.unlink()


class SafeToolset:
    def __init__(self, context: ToolContext):
        self.context = context
        self.coverage = CoverageAnalyzer(context.config)

    @property
    def active_root(self) -> Path:
        return self.context.active_worktree or self.context.repo_root

    @property
    def active_root_resolved(self) -> Path:
        return self.active_root.resolve()

    @property
    def maven_logs_dir(self) -> Path:
        return self.context.logger.logs_dir / "maven"

    def _remove_merge_blockers(self) -> None:
        remove_merge_blockers(self.context.clone_root)

    def _resolve_active_path(self, path_value: str) -> Path:
        path = Path(path_value)
        resolved = path if path.is_absolute() else (self.active_root / path)
        resolved = resolved.resolve()
        try:
            resolved.relative_to(self.active_root_resolved)
        except ValueError:
            raise ValueError(f"Path outside allowed root: {path_value}")
        return resolved

    def _is_within_test_tree(self, target: Path) -> bool:
        try:
            relative = target.relative_to(self.active_root_resolved)
        except ValueError:
            return False
        parts = relative.parts
        for index in range(len(parts) - 2):
            if parts[index : index + 3] == ("src", "test", "java"):
                return True
        return False

    def read_file(self, file_path: str) -> str:
        with self.context.logger.step(
            "tool.read_file",
            subagent_id=self.context.subagent_id,
            file_path=file_path,
            details={"requested_path": file_path, "active_root": str(self.active_root)},
        ) as step:
            target = self._resolve_active_path(file_path)
            content = target.read_text(encoding="utf-8")
            step["summary"] = f"Read {target.relative_to(self.active_root_resolved)}"
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
            relative = str(target.relative_to(self.active_root_resolved))
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
            step["summary"] = f"Listed {root.relative_to(self.active_root_resolved)}"
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

    def write_new_test_file(self, file_path: str, content: str) -> str:
        with self.context.logger.step(
            "tool.write_new_test_file",
            subagent_id=self.context.subagent_id,
            file_path=file_path,
            details={"requested_path": file_path, "content_chars": len(content)},
        ) as step:
            target = self._resolve_active_path(file_path)
            if not self._is_within_test_tree(target):
                raise ValueError("write_new_test_file may only write inside src/test/java")
            relative = str(target.relative_to(self.active_root_resolved))
            if target.exists() and relative not in self.context.written_files:
                raise ValueError(f"Refusing to overwrite existing file: {file_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if relative not in self.context.written_files:
                self.context.written_files.append(relative)
            step["summary"] = f"Wrote {relative}"
            step["resolved_path"] = str(target)
            step["allowed_test_tree"] = "src/test/java"
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
            if not self.context.written_files:
                step["summary"] = "No generated files to commit"
                return ""
            for file_path in self.context.written_files:
                run_command(["git", "add", file_path], cwd=self.context.active_worktree)
            status = run_command(["git", "diff", "--cached", "--name-only"], cwd=self.context.active_worktree)
            if not status.stdout.strip():
                step["summary"] = "No changes to commit"
                step["git_staged"] = status.stdout
                return ""
            commit = run_command(["git", "commit", "-m", message], cwd=self.context.active_worktree)
            if not commit.ok:
                step["stderr"] = commit.stderr[:800]
                raise RuntimeError(commit.stderr or commit.stdout)
            rev = run_command(["git", "rev-parse", "HEAD"], cwd=self.context.active_worktree)
            commit_hash = rev.stdout.strip()
            step["summary"] = f"Committed {commit_hash[:7]}"
            step["git_staged"] = status.stdout[:800]
            return commit_hash

    def integrate_worktree_result(self, commit_hash: str) -> str:
        with self.context.logger.step(
            "tool.integrate_worktree_result",
            subagent_id=self.context.subagent_id,
            details={"commit_hash": commit_hash},
        ) as step:
            self._remove_merge_blockers()
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
                *self.context.config.maven_command(
                    "-q",
                    f"-Dtest={class_name}",
                    "test",
                )
            ]
            result = run_command(cmd, cwd=module_root)
            log_paths = write_command_logs(
                self.maven_logs_dir,
                f"{self.context.subagent_id or 'run'}-{class_name}-single-test",
                result,
            )
            self.context.last_single_test_exit_code = result.exit_code
            self.context.last_single_test_passing_count = _count_passing_tests(
                result.stdout + "\n" + result.stderr, module_root, class_name
            )
            step["summary"] = f"Ran {class_name}"
            step["resolved_path"] = str(target)
            step["module_root"] = str(module_root)
            step["command"] = sanitize_command(cmd)
            step["exit_code"] = result.exit_code
            step["passing_count"] = self.context.last_single_test_passing_count
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            step["maven_log_paths"] = log_paths
            if result.ok:
                return f"success: single test {class_name} passed ({self.context.last_single_test_passing_count} passing)"
            return (result.stdout + "\n" + result.stderr).strip()

    def run_project_tests_with_coverage(self) -> str:
        with self.context.logger.step(
            "tool.run_project_tests_with_coverage",
            subagent_id=self.context.subagent_id,
            details={"repo_root": str(self.active_root)},
        ) as step:
            result, reports, log_paths = self.coverage.run_tests_with_coverage(
                self.active_root,
                maven_logs_dir=self.maven_logs_dir,
                log_prefix=f"{self.context.subagent_id or 'run'}-project-coverage",
            )
            self.context.last_project_test_exit_code = result.exit_code
            step["summary"] = f"Reports: {len(reports)}"
            step["exit_code"] = result.exit_code
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            step["report_paths"] = [item.report_path for item in reports[:10]]
            step["maven_log_paths"] = log_paths
            if result.ok:
                return "success: project tests with coverage passed"
            return (result.stdout + "\n" + result.stderr).strip()

    def cleanup_worktree(self) -> None:
        if self.context.active_worktree and self.context.active_worktree.exists():
            run_command(["git", "worktree", "remove", str(self.context.active_worktree), "--force"], cwd=self.context.clone_root)
            shutil.rmtree(self.context.active_worktree, ignore_errors=True)

    def build_dspy_tools(self) -> list[dspy.Tool]:
        if dspy is None:
            raise RuntimeError("DSPy is not installed.")
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
        if dspy is None:
            raise RuntimeError("DSPy is not installed.")
        return [
            dspy.Tool(self.read_file),
            dspy.Tool(self.read_folder_structure),
            dspy.Tool(self.search_occurrences),
            dspy.Tool(self.search_file),
            dspy.Tool(self.run_project_tests_with_coverage),
        ]
