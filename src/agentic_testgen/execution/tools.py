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
    written_file_test_counts: dict[str, int] = field(default_factory=dict)
    last_written_test_method_count: int = 0
    last_single_test_exit_code: int | None = None
    last_single_test_executed_count: int = 0
    last_single_test_passing_count: int = 0
    last_project_test_exit_code: int | None = None


_TESTS_RUN_RE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
)
_JUNIT_METHOD_ANNOTATIONS = {"Test", "ParameterizedTest", "RepeatedTest", "TestFactory", "TestTemplate"}
_FOLDER_STRUCTURE_IGNORED_DIRS = {
    ".git",
    "target",
    "build",
    "out",
    "generated-resources",
    "generated-resorruces",
    "node_modules",
}


def _is_junit_method_annotation(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("@"):
        return False
    token = stripped[1:].split("(", 1)[0].strip()
    if not token:
        return False
    return token.split(".")[-1] in _JUNIT_METHOD_ANNOTATIONS


def _count_declared_junit_tests(content: str) -> int:
    count = 0
    pending_annotation = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if _is_junit_method_annotation(stripped):
            pending_annotation = True
            continue
        if pending_annotation and stripped.startswith("@"):
            continue
        if pending_annotation and _looks_like_java_method_declaration(stripped):
            count += 1
            pending_annotation = False
            continue
        pending_annotation = False
    return count


def _looks_like_java_method_declaration(line: str) -> bool:
    if "(" not in line or ")" not in line:
        return False
    if ";" in line:
        return False
    disallowed = ("class ", "interface ", "enum ", "record ", "if ", "for ", "while ", "switch ", "catch ")
    if any(token in line for token in disallowed):
        return False
    before_paren = line.split("(", 1)[0].strip()
    if not before_paren:
        return False
    return " " in before_paren
_FOLDER_STRUCTURE_MAX_DEPTH = 20


def _parse_test_counts(output: str, module_root: Path, class_name: str) -> tuple[int, int]:
    """Return executed/passing test counts for class_name.

    Tries stdout/stderr first (available when tests fail or -q is absent),
    then falls back to the surefire XML report (always written on disk).
    """
    for m in _TESTS_RUN_RE.finditer(output):
        run, failures, errors, skipped = (int(m.group(i)) for i in range(1, 5))
        passing = max(0, run - failures - errors - skipped)
        return run, passing

    # -q suppresses the summary line when all tests pass; parse the XML instead
    for xml_path in (module_root / "target" / "surefire-reports").glob(f"TEST-*{class_name}*.xml"):
        try:
            text = xml_path.read_text(encoding="utf-8")
            m = re.search(r'<testsuite[^>]+tests="(\d+)"[^>]+failures="(\d+)"[^>]+errors="(\d+)"[^>]+skipped="(\d+)"', text)
            if not m:
                m = re.search(r'tests="(\d+)".*?failures="(\d+)".*?errors="(\d+)".*?skipped="(\d+)"', text)
            if m:
                run, failures, errors, skipped = (int(m.group(i)) for i in range(1, 5))
                passing = max(0, run - failures - errors - skipped)
                return run, passing
        except OSError:
            continue
    return 0, 0


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
        """Read the full text content of a source file.

        Args:
            file_path: Absolute path to the file
                (e.g. /home/user/project/src/main/java/Foo.java on Unix or
                C:\\Users\\user\\project\\src\\main\\java\\Foo.java on Windows).
                Relative paths are resolved against the active repository root,
                but absolute paths are strongly preferred.
        """
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
        """Resolve a file path and return its path relative to the repository root.

        Args:
            file_path: Absolute path to the file to locate
                (e.g. /home/user/project/src/main/java/Foo.java).
                Must be an absolute path; relative paths are resolved against
                the active repository root.
        """
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

    def read_folder_structure(self, folder_path: str = ".") -> str:
        """Return a YAML tree of the folder's contents (ignores build artifacts).

        Args:
            folder_path: Absolute path to the folder to list
                (e.g. /home/user/project/src on Unix or
                C:\\Users\\user\\project\\src on Windows).
                Defaults to the active repository root when omitted.
                Relative paths are resolved against the active repository root,
                but absolute paths are strongly preferred.
        """
        with self.context.logger.step(
            "tool.read_folder_structure",
            subagent_id=self.context.subagent_id,
            file_path=folder_path,
            details={"folder_path": folder_path},
        ) as step:
            root = self._resolve_active_path(folder_path)
            entries: list[tuple[tuple[str, ...], bool]] = []
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part in _FOLDER_STRUCTURE_IGNORED_DIRS for part in relative.parts):
                    continue
                depth = len(relative.parts)
                if depth > _FOLDER_STRUCTURE_MAX_DEPTH:
                    continue
                entries.append((relative.parts, path.is_dir()))
            preview_entries = entries[:500]

            tree: dict[str, dict | None] = {}
            for parts, is_dir in preview_entries:
                node = tree
                for index, part in enumerate(parts):
                    is_last = index == len(parts) - 1
                    key = f"{part}/" if (not is_last or is_dir) else part
                    existing = node.get(key)
                    if is_last:
                        if key not in node:
                            node[key] = {} if is_dir else None
                        elif isinstance(existing, dict) and not is_dir:
                            # Keep richer directory node if both were seen.
                            node[key] = existing
                        break
                    if not isinstance(existing, dict):
                        existing = {}
                        node[key] = existing
                    node = existing

            summary = "[]"
            if preview_entries:
                def _render_yaml(node: dict[str, dict | None], indent: int = 0) -> list[str]:
                    lines: list[str] = []
                    prefix = "  " * indent
                    for key in sorted(node):
                        value = node[key]
                        if isinstance(value, dict):
                            lines.append(f"{prefix}{key}:")
                            if value:
                                lines.extend(_render_yaml(value, indent + 1))
                        else:
                            lines.append(f"{prefix}{key}:")
                    return lines

                summary = "\n".join(_render_yaml(tree))
            step["summary"] = f"Listed {root.relative_to(self.active_root_resolved)}"
            step["resolved_path"] = str(root)
            step["entry_count"] = len(entries)
            step["preview"] = ["/".join(parts) + ("/" if is_dir else "") for parts, is_dir in preview_entries[:20]]
            return summary

    def search_occurrences(self, query: str, folder_path: str = ".") -> str:
        """Search for a text pattern (ripgrep / grep) across a folder tree.

        Args:
            query: The text or regex pattern to search for.
            folder_path: Absolute path to the folder to search
                (e.g. /home/user/project/src).
                Defaults to the active repository root when omitted.
                Relative paths are resolved against the active repository root,
                but absolute paths are strongly preferred.
        """
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
        """Write a new Java test file inside the src/test/java tree.

        Only files under src/test/java are permitted.  The file must not already
        exist unless it was written by this agent in the current session.

        Args:
            file_path: Absolute path where the test file should be written
                (e.g. /home/user/project/src/test/java/com/example/FooTest.java).
                Must be an absolute path; a valid absolute Unix path starts with /
                and a valid absolute Windows path starts with a drive letter
                (e.g. C:\\).  If a relative path is given it is resolved against
                the active repository root.
            content: Full Java source code for the test class.
        """
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
            created_test_count = _count_declared_junit_tests(content)
            self.context.last_written_test_method_count = created_test_count
            self.context.written_file_test_counts[relative] = created_test_count
            if relative not in self.context.written_files:
                self.context.written_files.append(relative)
            step["summary"] = f"Wrote {relative}"
            step["resolved_path"] = str(target)
            step["allowed_test_tree"] = "src/test/java"
            step["created_test_count"] = created_test_count
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
        """Run a single JUnit test class using Maven and return the output.

        Args:
            test_file_path: Absolute path to the .java test file to execute
                (e.g. /home/user/project/src/test/java/com/example/FooTest.java).
                Must be an absolute path; a valid absolute Unix path starts with /
                and a valid absolute Windows path starts with a drive letter
                (e.g. C:\\).  If a relative path is given it is resolved against
                the active repository root.
        """
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
            executed_count, passing_count = _parse_test_counts(
                result.stdout + "\n" + result.stderr, module_root, class_name
            )
            self.context.last_single_test_exit_code = result.exit_code
            self.context.last_single_test_executed_count = executed_count
            self.context.last_single_test_passing_count = passing_count
            step["summary"] = f"Ran {class_name}"
            step["resolved_path"] = str(target)
            step["module_root"] = str(module_root)
            step["command"] = sanitize_command(cmd)
            step["exit_code"] = result.exit_code
            step["executed_count"] = self.context.last_single_test_executed_count
            step["passing_count"] = self.context.last_single_test_passing_count
            step["stdout_preview"] = result.stdout[:500]
            step["stderr_preview"] = result.stderr[:500]
            step["maven_log_paths"] = log_paths
            if result.ok:
                return (
                    f"success: single test {class_name} passed "
                    f"({self.context.last_single_test_passing_count}/{self.context.last_single_test_executed_count} passing)"
                )
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

    def build_analysis_dspy_tools(self) -> list[dspy.Tool]:
        """Read-only tools for RepoAnalysisAgent."""
        if dspy is None:
            raise RuntimeError("DSPy is not installed.")
        return [
            dspy.Tool(self.read_file),
            dspy.Tool(self.read_folder_structure),
            dspy.Tool(self.search_occurrences),
            dspy.Tool(self.search_file),
        ]

    def build_writing_dspy_tools(self) -> list[dspy.Tool]:
        """Write and run tools for TestWritingAgent."""
        if dspy is None:
            raise RuntimeError("DSPy is not installed.")
        return [
            dspy.Tool(self.write_new_test_file),
            dspy.Tool(self.run_single_test),
        ]
