import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tests._path_setup  # noqa: F401

from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.execution.tools import SafeToolset, ToolContext
from agentic_testgen.core.utils import CommandResult


class ToolWriteGuardTests(unittest.TestCase):
    def _toolset(self, root: Path) -> SafeToolset:
        logger = RunLogger("test-run", root / "logs", console_enabled=False)
        context = ToolContext(
            run_id="test-run",
            repo_root=root,
            clone_root=root,
            worktrees_root=root / "worktrees",
            config=AppConfig(),
            logger=logger,
            active_worktree=root,
        )
        return SafeToolset(context)

    def test_allows_write_in_module_specific_test_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "module-a" / "src" / "test" / "java").mkdir(parents=True)
            toolset = self._toolset(root)
            relative = toolset.write_new_test_file(
                "module-b/src/test/java/com/example/GeneratedTest.java",
                "class GeneratedTest {}",
            )
            self.assertEqual("module-b/src/test/java/com/example/GeneratedTest.java", relative)
            self.assertTrue((root / relative).exists())

    def test_allows_absolute_path_in_test_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toolset = self._toolset(root)
            target = root / "module-b" / "src" / "test" / "java" / "com" / "example" / "GeneratedTest.java"
            relative = toolset.write_new_test_file(
                str(target),
                "class GeneratedTest {}",
            )
            self.assertEqual("module-b/src/test/java/com/example/GeneratedTest.java", relative)
            self.assertTrue(target.exists())

    def test_rejects_write_outside_test_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toolset = self._toolset(root)
            with self.assertRaisesRegex(ValueError, "src/test/java"):
                toolset.write_new_test_file(
                    "module-b/src/main/java/com/example/GeneratedTest.java",
                    "class GeneratedTest {}",
                )

    def test_allows_rewrite_for_newly_created_test_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toolset = self._toolset(root)
            relative_path = "module-b/src/test/java/com/example/GeneratedTest.java"
            toolset.write_new_test_file(
                relative_path,
                "class GeneratedTest {}",
            )
            toolset.write_new_test_file(
                relative_path,
                "class GeneratedTest { void update() {} }",
            )
            content = (root / relative_path).read_text(encoding="utf-8")
            self.assertIn("update", content)

    def test_tracks_created_junit_test_method_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toolset = self._toolset(root)
            relative_path = "module-b/src/test/java/com/example/GeneratedTest.java"
            toolset.write_new_test_file(
                relative_path,
                "\n".join(
                    [
                        "import org.junit.jupiter.api.Test;",
                        "import org.junit.jupiter.params.ParameterizedTest;",
                        "class GeneratedTest {",
                        "    @Test",
                        "    void firstTest() {}",
                        "    @ParameterizedTest",
                        "    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2})",
                        "    void secondTest(int value) {}",
                        "    void helper() {}",
                        "}",
                    ]
                ),
            )
            self.assertEqual(2, toolset.context.last_written_test_method_count)
            self.assertEqual(2, toolset.context.written_file_test_counts[relative_path])

    def test_run_single_test_parses_console_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            module_root = root / "module-b"
            test_path = module_root / "src" / "test" / "java" / "com" / "example" / "GeneratedTest.java"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            (module_root / "pom.xml").write_text("<project/>", encoding="utf-8")
            test_path.write_text("class GeneratedTest {}", encoding="utf-8")
            toolset = self._toolset(root)
            with patch("agentic_testgen.execution.tools.run_command") as mocked_run, patch(
                "agentic_testgen.execution.tools.write_command_logs",
                return_value={},
            ):
                mocked_run.return_value = CommandResult(
                    args=["mvn", "test"],
                    exit_code=1,
                    stdout="Tests run: 3, Failures: 1, Errors: 0, Skipped: 1",
                    stderr="",
                    duration_seconds=0.01,
                )
                toolset.run_single_test(str(test_path))
            self.assertEqual(3, toolset.context.last_single_test_executed_count)
            self.assertEqual(1, toolset.context.last_single_test_passing_count)

    def test_run_single_test_falls_back_to_surefire_xml_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            module_root = root / "module-b"
            test_path = module_root / "src" / "test" / "java" / "com" / "example" / "GeneratedTest.java"
            surefire_dir = module_root / "target" / "surefire-reports"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            surefire_dir.mkdir(parents=True, exist_ok=True)
            (module_root / "pom.xml").write_text("<project/>", encoding="utf-8")
            test_path.write_text("class GeneratedTest {}", encoding="utf-8")
            (surefire_dir / "TEST-com.example.GeneratedTest.xml").write_text(
                '<testsuite tests="4" failures="1" errors="1" skipped="0"></testsuite>',
                encoding="utf-8",
            )
            toolset = self._toolset(root)
            with patch("agentic_testgen.execution.tools.run_command") as mocked_run, patch(
                "agentic_testgen.execution.tools.write_command_logs",
                return_value={},
            ):
                mocked_run.return_value = CommandResult(
                    args=["mvn", "test"],
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.01,
                )
                toolset.run_single_test(str(test_path))
            self.assertEqual(4, toolset.context.last_single_test_executed_count)
            self.assertEqual(2, toolset.context.last_single_test_passing_count)


if __name__ == "__main__":
    unittest.main()
