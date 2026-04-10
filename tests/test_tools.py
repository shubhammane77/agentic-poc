import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.config import AppConfig
from agentic_testgen.logging import RunLogger
from agentic_testgen.tools import SafeToolset, ToolContext


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


if __name__ == "__main__":
    unittest.main()
