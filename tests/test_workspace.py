import os
import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.workspace import WorkspaceManager


class WorkspaceTests(unittest.TestCase):
    def test_copy_local_repo_handles_dangling_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source-repo"
            source.mkdir(parents=True)
            (source / "pom.xml").write_text("<project/>", encoding="utf-8")
            dangling_link = source / "broken-link"
            try:
                os.symlink(str(source / "missing-target"), str(dangling_link))
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation not supported in this environment")
            destination = root / "destination-repo"
            manager = WorkspaceManager(root / "runs")
            manager.copy_local_repo(source, destination)
            self.assertTrue((destination / "pom.xml").exists())

    def test_copy_local_repo_skips_build_artifacts_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source-repo"
            (source / "target").mkdir(parents=True)
            (source / "target" / "jacoco.xml").write_text("<report/>", encoding="utf-8")
            (source / "pom.xml").write_text("<project/>", encoding="utf-8")
            destination = root / "destination-repo"
            manager = WorkspaceManager(root / "runs")
            manager.copy_local_repo(source, destination)
            self.assertTrue((destination / "pom.xml").exists())
            self.assertFalse((destination / "target").exists())


if __name__ == "__main__":
    unittest.main()
