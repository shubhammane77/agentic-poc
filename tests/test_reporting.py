import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import tests._path_setup  # noqa: F401

from agentic_testgen.models import RepoContext
from agentic_testgen.reporting import ReportWriter


class ReportingTests(unittest.TestCase):
    def test_writes_xlsx_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ReportWriter(Path(tmpdir))
            repo_context = RepoContext(
                run_id="run_123",
                repo_url="https://gitlab.example.com/group/project.git",
                repo_name="project",
                clone_path=Path(tmpdir) / "clone",
                workspace_root=Path(tmpdir),
                source_type="fixture",
            )
            workbook = writer.write_workbook(repo_context, [], [], [])
            self.assertTrue(workbook.exists())
            with ZipFile(workbook) as handle:
                names = set(handle.namelist())
            self.assertIn("xl/workbook.xml", names)
            self.assertIn("xl/worksheets/sheet1.xml", names)


if __name__ == "__main__":
    unittest.main()
