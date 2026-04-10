import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import tests._path_setup  # noqa: F401

from agentic_testgen.models import CoverageComparison, GlobalCoverageSummary, RepoContext
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

    def test_writes_coverage_comparison_artifacts(self) -> None:
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
            comparison = CoverageComparison(
                before=GlobalCoverageSummary(covered_lines=10, missed_lines=10, coverage_percent=50.0, report_count=1),
                after=GlobalCoverageSummary(covered_lines=14, missed_lines=6, coverage_percent=70.0, report_count=1),
                percentage_increase=20.0,
                covered_line_increase=4,
                missed_line_reduction=4,
            )
            workbook = writer.write_workbook(repo_context, [], [], [], coverage_comparison=comparison)
            comparison_path = writer.write_coverage_comparison(comparison)
            self.assertTrue(comparison_path.exists())
            self.assertIn("Percentage increase: 20.0%", comparison_path.read_text(encoding="utf-8"))
            with ZipFile(workbook) as handle:
                sheet = handle.read("xl/worksheets/sheet5.xml").decode("utf-8")
            self.assertIn("20.0", sheet)


if __name__ == "__main__":
    unittest.main()
