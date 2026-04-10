import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.config import AppConfig
from agentic_testgen.coverage import CoverageAnalyzer
from agentic_testgen.models import GlobalCoverageSummary


FIXTURE_ROOT = Path("tests/fixtures/repos/simple-service")


class CoverageAnalyzerTests(unittest.TestCase):
    def test_parses_jacoco_xml_into_records(self) -> None:
        analyzer = CoverageAnalyzer(AppConfig())
        records = analyzer.parse_jacoco_xml(FIXTURE_ROOT / "target/site/jacoco/jacoco.xml", FIXTURE_ROOT)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("src/main/java/com/example/Calculator.java", record.file_path)
        self.assertGreater(record.missed_lines, 0)

    def test_builds_ranked_work_items(self) -> None:
        analyzer = CoverageAnalyzer(AppConfig())
        records = analyzer.collect_reports(FIXTURE_ROOT)
        items = analyzer.build_work_items(records)
        self.assertEqual(1, len(items))
        self.assertEqual(1, items[0].priority_rank)
        self.assertEqual("src/main/java/com/example/Calculator.java", items[0].file_path)

    def test_summarizes_and_compares_global_coverage(self) -> None:
        analyzer = CoverageAnalyzer(AppConfig())
        records = analyzer.collect_reports(FIXTURE_ROOT)
        summary = analyzer.summarize_global_coverage(records)
        self.assertEqual(len(records), summary.report_count)
        self.assertGreater(summary.covered_lines, 0)
        improved = GlobalCoverageSummary(
            covered_lines=summary.covered_lines + 2,
            missed_lines=max(0, summary.missed_lines - 2),
            coverage_percent=min(100.0, round(summary.coverage_percent + 5.0, 2)),
            report_count=summary.report_count,
        )
        comparison = analyzer.compare_global_coverage(summary, improved)
        self.assertEqual(5.0, comparison.percentage_increase)
        self.assertEqual(2, comparison.covered_line_increase)
        self.assertEqual(2, comparison.missed_line_reduction)


if __name__ == "__main__":
    unittest.main()
