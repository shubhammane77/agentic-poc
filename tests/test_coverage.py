import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.config import AppConfig
from agentic_testgen.coverage import CoverageAnalyzer


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


if __name__ == "__main__":
    unittest.main()
