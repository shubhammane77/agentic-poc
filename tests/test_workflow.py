import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig


class WorkflowTests(unittest.TestCase):
    def test_runs_local_fixture_without_model_and_produces_artifacts(self) -> None:
        fixture = Path("tests/fixtures/repos/simple-service")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            result = workflow.run_from_local_path(fixture, source_name="simple-service")
            self.assertTrue(Path(result.overview_path).exists())
            self.assertTrue(Path(result.workbook_path).exists())
            self.assertGreaterEqual(len(result.work_items), 1)
            self.assertEqual("failed", result.subagent_results[0].status)

    def test_reruns_global_coverage_and_writes_comparison_report(self) -> None:
        fixture = Path("tests/fixtures/repos/simple-service")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            result = workflow.run_from_local_path(fixture, source_name="simple-service")
            comparison = workflow.rerun_after_merge_coverage(result.run_id)
            self.assertIsNotNone(comparison)
            self.assertTrue((result.repo_context.workspace_root / "artifacts" / "coverage-comparison.md").exists())
            self.assertTrue((result.repo_context.workspace_root / "artifacts" / "summary.json").exists())
            self.assertEqual(0.0, comparison.percentage_increase)


if __name__ == "__main__":
    unittest.main()
