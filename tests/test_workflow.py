import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig
from agentic_testgen.models import FileWorkItem, RepoContext


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

    def test_subagent_objective_includes_testing_stack(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig())
        repo_context = RepoContext(
            run_id="run_123",
            repo_url="https://gitlab.example.com/group/project.git",
            repo_name="project",
            clone_path=Path("/tmp/project"),
            workspace_root=Path("/tmp/workspace"),
            test_framework="junit4",
            test_framework_version="4.13.2",
        )
        item = FileWorkItem(
            file_path="src/main/java/com/example/LegacyService.java",
            module=".",
            coverage_percent=40.0,
            covered_lines=4,
            missed_lines=6,
            missed_line_numbers=[10, 11, 15],
        )
        objective = workflow._subagent_objective(
            repo_context,
            item,
            "/tmp/project/src/test/java/com/example/LegacyServiceGeneratedTestIter1.java",
            1,
            [],
            [],
        )
        self.assertIn("Testing stack: junit4 4.13.2", objective)
        self.assertIn("Match the repository's testing framework and version.", objective)


if __name__ == "__main__":
    unittest.main()
