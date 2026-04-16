import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig, MlflowSettings
from agentic_testgen.models import CoverageRecord, FileWorkItem, RepoContext
from agentic_testgen.utils import CommandResult


class WorkflowTests(unittest.TestCase):
    def test_runs_local_fixture_without_model_and_produces_artifacts(self) -> None:
        fixture = Path("tests/fixtures/repos/simple-service")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            workflow.coverage.run_tests_with_coverage = lambda *args, **kwargs: (
                CommandResult(args=["mvn"], exit_code=0, stdout="", stderr="", duration_seconds=0.01),
                [
                    CoverageRecord(
                        file_path="src/main/java/com/example/Calculator.java",
                        module=".",
                        covered_lines=3,
                        missed_lines=2,
                        coverage_percent=60.0,
                        missed_line_numbers=[10, 11],
                    )
                ],
                {},
            )
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
                mlflow=MlflowSettings(enabled=False),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            result = workflow.run_from_local_path(fixture, source_name="simple-service")
            comparison = workflow.rerun_after_merge_coverage(result.run_id)
            self.assertIsNotNone(comparison)
            self.assertTrue((result.repo_context.workspace_root / "artifacts" / "coverage-comparison.md").exists())
            self.assertTrue((result.repo_context.workspace_root / "artifacts" / "summary.json").exists())
            self.assertEqual(0.0, comparison.percentage_increase)

    def test_subagent_objective_avoids_framework_specific_guidance(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig(mlflow=MlflowSettings(enabled=False)))
        repo_context = RepoContext(
            run_id="run_123",
            repo_url="https://gitlab.example.com/group/project.git",
            repo_name="project",
            clone_path=Path("/tmp/project"),
            workspace_root=Path("/tmp/workspace"),
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
            Path("/tmp/workspace/artifacts/coverage-context.md"),
            ["if (input == null) {", "return repository.findById(id);"],
        )
        self.assertNotIn("Testing stack:", objective)
        self.assertNotIn("Match the repository's testing framework and version.", objective)
        self.assertIn("Coverage context artifact:", objective)
        self.assertIn("Uncovered code snippets in assigned file:", objective)
        self.assertIn("if (input == null) {", objective)


if __name__ == "__main__":
    unittest.main()
