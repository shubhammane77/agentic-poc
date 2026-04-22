import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import tests._path_setup  # noqa: F401

from agentic_testgen.agents.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.core.config import AppConfig, MlflowSettings
from agentic_testgen.core.models import CoverageRecord, FileWorkItem, RepoContext
from agentic_testgen.core.utils import CommandResult


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

    def test_run_from_gitlab_runs_maven_install_after_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            expected_result = MagicMock(name="workflow-result")

            with patch("agentic_testgen.agents.agents.GitLabRepositoryManager") as manager_cls, patch(
                "agentic_testgen.agents.agents.run_command"
            ) as mocked_run, patch.object(workflow, "_execute", return_value=expected_result) as mocked_execute:
                manager = manager_cls.return_value

                def _clone(repo_url: str, destination: Path) -> CommandResult:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / ".git").mkdir(parents=True, exist_ok=True)
                    (destination / "pom.xml").write_text("<project/>", encoding="utf-8")
                    return CommandResult(args=["git", "clone"], exit_code=0, stdout="", stderr="", duration_seconds=0.01)

                manager.clone.side_effect = _clone
                mocked_run.return_value = CommandResult(
                    args=["mvn", "install"],
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.01,
                )

                result = workflow.run_from_gitlab("https://gitlab.example.com/group/project.git", run_id="run_123")

            self.assertIs(result, expected_result)
            self.assertTrue(mocked_execute.called)
            self.assertEqual(1, mocked_run.call_count)
            install_call = mocked_run.call_args
            self.assertEqual(config.maven_command("install"), install_call.args[0])
            self.assertIn("cwd", install_call.kwargs)
            self.assertTrue((Path(install_call.kwargs["cwd"]) / "pom.xml").exists())

    def test_run_from_gitlab_skips_maven_install_without_root_pom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            expected_result = MagicMock(name="workflow-result")

            with patch("agentic_testgen.agents.agents.GitLabRepositoryManager") as manager_cls, patch(
                "agentic_testgen.agents.agents.run_command"
            ) as mocked_run, patch.object(workflow, "_execute", return_value=expected_result):
                manager = manager_cls.return_value

                def _clone(repo_url: str, destination: Path) -> CommandResult:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / ".git").mkdir(parents=True, exist_ok=True)
                    return CommandResult(args=["git", "clone"], exit_code=0, stdout="", stderr="", duration_seconds=0.01)

                manager.clone.side_effect = _clone

                workflow.run_from_gitlab("https://gitlab.example.com/group/project.git", run_id="run_456")

            self.assertFalse(mocked_run.called)

    def test_run_from_gitlab_reuses_cached_clone_between_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            expected_result = MagicMock(name="workflow-result")

            with patch("agentic_testgen.agents.agents.GitLabRepositoryManager") as manager_cls, patch(
                "agentic_testgen.agents.agents.run_command"
            ) as mocked_run, patch.object(workflow, "_execute", return_value=expected_result) as mocked_execute:
                manager = manager_cls.return_value

                def _clone(repo_url: str, destination: Path) -> CommandResult:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / ".git").mkdir(parents=True, exist_ok=True)
                    (destination / "pom.xml").write_text("<project/>", encoding="utf-8")
                    return CommandResult(args=["git", "clone"], exit_code=0, stdout="", stderr="", duration_seconds=0.01)

                manager.clone.side_effect = _clone
                mocked_run.return_value = CommandResult(
                    args=["mvn", "install"],
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.01,
                )

                workflow.run_from_gitlab("https://gitlab.example.com/group/project.git", run_id="run_1")
                workflow.run_from_gitlab("https://gitlab.example.com/group/project.git", run_id="run_2")

            self.assertEqual(1, manager.clone.call_count)
            self.assertEqual(1, mocked_run.call_count)
            self.assertEqual(2, mocked_execute.call_count)


if __name__ == "__main__":
    unittest.main()
