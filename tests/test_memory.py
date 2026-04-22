import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.core.config import AppConfig, MlflowSettings
from agentic_testgen.core.models import CoverageRecord
from agentic_testgen.core.utils import CommandResult, read_json


class MemoryTests(unittest.TestCase):
    def test_run_writes_memory_and_project_memory(self) -> None:
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

            memory_path = result.repo_context.workspace_root / "artifacts" / "memory.json"
            repo_key = f"{result.repo_context.repo_name}::{result.repo_context.repo_url}"
            project_key = repo_key.replace("/", "_").replace(":", "_")
            project_memory_path = Path(tmpdir) / "memory" / f"{project_key}.json"

            self.assertTrue(memory_path.exists())
            self.assertTrue(project_memory_path.exists())

            run_memory = read_json(memory_path, default={})
            project_memory = read_json(project_memory_path, default={})

            self.assertGreaterEqual(len(run_memory.get("entries", [])), 1)
            failed_entries = [entry for entry in run_memory.get("entries", []) if entry.get("status") != "passed"]
            self.assertGreaterEqual(len(failed_entries), 1)
            self.assertGreaterEqual(len(failed_entries[0].get("failure_feedback", [])), 1)
            failed_lessons = [
                item.get("lesson")
                for item in project_memory.get("lessons", [])
                if item.get("status") == "failed"
            ]
            self.assertGreaterEqual(len(failed_lessons), 1)
            self.assertTrue(all(isinstance(item, str) for item in failed_lessons))


if __name__ == "__main__":
    unittest.main()
