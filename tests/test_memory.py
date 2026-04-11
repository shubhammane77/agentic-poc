import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig, MlflowSettings
from agentic_testgen.models import CoverageRecord
from agentic_testgen.utils import CommandResult, read_json


class MemoryTests(unittest.TestCase):
    def test_run_writes_memory_and_global_memory(self) -> None:
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
            global_memory_path = Path(tmpdir) / "global_memory.json"

            self.assertTrue(memory_path.exists())
            self.assertTrue(global_memory_path.exists())

            run_memory = read_json(memory_path, default={})
            global_memory = read_json(global_memory_path, default={})

            self.assertGreaterEqual(len(run_memory.get("failures", [])), 1)
            self.assertGreaterEqual(len(run_memory.get("failures", [])[0].get("failure_feedback", [])), 1)
            self.assertIn("repos", global_memory)
            repo_entries = list(global_memory["repos"].values())
            self.assertEqual(1, len(repo_entries))
            self.assertGreaterEqual(len(repo_entries[0].get("failures", [])), 1)


if __name__ == "__main__":
    unittest.main()
