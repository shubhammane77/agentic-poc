import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig
from agentic_testgen.utils import read_json


class MemoryTests(unittest.TestCase):
    def test_run_writes_memory_and_global_memory(self) -> None:
        fixture = Path("tests/fixtures/repos/simple-service")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
            )
            workflow = DaddySubagentsReflectiveWorkflow(config)
            result = workflow.run_from_local_path(fixture, source_name="simple-service")

            memory_path = result.repo_context.workspace_root / "artifacts" / "memory.json"
            global_memory_path = Path(tmpdir) / "global_memory.json"

            self.assertTrue(memory_path.exists())
            self.assertTrue(global_memory_path.exists())

            run_memory = read_json(memory_path, default={})
            global_memory = read_json(global_memory_path, default={})

            self.assertEqual("junit5", run_memory.get("test_framework"))
            self.assertGreaterEqual(len(run_memory.get("failures", [])), 1)
            self.assertIn("repos", global_memory)
            repo_entries = list(global_memory["repos"].values())
            self.assertEqual(1, len(repo_entries))
            self.assertGreaterEqual(len(repo_entries[0].get("failures", [])), 1)


if __name__ == "__main__":
    unittest.main()
