import unittest
from pathlib import Path
import tempfile

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig, MlflowSettings
from agentic_testgen.models import FileWorkItem, IntegrationDecision
from agentic_testgen.workspace import WorkspaceManager


class BatchingTests(unittest.TestCase):
    def test_limits_work_items_to_configured_top_n(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(
            AppConfig(max_files_per_run=5, mlflow=MlflowSettings(enabled=False))
        )
        items = [
            FileWorkItem(
                file_path=f"src/main/java/com/example/File{i}.java",
                module=".",
                coverage_percent=float(i),
                covered_lines=10,
                missed_lines=10,
                missed_line_numbers=[1, 2],
                priority_rank=i,
            )
            for i in range(1, 8)
        ]
        limited = workflow._apply_work_item_limit(items, None)
        self.assertEqual(5, len(limited))
        self.assertEqual(1, limited[0].priority_rank)
        self.assertEqual(5, limited[-1].priority_rank)

    def test_sorts_integrations_by_priority_rank(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig(mlflow=MlflowSettings(enabled=False)))
        decisions = [
            IntegrationDecision("a", "branch/a", "aaa", "pending_review", "b/File.java", "ok", priority_rank=2),
            IntegrationDecision("b", "branch/b", "bbb", "pending_review", "a/File.java", "ok", priority_rank=1),
        ]
        ordered = workflow._sort_integrations(decisions)
        self.assertEqual("bbb", ordered[0].commit_hash)
        self.assertEqual("aaa", ordered[1].commit_hash)

    def test_dedupes_work_items_by_file_path(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig(mlflow=MlflowSettings(enabled=False)))
        items = [
            FileWorkItem("src/main/java/A.java", ".", 10.0, 1, 9, [1], priority_rank=2),
            FileWorkItem("src/main/java/A.java", ".", 20.0, 2, 8, [2], priority_rank=1),
            FileWorkItem("src/main/java/B.java", ".", 30.0, 3, 7, [3], priority_rank=3),
        ]
        deduped = workflow._dedupe_work_items(items)
        self.assertEqual(2, len(deduped))
        self.assertEqual("src/main/java/A.java", deduped[0].file_path)
        self.assertEqual("src/main/java/B.java", deduped[1].file_path)

    def test_append_integration_replaces_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DaddySubagentsReflectiveWorkflow(
                AppConfig(workspace_root=Path(tmpdir), mlflow=MlflowSettings(enabled=False))
            )
            workspace = WorkspaceManager(Path(tmpdir)).create("run_test")
            first = IntegrationDecision("a", "branch/a", "aaa", "pending_review", "src/main/java/A.java", "ok", 1)
            second = IntegrationDecision("b", "branch/b", "bbb", "pending_review", "src/main/java/A.java", "ok", 2)
            workflow._append_integration(workspace, first)
            workflow._append_integration(workspace, second)
            queued = workflow._read_pending_integrations(workspace)
            self.assertEqual(1, len(queued))
            self.assertEqual("bbb", queued[0].commit_hash)


if __name__ == "__main__":
    unittest.main()
