import unittest

import tests._path_setup  # noqa: F401

from agentic_testgen.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.config import AppConfig
from agentic_testgen.models import FileWorkItem, IntegrationDecision


class BatchingTests(unittest.TestCase):
    def test_limits_work_items_to_configured_top_n(self) -> None:
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig(max_files_per_run=5))
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
        workflow = DaddySubagentsReflectiveWorkflow(AppConfig())
        decisions = [
            IntegrationDecision("a", "branch/a", "aaa", "pending_review", "b/File.java", "ok", priority_rank=2),
            IntegrationDecision("b", "branch/b", "bbb", "pending_review", "a/File.java", "ok", priority_rank=1),
        ]
        ordered = workflow._sort_integrations(decisions)
        self.assertEqual("bbb", ordered[0].commit_hash)
        self.assertEqual("aaa", ordered[1].commit_hash)


if __name__ == "__main__":
    unittest.main()
