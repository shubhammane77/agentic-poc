import unittest

import tests._path_setup  # noqa: F401

from agentic_testgen.agents.subagent_dispatcher import SubagentDispatcher


class SubagentDispatcherAggregationTests(unittest.TestCase):
    def test_aggregates_counts_across_multiple_candidates(self) -> None:
        created, successful = SubagentDispatcher._aggregate_iteration_test_counts(
            [
                (3, 2),
                (4, 4),
                (2, 5),  # Passing count is capped to created count per candidate.
            ]
        )
        self.assertEqual(9, created)
        self.assertEqual(8, successful)


if __name__ == "__main__":
    unittest.main()
