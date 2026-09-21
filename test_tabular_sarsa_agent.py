"""Focused tests for the Stage-2 SARSA update."""

from types import SimpleNamespace
import unittest

import numpy as np

from agent_code.tabular_sarsa_agent.callbacks import ACTIONS
from agent_code.tabular_sarsa_agent.train import ALPHA, GAMMA, update_sarsa


class SilentLogger:
    def debug(self, *args, **kwargs):
        pass


class SarsaTests(unittest.TestCase):
    def test_sarsa_uses_selected_next_action_not_maximum(self):
        old_state = ("old",)
        next_state = ("next",)
        table = {
            old_state: np.zeros(len(ACTIONS)),
            # RIGHT is maximal, but the policy actually selected DOWN.
            next_state: np.array([0.0, 8.0, 3.0, 0.0, 0.0, 0.0]),
        }
        agent = SimpleNamespace(q_table=table, logger=SilentLogger())

        update_sarsa(agent, old_state, "RIGHT", 2.0, next_state, "DOWN")

        expected = ALPHA * (2.0 + GAMMA * 3.0)
        self.assertAlmostEqual(table[old_state][ACTIONS.index("RIGHT")], expected)


if __name__ == "__main__":
    unittest.main()
