"""Focused unit tests for the Stage-1 tabular Q-learning agent."""

from types import SimpleNamespace
import unittest

import numpy as np

from agent_code.tabular_q_agent.callbacks import (
    ACTIONS,
    shortest_coin_directions,
    state_to_features,
)
from agent_code.tabular_q_agent.train import ALPHA, GAMMA, update_q_table


def game_state(position=(1, 1), coins=((3, 1),)):
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": list(coins),
        "self": ("learner", 0, True, position),
        "others": [],
        "user_input": None,
    }


class SilentLogger:
    def debug(self, *args, **kwargs):
        pass


class TabularQAgentTests(unittest.TestCase):
    def test_shortest_path_points_right(self):
        directions, distance = shortest_coin_directions(game_state())
        self.assertEqual(directions, {ACTIONS.index("RIGHT")})
        self.assertEqual(distance, 2)

    def test_state_is_small_and_hashable(self):
        features = state_to_features(game_state())
        self.assertEqual(len(features), 13)
        self.assertIsInstance(hash(features), int)

    def test_q_learning_update_matches_equation(self):
        old_state = (1, 1, 1, 1, 0, 1, 0, 0, 2)
        next_state = (1, 1, 1, 1, 0, 1, 0, 0, 1)
        table = {
            old_state: np.zeros(len(ACTIONS)),
            next_state: np.array([0.0, 4.0, 1.0, 0.0, 0.0, 0.0]),
        }
        agent = SimpleNamespace(q_table=table, logger=SilentLogger())
        reward = 2.0

        update_q_table(
            agent,
            old_state,
            "RIGHT",
            reward,
            next_state,
            game_state(position=(2, 1)),
        )

        expected = ALPHA * (reward + GAMMA * 4.0)
        self.assertAlmostEqual(table[old_state][ACTIONS.index("RIGHT")], expected)


if __name__ == "__main__":
    unittest.main()
