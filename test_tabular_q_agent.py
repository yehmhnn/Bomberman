"""Focused unit tests for the Stage-2 tabular Q-learning agent."""

from types import SimpleNamespace
import unittest

import numpy as np
import events as e

from agent_code.tabular_q_agent.callbacks import (
    ACTIONS,
    state_to_features,
    valid_action_indices,
)
from agent_code.tabular_q_agent.train import (
    ALPHA,
    GAMMA,
    reward_from_transition,
    update_q_table,
)
from agent_code.tabular_q_agent.shared.features import FEATURE_SIZE, nearest_coin
from agent_code.tabular_q_agent.shared.tabular import stage2_potential


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
        direction, distance = nearest_coin(game_state())
        self.assertEqual(direction, "RIGHT")
        self.assertEqual(distance, 2)

    def test_state_has_shared_fixed_size_and_is_hashable(self):
        features = state_to_features(game_state())
        self.assertEqual(len(features), FEATURE_SIZE)
        self.assertIsInstance(hash(features), int)

    def test_action_mask_blocks_immediate_blast_and_allows_safe_bomb(self):
        state = game_state(position=(3, 3), coins=())
        state["bombs"] = [((3, 1), 0)]
        allowed = {ACTIONS[index] for index in valid_action_indices(state)}
        self.assertNotIn("UP", allowed)
        self.assertIn("LEFT", allowed)

        state["bombs"] = []
        allowed = {ACTIONS[index] for index in valid_action_indices(state)}
        self.assertIn("BOMB", allowed)

    def test_action_mask_rejects_wait_that_would_forfeit_escape(self):
        state = game_state(position=(6, 6), coins=())
        field = -np.ones((13, 13), dtype=int)
        field[1:12, 1:12] = 0
        for x in range(3, 11):
            field[x, 5] = -1
            field[x, 7] = -1
        state["field"] = field
        state["explosion_map"] = np.zeros_like(field)
        state["bombs"] = [((6, 6), 3)]

        allowed = {ACTIONS[index] for index in valid_action_indices(state)}
        self.assertIn("RIGHT", allowed)
        self.assertNotIn("WAIT", allowed)

    def test_potential_improves_when_moving_toward_bombable_crate(self):
        far = game_state(position=(1, 3), coins=())
        far["field"][5, 3] = 1
        near = dict(far, self=("learner", 0, True, (2, 3)))
        self.assertGreater(stage2_potential(near), stage2_potential(far))

    def test_safe_useful_bomb_has_positive_immediate_reward(self):
        state = game_state(position=(3, 3), coins=())
        state["field"][5, 3] = 1
        reward = reward_from_transition(state, "BOMB", state, [e.BOMB_DROPPED])
        self.assertGreater(reward, 0.0)

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
