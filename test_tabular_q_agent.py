"""Focused unit tests for the Stage-2 tabular Q-learning agent."""

from types import SimpleNamespace
import unittest

import numpy as np
import events as e

from agent_code.tabular_q_agent.callbacks import (
    ACTIONS,
    q_values,
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
from agent_code.tabular_q_agent.shared.opponents import FEATURE_SIZE as OPPONENT_FEATURE_SIZE
from agent_code.tabular_q_agent.shared.tabular import (
    stage2_potential,
    stage34_potential,
    survivable_action_mask,
)


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
        self.assertEqual(len(features), FEATURE_SIZE + OPPONENT_FEATURE_SIZE)
        self.assertIsInstance(hash(features), int)

    def test_state_contains_opponent_direction_and_count(self):
        state = game_state(position=(3, 3), coins=())
        state["others"] = [("target", 0, True, (5, 3))]
        features = state_to_features(state)
        opponent_part = features[FEATURE_SIZE:]
        self.assertEqual(opponent_part[1], 1)  # nearest opponent is RIGHT
        self.assertEqual(opponent_part[8], 1)  # one opponent remains

    def test_unseen_opponent_state_inherits_stage2_values(self):
        stage2_state = tuple(range(FEATURE_SIZE))
        stage34_state = stage2_state + (0,) * OPPONENT_FEATURE_SIZE
        prior_values = np.arange(len(ACTIONS), dtype=float)
        current = {}
        inherited = q_values(current, stage34_state, [{stage2_state: prior_values}])
        np.testing.assert_array_equal(inherited, prior_values)
        self.assertIsNot(inherited, prior_values)

    def test_stage34_potential_rewards_moving_toward_opponent(self):
        far = game_state(position=(1, 3), coins=())
        far["others"] = [("target", 0, False, (5, 3))]
        near = dict(far, self=("learner", 0, True, (2, 3)))
        self.assertGreater(stage34_potential(near), stage34_potential(far))

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

    def test_action_mask_anticipates_an_armed_opponents_bomb(self):
        field = -np.ones((17, 17), dtype=int)
        field[1:16, 1:16] = 0
        for wall in ((7, 7), (7, 9), (6, 8), (9, 8), (8, 7), (8, 9)):
            field[wall] = -1

        armed = game_state(position=(7, 8), coins=())
        armed["field"] = field
        armed["explosion_map"] = np.zeros_like(field)
        armed["others"] = [("opponent", 0, True, (8, 8))]
        self.assertFalse(survivable_action_mask(armed)["WAIT"])

        unarmed = dict(armed, others=[("opponent", 0, False, (8, 8))])
        self.assertTrue(survivable_action_mask(unarmed)["WAIT"])

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

    def test_killing_opponent_has_large_task_reward(self):
        state = game_state(position=(3, 3), coins=())
        plain = reward_from_transition(state, "WAIT", None, [])
        kill = reward_from_transition(state, "WAIT", None, [e.KILLED_OPPONENT])
        self.assertAlmostEqual(kill - plain, 50.0)

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
