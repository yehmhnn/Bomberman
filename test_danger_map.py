"""Tests for framework-accurate blast geometry and escape feasibility."""

import unittest

import numpy as np

from agent_code.tabular_q_agent.danger import (
    blast_coordinates,
    can_escape_after_bomb,
    danger_steps,
)


def state(field, position=(3, 3), bombs=(), explosion=()):
    explosion_map = np.zeros_like(field)
    for tile in explosion:
        explosion_map[tile] = 1
    return {
        "field": field,
        "self": ("learner", 0, True, position),
        "others": [],
        "bombs": list(bombs),
        "explosion_map": explosion_map,
        "coins": [],
    }


class DangerMapTests(unittest.TestCase):
    def setUp(self):
        self.field = -np.ones((9, 9), dtype=int)
        self.field[1:8, 1:8] = 0

    def test_stone_stops_blast_but_crate_does_not(self):
        self.field[5, 3] = 1
        self.field[3, 2] = -1
        blast = set(blast_coordinates(self.field, (3, 3)))
        self.assertIn((5, 3), blast)
        self.assertIn((6, 3), blast)
        self.assertNotIn((3, 2), blast)
        self.assertNotIn((3, 1), blast)

    def test_countdown_zero_is_one_transition_away(self):
        game_state = state(self.field, bombs=[((3, 3), 0)])
        danger = danger_steps(game_state)
        self.assertEqual(danger[3, 3], 1)
        self.assertEqual(danger[6, 3], 1)
        self.assertTrue(np.isinf(danger[7, 7]))

    def test_active_explosion_is_dangerous_now(self):
        danger = danger_steps(state(self.field, explosion=[(2, 2)]))
        self.assertEqual(danger[2, 2], 0)

    def test_escape_requires_turning_out_of_blast_line(self):
        self.assertTrue(can_escape_after_bomb(state(self.field)))

        corridor = -np.ones((9, 9), dtype=int)
        corridor[1:5, 3] = 0
        trapped = state(corridor, position=(1, 3))
        self.assertFalse(can_escape_after_bomb(trapped))


if __name__ == "__main__":
    unittest.main()
