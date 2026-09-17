import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import events as e  # noqa: E402
from training.rewards import reward_from_transition  # noqa: E402


def open_field(size=13):
    return np.zeros((size, size), dtype=int)


def make_state(field, pos, coins=(), bombs=(), others=()):
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
    }


def test_coin_collected_is_strongly_positive():
    old = make_state(open_field(), (6, 6), coins=[(9, 6)])
    new = make_state(open_field(), (9, 6), coins=[])
    r = reward_from_transition(old, new, [e.COIN_COLLECTED])
    assert r > 5.0


def test_self_kill_is_strongly_negative():
    old = make_state(open_field(), (6, 6))
    r = reward_from_transition(old, None, [e.KILLED_SELF])
    assert r < -10.0


def test_moving_closer_to_a_coin_is_rewarded_over_moving_away():
    field = open_field()
    old = make_state(field, (6, 6), coins=[(9, 6)])
    closer = make_state(field, (7, 6), coins=[(9, 6)])
    farther = make_state(field, (5, 6), coins=[(9, 6)])
    r_closer = reward_from_transition(old, closer, [])
    r_farther = reward_from_transition(old, farther, [])
    assert r_closer > r_farther

def test_terminal_transitions_ignore_potential_shaping():
    field = open_field(17)
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
    old = make_state(field, (1, 1), coins=[(15, 15)])
    got_killed = reward_from_transition(old, None, [e.GOT_KILLED])
    self_kill = reward_from_transition(old, None, [e.KILLED_SELF])
    assert got_killed < 0
    assert self_kill < -10.0


def test_potential_routes_around_danger_not_through_it():
    from training.rewards import _goal_potential
    field = open_field(17)
    state = make_state(field, (6, 6), coins=[(9, 6)], bombs=[((10, 6), 0)])
    assert _goal_potential(state) < -3.0