import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.q_agent.state import N_FEATURES, build_state_vector  # noqa: E402


def make_state(field, pos, coins=(), bombs=(), others=()):
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros_like(field),
    }


def test_state_vector_has_combined_length():
    field = np.zeros((17, 17), dtype=int)
    state = make_state(field, (6, 6), coins=[(9, 6)])
    vector = build_state_vector(state)
    assert len(vector) == N_FEATURES == 34


def test_state_vector_reflects_a_nearby_armed_opponent():
    field = np.zeros((17, 17), dtype=int)
    state = make_state(field, (6, 6), others=[("o", 0, True, (6, 3))])
    vector = build_state_vector(state)
    assert vector[33] == 1  # nearest opponent (index 24-33 block) has a bomb available


def test_state_vector_with_multiple_opponents_does_not_crash():
    field = np.zeros((17, 17), dtype=int)
    others = [("a", 0, True, (6, 3)), ("b", 0, False, (10, 10)), ("c", 0, True, (2, 2))]
    state = make_state(field, (6, 6), others=others)
    vector = build_state_vector(state)
    assert len(vector) == N_FEATURES
    assert vector[32] == 3  # opponents alive, capped at 3
