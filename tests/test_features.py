import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.danger import danger_map  # noqa: E402
from shared.features import (  # noqa: E402
    FEATURE_SIZE,
    RECENT_POSITIONS_MAXLEN,
    bomb_hits_crate,
    build_feature_vector,
    nearest_coin,
    nearest_crate,
    track_position,
)

WALL = -1
CRATE = 1


def open_field(size=13):
    return np.zeros((size, size), dtype=int)


def make_state(field, pos, coins=(), bombs=(), others=(), bombs_left=True, explosion_map=None):
    return {
        "field": field,
        "self": ("me", 0, bombs_left, pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_nearest_coin_direction_and_distance():
    state = make_state(open_field(), (6, 6), coins=[(9, 6)])
    assert nearest_coin(state) == ("RIGHT", 3)


def test_nearest_coin_none_when_no_coins():
    state = make_state(open_field(), (6, 6))
    assert nearest_coin(state) == (None, None)


def test_nearest_crate_targets_the_adjacent_free_tile():
    field = open_field()
    field[9, 6] = CRATE
    state = make_state(field, (6, 6))
    assert nearest_crate(state) == ("RIGHT", 2)  # walks to (8, 6), next to the crate


def test_bomb_hits_crate():
    field = open_field()
    field[8, 6] = CRATE
    state = make_state(field, (6, 6))
    assert bomb_hits_crate(state, (6, 6)) is True
    assert bomb_hits_crate(state, (6, 2)) is False


def test_feature_vector_has_fixed_length():
    state = make_state(open_field(), (6, 6), coins=[(9, 6)])
    assert len(build_feature_vector(state)) == FEATURE_SIZE


def test_recent_position_bit():
    state = make_state(open_field(), (6, 6))
    assert build_feature_vector(state)[11] == 0
    assert build_feature_vector(state, recent_positions={(6, 6)})[11] == 1


def test_obstruction_is_separate_from_danger():
    field = open_field()
    field[7, 6] = WALL  # RIGHT of (6, 6): physically blocked
    state = make_state(field, (6, 6), bombs=[((6, 3), 0)])  # UP: walkable but about to be lethal
    vector = build_feature_vector(state)
    assert vector[0] == 0  # UP unsafe (danger)
    assert vector[6] == 0  # UP not obstructed
    assert vector[1] == 0  # RIGHT unsafe (obstructed)
    assert vector[7] == 1  # RIGHT obstructed

def test_nearest_coin_avoids_a_lethal_first_step():
    field = open_field(17)
    state = make_state(field, (6, 6), coins=[(9, 6)], bombs=[((10, 6), 0)])
    danger = danger_map(state)
    action, dist = nearest_coin(state, danger=danger)
    assert action != "RIGHT"
    assert dist is not None and dist > 3

def test_track_position_appends_and_respects_maxlen():
    recent = deque(maxlen=RECENT_POSITIONS_MAXLEN)
    for i in range(RECENT_POSITIONS_MAXLEN + 3):
        track_position(recent, (i, 0))
    assert len(recent) == RECENT_POSITIONS_MAXLEN
    assert recent[-1] == (RECENT_POSITIONS_MAXLEN + 2, 0)