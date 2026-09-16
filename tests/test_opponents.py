import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.opponents import (  # noqa: E402
    FEATURE_SIZE,
    bomb_hits_opponent,
    nearest_opponent,
    opponent_features,
    opponent_trapped_by_bomb,
    threatened_by_opponent_bomb,
    threatening_opponent_count,
)

WALL = -1


def open_field(size=17):
    # 17x17 matches settings.COLS/ROWS; large enough that BFS/blast scans from
    # inland test positions never walk off the edge of this borderless field
    # (unlike the real game, this helper has no boundary wall to stop them).
    return np.zeros((size, size), dtype=int)


def make_state(field, pos, others=(), bombs=(), bombs_left=True, explosion_map=None):
    return {
        "field": field,
        "self": ("me", 0, bombs_left, pos),
        "others": list(others),
        "bombs": list(bombs),
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_no_opponents_gives_a_harmless_zero_vector():
    state = make_state(open_field(), (6, 6))
    direction, distance = nearest_opponent(state)
    assert direction is None and distance is None
    vector = opponent_features(state)
    assert len(vector) == FEATURE_SIZE
    assert vector == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_direction_and_distance_to_nearest_opponent():
    state = make_state(open_field(), (6, 6), others=[("o", 0, True, (9, 6))])
    direction, distance = nearest_opponent(state)
    assert direction == "RIGHT"
    assert distance == 3


def test_nearest_opponent_picked_by_true_path_distance_not_manhattan():
    field = open_field()
    # (8, 6) is Manhattan-distance 2 from (6, 6), (6, 9) is Manhattan-distance
    # 3. Walling the single direct tile forces a detour (real distance 4) for
    # the Manhattan-closer opponent, so the truly nearest one is (6, 9).
    field[7, 6] = WALL
    state = make_state(
        field, (6, 6),
        others=[("near_by_manhattan", 0, True, (8, 6)), ("near_by_path", 0, True, (6, 9))],
    )
    direction, distance = nearest_opponent(state)
    assert direction == "DOWN"
    assert distance == 3


def test_bomb_hits_opponent_on_the_blast_cross():
    field = open_field()
    state = make_state(field, (5, 5), others=[("o", 0, True, (5, 7))])
    assert bomb_hits_opponent(state, (5, 5)) is True
    assert bomb_hits_opponent(state, (5, 5)) != bomb_hits_opponent(state, (0, 0))


def test_opponent_trapped_in_a_dead_end():
    field = open_field()
    for wx, wy in [(5, 4), (5, 6), (4, 5), (6, 4), (6, 6), (7, 5)]:
        field[wx, wy] = WALL
    state = make_state(field, (5, 5))
    assert opponent_trapped_by_bomb(state, (5, 5), (5, 5)) is True


def test_opponent_not_trapped_in_open_space():
    state = make_state(open_field(), (6, 6))
    assert opponent_trapped_by_bomb(state, (6, 6), (6, 6)) is False


def test_threatened_by_nearby_armed_opponent():
    state = make_state(open_field(), (6, 6), others=[("o", 0, True, (6, 8))])
    assert threatened_by_opponent_bomb(state) is True


def test_not_threatened_when_opponent_out_of_range():
    state = make_state(open_field(), (6, 6), others=[("o", 0, True, (6, 12))])
    assert threatened_by_opponent_bomb(state) is False


def test_not_threatened_when_opponent_has_no_bomb():
    state = make_state(open_field(), (6, 6), others=[("o", 0, False, (6, 8))])
    assert threatened_by_opponent_bomb(state) is False


def test_opponent_features_reports_count_and_bomb_availability():
    state = make_state(
        open_field(), (6, 6),
        others=[("farther_no_bomb", 0, False, (9, 6)), ("nearer_has_bomb", 0, True, (6, 8))],
    )
    vector = opponent_features(state)
    assert len(vector) == FEATURE_SIZE
    assert vector[8] == 2  # opponents alive
    assert vector[9] == 1  # unambiguously nearer opponent has a bomb available


def test_threatening_opponent_count_is_zero_with_no_opponents():
    state = make_state(open_field(), (6, 6))
    assert threatening_opponent_count(state) == 0


def test_threatening_opponent_count_counts_only_armed_opponents_in_range():
    state = make_state(
        open_field(), (6, 6),
        others=[
            ("armed_close", 0, True, (6, 8)),      # in range, armed -> counts
            ("unarmed_close", 0, False, (8, 6)),   # in range, unarmed -> doesn't count
            ("armed_far", 0, True, (6, 12)),       # armed but out of range -> doesn't count
        ],
    )
    assert threatening_opponent_count(state) == 1


def test_threatening_opponent_count_crossfire_from_two_opponents():
    state = make_state(
        open_field(), (6, 6),
        others=[("a", 0, True, (6, 8)), ("b", 0, True, (8, 6))],
    )
    assert threatening_opponent_count(state) == 2
