import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.danger import blast_coords, danger_map, is_safe  # noqa: E402

FREE, WALL, CRATE = 0, -1, 1


def open_field(size=9):
    return np.zeros((size, size), dtype=int)


def test_blast_in_open_space_is_a_13_tile_cross():
    field = open_field()
    coords = set(blast_coords(field, (4, 4), power=3))
    expected = {(4, 4)}
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, 4):
            expected.add((4 + dx * i, 4 + dy * i))
    assert coords == expected
    assert len(coords) == 13


def test_wall_stops_the_blast():
    field = open_field()
    field[6, 4] = WALL
    coords = set(blast_coords(field, (4, 4), power=3))
    assert (5, 4) in coords
    assert (6, 4) not in coords
    assert (7, 4) not in coords


def test_crate_does_not_stop_the_blast():
    field = open_field()
    field[6, 4] = CRATE
    coords = set(blast_coords(field, (4, 4), power=3))
    assert (5, 4) in coords
    assert (6, 4) in coords
    assert (7, 4) in coords


def _open_state(bombs=(), explosion_map=None):
    field = open_field()
    return {
        "field": field,
        "bombs": list(bombs),
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_bomb_about_to_explode_is_lethal_now_and_next_step():
    state = _open_state(bombs=[((4, 4), 0)])
    danger = danger_map(state, bomb_power=3)
    for pos in blast_coords(state["field"], (4, 4), 3):
        assert danger[pos] == {0, 1}
    assert (0, 0) not in danger


def test_bomb_with_countdown_shifts_the_window():
    state = _open_state(bombs=[((4, 4), 2)])
    danger = danger_map(state, bomb_power=3)
    assert danger[(4, 4)] == {2, 3}
    assert danger[(5, 4)] == {2, 3}


def test_explosion_map_value_one_is_lethal_this_step_only():
    exp = np.zeros((9, 9), dtype=int)
    exp[4, 4] = 1
    state = _open_state(explosion_map=exp)
    danger = danger_map(state, bomb_power=3)
    assert danger[(4, 4)] == {0}


def test_explosion_map_zero_means_safe():
    state = _open_state(explosion_map=np.zeros((9, 9), dtype=int))
    danger = danger_map(state, bomb_power=3)
    assert danger == {}


def test_overlapping_danger_sources_are_merged():
    exp = np.zeros((9, 9), dtype=int)
    exp[4, 4] = 1
    state = _open_state(bombs=[((4, 4), 3)], explosion_map=exp)
    danger = danger_map(state, bomb_power=3)
    assert danger[(4, 4)] == {0, 3, 4}


def test_is_safe_helper():
    danger = {(4, 4): {0, 1}}
    assert not is_safe(danger, (4, 4), 0)
    assert not is_safe(danger, (4, 4), 1)
    assert is_safe(danger, (4, 4), 2)
    assert is_safe(danger, (9, 9), 0)

def test_two_bombs_with_overlapping_blast_are_merged():
    field = np.zeros((13, 13), dtype=int)
    state = {
        "field": field,
        "bombs": [((4, 6), 0), ((8, 6), 3)],
        "explosion_map": np.zeros_like(field),
    }
    danger = danger_map(state, bomb_power=3)
    assert danger[(6, 6)] == {0, 1, 3, 4}