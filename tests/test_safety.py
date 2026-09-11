import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.danger import danger_map  # noqa: E402
from shared.safety import earliest_danger, escape_exists, safe_action_mask, tile_free  # noqa: E402

WALL = -1


def open_field(size=13):
    return np.zeros((size, size), dtype=int)


def make_state(field, pos, bombs=(), others=(), bombs_left=True, explosion_map=None):
    return {
        "field": field,
        "self": ("me", 0, bombs_left, pos),
        "others": list(others),
        "bombs": list(bombs),
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_mask_all_true_in_open_space():
    state = make_state(open_field(), (6, 6))
    mask = safe_action_mask(state)
    assert all(mask[a] for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"))


def test_mask_blocks_wall():
    field = open_field()
    field[6, 5] = WALL  # UP of (6, 6)
    state = make_state(field, (6, 6))
    mask = safe_action_mask(state)
    assert mask["UP"] is False
    assert mask["DOWN"] is True


def test_mask_blocks_targets_on_the_blast_cross():
    field = open_field()
    state = make_state(field, (5, 5), bombs=[((6, 6), 0)])
    mask = safe_action_mask(state)
    assert mask["RIGHT"] is False  # (6, 5) is on the bomb's column
    assert mask["DOWN"] is False   # (5, 6) is on the bomb's row
    assert mask["UP"] is True
    assert mask["LEFT"] is True


def test_mask_blocks_current_explosion_on_wait():
    field = open_field()
    exp = np.zeros_like(field)
    exp[6, 6] = 1
    state = make_state(field, (6, 6), explosion_map=exp)
    mask = safe_action_mask(state)
    assert mask["WAIT"] is False
    assert mask["BOMB"] is False  # same tile, same reason


def test_mask_bomb_action_needs_bombs_left():
    state = make_state(open_field(), (6, 6), bombs_left=False)
    mask = safe_action_mask(state)
    assert mask["BOMB"] is False


def test_earliest_danger():
    danger = {(4, 4): {2, 3}}
    assert earliest_danger(danger, (4, 4)) == 2
    assert earliest_danger(danger, (0, 0)) is None


def test_escape_exists_in_open_space():
    field = open_field()
    state = make_state(field, (6, 6), bombs=[((6, 6), 4)])  # just dropped, full fuse
    danger = danger_map(state, bomb_power=3)
    assert escape_exists(state, danger) is True


def test_no_escape_in_a_short_dead_end():
    field = open_field()
    for wx, wy in [(5, 4), (5, 6), (4, 5), (6, 4), (6, 6), (7, 5)]:
        field[wx, wy] = WALL
    state = make_state(field, (5, 5), bombs=[((5, 5), 4)])
    danger = danger_map(state, bomb_power=3)
    assert escape_exists(state, danger) is False


def test_tile_free_checks_others_and_bombs():
    field = open_field()
    state = make_state(field, (6, 6), others=[("o", 0, True, (7, 6))])
    assert tile_free(state, (7, 6)) is False
    assert tile_free(state, (5, 6)) is True

def test_mask_blocks_bomb_with_no_escape():
    field = open_field()
    for wx, wy in [(5, 4), (5, 6), (4, 5), (6, 4), (6, 6), (7, 5)]:
        field[wx, wy] = WALL
    state = make_state(field, (5, 5))
    mask = safe_action_mask(state)
    assert mask["BOMB"] is False