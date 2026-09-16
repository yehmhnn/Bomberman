import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.dqn_agent.model import CROSSFIRE_COUNT_IDX, STATE_SIZE  # noqa: E402
from agent_code.dqn_agent.symmetry import (  # noqa: E402
    SYMMETRY_NAMES,
    _ACTION_MAPS,
    _INDEX_MAPS,
    augment_transition,
)

DIRECTION_GROUPS = [(0, 1, 2, 3), (6, 7, 8, 9), (12, 13, 14, 15), (16, 17, 18, 19), (24, 25, 26, 27)]
SCALAR_INDICES = [i for i in range(STATE_SIZE) if not any(i in g for g in DIRECTION_GROUPS)]


def _one_hot_state(direction_slot_in_group=0):
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    for group in DIRECTION_GROUPS:
        state[group[direction_slot_in_group]] = 1
    state[20] = 3.0                     # an arbitrary scalar, must never move or change
    state[CROSSFIRE_COUNT_IDX] = 2.0    # the crossfire count (appended last), likewise invariant
    return state


def test_identity_is_a_no_op():
    state = _one_hot_state(1)
    assert np.array_equal(state[_INDEX_MAPS["identity"]], state)


def test_rotations_cycle_all_direction_groups_together_and_preserve_scalars():
    state = _one_hot_state(0)  # every direction group's UP slot is hot
    for name, expected_slot in [("rot90", 1), ("rot180", 2), ("rot270", 3)]:
        transformed = state[_INDEX_MAPS[name]]
        for group in DIRECTION_GROUPS:
            hot = [i for i, v in enumerate(transformed[list(group)]) if v == 1]
            assert hot == [expected_slot], f"{name} group {group}: expected slot {expected_slot}, got {hot}"
        for i in SCALAR_INDICES:
            assert transformed[i] == state[i], f"{name} changed invariant scalar at index {i}"


def test_rot90_applied_four_times_is_identity():
    composed = np.arange(STATE_SIZE)
    for _ in range(4):
        composed = composed[_INDEX_MAPS["rot90"]]
    assert np.array_equal(composed, _INDEX_MAPS["identity"])


def test_every_flip_is_its_own_inverse():
    for name in ("flip_x", "flip_y", "flip_diag", "flip_anti"):
        twice = np.arange(STATE_SIZE)[_INDEX_MAPS[name]][_INDEX_MAPS[name]]
        assert np.array_equal(twice, _INDEX_MAPS["identity"])


def test_all_eight_transforms_are_distinct_and_are_valid_permutations():
    seen = set()
    for name in SYMMETRY_NAMES:
        idx = tuple(_INDEX_MAPS[name].tolist())
        assert sorted(idx) == list(range(STATE_SIZE)), f"{name} is not a valid permutation"
        seen.add(idx)
    assert len(seen) == 8, "the 8 dihedral transforms should all be distinct on this vector"


def test_action_direction_always_matches_the_state_transform():
    # For every symmetry, wherever a direction group's "hot" slot moves to,
    # the same-direction action must move to that same slot -- state and
    # action have to agree, or the network learns to associate a move with
    # the wrong resulting geometry.
    for start_slot, action in enumerate(("UP", "RIGHT", "DOWN", "LEFT")):
        state = _one_hot_state(start_slot)
        for name in SYMMETRY_NAMES:
            transformed = state[_INDEX_MAPS[name]]
            hot_slot = int(np.argmax(transformed[0:4]))
            mapped_action = _ACTION_MAPS[name][("UP", "RIGHT", "DOWN", "LEFT").index(action)]
            assert ("UP", "RIGHT", "DOWN", "LEFT")[hot_slot] == mapped_action


def test_wait_and_bomb_are_never_remapped():
    for name in SYMMETRY_NAMES:
        assert _ACTION_MAPS[name][4] == "WAIT"
        assert _ACTION_MAPS[name][5] == "BOMB"


def test_crossfire_count_index_is_never_permuted():
    # The regression this guards against: symmetry.py once hardcoded the
    # vector length from before this feature was appended, silently
    # truncating it out of every augmented transition.
    for name in SYMMETRY_NAMES:
        assert _INDEX_MAPS[name][CROSSFIRE_COUNT_IDX] == CROSSFIRE_COUNT_IDX


def test_augment_transition_returns_eight_variants_first_is_identity():
    state = _one_hot_state(0)
    next_state = _one_hot_state(2)
    variants = augment_transition(state, "UP", next_state)
    assert len(variants) == 8
    s0, a0, ns0 = variants[0]
    assert np.array_equal(s0, state) and a0 == "UP" and np.array_equal(ns0, next_state)
