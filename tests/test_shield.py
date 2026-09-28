import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.dqn_agent.model import _augment_with_opponent_threats, action_mask_vector, shield_mask  # noqa: E402
from shared.safety import safe_action_mask  # noqa: E402

WALL = -1
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def bordered_field(size=17):
    # A real board always has a wall border; danger/escape BFS relies on it
    # to terminate instead of walking off the array (see tests/test_opponents.py).
    field = np.zeros((size, size), dtype=int)
    field[:1, :] = WALL
    field[-1:, :] = WALL
    field[:, :1] = WALL
    field[:, -1:] = WALL
    return field


def make_state(field, pos, bombs=(), others=(), bombs_left=True, explosion_map=None):
    return {
        "field": field,
        "self": ("me", 0, bombs_left, pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": [],
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_open_space_no_danger_everything_allowed():
    state = make_state(bordered_field(), (8, 8))
    mask = shield_mask(state)
    assert all(mask[a] for a in ACTIONS)


def test_bomb_rejected_in_a_dead_end_pocket_even_though_immediately_safe():
    # (7, 8) is a one-tile dead-end reachable only via (8, 8); stepping into
    # it or waiting is fine (nothing is dangerous yet), but placing a bomb
    # at (8, 8) leaves nowhere to run once its blast covers the only exit --
    # exactly the gap shared.safety.safe_action_mask's step-0-only check
    # misses, since escape "exists" at step 0 but not for the full horizon.
    field = bordered_field()
    for wx, wy in [(7, 7), (7, 9), (6, 8), (9, 8), (8, 7), (8, 9)]:
        field[wx, wy] = WALL
    state = make_state(field, (8, 8))

    mask = shield_mask(state)
    assert mask["BOMB"] is False
    assert mask["LEFT"] is True  # stepping into the dead end itself is still fine
    assert mask["WAIT"] is True


def test_shield_is_a_subset_of_the_weaker_shared_mask():
    field = bordered_field()
    for wx, wy in [(7, 7), (7, 9), (6, 8), (9, 8), (8, 7), (8, 9)]:
        field[wx, wy] = WALL
    state = make_state(field, (8, 8))

    strong = shield_mask(state)
    weak = safe_action_mask(state)
    for action in ACTIONS:
        assert (not strong[action]) or weak[action], (
            f"shield allowed {action} but the weaker immediate-safety mask did not"
        )


def test_action_mask_vector_degrades_to_weak_mask_when_shield_finds_nothing():
    # A sealed 2-cell pocket: (7, 8)-(8, 8), walled top/bottom/left, with a
    # bomb sitting at (9, 8) blocking the only far exit. Countdown 3 means
    # neither cell is dangerous *yet* (weak/immediate mask sees WAIT and
    # moving to (8, 8) as fine right now), but nothing in this 2-cell pocket
    # escapes before the blast covers both cells at step 3 -- exactly the
    # gap the full-horizon shield exists to catch, correctly finding no
    # action survives it.
    field = bordered_field()
    for wx, wy in [(7, 7), (7, 9), (8, 7), (8, 9), (6, 8)]:
        field[wx, wy] = WALL
    state = make_state(field, (7, 8), bombs=[((9, 8), 3)])

    weak_from_shared = safe_action_mask(state)
    assert weak_from_shared["WAIT"] is True  # confirms this is the "looks fine now" case, not just blocked
    assert weak_from_shared["RIGHT"] is True

    strong = shield_mask(state)
    assert not any(strong.values())

    vector = action_mask_vector(state)
    assert vector.dtype == bool
    assert len(vector) == len(ACTIONS)
    assert bool(vector[ACTIONS.index("WAIT")]) is True  # degraded to the weak mask, not left all-False


def test_action_mask_vector_matches_shield_when_shield_is_non_empty():
    state = make_state(bordered_field(), (8, 8))
    vector = action_mask_vector(state)
    assert list(vector) == [True] * len(ACTIONS)


def test_shield_anticipates_an_armed_adjacent_opponents_next_bomb():
    # With BOMB_TIMER=4, a bomb dropped in the open gives plenty of time to
    # flee even reacting after the fact -- anticipation only matters in a
    # confined space, same insight as the dead-end-pocket tests above. Reuse
    # that exact geometry: a 1-tile pocket at (7, 8) whose only exit, (8, 8),
    # is occupied by an ARMED opponent instead of by our own bomb. No real
    # bomb exists in game_state["bombs"] yet -- the opponent could drop one
    # on their very next turn and there would be no way out of the pocket
    # (its only exit tile is the opponent's own square). The reactive-only
    # mask has no way to see this coming; the shield should.
    field = bordered_field()
    for wx, wy in [(7, 7), (7, 9), (6, 8), (9, 8), (8, 7), (8, 9)]:
        field[wx, wy] = WALL
    state = make_state(field, (7, 8), others=[("armed", 0, True, (8, 8))])

    weak = safe_action_mask(state)
    assert weak["WAIT"] is True, "sanity check: the reactive-only mask sees nothing wrong yet"

    strong = shield_mask(state)
    assert strong["WAIT"] is False, "the shield should see that the opponent could seal this pocket"


def test_shield_does_not_restrict_movement_near_an_unarmed_opponent():
    field = bordered_field()
    for wx, wy in [(7, 7), (7, 9), (6, 8), (9, 8), (8, 7), (8, 9)]:
        field[wx, wy] = WALL
    state = make_state(field, (7, 8), others=[("unarmed", 0, False, (8, 8))])
    strong = shield_mask(state)
    assert strong["WAIT"] is True


def test_augment_treats_opponent_neighbors_as_potentially_occupied():
    # shared.safety._occupied/escape_exists only ever see an opponent's
    # *current* tile, fixed for the whole multi-step BFS -- they have no way
    # to know the opponent could walk into a tile our escape plan is
    # counting on being free one step from now. The augmentation should add
    # a phantom occupant at each of the opponent's neighboring tiles (shaped
    # like a real (name, score, bombs_left, pos) tuple, so the existing
    # shared.safety._occupied picks it up for free) so escape-route planning
    # accounts for that risk. Applies even to an unarmed opponent, since
    # walking into our path doesn't require a bomb.
    field = bordered_field()
    state = make_state(field, (8, 8), others=[("unarmed", 0, False, (10, 10))])

    augmented = _augment_with_opponent_threats(state)
    phantom_positions = {pos for (name, _, _, pos) in augmented["others"] if name == "__phantom__"}

    assert phantom_positions == {(9, 10), (11, 10), (10, 9), (10, 11)}
    # the opponent's own tile must still be present, untouched
    assert ("unarmed", 0, False, (10, 10)) in augmented["others"]


def test_augment_clips_opponent_neighbors_to_the_board():
    # An opponent hugging the border must not generate a phantom tile
    # outside the field array -- that would crash the very next
    # field[pos]/danger.get(pos) lookup.
    field = bordered_field()
    state = make_state(field, (8, 8), others=[("unarmed", 0, False, (1, 1))])

    augmented = _augment_with_opponent_threats(state)
    phantom_positions = {pos for (name, _, _, pos) in augmented["others"] if name == "__phantom__"}

    width, height = field.shape
    assert all(0 <= x < width and 0 <= y < height for x, y in phantom_positions)
    assert phantom_positions == {(0, 1), (2, 1), (1, 0), (1, 2)}
