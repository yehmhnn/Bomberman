import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.dqn_agent.lookahead import simulate_own_action  # noqa: E402

WALL = -1


def bordered_field(size=17):
    field = np.zeros((size, size), dtype=int)
    field[:1, :] = WALL
    field[-1:, :] = WALL
    field[:, :1] = WALL
    field[:, -1:] = WALL
    return field


def make_state(field, pos, bombs=(), others=(), coins=(), bombs_left=True, explosion_map=None):
    return {
        "field": field,
        "self": ("me", 0, bombs_left, pos),
        "others": list(others),
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": explosion_map if explosion_map is not None else np.zeros_like(field),
    }


def test_move_into_free_tile_updates_position():
    state = make_state(bordered_field(), (8, 8))
    result = simulate_own_action(state, "RIGHT")
    assert result["self"][3] == (9, 8)


def test_move_into_wall_stays_put():
    field = bordered_field()
    field[9, 8] = WALL
    state = make_state(field, (8, 8))
    result = simulate_own_action(state, "RIGHT")
    assert result["self"][3] == (8, 8)


def test_wait_keeps_position_and_bombs_left():
    state = make_state(bordered_field(), (8, 8), bombs_left=True)
    result = simulate_own_action(state, "WAIT")
    assert result["self"] == ("me", 0, True, (8, 8))


def test_bomb_action_places_bomb_and_clears_bombs_left():
    # BOMB_TIMER == 4, but by the time this shows up in the *next* reported
    # game_state, it has already gone through this step's own "evaluate
    # world: bombs, explosions" pass (see the callback loop order in the
    # project PDF) -- same reasoning shared.safety._can_escape_own_bomb's
    # comment gives for why *that* function uses BOMB_TIMER un-ticked (it
    # reasons about the current instant, before this step's pass; this
    # function reasons about the state one full step later).
    state = make_state(bordered_field(), (8, 8), bombs_left=True)
    result = simulate_own_action(state, "BOMB")
    assert result["self"][2] is False  # bombs_left
    assert ((8, 8), 3) in result["bombs"]


def test_bomb_action_ignored_when_none_left():
    state = make_state(bordered_field(), (8, 8), bombs_left=False)
    result = simulate_own_action(state, "BOMB")
    assert result["bombs"] == []
    assert result["self"][2] is False


def test_existing_bomb_countdown_ticks_down():
    state = make_state(bordered_field(), (8, 8), bombs=[((5, 5), 3)])
    result = simulate_own_action(state, "WAIT")
    assert result["bombs"] == [((5, 5), 2)]


def test_bomb_at_zero_explodes_into_explosion_map():
    field = bordered_field()
    state = make_state(field, (8, 8), bombs=[((5, 5), 0)])
    result = simulate_own_action(state, "WAIT")
    assert result["bombs"] == []  # the exploding bomb is gone
    assert result["explosion_map"][5, 5] == 1  # EXPLOSION_TIMER - 1
    assert result["explosion_map"][6, 5] == 1  # within blast_coords


def test_existing_explosion_decays_and_clips_at_zero():
    field = bordered_field()
    explosion_map = np.zeros_like(field)
    explosion_map[5, 5] = 1
    state = make_state(field, (8, 8), explosion_map=explosion_map)
    result = simulate_own_action(state, "WAIT")
    assert result["explosion_map"][5, 5] == 0


def test_overlapping_explosion_takes_the_max_not_the_sum():
    field = bordered_field()
    explosion_map = np.zeros_like(field)
    explosion_map[5, 5] = 2  # will decay to 1 this step
    state = make_state(field, (8, 8), bombs=[((5, 5), 0)], explosion_map=explosion_map)
    result = simulate_own_action(state, "WAIT")
    assert result["explosion_map"][5, 5] == 1  # max(2-1, EXPLOSION_TIMER-1) == max(1,1)


def test_walking_onto_a_coin_collects_it():
    state = make_state(bordered_field(), (8, 8), coins=[(9, 8), (3, 3)])
    result = simulate_own_action(state, "RIGHT")
    assert result["coins"] == [(3, 3)]


def test_does_not_mutate_the_input_state():
    field = bordered_field()
    original_bombs = [((5, 5), 3)]
    state = make_state(field, (8, 8), bombs=original_bombs)
    simulate_own_action(state, "BOMB")
    assert state["bombs"] == original_bombs  # caller's list untouched
    assert state["self"] == ("me", 0, True, (8, 8))  # caller's self untouched
