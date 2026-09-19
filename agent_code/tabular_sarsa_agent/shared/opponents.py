"""Opponent features vendored with the submission-safe tabular agent."""

from settings import BOMB_POWER, BOMB_TIMER

from .danger import blast_coords, danger_map
from .features import bfs_target
from .safety import _occupied, escape_exists

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
FEATURE_SIZE = 10


def _nearest_opponent_info(game_state, occupied, danger):
    best = None
    for opponent in game_state["others"]:
        direction, distance = bfs_target(game_state, occupied, {opponent[3]}, danger)
        if distance is not None and (best is None or distance < best[0]):
            best = (distance, direction, opponent)
    return best


def nearest_opponent(game_state, occupied=None, danger=None):
    occupied = _occupied(game_state) if occupied is None else occupied
    info = _nearest_opponent_info(game_state, occupied, danger)
    return (None, None) if info is None else (info[1], info[0])


def bomb_hits_opponent(game_state, pos):
    targets = {opponent[3] for opponent in game_state["others"]}
    return any(
        tile in targets
        for tile in blast_coords(game_state["field"], pos, BOMB_POWER)
    )


def opponent_trapped_by_bomb(game_state, bomb_pos, opponent_pos):
    hypothetical = dict(
        game_state,
        bombs=game_state["bombs"] + [(bomb_pos, BOMB_TIMER)],
    )
    return not escape_exists(
        hypothetical,
        danger_map(hypothetical, BOMB_POWER),
        start=opponent_pos,
    )


def threatened_by_opponent_bomb(game_state, pos=None):
    pos = game_state["self"][3] if pos is None else pos
    return any(
        bombs_left
        and pos in blast_coords(game_state["field"], opponent_pos, BOMB_POWER)
        for _, _, bombs_left, opponent_pos in game_state["others"]
    )


def threatening_opponent_count(game_state, pos=None):
    pos = game_state["self"][3] if pos is None else pos
    return sum(
        1
        for _, _, bombs_left, opponent_pos in game_state["others"]
        if bombs_left
        and pos in blast_coords(game_state["field"], opponent_pos, BOMB_POWER)
    )


def opponent_features(game_state):
    danger = danger_map(game_state)
    occupied = _occupied(game_state)
    position = game_state["self"][3]
    info = _nearest_opponent_info(game_state, occupied, danger)
    direction = info[1] if info else None
    distance = info[0] if info else None
    trapped = (
        opponent_trapped_by_bomb(game_state, position, info[2][3])
        if info else False
    )
    nearest_has_bomb = bool(info[2][2]) if info else False

    vector = [int(direction == action) for action in ACTIONS]
    vector.extend([
        0 if distance is None else min(distance, 10),
        int(bomb_hits_opponent(game_state, position)),
        int(trapped),
        int(threatened_by_opponent_bomb(game_state, position)),
        min(len(game_state["others"]), 3),
        int(nearest_has_bomb),
    ])
    if len(vector) != FEATURE_SIZE:
        raise ValueError(f"feature vector length {len(vector)} != {FEATURE_SIZE}")
    return vector
