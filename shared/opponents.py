"""Opponent targeting, threat, and trapping features.

Built on top of shared.danger / shared.safety / shared.features exactly the
way coin/crate targeting is, so the DQN and any tabular/linear agent can
consume the same, once-verified primitives instead of each re-deriving blast
timing.
"""

from settings import BOMB_POWER, BOMB_TIMER
from shared.danger import blast_coords, danger_map
from shared.safety import _occupied, escape_exists
from shared.features import bfs_target

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
FEATURE_SIZE = 10
# 0-3 direction to nearest living opponent, one-hot (UP, RIGHT, DOWN, LEFT)
# 4   distance to nearest opponent, capped at 10
# 5   dropping a bomb at my own tile now would hit an opponent
# 6   the nearest opponent would have no escape if I bombed my tile now
# 7   I am on a tile some opponent could hit by bombing their own tile now
# 8   opponents alive, capped at 3
# 9   the nearest opponent currently has a bomb available


def _nearest_opponent_info(game_state, occupied, danger):
    """(distance, direction, opponent_tuple) for the true BFS-nearest
    opponent, or None if there are no opponents left. One bfs_target call per
    opponent (at most 3) rather than a single multi-goal search, so we know
    *which* opponent was nearest and can reuse that identity below.
    """
    best = None
    for opponent in game_state["others"]:
        direction, distance = bfs_target(game_state, occupied, {opponent[3]}, danger)
        if distance is None:
            continue
        if best is None or distance < best[0]:
            best = (distance, direction, opponent)
    return best


def nearest_opponent(game_state, occupied=None, danger=None):
    occupied = _occupied(game_state) if occupied is None else occupied
    info = _nearest_opponent_info(game_state, occupied, danger)
    return (None, None) if info is None else (info[1], info[0])


def bomb_hits_opponent(game_state, pos):
    field = game_state["field"]
    targets = {o[3] for o in game_state["others"]}
    return any(p in targets for p in blast_coords(field, pos, BOMB_POWER))


def opponent_trapped_by_bomb(game_state, bomb_pos, opponent_pos, bomb_power=BOMB_POWER):
    """Would an opponent standing at opponent_pos have no escape if I placed
    a bomb at bomb_pos right now? Mirrors shared.safety._can_escape_own_bomb:
    the hypothetical bomb uses BOMB_TIMER (not BOMB_TIMER - 1), since it would
    be created before this same step's update_bombs pass, exactly like any
    bomb already in game_state["bombs"].
    """
    bombs = game_state["bombs"] + [(bomb_pos, BOMB_TIMER)]
    hypothetical = dict(game_state, bombs=bombs)
    new_danger = danger_map(hypothetical, bomb_power)
    return not escape_exists(hypothetical, new_danger, start=opponent_pos)


def threatened_by_opponent_bomb(game_state, pos=None, bomb_power=BOMB_POWER):
    """Could a living opponent with a bomb available hit `pos` (default: my
    own tile) by dropping a bomb at their current tile right now? This flags
    the threat a step before it would appear in game_state["bombs"].
    """
    field = game_state["field"]
    pos = game_state["self"][3] if pos is None else pos
    for _, _, bombs_left, opponent_pos in game_state["others"]:
        if bombs_left and pos in blast_coords(field, opponent_pos, bomb_power):
            return True
    return False


def opponent_features(game_state):
    danger = danger_map(game_state)
    occupied = _occupied(game_state)
    pos = game_state["self"][3]
    others = game_state["others"]

    info = _nearest_opponent_info(game_state, occupied, danger)
    direction = info[1] if info else None
    distance = info[0] if info else None
    trapped = opponent_trapped_by_bomb(game_state, pos, info[2][3]) if info else False
    nearest_has_bomb = bool(info[2][2]) if info else False

    vector = [int(direction == a) for a in ACTIONS]
    vector.append(0 if distance is None else min(distance, 10))
    vector.append(int(bomb_hits_opponent(game_state, pos)))
    vector.append(int(trapped))
    vector.append(int(threatened_by_opponent_bomb(game_state, pos)))
    vector.append(min(len(others), 3))
    vector.append(int(nearest_has_bomb))
    if len(vector) != FEATURE_SIZE:
        raise ValueError(f"feature vector length {len(vector)} != FEATURE_SIZE {FEATURE_SIZE}")
    return vector
