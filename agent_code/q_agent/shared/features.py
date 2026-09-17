"""Coin/crate targeting + the assembled fixed-size feature vector."""

from collections import deque

from settings import BOMB_POWER
from .danger import blast_coords, danger_map
from .safety import MOVE, _occupied, earliest_danger, safe_action_mask, tile_free

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
FEATURE_SIZE = 24
# index layout of build_feature_vector's output:
#  0- 3 move safety mask      (UP, RIGHT, DOWN, LEFT)
#  4    WAIT safety
#  5    BOMB safety (safe now AND escapable)
#  6- 9 physical obstruction  (UP, RIGHT, DOWN, LEFT)
# 10    bombing here would hit a crate
# 11    standing on a recently visited tile
# 12-15 direction to nearest coin, one-hot         (UP, RIGHT, DOWN, LEFT)
# 16-19 direction to nearest bombable spot, one-hot
# 20    distance to nearest coin, capped at 10
# 21    distance to nearest bombable spot, capped at 10
# 22    current tile is in a known blast window
# 23    steps until that danger, capped at 5
RECENT_POSITIONS_MAXLEN = 8


def track_position(recent_positions, pos):
    """Append the agent's current position. recent_positions must start as
    collections.deque(maxlen=RECENT_POSITIONS_MAXLEN). Call this BEFORE
    build_feature_vector for the same step, so bit 11 includes the current tile.
    """
    recent_positions.append(pos)
    return recent_positions


def bfs_target(game_state, occupied, goals, danger=None):
    """Shortest-path first-step direction + distance to the nearest tile in
    goals. Shared by coin/crate targeting here and opponent targeting in
    shared/opponents.py — kept public so it isn't duplicated a third time.
    """
    start = game_state["self"][3]
    if not goals:
        return None, None
    if start in goals:
        return None, 0
    seen = {start}
    queue = deque([(start, None, 0)])
    while queue:
        pos, first, dist = queue.popleft()
        for action, (dx, dy) in MOVE.items():
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in seen:
                continue
            if npos not in goals and not tile_free(game_state, npos, occupied):
                continue
            if first is None and danger is not None and 0 in danger.get(npos, ()):
                continue  # don't suggest a first step that is lethal right now
            nfirst = first if first is not None else action
            ndist = dist + 1
            if npos in goals:
                return nfirst, ndist
            seen.add(npos)
            queue.append((npos, nfirst, ndist))
    return None, None


def _crate_adjacent_tiles(field):
    w, h = field.shape
    goals = set()
    for cx in range(w):
        for cy in range(h):
            if field[cx, cy] == 1:
                for dx, dy in MOVE.values():
                    n = (cx + dx, cy + dy)
                    if 0 <= n[0] < w and 0 <= n[1] < h and field[n] == 0:
                        goals.add(n)
    return goals


def nearest_coin(game_state, occupied=None, danger=None):
    occupied = _occupied(game_state) if occupied is None else occupied
    return bfs_target(game_state, occupied, set(game_state["coins"]), danger)


def nearest_crate(game_state, occupied=None, danger=None):
    occupied = _occupied(game_state) if occupied is None else occupied
    return bfs_target(game_state, occupied, _crate_adjacent_tiles(game_state["field"]), danger)


def bomb_hits_crate(game_state, pos):
    field = game_state["field"]
    return any(field[p] == 1 for p in blast_coords(field, pos, BOMB_POWER))


def build_feature_vector(game_state, recent_positions=None):
    """recent_positions: a deque(maxlen=RECENT_POSITIONS_MAXLEN) built with
    track_position, or None if not tracked. Call track_position with the
    current position before calling this function for the same step.
    """
    if recent_positions is not None:
        assert len(recent_positions) <= RECENT_POSITIONS_MAXLEN

    danger = danger_map(game_state)
    mask = safe_action_mask(game_state, danger)
    occupied = _occupied(game_state)
    pos = game_state["self"][3]

    coin_action, coin_dist = nearest_coin(game_state, occupied, danger)
    crate_action, crate_dist = nearest_crate(game_state, occupied, danger)
    here = earliest_danger(danger, pos)

    vector = [int(mask[a]) for a in ACTIONS]
    vector.append(int(mask["WAIT"]))
    vector.append(int(mask["BOMB"]))
    vector += [
        int(not tile_free(game_state, (pos[0] + MOVE[a][0], pos[1] + MOVE[a][1]), occupied))
        for a in ACTIONS
    ]
    vector.append(int(bomb_hits_crate(game_state, pos)))
    vector.append(int(recent_positions is not None and pos in recent_positions))
    vector += [int(coin_action == a) for a in ACTIONS]
    vector += [int(crate_action == a) for a in ACTIONS]
    vector.append(0 if coin_dist is None else min(coin_dist, 10))
    vector.append(0 if crate_dist is None else min(crate_dist, 10))
    vector.append(int(here is not None))
    vector.append(0 if here is None else min(here, 5))
    if len(vector) != FEATURE_SIZE:
        raise ValueError(f"feature vector length {len(vector)} != FEATURE_SIZE {FEATURE_SIZE}")
    return vector