"""Action-level safety checks on top of shared.danger."""

from collections import deque

from settings import BOMB_POWER, BOMB_TIMER
from .danger import danger_map

MOVE = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


def _occupied(game_state):
    occ = {pos for pos, _ in game_state["bombs"]}
    occ.update(o[3] for o in game_state["others"])
    return occ


def tile_free(game_state, pos, occupied=None):
    if game_state["field"][pos] != 0:
        return False
    if occupied is None:
        occupied = _occupied(game_state)
    return pos not in occupied


def earliest_danger(danger, pos):
    steps = danger.get(pos)
    return min(steps) if steps else None


def _can_escape_own_bomb(game_state, pos, bomb_power):
    bombs = game_state["bombs"] + [(pos, BOMB_TIMER)] # not -1! : hits the same update_bombs pass as this step
    hypothetical = dict(game_state, bombs=bombs)
    new_danger = danger_map(hypothetical, bomb_power)
    return escape_exists(hypothetical, new_danger, pos)


def safe_action_mask(game_state, danger=None, bomb_power=BOMB_POWER):
    if danger is None:
        danger = danger_map(game_state, bomb_power)
    x, y = game_state["self"][3]
    occupied = _occupied(game_state)
    here_safe = 0 not in danger.get((x, y), ())

    mask = {}
    for action, (dx, dy) in MOVE.items():
        target = (x + dx, y + dy)
        mask[action] = tile_free(game_state, target, occupied) and 0 not in danger.get(target, ())
    mask["WAIT"] = here_safe
    mask["BOMB"] = (
        bool(game_state["self"][2])
        and here_safe
        and _can_escape_own_bomb(game_state, (x, y), bomb_power)
    )
    return mask


def escape_exists(game_state, danger, start=None):
    occupied = _occupied(game_state)
    start = start or game_state["self"][3]
    horizon = max((s for steps in danger.values() for s in steps), default=-1) + 1

    seen = {(start, 0)}
    queue = deque([(start, 0)])
    while queue:
        pos, step = queue.popleft()
        if step >= horizon:
            return True
        for npos in [pos] + [(pos[0] + dx, pos[1] + dy) for dx, dy in MOVE.values()]:
            if npos != pos and not tile_free(game_state, npos, occupied):
                continue
            # start (relative-step 0) is already verified safe by the caller
            # (e.g. safe_action_mask's `here_safe`/`0 not in danger.get(target)`
            # checks); this expansion computes the state at relative-step
            # step+1, so that -- not the current step -- is what must be
            # checked against danger. Checking `step` here under-counts by
            # one and can approve a move sequence that is one step too slow
            # to clear a bomb's blast (e.g. WAIT once then flee, when only
            # fleeing immediately actually escapes in time).
            if (step + 1) in danger.get(npos, ()):
                continue
            state = (npos, step + 1)
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return False