"""Pure functions for Bomberman blast geometry and time-aware danger."""

from collections import deque

import numpy as np


MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
BOMB_POWER = 3


def blast_coordinates(field, origin, power=BOMB_POWER):
    x, y = origin
    blast = [(x, y)]
    for dx, dy in MOVE_DELTAS:
        for distance in range(1, power + 1):
            tile = (x + dx * distance, y + dy * distance)
            if field[tile] == -1:
                break
            blast.append(tile)
    return blast


def danger_steps(game_state):
    field = game_state["field"]
    danger = np.full(field.shape, np.inf)
    danger[game_state["explosion_map"] > 0] = 0
    for position, timer in game_state["bombs"]:
        explosion_step = timer + 1
        for tile in blast_coordinates(field, position):
            danger[tile] = min(danger[tile], explosion_step)
    return danger


def can_escape_after_bomb(game_state, max_moves=4):
    field = game_state["field"]
    start = game_state["self"][3]
    own_blast = set(blast_coordinates(field, start))
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])
    blocked.add(start)
    known_danger = danger_steps(game_state)
    queue = deque([(start, 0)])
    visited = {(start, 0)}
    while queue:
        position, elapsed = queue.popleft()
        if elapsed > 0 and position not in own_blast:
            return True
        if elapsed == max_moves:
            continue
        for dx, dy in MOVE_DELTAS:
            target = (position[0] + dx, position[1] + dy)
            arrival = elapsed + 1
            if (
                field[target] == 0
                and target not in blocked
                and known_danger[target] > arrival
                and (target, arrival) not in visited
            ):
                visited.add((target, arrival))
                queue.append((target, arrival))
    return False
