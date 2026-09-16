"""Blast-time danger map for the current game state."""

from settings import BOMB_POWER

WALL = -1
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # UP, RIGHT, DOWN, LEFT


def blast_coords(field, pos, power):
    x, y = pos
    coords = [(x, y)]
    for dx, dy in DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == WALL:
                break
            coords.append((nx, ny))
    return coords


def danger_map(game_state, bomb_power=BOMB_POWER):
    field = game_state["field"]
    danger = {}

    def mark(pos, steps):
        if steps:
            danger.setdefault(pos, set()).update(steps)

    for (bx, by), countdown in game_state["bombs"]:
        for pos in blast_coords(field, (bx, by), bomb_power):
            mark(pos, {countdown, countdown + 1})

    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None:
        xs, ys = explosion_map.nonzero()
        for x, y in zip(xs, ys):
            mark((int(x), int(y)), set(range(int(explosion_map[x, y]))))

    return danger


def is_safe(danger, pos, step):
    return step not in danger.get(pos, ())