"""Action selection and state representation for a tabular Q-learning agent.

This first version intentionally targets the ``coin-heaven`` scenario. It learns
movement and coin collection before bombs and opponents are introduced.
"""

from collections import deque
from pathlib import Path
import pickle

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
MODEL_FILE = Path(__file__).with_name("q_table.pkl")


def setup(self):
    """Load learned action values, or start with an empty Q table."""
    if MODEL_FILE.is_file():
        with MODEL_FILE.open("rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded %d states from %s", len(self.q_table), MODEL_FILE)
    else:
        self.q_table = {}
        self.logger.info("Starting with an empty Q table")

    # A dedicated generator makes random tie-breaking and exploration seedable.
    self.rng = np.random.default_rng()
    self.epsilon = 0.0


def act(self, game_state: dict) -> str:
    """Choose an action with an epsilon-greedy policy.

    During evaluation ``epsilon`` is zero. During training, ``train.py`` updates
    it after every round.
    """
    state = state_to_features(game_state)
    allowed = valid_action_indices(game_state)

    if self.train and self.rng.random() < self.epsilon:
        action_index = int(self.rng.choice(allowed))
        self.logger.debug("Exploring with epsilon %.3f", self.epsilon)
    else:
        values = q_values(self.q_table, state)
        best_value = np.max(values[allowed])
        best_actions = allowed[np.isclose(values[allowed], best_value)]
        action_index = int(self.rng.choice(best_actions))

    return ACTIONS[action_index]


def q_values(q_table: dict, state: tuple) -> np.ndarray:
    """Return Q(s, ·), creating six zero estimates for an unseen state."""
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS), dtype=np.float64)
    return q_table[state]


def valid_action_indices(game_state: dict) -> np.ndarray:
    """Return executable Stage-1 actions (four moves plus WAIT).

    Masking walls is part of the environment interface, not the learned policy.
    BOMB is intentionally disabled until the Stage-2 danger representation is
    implemented.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]
    occupied = {position for position, _ in game_state["bombs"]}
    occupied.update(other[3] for other in game_state["others"])

    allowed = []
    for index, (dx, dy) in enumerate(MOVE_DELTAS):
        target = (x + dx, y + dy)
        if field[target] == 0 and target not in occupied:
            allowed.append(index)
    allowed.append(ACTIONS.index("WAIT"))
    return np.asarray(allowed, dtype=np.int64)


def state_to_features(game_state: dict) -> tuple:
    """Compress a large game state into a small, hashable Markov-state proxy.

    The tuple contains four passability bits, four bits identifying first steps
    on shortest paths to the nearest coin, and a coarse distance bucket. States
    with the same tuple share the same row of the Q table.
    """
    if game_state is None:
        return None

    allowed = set(valid_action_indices(game_state))
    passable = tuple(int(index in allowed) for index in range(4))
    directions, distance = shortest_coin_directions(game_state)
    coin_directions = tuple(int(index in directions) for index in range(4))
    distance_bucket = min(distance, 5) if distance is not None else 0
    return passable + coin_directions + (distance_bucket,)


def shortest_coin_directions(game_state: dict) -> tuple:
    """Find all first moves belonging to a shortest path to any visible coin."""
    coins = set(game_state["coins"])
    if not coins:
        return set(), None

    field = game_state["field"]
    start = game_state["self"][3]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])

    queue = deque()
    visited_distance = {start: 0}
    for direction, (dx, dy) in enumerate(MOVE_DELTAS):
        neighbor = (start[0] + dx, start[1] + dy)
        if field[neighbor] == 0 and neighbor not in blocked:
            queue.append((neighbor, 1, direction))
            visited_distance[neighbor] = 1

    best_distance = None
    best_directions = set()
    while queue:
        position, distance, first_direction = queue.popleft()
        if best_distance is not None and distance > best_distance:
            break
        if position in coins:
            best_distance = distance
            best_directions.add(first_direction)
            continue

        for dx, dy in MOVE_DELTAS:
            neighbor = (position[0] + dx, position[1] + dy)
            new_distance = distance + 1
            old_distance = visited_distance.get(neighbor)
            if (
                field[neighbor] == 0
                and neighbor not in blocked
                and (old_distance is None or new_distance <= old_distance)
            ):
                visited_distance[neighbor] = new_distance
                queue.append((neighbor, new_distance, first_direction))

    return best_directions, best_distance


def coin_potential(game_state: dict) -> float:
    """Potential Phi(s) = negative shortest-path distance to a visible coin."""
    if game_state is None:
        return 0.0
    _, distance = shortest_coin_directions(game_state)
    return -float(distance) if distance is not None else 0.0
