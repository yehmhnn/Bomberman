"""On-policy action selection and features for the Stage-1 SARSA agent."""

from collections import deque
from pathlib import Path
import pickle

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
MODEL_FILE = Path(__file__).with_name("q_table.pkl")


def setup(self):
    """Load the SARSA table or initialize an empty one."""
    if MODEL_FILE.is_file():
        with MODEL_FILE.open("rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded %d states from %s", len(self.q_table), MODEL_FILE)
    else:
        self.q_table = {}
        self.logger.info("Starting with an empty SARSA table")

    self.rng = np.random.default_rng()
    self.epsilon = 0.0
    self.planned_state = None
    self.planned_action = None


def act(self, game_state: dict) -> str:
    """Execute the action already sampled for SARSA, or sample a new one.

    SARSA's update needs the actual next action a'. During training, ``train.py``
    samples that action when it observes s' and stores it here. The following
    call executes exactly that stored action, keeping the update on-policy.
    """
    state = state_to_features(game_state)
    if self.train and state == self.planned_state and self.planned_action is not None:
        action = self.planned_action
        self.planned_state = None
        self.planned_action = None
        return action
    return choose_action(self, game_state)


def choose_action(self, game_state: dict) -> str:
    """Sample one action from the current epsilon-greedy policy."""
    state = state_to_features(game_state)
    allowed = valid_action_indices(game_state)
    if self.train and self.rng.random() < self.epsilon:
        action_index = int(self.rng.choice(allowed))
    else:
        values = q_values(self.q_table, state)
        best_value = np.max(values[allowed])
        best_actions = allowed[np.isclose(values[allowed], best_value)]
        action_index = int(self.rng.choice(best_actions))
    return ACTIONS[action_index]


def q_values(q_table: dict, state: tuple) -> np.ndarray:
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS), dtype=np.float64)
    return q_table[state]


def valid_action_indices(game_state: dict) -> np.ndarray:
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
    if game_state is None:
        return None
    allowed = set(valid_action_indices(game_state))
    passable = tuple(int(index in allowed) for index in range(4))
    directions, distance = shortest_coin_directions(game_state)
    coin_directions = tuple(int(index in directions) for index in range(4))
    distance_bucket = min(distance, 5) if distance is not None else 0
    return passable + coin_directions + (distance_bucket,)


def shortest_coin_directions(game_state: dict) -> tuple:
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
    if game_state is None:
        return 0.0
    _, distance = shortest_coin_directions(game_state)
    return -float(distance) if distance is not None else 0.0
