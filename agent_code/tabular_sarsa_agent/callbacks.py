"""On-policy action selection for the opponent-aware SARSA agent."""

import os
from pathlib import Path
import pickle

import numpy as np


from .shared.tabular import ACTIONS, state_to_features, valid_action_indices

MODEL_STAGE = int(os.environ.get("TABULAR_STAGE", "4"))
if MODEL_STAGE not in (3, 4):
    raise ValueError("TABULAR_STAGE must be 3 or 4")
MODEL_FILE = Path(__file__).with_name(f"q_table_stage{MODEL_STAGE}.pkl")
PRIOR_MODEL_FILES = (
    ([Path(__file__).with_name("q_table_stage3.pkl")] if MODEL_STAGE == 4 else [])
    + [Path(__file__).with_name("q_table_stage2.pkl")]
)


def setup(self):
    """Load the SARSA table or initialize an empty one."""
    if MODEL_FILE.is_file():
        with MODEL_FILE.open("rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded %d states from %s", len(self.q_table), MODEL_FILE)
    else:
        self.q_table = {}
        self.logger.info("Starting with an empty SARSA table")

    self.prior_q_tables = []
    for prior_file in PRIOR_MODEL_FILES:
        if prior_file == MODEL_FILE or not prior_file.is_file():
            continue
        with prior_file.open("rb") as file:
            self.prior_q_tables.append(pickle.load(file))
        self.logger.info("Using %s to initialize unseen states", prior_file.name)

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
        values = q_values(self.q_table, state, self.prior_q_tables)
        best_value = np.max(values[allowed])
        best_actions = allowed[np.isclose(values[allowed], best_value)]
        action_index = int(self.rng.choice(best_actions))
    return ACTIONS[action_index]


def q_values(q_table: dict, state: tuple, prior_q_tables=()) -> np.ndarray:
    if state not in q_table:
        initial = None
        for prior in prior_q_tables:
            if state in prior:
                initial = prior[state]
                break
            stage2_state = state[:24]
            if stage2_state in prior:
                initial = prior[stage2_state]
                break
        q_table[state] = (
            np.zeros(len(ACTIONS), dtype=np.float64)
            if initial is None else np.asarray(initial, dtype=np.float64).copy()
        )
    return q_table[state]
