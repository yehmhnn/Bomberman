"""Action selection for the Stage-2 tabular Q-learning agent."""

from pathlib import Path
import pickle

import numpy as np


from shared.tabular import ACTIONS, state_to_features, valid_action_indices

MODEL_FILE = Path(__file__).with_name("q_table_stage2.pkl")


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
