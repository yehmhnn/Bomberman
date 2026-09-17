"""Linear Q-learning agent on the shared feature + opponent-awareness vector."""
from collections import deque
from pathlib import Path

import numpy as np

from .shared.features import RECENT_POSITIONS_MAXLEN, track_position
from .shared.safety import safe_action_mask
from .q_learning import ACTIONS, LinearQ
from .state import N_FEATURES, build_state_vector

MODEL_FILE = Path(__file__).with_name("model.npz")


def setup(self):
    if MODEL_FILE.is_file():
        self.q = LinearQ.load(MODEL_FILE, n_features=N_FEATURES)
        self.logger.info("Loaded weights from %s", MODEL_FILE)
    else:
        self.q = LinearQ(n_features=N_FEATURES)
        self.logger.info("Starting with zero-initialized weights")
    self.rng = np.random.default_rng()
    self.epsilon = 0.0
    self.recent_positions = deque(maxlen=RECENT_POSITIONS_MAXLEN)


def act(self, game_state):
    track_position(self.recent_positions, game_state["self"][3])
    features = build_state_vector(game_state, self.recent_positions)

    mask = safe_action_mask(game_state)
    allowed = tuple(a for a in ACTIONS if mask[a]) or ("WAIT",)

    if self.train and self.rng.random() < self.epsilon:
        action = str(self.rng.choice(allowed))
    else:
        action, _ = self.q.best_action(features, allowed)
        if action is None:
            action = allowed[0]

    self.last_features = features
    return action