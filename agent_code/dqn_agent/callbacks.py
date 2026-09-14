"""Dueling Double-DQN agent: inference-side setup and action selection."""

from pathlib import Path

import numpy as np
import torch

from shared.features import track_position

from .model import DuelingQNetwork, action_mask_vector, new_recent_positions, select_action, state_to_vector

MODEL_FILE = Path(__file__).with_name("model.pt")


def setup(self):
    torch.set_num_threads(1)  # the tournament guarantees exactly one CPU thread
    self.device = torch.device("cpu")
    self.model = DuelingQNetwork().to(self.device)
    if MODEL_FILE.is_file():
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
        self.logger.info("Loaded trained weights from %s", MODEL_FILE)
    else:
        self.logger.info("No saved model found at %s, starting from random weights", MODEL_FILE)
    self.model.eval()

    self.rng = np.random.default_rng()
    self.recent_positions = new_recent_positions()
    self.epsilon = 0.0  # train.py raises this during training; evaluation stays greedy


def act(self, game_state: dict) -> str:
    track_position(self.recent_positions, game_state["self"][3])
    state_vec = state_to_vector(game_state, self.recent_positions)
    mask = action_mask_vector(game_state)

    with torch.no_grad():
        q_values = self.model(torch.from_numpy(state_vec).unsqueeze(0)).squeeze(0).numpy()

    action = select_action(q_values, mask, self.epsilon if self.train else 0.0, self.rng)
    if self.train:
        # train.py's game_events_occurred reuses this instead of recomputing
        # the same BFS/danger-map features a second time.
        self.last_state_vec = state_vec

    self.logger.debug("q=%s mask=%s epsilon=%.3f -> %s", q_values, mask, self.epsilon, action)
    return action
