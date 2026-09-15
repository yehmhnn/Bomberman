"""Dueling Double-DQN agent: inference-side setup and action selection."""

from pathlib import Path

import numpy as np
import torch

from shared.features import track_position

from .model import DuelingQNetwork, action_mask_vector, new_recent_positions, select_action, state_to_vector

MODEL_FILE = Path(__file__).with_name("model.pt")


def load_checkpoint():
    """Returns the checkpoint dict (model_state, optimizer_state, total_steps,
    training_round) if MODEL_FILE exists, else None. A single dict format
    shared by callbacks.py (reads model_state) and train.py (reads the rest),
    so training state survives across separate process runs -- resuming a
    Kaggle session or moving from coin-heaven to classic doesn't silently
    reset epsilon back to 1.0 and lose optimizer momentum.
    """
    return torch.load(MODEL_FILE, map_location="cpu") if MODEL_FILE.is_file() else None


def setup(self):
    torch.set_num_threads(1)  # the tournament guarantees exactly one CPU thread
    self.device = torch.device("cpu")
    self.model = DuelingQNetwork().to(self.device)
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        self.model.load_state_dict(checkpoint["model_state"])
        self.logger.info(
            "Loaded trained weights from %s (total_steps=%d)",
            MODEL_FILE, checkpoint.get("total_steps", 0),
        )
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

    # Per-step DEBUG logging at training scale (millions of steps) grows the
    # log file unbounded -- agents.py's FileHandler has no rotation/cap, and
    # a 400MB+ log file previously appears to have brought a training run
    # down. Throttle to keep some visibility without the unbounded growth;
    # evaluation runs (self.train=False) are short, so log every step there.
    if not self.train or self.total_steps % 500 == 0:
        self.logger.debug("step=%s q=%s mask=%s epsilon=%.3f -> %s", getattr(self, "total_steps", "-"), q_values, mask, self.epsilon, action)
    return action
