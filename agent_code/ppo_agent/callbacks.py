"""Inference and on-policy action sampling for PPO."""

from pathlib import Path

import numpy as np
import torch

from .model import (
    ACTIONS,
    ActorCritic,
    action_mask_vector,
    masked_distribution,
    new_recent_positions,
    state_to_vector,
)
from .shared.features import track_position

MODEL_FILE = Path(__file__).with_name("model.pt")


def load_checkpoint():
    return torch.load(MODEL_FILE, map_location="cpu") if MODEL_FILE.is_file() else None


def setup(self):
    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.model = ActorCritic().to(self.device)
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        self.model.load_state_dict(checkpoint["model_state"])
        self.logger.info(
            "Loaded PPO checkpoint at round=%d step=%d",
            checkpoint.get("training_round", 0),
            checkpoint.get("total_steps", 0),
        )
    else:
        self.logger.info("No PPO checkpoint found; using fresh parameters")

    self.model.train(self.train)
    self.rng = np.random.default_rng()
    self.recent_positions = new_recent_positions()
    self.last_state_vec = None
    self.last_action_index = None
    self.last_log_probability = None
    self.last_value = None
    self.last_mask = None


def act(self, game_state: dict) -> str:
    track_position(self.recent_positions, game_state["self"][3])
    state = state_to_vector(game_state, self.recent_positions)
    mask = action_mask_vector(game_state)
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)

    with torch.no_grad():
        logits, value = self.model(state_tensor)
        distribution = masked_distribution(logits, mask_tensor)
        if self.train:
            action = distribution.sample()
        else:
            masked_logits = logits.masked_fill(~mask_tensor, -1e9)
            action = masked_logits.argmax(dim=-1)

    action_index = int(action.item())
    if self.train:
        self.last_state_vec = state
        self.last_action_index = action_index
        self.last_log_probability = float(distribution.log_prob(action).item())
        self.last_value = float(value.item())
        self.last_mask = mask
    return ACTIONS[action_index]
