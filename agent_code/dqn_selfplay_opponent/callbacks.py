"""Frozen self-play opponent for training dqn_agent.

Loads a random snapshot from dqn_agent's checkpoint pool
(agent_code/dqn_agent/snapshots/) and plays it greedily, never
training. Used as one or more opponent slots during dqn_agent's
training so it faces a diverse set of its own past strategies
(fictitious self-play) instead of only the four fixed, always-
identical provided agents -- the actual tournament is against other
teams' unknown agents, so training exclusively against a small fixed
set risks learning to exploit their specific quirks rather than
robust general play.

Deliberately has no train.py: this agent is never meant to occupy a
--train slot. Placing it anywhere in --agents *after* the first N in
--train N is what makes it act as a fixed opponent.
"""

import random
from pathlib import Path

import numpy as np
import torch

from agent_code.dqn_agent.model import DuelingQNetwork, action_mask_vector, new_recent_positions, select_action, state_to_vector
from shared.features import track_position

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "dqn_agent" / "snapshots"
FALLBACK_MODEL = Path(__file__).resolve().parent.parent / "dqn_agent" / "model.pt"


def _pick_checkpoint():
    """A snapshot (plain state_dict) if the pool has any, else dqn_agent's
    own current checkpoint (a {"model_state": ..., ...} dict) as a fallback
    for early training before any snapshot has been saved yet.
    """
    candidates = list(SNAPSHOT_DIR.glob("snapshot_round*.pt")) if SNAPSHOT_DIR.is_dir() else []
    if candidates:
        return random.choice(candidates), "state_dict"
    if FALLBACK_MODEL.is_file():
        return FALLBACK_MODEL, "checkpoint_dict"
    return None, None


def setup(self):
    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.model = DuelingQNetwork().to(self.device)

    path, kind = _pick_checkpoint()
    if path is not None:
        data = torch.load(path, map_location=self.device)
        state_dict = data["model_state"] if kind == "checkpoint_dict" else data
        self.model.load_state_dict(state_dict)
        self.logger.info("Loaded self-play opponent snapshot %s", path.name)
    else:
        self.logger.info("No snapshot or model found yet; opponent starts from random weights")
    self.model.eval()
    self.model.zero_noise()  # deterministic: represents this past version's best strategy, not noisy exploration

    self.rng = np.random.default_rng()
    self.recent_positions = new_recent_positions()


def act(self, game_state: dict) -> str:
    track_position(self.recent_positions, game_state["self"][3])
    state_vec = state_to_vector(game_state, self.recent_positions)
    mask = action_mask_vector(game_state)
    with torch.no_grad():
        q_values = self.model.q_values(torch.from_numpy(state_vec).unsqueeze(0)).squeeze(0).numpy()
    return select_action(q_values, mask, 0.0, self.rng)  # always greedy: its best learned strategy
