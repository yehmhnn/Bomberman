"""Dueling Double-DQN network and state encoding shared by callbacks.py and train.py.

Kept separate from callbacks.py/train.py so both can import it without either
depending on the other; also makes the network architecture independently
testable.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from shared.features import RECENT_POSITIONS_MAXLEN, build_feature_vector
from shared.features import FEATURE_SIZE as SCALAR_FEATURE_SIZE
from shared.opponents import FEATURE_SIZE as OPPONENT_FEATURE_SIZE
from shared.opponents import opponent_features
from shared.safety import safe_action_mask

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
STATE_SIZE = SCALAR_FEATURE_SIZE + OPPONENT_FEATURE_SIZE  # 24 + 10 = 34

# Indices into the state vector produced by state_to_vector, needed by
# train.py's reward shaping. Mirrors the layout comments in
# shared/features.py; kept here instead of re-deriving them from game_state.
COIN_DIST_IDX = 20
CRATE_DIST_IDX = 21
IN_DANGER_IDX = 22


def new_recent_positions():
    return deque(maxlen=RECENT_POSITIONS_MAXLEN)


def state_to_vector(game_state, recent_positions=None):
    """Fixed-size float32 feature vector: the shared coin/crate/safety
    features concatenated with opponent targeting/threat/trapping features.
    """
    vector = build_feature_vector(game_state, recent_positions) + opponent_features(game_state)
    return np.asarray(vector, dtype=np.float32)


def action_mask_vector(game_state):
    """Boolean array over ACTIONS from the shared safety mask (physically
    legal AND not immediately lethal per shared.safety.safe_action_mask).
    """
    mask = safe_action_mask(game_state)
    return np.array([mask[a] for a in ACTIONS], dtype=bool)


class DuelingQNetwork(nn.Module):
    """Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).

    Splitting value and advantage matters here because many Bomberman states
    have similar value regardless of which of the 6 actions is taken (e.g.
    nothing dangerous nearby, several safe moves) -- the dueling head avoids
    wasting capacity re-learning V from scratch for every action.
    """

    def __init__(self, state_size=STATE_SIZE, n_actions=len(ACTIONS), hidden=(256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(state_size, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.value_head = nn.Linear(h2, 1)
        self.advantage_head = nn.Linear(h2, n_actions)

    def forward(self, x):
        z = self.trunk(x)
        value = self.value_head(z)
        advantage = self.advantage_head(z)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def select_action(q_values, mask, epsilon, rng):
    """Epsilon-greedy over ACTIONS, restricted to `mask` where possible.

    q_values: array of length len(ACTIONS).
    mask: boolean array of length len(ACTIONS); True = allowed by the shared
    safety mask. If nothing is masked True (a genuinely inescapable spot),
    fall back to considering all actions rather than returning nothing.
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    allowed = np.flatnonzero(mask) if mask.any() else np.arange(len(ACTIONS))

    if rng.random() < epsilon:
        return ACTIONS[int(rng.choice(allowed))]

    best = allowed[np.isclose(q_values[allowed], q_values[allowed].max())]
    return ACTIONS[int(rng.choice(best))]
