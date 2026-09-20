"""Actor-critic network and DQN-compatible state/action representation."""

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from settings import BOMB_POWER, BOMB_TIMER

from .shared.danger import danger_map
from .shared.features import RECENT_POSITIONS_MAXLEN, build_feature_vector
from .shared.features import FEATURE_SIZE as SCALAR_FEATURE_SIZE
from .shared.opponents import FEATURE_SIZE as OPPONENT_FEATURE_SIZE
from .shared.opponents import opponent_features, threatening_opponent_count
from .shared.safety import (
    MOVE,
    _can_escape_own_bomb,
    _occupied,
    escape_exists,
    safe_action_mask,
    tile_free,
)

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
STATE_SIZE = SCALAR_FEATURE_SIZE + OPPONENT_FEATURE_SIZE + 1

COIN_DIST_IDX = 20
CRATE_DIST_IDX = 21
IN_DANGER_IDX = 22
OPPONENT_DIST_IDX = SCALAR_FEATURE_SIZE + 4
BOMB_HITS_OPPONENT_IDX = SCALAR_FEATURE_SIZE + 5
OPPONENT_TRAPPED_IDX = SCALAR_FEATURE_SIZE + 6


def new_recent_positions():
    return deque(maxlen=RECENT_POSITIONS_MAXLEN)


def state_to_vector(game_state, recent_positions=None):
    vector = (
        build_feature_vector(game_state, recent_positions)
        + opponent_features(game_state)
        + [threatening_opponent_count(game_state)]
    )
    return np.asarray(vector, dtype=np.float32)


def _augment_with_opponent_threats(game_state):
    hypothetical_bombs = [
        (position, BOMB_TIMER)
        for _, _, bombs_left, position in game_state["others"]
        if bombs_left
    ]
    if not hypothetical_bombs:
        return game_state
    return dict(game_state, bombs=game_state["bombs"] + hypothetical_bombs)


def shield_mask(game_state):
    """Require every selected action to leave at least one escape path."""
    x, y = game_state["self"][3]
    augmented = _augment_with_opponent_threats(game_state)
    occupied = _occupied(augmented)
    danger = danger_map(augmented)
    here_safe = 0 not in danger.get((x, y), ())

    mask = {}
    for action, (dx, dy) in MOVE.items():
        target = (x + dx, y + dy)
        mask[action] = (
            tile_free(augmented, target, occupied)
            and 0 not in danger.get(target, ())
            and escape_exists(augmented, danger, start=target)
        )
    mask["WAIT"] = here_safe and escape_exists(augmented, danger, start=(x, y))
    mask["BOMB"] = (
        bool(game_state["self"][2])
        and here_safe
        and _can_escape_own_bomb(augmented, (x, y), BOMB_POWER)
    )
    return mask


def action_mask_vector(game_state):
    mask = shield_mask(game_state)
    if not any(mask.values()):
        mask = safe_action_mask(game_state)
    return np.asarray([mask[action] for action in ACTIONS], dtype=bool)


class ActorCritic(nn.Module):
    def __init__(self, state_size=STATE_SIZE, hidden=(128, 128)):
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(state_size, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(h2, len(ACTIONS))
        self.value_head = nn.Linear(h2, 1)
        self.apply(self._initialize)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, states):
        hidden = self.trunk(states)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def masked_distribution(logits, masks):
    """Categorical policy with invalid/unsafe actions assigned zero mass."""
    masks = masks.bool()
    if masks.ndim == 1:
        masks = masks.unsqueeze(0)
        logits = logits.unsqueeze(0) if logits.ndim == 1 else logits
    no_action = ~masks.any(dim=-1)
    if no_action.any():
        masks = masks.clone()
        masks[no_action] = True
    return Categorical(logits=logits.masked_fill(~masks, -1e9))
