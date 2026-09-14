"""Dueling Double-DQN network and state encoding shared by callbacks.py and train.py.

Kept separate from callbacks.py/train.py so both can import it without either
depending on the other; also makes the network architecture independently
testable.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from settings import BOMB_POWER
from shared.danger import danger_map
from shared.features import RECENT_POSITIONS_MAXLEN, build_feature_vector
from shared.features import FEATURE_SIZE as SCALAR_FEATURE_SIZE
from shared.opponents import FEATURE_SIZE as OPPONENT_FEATURE_SIZE
from shared.opponents import opponent_features
from shared.safety import MOVE, _can_escape_own_bomb, _occupied, escape_exists, safe_action_mask, tile_free

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
STATE_SIZE = SCALAR_FEATURE_SIZE + OPPONENT_FEATURE_SIZE  # 24 + 10 = 34

# Indices into the state vector produced by state_to_vector, needed by
# train.py's reward shaping. Mirrors the layout comments in
# shared/features.py; kept here instead of re-deriving them from game_state.
COIN_DIST_IDX = 20
CRATE_DIST_IDX = 21
IN_DANGER_IDX = 22
BOMB_HITS_OPPONENT_IDX = SCALAR_FEATURE_SIZE + 5   # shared/opponents.py index 5
OPPONENT_TRAPPED_IDX = SCALAR_FEATURE_SIZE + 6     # shared/opponents.py index 6


def new_recent_positions():
    return deque(maxlen=RECENT_POSITIONS_MAXLEN)


def state_to_vector(game_state, recent_positions=None):
    """Fixed-size float32 feature vector: the shared coin/crate/safety
    features concatenated with opponent targeting/threat/trapping features.
    """
    vector = build_feature_vector(game_state, recent_positions) + opponent_features(game_state)
    return np.asarray(vector, dtype=np.float32)


def shield_mask(game_state):
    """A genuine multi-step safety shield.

    shared.safety.safe_action_mask only checks whether the destination tile
    is dangerous *this instant* (relative step 0). That lets an agent take a
    sequence of individually-"safe-right-now" moves into a dead end that
    becomes lethal a few steps later -- exactly the failure mode behind a
    high self-kill rate. This additionally requires that an escape route
    still exists *after* taking the action, using the same danger map and
    BFS reachability check shared.safety already uses for the BOMB action,
    generalized to every action. Strictly stronger (a subset of) safe_
    action_mask's True set, so it can legitimately mask everything out in a
    genuinely hopeless spot -- action_mask_vector below degrades gracefully
    when that happens instead of returning an all-False mask.
    """
    x, y = game_state["self"][3]
    occupied = _occupied(game_state)
    danger = danger_map(game_state)
    here_safe = 0 not in danger.get((x, y), ())

    mask = {}
    for action, (dx, dy) in MOVE.items():
        target = (x + dx, y + dy)
        mask[action] = (
            tile_free(game_state, target, occupied)
            and 0 not in danger.get(target, ())
            and escape_exists(game_state, danger, start=target)
        )
    mask["WAIT"] = here_safe and escape_exists(game_state, danger, start=(x, y))
    mask["BOMB"] = (
        bool(game_state["self"][2])
        and here_safe
        and _can_escape_own_bomb(game_state, (x, y), BOMB_POWER)
    )
    return mask


def action_mask_vector(game_state):
    """Boolean array over ACTIONS: the strong shield above, degrading to the
    shared (weaker, immediate-safety-only) mask when the shield finds no
    action survives full lookahead -- a genuinely hopeless spot should still
    prefer "safe for now" over an arbitrary choice.
    """
    mask = shield_mask(game_state)
    if not any(mask.values()):
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
