"""Dueling Double-DQN network and state encoding shared by callbacks.py and train.py.

Kept separate from callbacks.py/train.py so both can import it without either
depending on the other; also makes the network architecture independently
testable.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from settings import BOMB_POWER, BOMB_TIMER
from shared.danger import danger_map
from shared.features import RECENT_POSITIONS_MAXLEN, build_feature_vector
from shared.features import FEATURE_SIZE as SCALAR_FEATURE_SIZE
from shared.opponents import FEATURE_SIZE as OPPONENT_FEATURE_SIZE
from shared.opponents import opponent_features, threatening_opponent_count
from shared.safety import MOVE, _can_escape_own_bomb, _occupied, escape_exists, safe_action_mask, tile_free

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
STATE_SIZE = SCALAR_FEATURE_SIZE + OPPONENT_FEATURE_SIZE + 1  # 24 + 10 + 1 = 35

# Indices into the state vector produced by state_to_vector, needed by
# train.py's reward shaping. Mirrors the layout comments in
# shared/features.py; kept here instead of re-deriving them from game_state.
COIN_DIST_IDX = 20
CRATE_DIST_IDX = 21
IN_DANGER_IDX = 22
BOMB_HITS_OPPONENT_IDX = SCALAR_FEATURE_SIZE + 5   # shared/opponents.py index 5
OPPONENT_TRAPPED_IDX = SCALAR_FEATURE_SIZE + 6     # shared/opponents.py index 6
CROSSFIRE_COUNT_IDX = SCALAR_FEATURE_SIZE + OPPONENT_FEATURE_SIZE  # appended last, index 34


def new_recent_positions():
    return deque(maxlen=RECENT_POSITIONS_MAXLEN)


def state_to_vector(game_state, recent_positions=None):
    """Fixed-size float32 feature vector: the shared coin/crate/safety
    features, opponent targeting/threat/trapping features, and a crossfire
    count (how many armed opponents could hit my tile right now) appended
    last -- being caught between two simultaneous potential blasts is a
    materially worse spot than either alone, and the network otherwise has
    no direct signal for "surrounded" versus "one threat, one escape."
    """
    vector = (
        build_feature_vector(game_state, recent_positions)
        + opponent_features(game_state)
        + [threatening_opponent_count(game_state)]
    )
    return np.asarray(vector, dtype=np.float32)


def _augment_with_opponent_threats(game_state):
    """game_state with each armed opponent's current tile added as a
    hypothetical bomb (same BOMB_TIMER convention used for our own
    about-to-be-placed bomb -- see shared.safety._can_escape_own_bomb).

    The shield otherwise only reacts to bombs already in game_state["bombs"];
    an opponent standing next to us could drop one on their very next turn
    with zero warning. Merging this into "bombs" before computing danger
    means every downstream check (move safety, escape_exists, our own BOMB
    action's escape check) automatically accounts for it too, since they all
    key off game_state["bombs"].
    """
    opponent_bombs = [(pos, BOMB_TIMER) for (_, _, bombs_left, pos) in game_state["others"] if bombs_left]
    if not opponent_bombs:
        return game_state
    return dict(game_state, bombs=game_state["bombs"] + opponent_bombs)


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
    """Boolean array over ACTIONS: the strong shield above, degrading to the
    shared (weaker, immediate-safety-only) mask when the shield finds no
    action survives full lookahead -- a genuinely hopeless spot should still
    prefer "safe for now" over an arbitrary choice.
    """
    mask = shield_mask(game_state)
    if not any(mask.values()):
        mask = safe_action_mask(game_state)
    return np.array([mask[a] for a in ACTIONS], dtype=bool)


N_QUANTILES = 32


class DuelingQNetwork(nn.Module):
    """Distributional (quantile regression) dueling Double-DQN.

    Rather than a single expected Q(s,a), outputs N_QUANTILES estimates of
    the return distribution per action (QR-DQN: Dabney, Rowland, Bellemare &
    Munos, 2018), combined with the same dueling decomposition as before,
    applied per-quantile: Q_tau(s,a) = V_tau(s) + (A_tau(s,a) - mean_a
    A_tau(s,a)). Modeling the full distribution rather than just its mean
    lets the policy tell "high average value but occasionally fatal" apart
    from "reliably good" -- directly the kill-more/die-less tradeoff that
    tuning scalar reward magnitudes alone can only blunt-instrument.

    forward() returns the full (batch, n_actions, N_QUANTILES) tensor for
    training; q_values() collapses it to the familiar (batch, n_actions)
    mean for action selection, so act()/select_action's interface doesn't
    need to change.
    """

    def __init__(self, state_size=STATE_SIZE, n_actions=len(ACTIONS), hidden=(256, 128), n_quantiles=N_QUANTILES):
        super().__init__()
        h1, h2 = hidden
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.trunk = nn.Sequential(
            nn.Linear(state_size, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.value_head = nn.Linear(h2, n_quantiles)
        self.advantage_head = nn.Linear(h2, n_actions * n_quantiles)
        # tau_i = (i + 0.5) / N: the quantile fraction each output slot targets.
        self.register_buffer("tau", (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles)

    def forward(self, x):
        z = self.trunk(x)
        value = self.value_head(z).unsqueeze(1)  # (batch, 1, n_quantiles)
        advantage = self.advantage_head(z).view(-1, self.n_actions, self.n_quantiles)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def q_values(self, x):
        return self.forward(x).mean(dim=-1)


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
