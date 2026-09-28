"""Shared Stage-2 representation for the tabular Q-learning and SARSA agents."""

import numpy as np

from shared.danger import danger_map
from shared.features import build_feature_vector, nearest_coin, nearest_crate
from shared.safety import MOVE, earliest_danger, escape_exists, safe_action_mask


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def state_to_features(game_state):
    """Return the shared 24-value board summary as a hashable table key.

    The representation includes safe actions, physical obstructions, useful
    bombing, directions/distances to coins and bombable crates, and current
    blast danger.  The recent-position bit is deliberately left disabled: a
    tabular state must be reproduced identically in ``act`` and in the
    subsequent learning callback.
    """
    if game_state is None:
        return None
    vector = build_feature_vector(game_state, recent_positions=None)
    survivable = survivable_action_mask(game_state)
    vector[:6] = [int(survivable[action]) for action in ACTIONS]
    return tuple(vector)


def survivable_action_mask(game_state):
    """Require each allowed action to preserve a complete future escape path."""
    danger = danger_map(game_state)
    mask = safe_action_mask(game_state, danger)
    position = game_state["self"][3]

    for action in ACTIONS[:5]:
        if not mask[action]:
            continue
        if action == "WAIT":
            target = position
        else:
            dx, dy = MOVE[action]
            target = (position[0] + dx, position[1] + dy)
        mask[action] = escape_exists(game_state, danger, start=target)
    return mask


def valid_action_indices(game_state):
    """Indices of actions that are immediately safe and physically possible."""
    mask = survivable_action_mask(game_state)
    allowed = [index for index, action in enumerate(ACTIONS) if mask[action]]

    # In a position with no survivable choice, WAIT is still an executable
    # action and keeps action selection / bootstrapping well-defined.
    if not allowed:
        allowed = [ACTIONS.index("WAIT")]
    return np.asarray(allowed, dtype=np.int64)


def stage2_potential(game_state):
    """Potential Phi(s): progress to a useful target minus current danger.

    A target is either a visible coin or a free tile from which a crate can be
    bombed.  Because reward shaping uses gamma*Phi(s') - Phi(s), this dense
    progress signal does not by itself change the optimal policy.
    """
    if game_state is None:
        return 0.0

    danger = danger_map(game_state)
    _, coin_distance = nearest_coin(game_state, danger=danger)
    _, crate_distance = nearest_crate(game_state, danger=danger)
    distances = [distance for distance in (coin_distance, crate_distance) if distance is not None]
    goal_term = -float(min(distances)) if distances else 0.0

    position = game_state["self"][3]
    danger_step = earliest_danger(danger, position)
    danger_term = 0.0 if danger_step is None else -4.0 / (danger_step + 1.0)
    return goal_term + danger_term
