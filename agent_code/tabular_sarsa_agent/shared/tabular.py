"""Shared representation for the tabular Q-learning and SARSA agents."""

import numpy as np

from settings import BOMB_TIMER

from .danger import danger_map
from .features import build_feature_vector, nearest_coin, nearest_crate
from .opponents import nearest_opponent, opponent_features, threatening_opponent_count
from .safety import MOVE, earliest_danger, escape_exists, safe_action_mask


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def _augment_with_opponent_threats(game_state):
    """Treat every armed opponent as a possible next-turn bomb source.

    This is a safety-only hypothetical: it does not claim the opponent will
    certainly bomb.  It prevents the agent from choosing a move whose only
    escape route would disappear if a nearby opponent used its available
    bomb on the next turn.
    """
    possible_bombs = [
        (position, BOMB_TIMER)
        for _, _, bombs_left, position in game_state["others"]
        if bombs_left
    ]
    if not possible_bombs:
        return game_state
    return dict(game_state, bombs=game_state["bombs"] + possible_bombs)


def state_to_features(game_state):
    """Return the shared 34-value board summary as a hashable table key.

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
    vector += opponent_features(game_state)
    return tuple(vector)


def survivable_action_mask(game_state):
    """Require actions to survive existing and possible opponent bombs."""
    augmented = _augment_with_opponent_threats(game_state)
    danger = danger_map(augmented)
    mask = safe_action_mask(augmented, danger)
    position = game_state["self"][3]

    for action in ACTIONS[:5]:
        if not mask[action]:
            continue
        if action == "WAIT":
            target = position
        else:
            dx, dy = MOVE[action]
            target = (position[0] + dx, position[1] + dy)
        mask[action] = escape_exists(augmented, danger, start=target)
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


def stage34_potential(game_state):
    if game_state is None:
        return 0.0

    direction, distance = nearest_opponent(game_state)
    del direction
    pursuit_term = 0.0 if distance is None else -0.5 * float(distance)
    threat_term = -2.0 * float(threatening_opponent_count(game_state))
    return stage2_potential(game_state) + pursuit_term + threat_term
