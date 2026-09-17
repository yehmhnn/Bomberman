"""Shared reward shaping."""
import events as e
from shared.danger import danger_map
from shared.features import nearest_coin, nearest_crate

GAMMA = 0.95

EVENT_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.KILLED_OPPONENT: 20.0,
    e.KILLED_SELF: -15.0,
    e.GOT_KILLED: -20.0,
    e.CRATE_DESTROYED: 5.0,
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.2,
}
STEP_PENALTY = -0.02


def _goal_potential(game_state):
    """None at a terminal state: shaping is skipped there, never zeroed."""
    if game_state is None:
        return None
    danger = danger_map(game_state)
    _, coin_dist = nearest_coin(game_state, danger=danger)
    _, crate_dist = nearest_crate(game_state, danger=danger)
    dists = [d for d in (coin_dist, crate_dist) if d is not None]
    if not dists:
        return 0.0
    return -float(min(min(dists), 10))


def reward_from_transition(old_game_state, new_game_state, events):
    reward = STEP_PENALTY + sum(EVENT_REWARDS.get(ev, 0.0) for ev in events)
    old_phi = _goal_potential(old_game_state)
    new_phi = _goal_potential(new_game_state)
    if old_phi is not None and new_phi is not None:
        reward += GAMMA * new_phi - old_phi
    return reward