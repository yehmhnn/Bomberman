"""One-step tabular Q-learning updates for the Stage-2 agent."""

from pathlib import Path
import pickle

import numpy as np

import events as e
from shared.features import bomb_hits_crate
from shared.tabular import stage2_potential
from .callbacks import (
    ACTIONS,
    MODEL_FILE,
    q_values,
    state_to_features,
    valid_action_indices,
)


ALPHA = 0.20
GAMMA = 0.90
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
SAVE_EVERY_ROUNDS = 1

EVENT_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.CRATE_DESTROYED: 2.0,
    e.KILLED_SELF: -30.0,
    e.GOT_KILLED: -30.0,
    e.INVALID_ACTION: -5.0,
    e.WAITED: -0.2,
    e.BOMB_DROPPED: -0.25,
}
STEP_REWARD = -0.05
# Placing a bomb temporarily makes Phi(s') worse because the agent enters its
# own future blast window.  This bonus must exceed that immediate danger term;
# the action mask already guarantees that an escape route exists.
USEFUL_BOMB_BONUS = 3.0


def setup_training(self):
    """Initialize counters and the exploration rate for training."""
    self.training_round = 0
    self.epsilon = EPSILON_START


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    """Learn from the transition (s, a, r, s') produced by one game step."""
    reward = reward_from_transition(old_game_state, self_action, new_game_state, events)
    update_q_table(
        self,
        state_to_features(old_game_state),
        self_action,
        reward,
        state_to_features(new_game_state),
        new_game_state,
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list,
):
    """Apply the terminal update, decay exploration, and save periodically."""
    reward = reward_from_transition(last_game_state, last_action, None, events)
    update_q_table(
        self,
        state_to_features(last_game_state),
        last_action,
        reward,
        None,
        None,
    )

    self.training_round += 1
    self.epsilon = max(
        EPSILON_MIN,
        EPSILON_START * EPSILON_DECAY ** self.training_round,
    )
    if self.training_round % SAVE_EVERY_ROUNDS == 0:
        save_model(self)


def update_q_table(
    self,
    state: tuple,
    action: str,
    reward: float,
    next_state: tuple,
    next_game_state: dict,
):
    """Apply Q(s,a) <- Q(s,a) + alpha [target - Q(s,a)]."""
    if state is None or action not in ACTIONS:
        return

    action_index = ACTIONS.index(action)
    current_q = q_values(self.q_table, state)[action_index]

    if next_state is None:
        target = reward
    else:
        next_values = q_values(self.q_table, next_state)
        allowed = valid_action_indices(next_game_state)
        target = reward + GAMMA * np.max(next_values[allowed])

    td_error = target - current_q
    self.q_table[state][action_index] += ALPHA * td_error
    self.logger.debug(
        "Q update: action=%s reward=%.3f target=%.3f error=%.3f new_q=%.3f",
        action,
        reward,
        target,
        td_error,
        self.q_table[state][action_index],
    )


def reward_from_transition(
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    events: list,
) -> float:
    """Combine task reward with policy-invariant potential-based shaping.

    F(s,s') = gamma * Phi(s') - Phi(s), where Phi is negative distance to a
    useful coin/crate target plus a danger penalty. This provides dense
    feedback while preserving the optimal policy of the discounted MDP.
    """
    reward = STEP_REWARD + sum(EVENT_REWARDS.get(event, 0.0) for event in events)
    if action == "BOMB" and bomb_hits_crate(old_game_state, old_game_state["self"][3]):
        reward += USEFUL_BOMB_BONUS
    shaping = GAMMA * stage2_potential(new_game_state) - stage2_potential(old_game_state)
    return reward + shaping


def save_model(self):
    """Persist the learned table inside the agent directory."""
    temporary_file = Path(str(MODEL_FILE) + ".tmp")
    with temporary_file.open("wb") as file:
        pickle.dump(self.q_table, file)
    temporary_file.replace(MODEL_FILE)
