"""One-step tabular SARSA updates for the Stage-2 agent."""

from pathlib import Path
import pickle

import numpy as np

import events as e
from shared.features import bomb_hits_crate
from shared.tabular import stage2_potential
from .callbacks import (
    ACTIONS,
    MODEL_FILE,
    choose_action,
    q_values,
    state_to_features,
)


ALPHA = 0.20
GAMMA = 0.90
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
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
# Compensate for the immediate danger-potential drop after a safe crate bomb.
USEFUL_BOMB_BONUS = 3.0


def setup_training(self):
    self.training_round = 0
    self.epsilon = EPSILON_START
    self.planned_state = None
    self.planned_action = None


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    """Sample a' from the current policy, then update using Q(s', a')."""
    next_action = choose_action(self, new_game_state)
    next_state = state_to_features(new_game_state)
    reward = reward_from_transition(old_game_state, self_action, new_game_state, events)
    update_sarsa(
        self,
        state_to_features(old_game_state),
        self_action,
        reward,
        next_state,
        next_action,
    )

    # The sampled a' must become the action actually executed on the next step.
    self.planned_state = next_state
    self.planned_action = next_action


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list,
):
    reward = reward_from_transition(last_game_state, last_action, None, events)
    update_sarsa(
        self,
        state_to_features(last_game_state),
        last_action,
        reward,
        None,
        None,
    )
    self.planned_state = None
    self.planned_action = None
    self.training_round += 1
    self.epsilon = max(
        EPSILON_MIN,
        EPSILON_START * EPSILON_DECAY ** self.training_round,
    )
    save_model(self)


def update_sarsa(
    self,
    state: tuple,
    action: str,
    reward: float,
    next_state: tuple,
    next_action: str,
):
    """Apply Q(s,a) <- Q(s,a) + alpha[r + gamma Q(s',a') - Q(s,a)]."""
    if state is None or action not in ACTIONS:
        return

    action_index = ACTIONS.index(action)
    current_q = q_values(self.q_table, state)[action_index]
    if next_state is None:
        target = reward
    else:
        next_action_index = ACTIONS.index(next_action)
        target = reward + GAMMA * q_values(self.q_table, next_state)[next_action_index]

    td_error = target - current_q
    self.q_table[state][action_index] += ALPHA * td_error
    self.logger.debug(
        "SARSA update: action=%s reward=%.3f target=%.3f error=%.3f new_q=%.3f",
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
    reward = STEP_REWARD + sum(EVENT_REWARDS.get(event, 0.0) for event in events)
    if action == "BOMB" and bomb_hits_crate(old_game_state, old_game_state["self"][3]):
        reward += USEFUL_BOMB_BONUS
    shaping = GAMMA * stage2_potential(new_game_state) - stage2_potential(old_game_state)
    return reward + shaping


def save_model(self):
    temporary_file = Path(str(MODEL_FILE) + ".tmp")
    with temporary_file.open("wb") as file:
        pickle.dump(self.q_table, file)
    temporary_file.replace(MODEL_FILE)
