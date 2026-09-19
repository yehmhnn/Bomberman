"""Clipped PPO with generalized advantage estimation (GAE)."""

from pathlib import Path

import numpy as np
import torch
from torch import optim

import events as e

from .callbacks import MODEL_FILE, load_checkpoint
from .model import (
    ACTIONS,
    BOMB_HITS_OPPONENT_IDX,
    COIN_DIST_IDX,
    CRATE_DIST_IDX,
    IN_DANGER_IDX,
    OPPONENT_TRAPPED_IDX,
    STATE_SIZE,
    action_mask_vector,
    masked_distribution,
    state_to_vector,
)

GAMMA = 0.95
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.20
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
ROLLOUT_SIZE = 1024
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
SAVE_EVERY_ROUNDS = 25

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.CRATE_DESTROYED: 0.2,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.05,
    e.SURVIVED_ROUND: 1.0,
}
STEP_REWARD = -0.01
SNIPER_HIT_BONUS = 1.5
SNIPER_TRAP_BONUS = 3.0


def setup_training(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.rollout = []
    self.train_rng = np.random.default_rng()
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.training_round = int(checkpoint.get("training_round", 0))
    else:
        self.total_steps = 0
        self.training_round = 0


def _potential(vector):
    if vector is None:
        return 0.0
    goal = -min(vector[COIN_DIST_IDX], vector[CRATE_DIST_IDX])
    danger = -2.0 if vector[IN_DANGER_IDX] > 0 else 0.0
    return float(goal + danger)


def _reward(old_vector, new_vector, events, action):
    reward = STEP_REWARD + sum(EVENT_REWARDS.get(event, 0.0) for event in events)
    if action == "BOMB":
        if old_vector[BOMB_HITS_OPPONENT_IDX] > 0:
            reward += SNIPER_HIT_BONUS
        if old_vector[OPPONENT_TRAPPED_IDX] > 0:
            reward += SNIPER_TRAP_BONUS
    return reward + GAMMA * _potential(new_vector) - _potential(old_vector)


def _value(self, vector):
    with torch.no_grad():
        _, value = self.model(torch.from_numpy(vector).unsqueeze(0))
    return float(value.item())


def _append_transition(self, new_game_state, events, action, done):
    if self.last_state_vec is None or action not in ACTIONS:
        return
    if done:
        new_vector = None
        next_value = 0.0
    else:
        new_vector = state_to_vector(new_game_state, self.recent_positions)
        next_value = _value(self, new_vector)

    self.rollout.append({
        "state": self.last_state_vec,
        "action": self.last_action_index,
        "old_log_probability": self.last_log_probability,
        "value": self.last_value,
        "next_value": next_value,
        "reward": _reward(self.last_state_vec, new_vector, events, action),
        "done": float(done),
        "mask": self.last_mask,
    })
    self.total_steps += 1
    if len(self.rollout) >= ROLLOUT_SIZE:
        _update(self)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    _append_transition(self, new_game_state, events, self_action, done=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    _append_transition(self, None, events, last_action, done=True)
    self.training_round += 1
    if self.training_round % SAVE_EVERY_ROUNDS == 0:
        if self.rollout:
            _update(self)
        save_model(self)


def _advantages_and_returns(rollout):
    advantages = np.zeros(len(rollout), dtype=np.float32)
    gae = 0.0
    for index in reversed(range(len(rollout))):
        transition = rollout[index]
        alive = 1.0 - transition["done"]
        delta = (
            transition["reward"]
            + GAMMA * alive * transition["next_value"]
            - transition["value"]
        )
        gae = delta + GAMMA * GAE_LAMBDA * alive * gae
        advantages[index] = gae
    values = np.asarray([item["value"] for item in rollout], dtype=np.float32)
    return advantages, advantages + values


def _update(self):
    rollout = self.rollout
    advantages, returns = _advantages_and_returns(rollout)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.from_numpy(np.stack([item["state"] for item in rollout]))
    actions = torch.tensor([item["action"] for item in rollout], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item["old_log_probability"] for item in rollout], dtype=torch.float32
    )
    masks = torch.from_numpy(np.stack([item["mask"] for item in rollout]))
    advantages_tensor = torch.from_numpy(advantages)
    returns_tensor = torch.from_numpy(returns)

    indices = np.arange(len(rollout))
    for _ in range(UPDATE_EPOCHS):
        self.train_rng.shuffle(indices)
        for start in range(0, len(indices), MINIBATCH_SIZE):
            batch = indices[start:start + MINIBATCH_SIZE]
            logits, values = self.model(states[batch])
            distribution = masked_distribution(logits, masks[batch])
            new_log_probabilities = distribution.log_prob(actions[batch])
            ratio = torch.exp(new_log_probabilities - old_log_probabilities[batch])
            unclipped = ratio * advantages_tensor[batch]
            clipped = torch.clamp(
                ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON
            ) * advantages_tensor[batch]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (returns_tensor[batch] - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + VALUE_COEFFICIENT * value_loss
                - ENTROPY_COEFFICIENT * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

    self.logger.info(
        "PPO update step=%d rollout=%d advantage_mean=%.3f",
        self.total_steps,
        len(rollout),
        float(advantages.mean()),
    )
    self.rollout = []


def save_model(self):
    checkpoint = {
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "total_steps": self.total_steps,
        "training_round": self.training_round,
    }
    temporary = Path(str(MODEL_FILE) + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(MODEL_FILE)
