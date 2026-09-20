"""Clipped PPO with generalized advantage estimation (GAE)."""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch import optim

import events as e

from .callbacks import MODEL_FILE, MODEL_STAGE, load_checkpoint
from .model import (
    ACTIONS,
    BOMB_HITS_OPPONENT_IDX,
    COIN_DIST_IDX,
    CRATE_DIST_IDX,
    IN_DANGER_IDX,
    OPPONENT_DIST_IDX,
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
ENTROPY_COEFFICIENT = float(os.environ.get("PPO_ENTROPY_COEFFICIENT", "0.01"))
if ENTROPY_COEFFICIENT < 0:
    raise ValueError("PPO_ENTROPY_COEFFICIENT must be non-negative")
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
    self.train_rng = np.random.default_rng(self.ppo_seed + 1)
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        resuming_stage = bool(checkpoint.get("_is_current_stage", True))
        if resuming_stage and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.training_round = int(checkpoint.get("training_round", 0))
        self.stage_training_round = (
            int(checkpoint.get("stage_training_round", self.training_round))
            if resuming_stage else 0
        )
        self.update_count = int(checkpoint.get("update_count", 0)) if resuming_stage else 0
        self.optimized_steps = int(checkpoint.get("optimized_steps", 0))
        if resuming_stage:
            self.rollout = _restore_rollout(checkpoint.get("rollout", []))
    else:
        self.total_steps = 0
        self.training_round = 0
        self.stage_training_round = 0
        self.update_count = 0
        self.optimized_steps = 0


def _potential(vector, stage=MODEL_STAGE):
    if vector is None:
        return 0.0
    # The feature representation uses zero both when a target is absent and
    # when its distance is zero. An absent target must not win the minimum and
    # erase the distance signal from the other target type.
    distances = [
        float(vector[index])
        for index in (COIN_DIST_IDX, CRATE_DIST_IDX)
        if vector[index] > 0
    ]
    goal = -min(distances) if distances else 0.0
    opponent_distance = float(vector[OPPONENT_DIST_IDX])
    hunt = -opponent_distance if stage >= 3 and opponent_distance > 0 else 0.0
    danger = -2.0 if vector[IN_DANGER_IDX] > 0 else 0.0
    return float(goal + hunt + danger)


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
        "events": tuple(events),
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
    self.stage_training_round += 1
    if self.training_round % SAVE_EVERY_ROUNDS == 0:
        if self.rollout:
            _update(self)
    # The seeded curriculum runner changes process between board seeds. Save
    # unfinished rollout transitions so those boundaries do not discard data
    # or force a tiny PPO update after every episode.
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
    raw_advantages, returns = _advantages_and_returns(rollout)
    advantages = (
        raw_advantages - raw_advantages.mean()
    ) / (raw_advantages.std() + 1e-8)

    states = torch.from_numpy(np.stack([item["state"] for item in rollout]))
    actions = torch.tensor([item["action"] for item in rollout], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item["old_log_probability"] for item in rollout], dtype=torch.float32
    )
    masks = torch.from_numpy(np.stack([item["mask"] for item in rollout]))
    advantages_tensor = torch.from_numpy(advantages)
    returns_tensor = torch.from_numpy(returns)

    return_variance = float(np.var(returns))
    explained_variance = (
        0.0 if return_variance < 1e-8
        else 1.0 - float(np.var(returns - np.asarray(
            [item["value"] for item in rollout], dtype=np.float32
        ))) / return_variance
    )
    policy_losses = []
    value_losses = []
    entropies = []
    approximate_kls = []
    clip_fractions = []

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

            with torch.no_grad():
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approximate_kls.append(float(
                    (old_log_probabilities[batch] - new_log_probabilities).mean().item()
                ))
                clip_fractions.append(float(
                    ((ratio - 1.0).abs() > CLIP_EPSILON).float().mean().item()
                ))

    self.update_count += 1
    self.optimized_steps += len(rollout)
    _write_diagnostic(
        self,
        rollout,
        raw_advantages,
        returns,
        policy_losses,
        value_losses,
        entropies,
        approximate_kls,
        clip_fractions,
        explained_variance,
    )
    self.logger.info(
        "PPO update step=%d rollout=%d advantage_mean=%.3f entropy=%.3f kl=%.5f",
        self.total_steps,
        len(rollout),
        float(raw_advantages.mean()),
        float(np.mean(entropies)),
        float(np.mean(approximate_kls)),
    )
    self.rollout = []


def _write_diagnostic(
    self,
    rollout,
    advantages,
    returns,
    policy_losses,
    value_losses,
    entropies,
    approximate_kls,
    clip_fractions,
    explained_variance,
):
    """Append one compact, report-ready row per PPO update."""
    path = MODEL_FILE.with_suffix(".diagnostics.csv")
    actions = np.bincount(
        [item["action"] for item in rollout], minlength=len(ACTIONS)
    ) / len(rollout)
    all_events = [event for item in rollout for event in item.get("events", ())]
    row = {
        "update": self.update_count,
        "total_steps": self.total_steps,
        "optimized_steps": self.optimized_steps,
        "training_round": self.training_round,
        "stage_training_round": self.stage_training_round,
        "rollout_size": len(rollout),
        "reward_mean": float(np.mean([item["reward"] for item in rollout])),
        "return_mean": float(np.mean(returns)),
        "advantage_mean": float(np.mean(advantages)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approximate_kl": float(np.mean(approximate_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
        "explained_variance": explained_variance,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "coins": all_events.count(e.COIN_COLLECTED),
        "crates": all_events.count(e.CRATE_DESTROYED),
        "kills": all_events.count(e.KILLED_OPPONENT),
        "self_kills": all_events.count(e.KILLED_SELF),
        "deaths": all_events.count(e.GOT_KILLED),
    }
    row.update({f"action_{action.lower()}": float(actions[index])
                for index, action in enumerate(ACTIONS)})
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_model(self):
    checkpoint = {
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "total_steps": self.total_steps,
        "training_round": self.training_round,
        "stage_training_round": self.stage_training_round,
        "update_count": self.update_count,
        "optimized_steps": self.optimized_steps,
        "rollout": _serializable_rollout(self.rollout),
        "entropy_coefficient": ENTROPY_COEFFICIENT,
    }
    temporary = Path(str(MODEL_FILE) + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(MODEL_FILE)


def _serializable_rollout(rollout):
    """Convert NumPy arrays to tensors accepted by safe ``torch.load``."""
    saved = []
    for transition in rollout:
        item = dict(transition)
        item["state"] = torch.from_numpy(np.asarray(item["state"]))
        item["mask"] = torch.from_numpy(np.asarray(item["mask"]))
        saved.append(item)
    return saved


def _restore_rollout(rollout):
    restored = []
    for transition in rollout:
        item = dict(transition)
        for key in ("state", "mask"):
            value = item[key]
            if isinstance(value, torch.Tensor):
                item[key] = value.cpu().numpy()
        restored.append(item)
    return restored
