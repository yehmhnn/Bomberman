"""Double-DQN training: n-step replay buffer, target network, reward shaping."""

from collections import deque
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
    DuelingQNetwork,
    state_to_vector,
)
from .replay import PrioritizedReplayBuffer
from .symmetry import augment_transition

GAMMA = 0.95
N_STEP = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
MIN_REPLAY_BEFORE_TRAINING = 1_000
TRAIN_EVERY_STEPS = 4
TARGET_SYNC_EVERY_STEPS = 2_000
GRAD_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 150_000

SAVE_EVERY_ROUNDS = 25
SNAPSHOT_DIR = Path(__file__).with_name("snapshots")
SNAPSHOT_EVERY_ROUNDS = 1000
MAX_SNAPSHOTS = 15  # small checkpoints (~0.5MB each); caps disk use, not a meaningful memory concern

# Official-score events kept at (roughly) their real point values so shaping
# stays a dense hint on top of the true objective, not a replacement for it.
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

# Reward for choosing BOMB when the feature vector already shows it would hit
# (or fully trap) the nearest opponent -- both indices are read from the
# state *before* the action, and BOMB is only ever selectable when the
# shield's escape check passed, so this rewards a safe kill opportunity, not
# recklessness. Trapping (near-certain kill) is worth more than merely being
# in blast range (the opponent can still walk out before it detonates).
SNIPER_HIT_BONUS = 1.5
SNIPER_TRAP_BONUS = 3.0


def setup_training(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.target_model = DuelingQNetwork().to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.replay = PrioritizedReplayBuffer(REPLAY_CAPACITY)  # not persisted: refills after a resume
    self.n_step_buffer = deque(maxlen=N_STEP)
    self.train_rng = np.random.default_rng()
    self.last_state_vec = None

    checkpoint = load_checkpoint()
    if checkpoint is not None and "optimizer_state" in checkpoint:
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_steps = checkpoint["total_steps"]
        self.training_round = checkpoint["training_round"]
        self.logger.info(
            "Resuming training at step=%d round=%d", self.total_steps, self.training_round,
        )
    else:
        self.total_steps = 0
        self.training_round = 0
    self.epsilon = _epsilon_for_step(self.total_steps)


def _potential_from_vector(vec):
    """Potential Phi(s), built from features already computed for this step
    (never recomputed from game_state). Combines negative safe-path distance
    to the nearer of {coin, useful crate} with a penalty for currently
    standing in a known blast window.

    Potential-based (Ng, Harada & Russell 1999): reward = STEP_REWARD +
    events + gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged while
    densifying the otherwise sparse official-score reward.
    """
    goal_term = -min(vec[COIN_DIST_IDX], vec[CRATE_DIST_IDX])
    danger_term = -2.0 if vec[IN_DANGER_IDX] > 0 else 0.0
    return float(goal_term + danger_term)


def _sniper_bonus(old_vec, action):
    if action != "BOMB":
        return 0.0
    bonus = 0.0
    if old_vec[BOMB_HITS_OPPONENT_IDX] > 0:
        bonus += SNIPER_HIT_BONUS
    if old_vec[OPPONENT_TRAPPED_IDX] > 0:
        bonus += SNIPER_TRAP_BONUS
    return bonus


def _shaped_reward(old_vec, new_vec, events, action):
    reward = STEP_REWARD + sum(EVENT_REWARDS.get(ev, 0.0) for ev in events)
    reward += _sniper_bonus(old_vec, action)
    new_potential = 0.0 if new_vec is None else _potential_from_vector(new_vec)
    reward += GAMMA * new_potential - _potential_from_vector(old_vec)
    return reward


def _epsilon_for_step(step):
    if step >= EPSILON_DECAY_STEPS:
        return EPSILON_END
    frac = step / EPSILON_DECAY_STEPS
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def _emit_n_step(self, k):
    """Pop the n_step_buffer's oldest entry, replaced by its k-step return
    and a bootstrap target k steps later (or, if that later entry is
    terminal, no bootstrap at all). Pushes all 8 dihedral-symmetric variants
    (see symmetry.py) rather than just the one actually played -- the board
    has 8-fold symmetry the network otherwise has no way to know about."""
    state0, action0, _, _, _ = self.n_step_buffer[0]
    discounted_reward = sum((GAMMA ** i) * self.n_step_buffer[i][2] for i in range(k))
    _, _, _, state_k, done_k = self.n_step_buffer[k - 1]
    for sym_state0, sym_action0, sym_state_k in augment_transition(state0, action0, state_k):
        self.replay.push((sym_state0, ACTIONS.index(sym_action0), discounted_reward, sym_state_k, done_k, k))


def _push_transition(self, state_vec, action, reward, next_state_vec, done):
    self.n_step_buffer.append((state_vec, action, reward, next_state_vec, done))
    if done:
        # Episode over: drain every pending transition, each truncated to
        # however many real steps remain until the terminal one.
        while self.n_step_buffer:
            _emit_n_step(self, len(self.n_step_buffer))
            self.n_step_buffer.popleft()
    elif len(self.n_step_buffer) == N_STEP:
        _emit_n_step(self, N_STEP)
        self.n_step_buffer.popleft()


def _quantile_huber_loss(predicted, target, tau, kappa=1.0):
    """Per-sample quantile regression loss (Dabney et al., 2018).

    predicted, target: (batch, N) quantile estimates. tau: (N,) the quantile
    fraction each *predicted* slot targets. Averaged over the N x N pairing
    of predicted vs target quantiles; not reduced over the batch dimension
    so callers can apply importance-sampling weights first.
    """
    u = target.unsqueeze(1) - predicted.unsqueeze(2)  # u[b, i, j] = target_j - predicted_i
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    quantile_weight = (tau.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    return (quantile_weight * huber).mean(dim=(1, 2))


def _optimize(self):
    batch, indices, is_weights = self.replay.sample(BATCH_SIZE, self.train_rng, self.total_steps)
    states = torch.from_numpy(np.stack([b[0] for b in batch]))
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([b[3] for b in batch]))
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    steps = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    weights = torch.from_numpy(is_weights)
    n_quantiles = self.model.n_quantiles

    quantiles = self.model(states)  # (batch, n_actions, N)
    action_index = actions.view(-1, 1, 1).expand(-1, 1, n_quantiles)
    predicted = quantiles.gather(1, action_index).squeeze(1)  # (batch, N)

    with torch.no_grad():
        # Double DQN: the online network's mean Q-values pick the next
        # action, the target network's full quantile distribution for that
        # action supplies the target -- decouples selection from evaluation
        # to cut the overestimation bias plain DQN has under sparse/spiky
        # rewards, same as before, just applied to a distribution now.
        next_actions = self.model.q_values(next_states).argmax(dim=1)
        next_action_index = next_actions.view(-1, 1, 1).expand(-1, 1, n_quantiles)
        next_quantiles = self.target_model(next_states).gather(1, next_action_index).squeeze(1)
        targets = rewards.unsqueeze(1) + (GAMMA ** steps).unsqueeze(1) * (1.0 - dones).unsqueeze(1) * next_quantiles

    # Priorities are updated from the raw TD error of the *means* (a scalar
    # proxy consistent with the pre-distributional priority scheme); the
    # training loss itself is the full quantile regression loss, weighted by
    # the prioritized-sampling importance-correction.
    self.replay.update_priorities(indices, (targets.mean(dim=1) - predicted.mean(dim=1)).detach().numpy())
    elementwise_loss = _quantile_huber_loss(predicted, targets, self.model.tau)
    loss = (elementwise_loss * weights).mean()

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
    self.optimizer.step()
    if self.total_steps % 500 == 0:  # see callbacks.act's comment: unbounded per-step logging is what broke a run
        self.logger.debug("train step=%d loss=%.4f replay=%d", self.total_steps, loss.item(), len(self.replay))


def _maybe_train(self):
    if len(self.replay) < MIN_REPLAY_BEFORE_TRAINING:
        return
    if self.total_steps % TRAIN_EVERY_STEPS != 0:
        return
    _optimize(self)
    if self.total_steps % TARGET_SYNC_EVERY_STEPS == 0:
        self.target_model.load_state_dict(self.model.state_dict())


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    new_vec = state_to_vector(new_game_state, self.recent_positions)
    reward = _shaped_reward(self.last_state_vec, new_vec, events, self_action)
    _push_transition(self, self.last_state_vec, self_action, reward, new_vec, done=False)

    self.total_steps += 1
    self.epsilon = _epsilon_for_step(self.total_steps)
    _maybe_train(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    reward = _shaped_reward(self.last_state_vec, None, events, last_action)
    terminal_vec = np.zeros(STATE_SIZE, dtype=np.float32)  # unused when done=True; never bootstrapped
    _push_transition(self, self.last_state_vec, last_action, reward, terminal_vec, done=True)
    _maybe_train(self)

    self.training_round += 1
    self.logger.info(
        "round=%d steps=%d epsilon=%.3f replay=%d",
        self.training_round, self.total_steps, self.epsilon, len(self.replay),
    )
    if self.training_round % SAVE_EVERY_ROUNDS == 0:
        save_model(self)
    if self.training_round % SNAPSHOT_EVERY_ROUNDS == 0:
        save_snapshot(self)


def save_model(self):
    checkpoint = {
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "total_steps": self.total_steps,
        "training_round": self.training_round,
    }
    temporary_file = Path(str(MODEL_FILE) + ".tmp")
    torch.save(checkpoint, temporary_file)
    temporary_file.replace(MODEL_FILE)


def save_snapshot(self):
    """Add the current weights to the self-play opponent pool (plain weights
    only -- dqn_selfplay_opponent never trains, so it needs no optimizer
    state). Fictitious self-play: sampling a diverse pool of past selves
    trains more robust play than a fixed set of always-identical opponents,
    and avoids the cyclic/overfit behavior pure single-opponent self-play is
    prone to (Heinrich & Silver, 2016; see also AlphaStar-style league
    training). Capped at MAX_SNAPSHOTS, oldest evicted first.
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    torch.save(self.model.state_dict(), SNAPSHOT_DIR / f"snapshot_round{self.training_round}.pt")
    snapshots = sorted(SNAPSHOT_DIR.glob("snapshot_round*.pt"), key=lambda p: p.stat().st_mtime)
    for stale in snapshots[:-MAX_SNAPSHOTS]:
        stale.unlink()
