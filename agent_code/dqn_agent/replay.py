"""Proportional prioritized experience replay (Schaul et al., 2016).

Standard uniform replay treats a rare self-kill transition the same as any
of the thousands of "survived this step" transitions around it -- exactly
the opposite of what we want when self-kills are the failure mode we're
trying to eliminate. Prioritized replay samples transitions with larger
|TD error| more often (with importance-sampling weights correcting the bias
this introduces), and death transitions additionally get a priority boost
on insertion so they're not diluted inside a ~100k-transition buffer before
their TD error has even been computed once.
"""

import numpy as np


class SumTree:
    """Binary tree where each leaf holds a priority and each internal node
    holds the sum of its children: O(log n) priority-proportional sampling
    and O(log n) priority updates. A flat array + np.random.choice(p=...)
    would be O(n) per call, which matters once this is sampled every few
    env steps against a 100k-capacity buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, cumulative):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            idx = left if cumulative <= self.tree[left] else right
            if idx == right:
                cumulative -= self.tree[left]
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    EPSILON = 1e-3          # keeps every priority strictly positive
    ALPHA = 0.6             # 0 = uniform sampling, 1 = fully greedy on |TD error|
    BETA_START = 0.4
    BETA_FRAMES = 200_000   # steps over which the IS-weight correction anneals to 1.0
    DEATH_PRIORITY_BOOST = 4.0
    DEATH_REWARD_THRESHOLD = -1.0  # below this + done=True counts as a death transition

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0

    def push(self, transition):
        _, _, reward, _, done, _ = transition
        priority = self.max_priority
        if done and reward < self.DEATH_REWARD_THRESHOLD:
            priority *= self.DEATH_PRIORITY_BOOST
        self.tree.add(priority ** self.ALPHA, transition)

    def sample(self, batch_size, rng, step):
        beta = min(1.0, self.BETA_START + step * (1.0 - self.BETA_START) / self.BETA_FRAMES)
        segment = self.tree.total / batch_size
        indices, transitions, priorities = [], [], []
        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            idx, priority, data = self.tree.get(rng.uniform(low, high))
            indices.append(idx)
            transitions.append(data)
            priorities.append(priority)

        probs = np.asarray(priorities) / self.tree.total
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + self.EPSILON) ** self.ALPHA
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, float(priority))
            self.max_priority = max(self.max_priority, float(priority))

    def __len__(self):
        return self.tree.size
