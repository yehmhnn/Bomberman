"""Factorized-Gaussian noisy linear layer (Fortunato et al., 2017, "Noisy
Networks for Exploration").

Standard epsilon-greedy explores by picking a uniformly random action a
fixed fraction of the time, regardless of state -- the same randomness
whether the agent is in a clearly-safe open area or a tight spot. Noisy
layers instead add learnable, resample-able noise to the network's own
weights, so exploration becomes state-dependent and the network can learn
where noise actually helps rather than acting randomly everywhere alike.
This is the last of the 6 Rainbow DQN components not already present here
(Double DQN, Dueling, Prioritized Replay, n-step, and Distributional/QR-DQN
are already in model.py/train.py).

Noise is controlled explicitly via reset_noise()/zero_noise() rather than
piggybacking on nn.Module.training/.eval(), since those are already used
elsewhere in this codebase for unrelated reasons and overloading them here
would be an easy source of subtle bugs. zero_noise() gives fully
deterministic (mu-only) behavior for the tournament policy; reset_noise()
resamples fresh noise for training-time exploration.
"""

import torch
from torch import nn


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / (self.in_features ** 0.5))
        self.bias_sigma.data.fill_(self.sigma0 / (self.out_features ** 0.5))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample fresh factorized noise -- call once per decision during
        training so exploration varies, never during eval."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def zero_noise(self):
        """Deterministic mu-only behavior, for the final tournament policy."""
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)
