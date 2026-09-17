import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.dqn_agent.noisy import NoisyLinear  # noqa: E402


def test_zero_noise_matches_plain_mu_only_linear():
    layer = NoisyLinear(10, 5)
    layer.zero_noise()
    x = torch.randn(3, 10)
    expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(layer(x), expected)


def test_reset_noise_changes_the_output():
    layer = NoisyLinear(10, 5)
    x = torch.randn(3, 10)
    layer.reset_noise()
    out1 = layer(x).clone()
    layer.reset_noise()
    out2 = layer(x).clone()
    assert not torch.allclose(out1, out2), "different noise samples should give different outputs"


def test_zero_noise_is_reproducible_across_calls():
    layer = NoisyLinear(10, 5)
    x = torch.randn(3, 10)
    layer.zero_noise()
    out1 = layer(x).clone()
    out2 = layer(x).clone()
    assert torch.allclose(out1, out2), "with zero noise, repeated calls must be identical"


def test_output_shape():
    layer = NoisyLinear(7, 4)
    x = torch.randn(6, 7)
    assert layer(x).shape == (6, 4)


def test_gradients_flow_to_mu_and_sigma_not_to_noise_buffers():
    layer = NoisyLinear(6, 3)
    layer.reset_noise()
    x = torch.randn(2, 6)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight_mu.grad is not None and layer.weight_mu.grad.abs().sum() > 0
    assert layer.weight_sigma.grad is not None and layer.weight_sigma.grad.abs().sum() > 0
    assert layer.bias_mu.grad is not None
    assert layer.bias_sigma.grad is not None
    # epsilon is a buffer, not a Parameter -- it must not accumulate a gradient at all
    assert not layer.weight_epsilon.requires_grad
    assert not layer.bias_epsilon.requires_grad


def test_factorized_noise_structure():
    # weight_epsilon must be the outer product of the in/out noise vectors
    # (weight_epsilon[i, j] = epsilon_out[i] * epsilon_in[j]), not
    # independent per-element noise (the more expensive non-factorized
    # variant) -- verify the rank-1 structure directly: each ROW, divided by
    # its own first element, must be constant across all columns (equal to
    # epsilon_in / epsilon_in[0], the same for every row since epsilon_out
    # cancels out). Different rows are allowed -- expected -- to have a
    # different overall scale (their own epsilon_out[i]).
    layer = NoisyLinear(8, 5)
    layer.reset_noise()
    we = layer.weight_epsilon
    normalized_rows = we / we[:, :1]  # each row divided by its own first entry
    assert torch.allclose(normalized_rows, normalized_rows[0].expand_as(normalized_rows), atol=1e-4)
