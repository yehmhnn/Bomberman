import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_code.dqn_agent.model import ACTIONS, STATE_SIZE, DuelingQNetwork  # noqa: E402
from agent_code.dqn_agent.train import _quantile_huber_loss  # noqa: E402


def test_network_output_shapes():
    net = DuelingQNetwork(n_quantiles=8)
    x = torch.randn(5, STATE_SIZE)
    quantiles = net(x)
    assert quantiles.shape == (5, len(ACTIONS), 8)
    q = net.q_values(x)
    assert q.shape == (5, len(ACTIONS))
    assert torch.allclose(q, quantiles.mean(dim=-1))


def test_dueling_combination_is_zero_centered_across_actions_per_quantile():
    # advantage - mean(advantage) must average to exactly zero across actions
    # for every quantile slot, same invariant as the original scalar dueling
    # head, just applied per-quantile now.
    net = DuelingQNetwork(n_quantiles=4)
    x = torch.randn(3, STATE_SIZE)
    quantiles = net(x)
    value_only = net.value_head(net.trunk(x)).unsqueeze(1)
    centered = quantiles - value_only
    assert torch.allclose(centered.mean(dim=1), torch.zeros(3, 4), atol=1e-5)


def test_quantile_huber_loss_is_zero_for_a_perfectly_predicted_point_mass():
    # For a point-mass target (every "sample" is the same value), every
    # predicted quantile should equal that value at the optimum, and every
    # (i, j) pairwise term is then exactly 0 -- unlike squared error, matching
    # predicted to target elementwise is *not* generally the zero-loss case
    # for quantile regression (the loss is over all N x N pairs, not the
    # diagonal), so this only holds for a degenerate constant target.
    tau = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    constant = torch.full((1, 8), 3.0)
    loss = _quantile_huber_loss(constant, constant, tau)
    assert torch.allclose(loss, torch.zeros(1), atol=1e-6)


def test_quantile_huber_loss_increases_with_distance():
    tau = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    predicted = torch.zeros(1, 8)
    close_target = torch.full((1, 8), 1.0)
    far_target = torch.full((1, 8), 5.0)
    assert _quantile_huber_loss(predicted, close_target, tau) < _quantile_huber_loss(predicted, far_target, tau)


def test_action_gather_extracts_each_samples_own_chosen_action():
    # Mirrors train.py._optimize's exact gather pattern in isolation: with a
    # distinct, easily-identified quantile vector per (batch, action) cell,
    # confirm gathering by a per-sample action index pulls out that sample's
    # own action -- not the same action for the whole batch, and not a
    # transposed/off-by-one action, either of which shape checks alone
    # wouldn't catch.
    batch, n_actions, n_quantiles = 4, len(ACTIONS), 3
    quantiles = torch.arange(batch * n_actions * n_quantiles, dtype=torch.float32).view(batch, n_actions, n_quantiles)
    actions = torch.tensor([0, 3, 5, 1])  # a different action index per batch row

    action_index = actions.view(-1, 1, 1).expand(-1, 1, n_quantiles)
    gathered = quantiles.gather(1, action_index).squeeze(1)

    for b in range(batch):
        assert torch.equal(gathered[b], quantiles[b, actions[b]])


def test_quantile_regression_recovers_the_true_distribution():
    """The real correctness test: does gradient descent on this loss against
    known samples actually converge to the right quantiles, not just have
    the right shape? Uses a plain learnable parameter vector (not a full
    network) as "predicted" so this isolates the loss function itself from
    everything else -- if this fails, the bug is in the loss/index math,
    not in feature extraction, replay, or any of the surrounding pipeline.
    """
    torch.manual_seed(0)
    n_quantiles = 32
    true_mean, true_std = 5.0, 2.0
    target_samples = torch.randn(1, 2000) * true_std + true_mean  # (1, 2000) "empirical distribution"
    tau = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles

    predicted = torch.nn.Parameter(torch.zeros(1, n_quantiles))
    optimizer = torch.optim.Adam([predicted], lr=0.1)
    for _ in range(500):
        optimizer.zero_grad()
        loss = _quantile_huber_loss(predicted, target_samples, tau).mean()
        loss.backward()
        optimizer.step()

    learned = predicted.detach()
    # Mean of the learned quantiles should approximate the true mean.
    assert abs(learned.mean().item() - true_mean) < 0.3
    # It should have learned a spread, not collapsed to a point estimate.
    assert (learned.max() - learned.min()).item() > true_std
    # Monotonic non-decreasing with tau is the defining shape of a quantile
    # function; allow a little slack for the stochastic-gradient noise
    # rather than requiring a perfectly sorted vector.
    diffs = learned[0, 1:] - learned[0, :-1]
    assert (diffs > -0.5).all(), "learned quantiles should be roughly non-decreasing with tau"
    # The median quantile slot (tau closest to 0.5) should sit near the true mean
    # (mean == median for a symmetric normal distribution).
    median_slot = int((tau - 0.5).abs().argmin())
    assert abs(learned[0, median_slot].item() - true_mean) < 0.5
