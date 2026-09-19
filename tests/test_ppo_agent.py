from types import SimpleNamespace

import numpy as np
import torch

from agent_code.ppo_agent.model import (
    ACTIONS,
    STATE_SIZE,
    ActorCritic,
    action_mask_vector,
    masked_distribution,
    state_to_vector,
)
from agent_code.ppo_agent.train import _advantages_and_returns


def game_state(position=(3, 3)):
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    return {
        "field": field,
        "self": ("ppo", 0, True, position),
        "others": [("opponent", 0, False, (5, 3))],
        "bombs": [],
        "coins": [(4, 3)],
        "explosion_map": np.zeros_like(field),
    }


def test_state_matches_dqn_feature_dimension():
    assert state_to_vector(game_state()).shape == (STATE_SIZE,)
    assert STATE_SIZE == 35


def test_actor_critic_has_policy_and_value_outputs():
    network = ActorCritic()
    logits, values = network(torch.zeros((4, STATE_SIZE)))
    assert logits.shape == (4, len(ACTIONS))
    assert values.shape == (4,)


def test_masked_distribution_never_samples_forbidden_action():
    logits = torch.full((1, len(ACTIONS)), 100.0)
    mask = torch.tensor([[False, False, True, False, False, False]])
    distribution = masked_distribution(logits, mask)
    samples = distribution.sample((100,))
    assert torch.all(samples == ACTIONS.index("DOWN"))


def test_action_mask_has_at_least_one_executable_action():
    mask = action_mask_vector(game_state())
    assert mask.dtype == bool
    assert mask.shape == (len(ACTIONS),)
    assert mask.any()


def test_gae_stops_bootstrapping_at_terminal_transition():
    rollout = [
        {"reward": 1.0, "value": 0.5, "next_value": 2.0, "done": 0.0},
        {"reward": 3.0, "value": 1.0, "next_value": 99.0, "done": 1.0},
    ]
    advantages, returns = _advantages_and_returns(rollout)
    assert np.isclose(advantages[1], 2.0)
    assert np.isclose(returns[1], 3.0)
