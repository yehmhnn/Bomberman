import io
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from agent_code.ppo_agent.model import (
    ACTIONS,
    CRATE_DIST_IDX,
    OPPONENT_DIST_IDX,
    STATE_SIZE,
    ActorCritic,
    action_mask_vector,
    masked_distribution,
    state_to_vector,
)
from agent_code.ppo_agent.train import (
    _advantages_and_returns,
    _potential,
    _restore_rollout,
    _serializable_rollout,
)


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


def test_unfinished_rollout_round_trips_through_safe_checkpoint_loader():
    rollout = [{
        "state": np.arange(STATE_SIZE, dtype=np.float32),
        "action": 2,
        "old_log_probability": -0.5,
        "value": 1.25,
        "next_value": 1.5,
        "reward": 0.2,
        "done": 0.0,
        "mask": np.array([True, False, True, False, True, False]),
        "events": ("MOVED_DOWN",),
    }]
    buffer = io.BytesIO()
    torch.save({"rollout": _serializable_rollout(rollout)}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    restored = _restore_rollout(loaded["rollout"])
    np.testing.assert_array_equal(restored[0]["state"], rollout[0]["state"])
    np.testing.assert_array_equal(restored[0]["mask"], rollout[0]["mask"])
    assert restored[0]["events"] == rollout[0]["events"]


def test_stage_variant_selects_a_separate_checkpoint_file():
    environment = os.environ.copy()
    environment["PPO_STAGE"] = "1"
    environment["PPO_VARIANT"] = "entropy02"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agent_code.ppo_agent.callbacks import MODEL_FILE; print(MODEL_FILE.name)",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "model_stage1_entropy02.pt"


def test_absent_coin_does_not_erase_crate_distance_potential():
    vector = np.zeros(STATE_SIZE, dtype=np.float32)
    vector[CRATE_DIST_IDX] = 5.0
    assert _potential(vector) == -5.0


def test_opponent_distance_potential_is_enabled_only_for_hunting_stages():
    vector = np.zeros(STATE_SIZE, dtype=np.float32)
    vector[OPPONENT_DIST_IDX] = 4.0
    assert _potential(vector, stage=2) == 0.0
    assert _potential(vector, stage=3) == -4.0
    assert _potential(vector, stage=4) == -4.0


def test_stage_variant_can_initialize_from_a_different_prior_variant():
    environment = os.environ.copy()
    environment["PPO_STAGE"] = "2"
    environment["PPO_VARIANT"] = "goalfix"
    environment["PPO_INIT_VARIANT"] = "entropy02"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agent_code.ppo_agent.callbacks import "
                "MODEL_FILE, PRIOR_MODEL_FILES; "
                "print(MODEL_FILE.name, PRIOR_MODEL_FILES[0].name)"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == (
        "model_stage2_goalfix.pt model_stage1_entropy02.pt"
    )
