import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.q_learning import LinearQ  # noqa: E402


def test_values_shape():
    q = LinearQ(n_features=4, actions=("UP", "DOWN"))
    q.weights = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    vals = q.values([2.0, 3.0, 0.0, 0.0])
    assert np.allclose(vals, [2.0, 3.0])


def test_best_action_respects_allowed():
    q = LinearQ(n_features=2, actions=("UP", "DOWN", "LEFT"))
    q.weights = np.array([[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    action, value = q.best_action([1.0, 1.0], allowed=("DOWN", "LEFT"))
    assert action in ("DOWN", "LEFT")  # UP is best overall but not allowed


def test_update_reduces_td_error():
    q = LinearQ(n_features=3, actions=("UP", "DOWN"))
    features = [1.0, 0.0, 0.0]
    errors = []
    for _ in range(50):
        err = q.update(features, "UP", reward=1.0, next_features=None, next_allowed=None,
                        alpha=0.1, gamma=0.9, done=True)
        errors.append(abs(err))
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


def test_save_and_load_roundtrip(tmp_path):
    q = LinearQ(n_features=3, actions=("UP", "DOWN"))
    q.weights[:] = 1.0
    q.bias[:] = 2.0
    path = str(tmp_path / "q.npz")
    q.save(path)
    loaded = LinearQ.load(path, n_features=3, actions=("UP", "DOWN"))
    assert np.allclose(loaded.weights, q.weights)
    assert np.allclose(loaded.bias, q.bias)

def test_best_action_with_empty_allowed_returns_none():
    q = LinearQ(n_features=2, actions=("UP", "DOWN"))
    action, value = q.best_action([1.0, 1.0], allowed=())
    assert action is None
    assert value == 0.0


def test_update_treats_no_safe_next_action_as_terminal():
    q = LinearQ(n_features=3, actions=("UP", "DOWN"))
    err = q.update([1.0, 0.0, 0.0], "UP", reward=-5.0, next_features=[1.0, 1.0, 1.0],
                    next_allowed=(), alpha=0.1, gamma=0.9, done=False)
    assert np.isfinite(err)
    assert np.all(np.isfinite(q.weights))
    assert np.all(np.isfinite(q.bias))


def test_sarsa_reduces_td_error():
    q = LinearQ(n_features=3, actions=("UP", "DOWN"))
    features = [1.0, 0.0, 0.0]
    errors = []
    for _ in range(50):
        err = q.update_sarsa(features, "UP", reward=1.0, next_features=None, next_action=None,
                              alpha=0.1, gamma=0.9, done=True)
        errors.append(abs(err))
    assert errors[-1] < errors[0]


def test_save_and_load_without_npz_suffix(tmp_path):
    q = LinearQ(n_features=3, actions=("UP", "DOWN"))
    q.weights[:] = 3.0
    q.bias[:] = -1.0
    path = str(tmp_path / "checkpoint")
    q.save(path)
    loaded = LinearQ.load(path, n_features=3, actions=("UP", "DOWN"))
    assert np.allclose(loaded.weights, q.weights)
    assert np.allclose(loaded.bias, q.bias)