"""Linear Q-function and one-step Q-learning update."""
import numpy as np

from .state import N_FEATURES

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
TD_ERROR_CLIP = 5.0


class LinearQ:
    def __init__(self, n_features=N_FEATURES, actions=ACTIONS):
        self.actions = actions
        self.weights = np.zeros((len(actions), n_features))
        self.bias = np.zeros(len(actions))

    def values(self, features):
        x = np.asarray(features, dtype=np.float64)
        return self.weights @ x + self.bias

    def best_action(self, features, allowed=None):
        if allowed is not None and len(allowed) == 0:
            return None, 0.0
        q = self.values(features)
        if allowed is not None:
            mask = np.full_like(q, -np.inf)
            for a in allowed:
                mask[self.actions.index(a)] = 0.0
            q = q + mask
        idx = int(np.argmax(q))
        return self.actions[idx], float(q[idx])

    def update(self, features, action, reward, next_features, next_allowed, alpha, gamma, done):
        x = np.asarray(features, dtype=np.float64)
        a_idx = self.actions.index(action)
        current = self.weights[a_idx] @ x + self.bias[a_idx]
        no_safe_next = next_allowed is not None and len(next_allowed) == 0
        if done or next_features is None or no_safe_next:
            target = reward
        else:
            _, best_next_q = self.best_action(next_features, next_allowed)
            target = reward + gamma * best_next_q
        td_error = float(np.clip(target - current, -TD_ERROR_CLIP, TD_ERROR_CLIP))
        self.weights[a_idx] += alpha * td_error * x
        self.bias[a_idx] += alpha * td_error
        return td_error

    def update_sarsa(self, features, action, reward, next_features, next_action, alpha, gamma, done):
        x = np.asarray(features, dtype=np.float64)
        a_idx = self.actions.index(action)
        current = self.weights[a_idx] @ x + self.bias[a_idx]
        if done or next_features is None or next_action is None:
            target = reward
        else:
            next_x = np.asarray(next_features, dtype=np.float64)
            next_a_idx = self.actions.index(next_action)
            target = reward + gamma * (self.weights[next_a_idx] @ next_x + self.bias[next_a_idx])
        td_error = float(np.clip(target - current, -TD_ERROR_CLIP, TD_ERROR_CLIP))
        self.weights[a_idx] += alpha * td_error * x
        self.bias[a_idx] += alpha * td_error
        return td_error

    def save(self, path):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(path, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, path, n_features=N_FEATURES, actions=ACTIONS):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        q = cls(n_features, actions)
        data = np.load(path)
        assert data["weights"].shape == (len(actions), n_features), (
            f"checkpoint shape {data['weights'].shape} != expected {(len(actions), n_features)}"
        )
        q.weights = data["weights"]
        q.bias = data["bias"]
        return q