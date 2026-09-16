"""Micro-benchmark: how long does an agent's act() take, worst case?"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def make_random_state(rng, size=17, n_crates=60, n_coins=9, n_bombs=3, n_others=3):
    field = np.zeros((size, size), dtype=int)
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
    for x in range(size):
        for y in range(size):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    free = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1) if field[x, y] == 0]
    rng.shuffle(free)
    for (x, y) in free[:n_crates]:
        field[x, y] = 1
    free = free[n_crates:]

    pos = free[0]
    others_pos = free[1:1 + n_others]
    coins = free[1 + n_others: 1 + n_others + n_coins]
    bomb_spots = free[1 + n_others + n_coins: 1 + n_others + n_coins + n_bombs]
    bombs = [(p, int(rng.integers(0, 5))) for p in bomb_spots]

    return {
        "round": 1, "step": 50,
        "field": field,
        "bombs": bombs,
        "explosion_map": np.zeros((size, size), dtype=int),
        "coins": coins,
        "self": ("me", 0, True, pos),
        "others": [(f"o{i}", 0, True, p) for i, p in enumerate(others_pos)],
        "user_input": None,
    }


class _FakeLogger:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass


class _FakeSelf:
    train = False
    logger = _FakeLogger()


def profile_agent(code_name, n_calls=300, seed=0):
    module = importlib.import_module(f"agent_code.{code_name}.callbacks")
    fake_self = _FakeSelf()
    module.setup(fake_self)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_calls):
        state = make_random_state(rng)
        t0 = time.perf_counter()
        module.act(fake_self, state)
        samples.append(time.perf_counter() - t0)

    arr = np.array(samples)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "rule_based_agent"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    result = profile_agent(name, n_calls=n)
    for k, v in result.items():
        print(f"{k}: {v}")
    status = "OK" if result["max"] < 0.5 else "OVER BUDGET"
    print(f"budget check (0.5s): {status}")