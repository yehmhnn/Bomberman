"""Frozen seed sets for reproducible training and evaluation.

TRAIN: used for training only.
VAL:   used for tuning and checkpoint selection.
TEST:  used once, for the final reported numbers. Never tune on these.

These values must not change after the first commit! every reported
result is tied to them.
"""
import hashlib
import random

_MASTER_SEED = 20260909
_rng = random.Random(_MASTER_SEED)
_pool = _rng.sample(range(1, 1_000_000), 400)

# random.sample is not guaranteed to give the same sequence across CPython
# versions, so pin the pool. If this fires, the seeds drifted and every
# held-out comparison would be invalid: freeze the pool as literals instead.
assert hashlib.sha256(repr(_pool).encode()).hexdigest()[:16] == "de76fc86869a7ad4", (
    "seed pool changed; do not run experiments until this is resolved"
)

TRAIN_SEEDS = tuple(_pool[:200])
VAL_SEEDS = tuple(_pool[200:300])
TEST_SEEDS = tuple(_pool[300:400])

ALL_SEEDS = {"train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS}

if __name__ == "__main__":
    for name, seeds in ALL_SEEDS.items():
        print(f"{name}: n={len(seeds)} first5={seeds[:5]}")