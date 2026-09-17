"""Reset the DQN's exploration schedule between curriculum stages.

Because train.py's epsilon is a pure function of cumulative total_steps
(persisted across process runs so a resumed session doesn't silently
restart exploration -- see agent_code/dqn_agent/train.py setup_training),
moving to a harder scenario after epsilon has already decayed to its floor
gives near-zero exploration of behaviour the new scenario actually needs
(e.g. BOMB was rarely useful on coin-heaven but is essential on classic).

This keeps the learned weights (warm start) but zeroes total_steps/
training_round so the epsilon schedule restarts, and drops the Adam
optimizer state since its momentum was tuned for a different reward
landscape. Standard curriculum-learning practice, not a workaround for a
bug: the alternative (not resetting) is the actual gap.

Usage:
    python scripts/dqn_curriculum_reset.py agent_code/dqn_agent/model.pt
"""

import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)

    model_file = Path(sys.argv[1])
    checkpoint = torch.load(model_file, map_location="cpu")
    before_steps = checkpoint.get("total_steps", 0)

    reset = {
        "model_state": checkpoint["model_state"],
        "total_steps": 0,
        "training_round": 0,
    }
    # No optimizer_state key: setup_training() falls back to a fresh Adam
    # optimizer when the checkpoint doesn't have one, same as a first run.

    torch.save(reset, model_file)
    print(f"{model_file}: kept trained weights, reset total_steps {before_steps} -> 0")


if __name__ == "__main__":
    main()
