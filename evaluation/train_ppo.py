"""Reproducible shared-seed curriculum runner for ``ppo_agent``.

The PPO checkpoint stores unfinished rollout transitions, so changing process
between seeds does not throw away on-policy data. Each subprocess receives an
explicit agent RNG seed in addition to the framework's board seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from seeds import TRAIN_SEEDS  # noqa: E402

LOG_DIR = REPO_ROOT / "evaluation" / "results" / "training"
DEFAULT_SCENARIOS = {1: "coin-heaven", 2: "loot-crate", 3: "classic", 4: "classic"}


def distribute(total, count):
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument(
        "--variant", default="",
        help="safe suffix for a separate checkpoint, e.g. entropy02",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument(
        "--fresh", action="store_true",
        help="remove only this stage's checkpoint and diagnostics before training",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if len(args.opponents) > 3:
        parser.error("at most three opponents are supported")
    if args.variant and not all(c.isalnum() or c in "_-" for c in args.variant):
        parser.error("--variant may contain only letters, digits, '_' and '-'")
    if args.entropy_coefficient < 0:
        parser.error("--entropy-coefficient must be non-negative")
    scenario = args.scenario or DEFAULT_SCENARIOS[args.stage]

    agent_dir = REPO_ROOT / "agent_code" / "ppo_agent"
    suffix = f"_{args.variant}" if args.variant else ""
    model_file = agent_dir / f"model_stage{args.stage}{suffix}.pt"
    diagnostics_file = agent_dir / f"model_stage{args.stage}{suffix}.diagnostics.csv"
    if args.fresh:
        for path in (model_file, diagnostics_file):
            if path.exists():
                path.unlink()

    env = os.environ.copy()
    env["PPO_STAGE"] = str(args.stage)
    env["PPO_VARIANT"] = args.variant
    env["PPO_ENTROPY_COEFFICIENT"] = str(args.entropy_coefficient)
    rounds_by_seed = distribute(args.episodes, len(TRAIN_SEEDS))
    started = time.perf_counter()
    completed = 0

    for index, (seed, rounds) in enumerate(zip(TRAIN_SEEDS, rounds_by_seed), 1):
        if rounds == 0:
            continue
        env["PPO_AGENT_SEED"] = str(seed)
        command = [
            sys.executable, "main.py", "play", "--no-gui",
            "--scenario", scenario, "--seed", str(seed),
            "--n-rounds", str(rounds), "--train", "1",
            "--agents", "ppo_agent", *args.opponents,
        ]
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"training failed for seed {seed}:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        completed += rounds
        print(
            f"[{args.tag}] seed {index}/{len(TRAIN_SEEDS)} "
            f"episodes {completed}/{args.episodes}",
            flush=True,
        )

    checkpoint = None
    if model_file.is_file():
        import torch
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    record = {
        "tag": args.tag,
        "git_commit": commit,
        "agent": "ppo_agent",
        "stage": args.stage,
        "scenario": scenario,
        "opponents": args.opponents,
        "episodes": args.episodes,
        "training_seed_count": sum(rounds > 0 for rounds in rounds_by_seed),
        "fresh": args.fresh,
        "variant": args.variant,
        "entropy_coefficient": args.entropy_coefficient,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "model_file": str(model_file.relative_to(REPO_ROOT)),
        "total_steps": None if checkpoint is None else checkpoint.get("total_steps"),
        "optimized_steps": None if checkpoint is None else checkpoint.get("optimized_steps"),
        "stage_training_round": None if checkpoint is None else checkpoint.get("stage_training_round"),
    }
    (LOG_DIR / f"{args.tag}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
