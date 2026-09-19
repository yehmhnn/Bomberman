"""Reproducible seeded curriculum runner for the two tabular agents.

Example (first Stage-3 phase):
    python evaluation/train_tabular.py --tag q_s3_peaceful \
      --agent tabular_q_agent --stage 3 --opponents \
      peaceful_agent peaceful_agent peaceful_agent --episodes 2000 --fresh
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

ALLOWED_AGENTS = ("tabular_q_agent", "tabular_sarsa_agent")
LOG_DIR = REPO_ROOT / "evaluation" / "results" / "training"


def distribute(total, count):
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--agent", choices=ALLOWED_AGENTS, required=True)
    parser.add_argument("--stage", choices=(3, 4), type=int, required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--fresh", action="store_true",
        help="delete only this agent's selected-stage table and exploration metadata",
    )
    parser.add_argument(
        "--reset-epsilon", action="store_true",
        help="restart epsilon decay but retain the selected-stage table",
    )
    args = parser.parse_args()

    if not 1 <= len(args.opponents) <= 3:
        parser.error("provide between one and three opponents")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    model_file = agent_dir / f"q_table_stage{args.stage}.pkl"
    meta_file = agent_dir / f"q_table_stage{args.stage}.meta.json"
    if args.fresh:
        for path in (model_file, meta_file):
            if path.exists():
                path.unlink()

    env = os.environ.copy()
    env["TABULAR_STAGE"] = str(args.stage)
    rounds_by_seed = distribute(args.episodes, len(TRAIN_SEEDS))
    started = time.time()
    completed = 0
    for index, (seed, rounds) in enumerate(zip(TRAIN_SEEDS, rounds_by_seed), 1):
        if rounds == 0:
            continue
        if args.reset_epsilon and completed == 0:
            env["TABULAR_RESET_EPSILON"] = "1"
        else:
            env.pop("TABULAR_RESET_EPSILON", None)
        command = [
            sys.executable, "main.py", "play", "--no-gui", "--scenario", "classic",
            "--seed", str(seed), "--n-rounds", str(rounds), "--train", "1",
            "--agents", args.agent, *args.opponents,
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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    record = {
        "tag": args.tag,
        "git_commit": commit,
        "agent": args.agent,
        "stage": args.stage,
        "scenario": "classic",
        "opponents": args.opponents,
        "episodes": args.episodes,
        "training_seed_count": len(TRAIN_SEEDS),
        "fresh": args.fresh,
        "reset_epsilon": args.reset_epsilon,
        "elapsed_seconds": round(time.time() - started, 2),
        "model_file": str(model_file.relative_to(REPO_ROOT)),
    }
    (LOG_DIR / f"{args.tag}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
