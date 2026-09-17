"""Run seeded headless matches and write one CSV row per agent per match.

Example:
    python evaluation/benchmark.py --tag base_4rule \
        --agents rule_based_agent rule_based_agent rule_based_agent rule_based_agent \
        --scenarios classic coin-heaven --split val
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from metrics import load_stats, rows_from_stats  # noqa: E402
from seeds import ALL_SEEDS  # noqa: E402

RAW_DIR = REPO_ROOT / "evaluation" / "results" / "raw"
STATS_DIR = REPO_ROOT / "evaluation" / "results" / "_stats"


def run_match(agents, scenario, seed, rounds=1):
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile("r", suffix=".json", dir=str(STATS_DIR), delete=False)
    fd.close()
    stats_path = Path(fd.name)
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--n-rounds", str(rounds),
        "--scenario", scenario,
        "--seed", str(seed),
        "--agents", *agents,
        "--save-stats", str(stats_path),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"match timed out after 180s (seed={seed}, {scenario})")
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"match failed (seed={seed}, {scenario}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    stats = load_stats(stats_path)
    try:
        stats_path.unlink()
    except OSError:
        pass
    return stats, wall


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="short label for this run; used as the CSV file name")
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--scenarios", nargs="+", default=["classic"])
    p.add_argument("--split", choices=sorted(ALL_SEEDS), default="val",
                   help="test is read only for final numbers; keep the default for development")
    p.add_argument("--limit", type=int, default=0, help="use only the first N seeds of the split (0 = all)")
    p.add_argument("--rounds", type=int, default=1)
    args = p.parse_args()

    if args.rounds != 1:
        p.error("--rounds must be 1; per-match parsing assumes a single round, loop over seeds instead")

    seeds = list(ALL_SEEDS[args.split])
    if args.limit:
        seeds = seeds[: args.limit]
    commit = git_commit()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / (args.tag + ".csv")

    writer = None
    n_rows = 0
    with open(str(out_path), "w", newline="", encoding="utf-8") as fh:
        for scenario in args.scenarios:
            for i, seed in enumerate(seeds, 1):
                stats, wall = run_match(args.agents, scenario, seed, args.rounds)
                meta = dict(
                    tag=args.tag, git_commit=commit, scenario=scenario, seed=seed,
                    split=args.split, rounds=args.rounds, wall_seconds=round(wall, 2),
                )
                rows = rows_from_stats(stats, args.agents, meta)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                n_rows += len(rows)
                print(f"[{scenario}] {i}/{len(seeds)} seed={seed} ({wall:.1f}s)")
    print(f"wrote {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()