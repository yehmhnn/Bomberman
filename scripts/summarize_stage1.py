"""Summarize JSON files produced by the Stage-1 experiment runner."""

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("tabular_q_agent", "tabular_sarsa_agent")


def summarize(runs, rounds_per_seed):
    total_rounds = len(runs) * rounds_per_seed
    summary = {"rounds": total_rounds}
    for metric in ("score", "coins", "moves", "invalid", "steps", "time"):
        totals = [run.get(metric, 0) for run in runs]
        per_seed_round = [value / rounds_per_seed for value in totals]
        summary[f"{metric}_per_round"] = sum(totals) / total_rounds
        summary[f"{metric}_per_round_stdev"] = (
            statistics.stdev(per_seed_round) if len(per_seed_round) > 1 else 0.0
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-per-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    summaries = {}
    for agent in AGENTS:
        runs = []
        for seed in args.seeds:
            result_file = ROOT / "results" / f"stage1_{agent}_{seed}.json"
            with result_file.open() as file:
                runs.append(json.load(file)["by_agent"][agent])
        summaries[agent] = summarize(runs, args.rounds_per_seed)

    summary_file = ROOT / "results" / "stage1_comparison.json"
    with summary_file.open("w") as file:
        json.dump(summaries, file, indent=2)

    print("\nStage-1 held-out evaluation")
    print("agent                     score/round   coins/round   steps/round")
    for agent, result in summaries.items():
        print(
            f"{agent:25}"
            f"{result['score_per_round']:13.3f}"
            f"{result['coins_per_round']:14.3f}"
            f"{result['steps_per_round']:14.3f}"
        )
    print(f"\nFull summary: {summary_file}")


if __name__ == "__main__":
    main()
