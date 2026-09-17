"""Summarize raw benchmark CSVs: per-match mean + bootstrap CI per (tag, scenario, agent_code)."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from metrics import bootstrap_ci  # noqa: E402

RAW_DIR = REPO_ROOT / "evaluation" / "results" / "raw"
SUMMARY_DIR = REPO_ROOT / "evaluation" / "results" / "summary"

METRICS = ["score", "coins", "kills", "suicides", "suicide", "crates",
           "invalid", "alive_steps", "rank", "win", "survived",
           "time", "think_time_mean"]


def load_rows(paths):
    rows = []
    for path in paths:
        with open(str(path), newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="*", default=[], help="raw CSV stems to include (empty = all)")
    args = p.parse_args()

    if args.tags:
        paths = [RAW_DIR / (t + ".csv") for t in args.tags]
    else:
        paths = sorted(p for p in RAW_DIR.glob("*.csv") if not p.stem.startswith("smoke"))
    if not paths:
        print("no CSV under evaluation/results/raw/. Run benchmark.py first.")
        return
    rows = load_rows(paths)

    # Collapse to one value per match before bootstrapping. The agent rows of a
    # self-play match are not independent (e.g. their coins sum to the board
    # total), so resampling rows would understate the CI. Average the same-code
    # agents within each match, then bootstrap over the per-match values.
    per_match = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["tag"], r["scenario"], r["agent_code"], r["seed"])
        for m in METRICS:
            per_match[key][m].append(float(r[m]))

    groups = defaultdict(lambda: defaultdict(list))
    for (tag, scenario, code, _seed), md in per_match.items():
        for m in METRICS:
            groups[(tag, scenario, code)][m].append(sum(md[m]) / len(md[m]))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / "summary.csv"
    with open(str(out), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "scenario", "agent_code", "n_matches", "metric", "mean", "ci_lo", "ci_hi"])
        for (tag, scenario, code), md in sorted(groups.items()):
            n = len(md["score"])
            for m in METRICS:
                mean, lo, hi = bootstrap_ci(md[m])
                w.writerow([tag, scenario, code, n, m,
                            f"{mean:.3f}", f"{lo:.3f}", f"{hi:.3f}"])

    print(f"\nwrote {out}\n")
    print("| tag | scenario | agent | n_matches | score (95% CI) | survival | suicide |")
    print("|---|---|---|--:|---|--:|--:|")
    for (tag, scenario, code), md in sorted(groups.items()):
        n = len(md["score"])
        s_m, s_lo, s_hi = bootstrap_ci(md["score"])
        surv = bootstrap_ci(md["survived"])[0]
        sui = bootstrap_ci(md["suicide"])[0]
        print(f"| {tag} | {scenario} | {code} | {n} | "
              f"{s_m:.2f} [{s_lo:.2f}, {s_hi:.2f}] | {surv:.2f} | {sui:.2f} |")


if __name__ == "__main__":
    main()