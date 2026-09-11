"""Parse bomberman --save-stats JSON into flat per-agent rows, plus a bootstrap-CI helper."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

# Keys that environment.py accumulates in lifetime_statistics
# (via note_stat calls and agents.py EVENT_STAT_MAP). "time" is the agent's
# cumulative think time in seconds over the turns it acted.
STAT_KEYS = ("score", "coins", "kills", "suicides", "crates", "bombs", "moves", "invalid", "steps", "time")


def load_stats(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def expected_agent_names(agents):
    """Reproduce environment.py setup_agents(): duplicated code names get a 0-based suffix."""
    counts = Counter(agents)
    seen = Counter()
    names = []
    for code in agents:
        if counts[code] > 1:
            names.append(f"{code}_{seen[code]}")
            seen[code] += 1
        else:
            names.append(code)
    return names


def rows_from_stats(stats, agents, meta):
    """Build one dict row per agent slot from a single match's stats JSON."""
    by_agent = stats["by_agent"]
    by_round = stats.get("by_round", {})
    round_steps = max((int(r.get("steps", 0)) for r in by_round.values()), default=0)

    rows = []
    for slot, (code, name) in enumerate(zip(agents, expected_agent_names(agents))):
        a = by_agent.get(name, {})
        alive_steps = int(a.get("steps", 0))
        suicides = int(a.get("suicides", 0))
        # An agent's 'steps' counts the turns it was alive; equal to the round
        # length (and no suicide) means it survived to the end.
        survived = round_steps > 0 and alive_steps >= round_steps and suicides == 0
        row = dict(meta)
        row.update(
            agent_slot=slot, agent_code=code, agent_name=name,
            round_steps=round_steps, alive_steps=alive_steps,
            survived=int(survived), suicide=int(suicides > 0),
        )
        for k in STAT_KEYS:
            row[k] = float(a.get(k, 0))
        # Mean per-move decision time; the tournament limit is 0.5 s per move.
        row["think_time_mean"] = row["time"] / alive_steps if alive_steps else float("nan")
        rows.append(row)

    # Within-match ranking on official score (1 = best); average rank for ties.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["score"], reverse=True)
    ranks = [0.0] * len(rows)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and rows[order[j + 1]]["score"] == rows[order[i]]["score"]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    for idx, r in enumerate(rows):
        r["rank"] = ranks[idx]
        r["win"] = int(ranks[idx] == 1.0 and ranks.count(1.0) == 1)
    return rows


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Return (mean, low, high); a 95% CI for alpha=0.05."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))