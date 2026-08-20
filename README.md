# Bomberman reinforcement-learning project

Course project for *Machine Learning Essentials*, Summer Semester 2026.
The code builds on the official [`ukoethe/bomberman_rl`](https://github.com/ukoethe/bomberman_rl)
framework. The goal is to develop, compare, and systematically evaluate at least two
learned Bomberman agents.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the framework without opening the GUI:

```bash
python main.py play --agents random_agent peaceful_agent --no-gui --n-rounds 2
```

To watch the supplied rule-based agent:

```bash
python main.py play
```

## Planned agents

1. `tabular_q_agent`: lecture-based Q-learning with compact, symmetry-aware state
   features and a staged curriculum.
2. `linear_q_agent`: approximate action-value learning with linear features and
   replay-based mini-batch updates.

The agents will be trained through the four project stages: coin collection,
crate destruction and safe bombing, hunting non-aggressive opponents, and full
competition. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the experiment plan and
success criteria.

## First training run

The initial `tabular_q_agent` intentionally handles only navigation and visible
coins. Train it on the first curriculum stage with:

```bash
python main.py play --agents tabular_q_agent --train 1 \
  --scenario coin-heaven --no-gui --n-rounds 1000
```

Its table is saved as `agent_code/tabular_q_agent/q_table.pkl` after every round,
so interrupted training can resume from the latest completed episode.
Bombs remain disabled until safe-bomb and danger-map features are implemented.

## Reproducibility

- Use fixed `--seed` values for paired comparisons.
- Save machine-readable results with `--save-stats`.
- Evaluate on held-out seeds and against the supplied baseline agents.
- Keep training-only utilities outside the final submitted agent directory.
- Record every experiment's commit, configuration, random seeds, and aggregate
  metrics.

## Repository/remotes

The official framework is configured as the `upstream` remote. After creating an
empty public GitHub repository, connect it as `origin` and push:

```bash
git remote add origin <YOUR_PUBLIC_REPOSITORY_URL>
git push -u origin main
```

Do not add the final report PDF to this repository. The tournament submission is
only the directory of the best trained agent under `agent_code/`, including its
trained parameters.

## Important constraints

- Agent decisions have a 0.5 second time limit and run on one CPU thread.
- The submitted agent may use at most 8 GB RAM and no multiprocessing.
- Framework modifications used during training are absent in official games.
- Any additional runtime dependency must be declared in `requirements.txt`.
- Agent code is due 21 September 2026 at 21:00; the report is due 28 September
  2026 at 21:00.
