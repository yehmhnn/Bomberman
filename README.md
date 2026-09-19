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

Current tabular opponent experiments and limitations are recorded in
[STAGE34_RESULTS.md](STAGE34_RESULTS.md).

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

To train Q-learning and SARSA from scratch and evaluate them on the same held-out
seeds:

```bash
scripts/run_stage1_experiment.sh --fresh 1000 100
```

The positional numbers are training rounds per agent and evaluation rounds per
held-out seed. `--fresh` explicitly removes previous Stage-1 tables first; omit
it to continue training existing tables.

The two agents deliberately share their state and reward representation. Their
controlled difference is the TD target: Q-learning uses the largest next-state
value, while SARSA uses the value of the next action sampled from its current
epsilon-greedy policy.

## Stage 2: crates and safe bombing

Stage 2 uses the verified functions in `shared/` to represent blast timing,
safe actions, coin/crate targets, and whether a bomb can be escaped. It stores
its tables separately as `q_table_stage2.pkl`, so training cannot overwrite the
Stage 1 models. Start the crate curriculum with:

```bash
python main.py play --agents tabular_q_agent --train 1 \
  --scenario loot-crate --no-gui --n-rounds 2000
python main.py play --agents tabular_sarsa_agent --train 1 \
  --scenario loot-crate --no-gui --n-rounds 2000
```

Then continue each saved table on the full board by changing the scenario to
`classic`. During evaluation, omit `--train 1`; this sets exploration to zero
and applies the same safety mask to the greedy policy.

The completed experiment and failure analysis are recorded in
[STAGE2_RESULTS.md](STAGE2_RESULTS.md).

## Stages 3 and 4: opponents and full competition

The tabular agents now use the same 34-value state: the 24 Stage-2 navigation,
crate and bomb-safety values plus 10 opponent direction, distance, trapping and
threat values. An unseen opponent-aware state is initialized from the matching
24-value Stage-2 entry, so the curriculum retains learned navigation and safe
bombing instead of starting from zero.

Run the same four curriculum phases for each of `tabular_q_agent` and
`tabular_sarsa_agent`. The runner cycles over all frozen training seeds and
records each run under `evaluation/results/training/`:

```bash
python evaluation/train_tabular.py --tag q_s3_peaceful \
  --agent tabular_q_agent --stage 3 --opponents \
  peaceful_agent --episodes 2000 --fresh
python evaluation/train_tabular.py --tag q_s3_coin \
  --agent tabular_q_agent --stage 3 --opponents \
  coin_collector_agent --episodes 3000
python evaluation/train_tabular.py --tag q_s4_rule \
  --agent tabular_q_agent --stage 4 --opponents \
  rule_based_agent rule_based_agent rule_based_agent --episodes 4000 --fresh
python evaluation/train_tabular.py --tag q_s4_mixed \
  --agent tabular_q_agent --stage 4 --opponents \
  rule_based_agent coin_collector_agent peaceful_agent --episodes 4000
```

Replace `tabular_q_agent` and the `q_` tag prefix with
`tabular_sarsa_agent` and `sarsa_` for the controlled SARSA runs. `--fresh`
only removes the selected Stage-3 or Stage-4 table; it does not touch the
Stage-1 or Stage-2 checkpoints. Stage 4 initializes unseen states from Stage 3.

Use `evaluation/benchmark.py` with the line-ups agreed in
`EXPERIMENT_PROTOCOL.md`. Run the full validation set for checkpoint selection
and use the test split exactly once after every method and hyperparameter is
frozen. Any proposed change to the shared line-ups belongs in a separate PR and
must be accepted before it is used for cross-method claims.

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
