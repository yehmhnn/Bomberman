# Stage-1 results

## Shared-protocol validation result

We re-evaluated both saved models with the team's shared benchmark harness on
all 100 frozen validation seeds. Each agent played alone on `coin-heaven`, with
exploration disabled. Scores are official collected-coin scores; confidence
intervals are bootstrap 95% intervals produced by `evaluation/aggregate.py`.

| Method | Mean score/coins | 95% confidence interval | Survival | Suicide |
|---|---:|---:|---:|---:|
| Tabular Q-learning | 5.59 | [4.40, 6.86] | 100% | 0% |
| Tabular SARSA | 47.13 | [45.26, 48.80] | 100% | 0% |

This confirms the original finding under the common evaluation protocol. SARSA
collected about 94.3% of the 50 available coins, while Q-learning collected
about 11.2%. Raw rows are stored as `stage1_tabular_q_val.csv` and
`stage1_tabular_sarsa_val.csv` in `evaluation/results/raw/`.

## Initial exploratory evaluation

The following experiment predates the shared protocol and uses its own seeds.
It is retained as development history, but should not be compared numerically
with experiments evaluated by `evaluation/benchmark.py`.

## Controlled evaluation after 200 training rounds

Both agents used the same nine-value state abstraction, potential-based reward
shaping, learning rate, discount, exploration schedule, training seed, and 200
training rounds. Evaluation used 20 rounds on each of five held-out seeds (100
rounds per method), with exploration disabled.

| Method | Coins/score per round | Steps per round |
|---|---:|---:|
| Tabular Q-learning | 6.06 | 400.00 |
| Tabular SARSA | 47.36 | 160.56 |

The board contains 50 coins. SARSA therefore collected about 94.7% of available
coins per evaluation round, while Q-learning usually reached the 400-step limit.

## Direct competition

In 50 shared `coin-heaven` rounds using seed 77:

| Method | Total coins | Coins per round |
|---|---:|---:|
| Tabular Q-learning | 427 | 8.54 |
| Tabular SARSA | 1,992 | 39.84 |

## Interpretation and next hypothesis

SARSA uses the value of the action actually selected by its exploratory policy,
whereas Q-learning uses the maximum next-state value. The current state is a
compressed proxy rather than a truly Markov state: many different boards map to
the same tuple. The working hypothesis is that Q-learning's maximum target
amplifies optimistic values under this state aliasing. Further runs, learning
curves, multiple training seeds, and hyperparameter sweeps are required before
claiming that SARSA is generally superior.

Reproduce the shared validation result with:

```bash
python evaluation/benchmark.py --tag stage1_tabular_q_val \
  --agents tabular_q_agent --scenarios coin-heaven --split val
python evaluation/benchmark.py --tag stage1_tabular_sarsa_val \
  --agents tabular_sarsa_agent --scenarios coin-heaven --split val
python evaluation/aggregate.py
```

Watch both trained policies compete with:

```bash
python main.py play --agents tabular_q_agent tabular_sarsa_agent \
  --scenario coin-heaven --n-rounds 1 --skip-frames
```
