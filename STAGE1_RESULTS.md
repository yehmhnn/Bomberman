# Preliminary Stage-1 results

These results validate the experiment pipeline and demonstrate learned behavior;
they are not the final hyperparameter study.

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

Reproduce the experiment with:

```bash
scripts/run_stage1_experiment.sh --fresh 200 20
```

Watch both trained policies compete with:

```bash
python main.py play --agents tabular_q_agent tabular_sarsa_agent \
  --scenario coin-heaven --n-rounds 1 --skip-frames
```
