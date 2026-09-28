# Stage-2 results: crates and safe bombing

## Protocol

Both tabular agents used the same 24-value state, rewards, safety mask,
hyperparameters, scenarios, and training seeds. Their only algorithmic
difference remained the temporal-difference target: Q-learning bootstraps from
the best safe next action, while SARSA bootstraps from its sampled next action.

The final curriculum was:

1. 1,000 `loot-crate` rounds from frozen training seed 94974.
2. 1,000 solo `classic` rounds from frozen training seed 69498.
3. Add the escape-preserving action mask, then continue for 1,000 solo
   `classic` rounds from frozen training seed 168237.

Evaluation used `evaluation/benchmark.py` on all 100 frozen validation seeds,
one solo `classic` match per seed, with exploration disabled. Test seeds were
not used.

## Final validation result

| Method | Score/coins | 95% CI | Crates | Bombs | Survival | Suicide | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tabular Q-learning | 3.96 | [3.48, 4.45] | 67.83 | 23.51 | 100% | 0% | 0.00 |
| Tabular SARSA | 4.55 | [3.98, 5.08] | 78.22 | 27.24 | 100% | 0% | 0.00 |

On paired validation boards, SARSA minus Q-learning was +0.59 score (bootstrap
95% CI [-0.13, 1.27]) and +10.39 crates ([0.97, 19.64]). Thus SARSA's score
advantage is not conclusive at this sample size, while its additional crate
destruction is supported by the paired interval. Both agents satisfy the Stage
2 safety objective.

## What failed and why

The first useful-bomb bonus was +0.75. In states where a safe bomb would hit a
crate, the learned mean BOMB values were negative (Q-learning -0.32, SARSA
-0.50). Dropping a bomb made the danger component of the potential immediately
worse, outweighing the action bonus. Both agents therefore learned safe walking
loops and scored almost zero.

Raising the safe useful-bomb bonus to +3.0 made bombing learnable, but SARSA
initially reached a 19% suicide rate. The original action mask proved that an
escape existed when BOMB was selected, but later moves were checked only for
immediate danger. The final mask accepts each subsequent move only when at least
one complete escape path remains. After training with those state bits, both
agents reached zero suicides without suppressing bombing.

## Reproduction

The final trained tables are `q_table_stage2.pkl` in the two tabular agent
directories. Re-run validation with:

```bash
python evaluation/benchmark.py --tag stage2_tabular_q_val \
  --agents tabular_q_agent --scenarios classic --split val
python evaluation/benchmark.py --tag stage2_tabular_sarsa_val \
  --agents tabular_sarsa_agent --scenarios classic --split val
python evaluation/aggregate.py --tags \
  stage2_tabular_q_val stage2_tabular_sarsa_val
```
