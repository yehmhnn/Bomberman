# PPO exploratory results

## Checkpoint

The same masked actor-critic checkpoint was trained for 25 smoke-test rounds,
1,000 rounds on `coin-heaven`, and 1,000 rounds on `loot-crate`: 2,025 rounds
and 736,732 environment transitions in total. These training runs did not yet
use the shared training-seed distributor, so they are exploratory. Every
reported evaluation below uses the unchanged shared benchmark code and only
validation seeds.

## Stage 1 checkpoint

After 1,025 total rounds, a 20-seed validation smoke evaluation on
`coin-heaven` produced 41.60 coins per game (95% bootstrap CI 37.55 to 45.25),
100% survival, and no suicides. This establishes that the PPO implementation
learns navigation; it is not a full 100-seed final comparison.

## Stage 2 checkpoint

After the additional 1,000 `loot-crate` rounds, the checkpoint was evaluated
on all 100 shared validation seeds in `loot-crate`:

| Metric | Mean | 95% CI |
|---|---:|---:|
| Score / coins | 14.44 | [11.01, 18.06] |
| Crates destroyed | 37.06 | [28.67, 46.35] |
| Survival | 100% | [100%, 100%] |
| Suicide | 0% | [0%, 0%] |
| Invalid actions | 0.00 | [0.00, 0.00] |
| Alive steps | 400.00 | [400.00, 400.00] |

The safety result is strong, but every game reached the 400-step limit. PPO
therefore learned a conservative bomb-safe policy without yet learning to
finish the board efficiently. The next PPO experiment should keep the same
architecture and first test more Stage-2 training plus a modest entropy or
time-pressure adjustment; changing algorithms before diagnosing the learning
curve would confound the comparison.

## Reproducible Stage 1 rerun

A new checkpoint was trained from scratch for 1,000 `coin-heaven` rounds,
distributed evenly across all 200 shared training seeds. PPO's sampling RNG
was explicitly tied to each board seed. The run observed and optimized 301,056
transitions in 314 updates; no partial rollout was left at the checkpoint.

Evaluation on all 100 validation seeds produced:

| Metric | Mean | 95% CI |
|---|---:|---:|
| Score / coins | 35.64 | [33.55, 37.69] |
| Survival | 100% | [100%, 100%] |
| Suicide | 0% | [0%, 0%] |
| Invalid actions | 0.00 | [0.00, 0.00] |

This is below the earlier 20-seed exploratory estimate of 41.60, demonstrating
why the complete shared validation set is needed. Training diagnostics explain
part of the remaining gap: mean policy entropy fell from 1.243 nats over the
first ten updates to 0.023 over the final ten. Bomb frequency fell from 7.80%
to 0.03%, and WAIT frequency from 24.82% to 0.02%, so PPO learned the correct
Stage-1 action types but became nearly deterministic before navigation was
optimal. The next controlled experiment should address exploration rather than
assuming that Stage 2 alone will repair navigation.

The first evaluation attempt also exposed a framework timing bug: callback
duration used the non-monotonic wall clock, so system suspend/clock adjustment
created two false 15-minute-plus think times. After replacing duration timing
with `perf_counter`, the complete 100-seed rerun reproduced every score while
mean action time was 0.000392 seconds and the maximum per-game mean was
0.000646 seconds.

## Stage 1 entropy ablation

The first controlled ablation changed only the entropy coefficient from 0.01
to 0.02. It was trained from scratch for the same 1,000 rounds over the same
200 training seeds, producing 263,677 optimized transitions in 278 updates.
Evaluation used the same 100 validation boards as the baseline:

| Entropy coefficient | Score / coins (95% CI) | Survival | Suicide |
|---:|---:|---:|---:|
| 0.01 | 35.64 [33.55, 37.69] | 100% | 0% |
| 0.02 | 39.67 [37.61, 41.65] | 100% | 0% |

Because the validation boards were paired, the most informative estimate is
the per-seed score difference: the 0.02 model gained 4.03 coins per game, with
a 95% paired bootstrap interval of [2.21, 5.98]. It scored higher on 37 boards,
tied on 55, and scored lower on 8. This is evidence that the improvement is not
just a different sample of boards.

The higher coefficient slowed but did not prevent policy collapse: mean
entropy over the last ten updates was 0.060 nats, compared with 0.023 for the
baseline. It also finished games sooner on average (343.71 versus 380.36
steps), with zero invalid actions and a mean action time of 0.000393 seconds.
The 0.02 checkpoint is therefore the preferred Stage-1 initialization for the
next curriculum experiment; further entropy tuning should remain a separate
ablation rather than changing the shared evaluation protocol.
