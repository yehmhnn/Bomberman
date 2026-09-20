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

## Reproducible Stage 2 curriculum run

The preferred entropy-0.02 Stage-1 checkpoint was continued for 2,000
`loot-crate` rounds, distributed evenly over the 200 shared training seeds.
This added 802,000 transitions, for 1,065,677 cumulative observed and optimized
transitions. The Stage-2 checkpoint ended with no unfinished rollout.

Evaluation on all 100 validation seeds produced:

| Metric | Mean | 95% CI |
|---|---:|---:|
| Score / coins | 5.49 | [3.71, 7.48] |
| Crates destroyed | 16.44 | [11.39, 22.05] |
| Survival | 100% | [100%, 100%] |
| Suicide | 0% | [0%, 0%] |
| Invalid actions | 0.00 | [0.00, 0.00] |
| Alive steps | 400.00 | [400.00, 400.00] |

The policy learned safe bomb use but not efficient board completion: it dropped
8.95 bombs per game on average, yet all 100 games reached the step limit. Mean
entropy over the final ten updates was only 0.069 nats. Although final training
updates contained more crate destruction than early updates, this did not
generalize into a strong deterministic validation policy.

This checkpoint should not yet be promoted into Stage 3. The next controlled
Stage-2 experiment should target the efficiency bottleneck while preserving
the action-safety mask—for example, stronger progress/time-pressure shaping or
an entropy schedule—then repeat the unchanged 100-seed validation. The older
exploratory Stage-2 checkpoint scored higher, but it was trained through a
different, unseeded history and therefore is evidence for possible headroom,
not a directly controlled comparison.

## Stage 2 goal-potential correction

Inspection exposed a reward-shaping error. Missing coin and crate targets are
encoded as distance zero, but the original potential used the minimum of both
encoded distances. Consequently, a missing coin erased the crate-distance
signal (`min(0, crate_distance) = 0`), and a missing crate similarly erased the
coin-distance signal. The corrected potential takes the minimum only over
positive, present-target distances.

A separate `goalfix` checkpoint was initialized from exactly the same
entropy-0.02 Stage-1 model and trained for the same 2,000 rounds over the same
200 training seeds. On the same 100 validation boards:

| Metric | Original Stage 2 | Goal fix | Paired change (95% CI) |
|---|---:|---:|---:|
| Score / coins | 5.49 | 8.28 | +2.79 [+0.53, +5.10] |
| Crates destroyed | 16.44 | 23.26 | +6.82 [+0.60, +13.18] |
| Survival | 100% | 100% | 0 pp |
| Suicide | 0% | 0% | 0 pp |
| Invalid actions | 0.00 | 0.00 | 0.00 |

The correction therefore improved both task outcomes without sacrificing
safety. It is still not sufficient to complete Stage 2: every validation game
reached 400 steps. This model is a better Stage-2 candidate, but further
training or an additional efficiency intervention should be measured before it
is used as the foundation for opponent hunting.

## Stage 2 entropy-0.05 ablation

The corrected goal potential was retained while the entropy coefficient alone
was raised from 0.02 to 0.05. The new checkpoint used the same Stage-1
initialization, 2,000-round budget, 200 training seeds, and 100 validation
boards. Final-ten-update entropy increased from 0.097 to 0.248 nats.

| Metric | Goal fix, entropy 0.02 | Goal fix, entropy 0.05 | Paired change (95% CI) |
|---|---:|---:|---:|
| Score / coins | 8.28 | 23.39 | +15.11 [+11.30, +18.76] |
| Crates destroyed | 23.26 | 59.97 | +36.71 [+27.36, +45.87] |
| Survival | 100% | 100% | 0 pp |
| Suicide | 0% | 0% | 0 pp |
| Invalid actions | 0.00 | 0.00 | 0.00 |

Higher exploration produced a large improvement without weakening safety, so
the entropy-0.05 checkpoint is the preferred Stage-2 model for Stage 3. All
games still reached the 400-step limit, meaning complete board clearance
remains an efficiency limitation rather than a safety limitation.

## Stage 3 peaceful-opponent baseline

The preferred Stage-2 checkpoint was continued for 2,000 `classic` rounds
against `peaceful_agent`, evenly distributed over the 200 training seeds. On
the 100 validation boards PPO achieved:

| Metric | Mean / rate |
|---|---:|
| Score | 1.56 |
| Coins | 1.11 |
| Opponent kills | 0.09 |
| Games with a kill | 9 / 100 |
| Win rate | 34% |
| Mean rank | 1.33 |
| Survival | 100% |
| Suicide / invalid actions | 0% / 0.00 |

This is safe but not a successful hunting policy. Inspection explains the
failure: the state includes opponent direction and distance, but the dense
potential reward includes only coin/crate distance. PPO is therefore rewarded
for resource navigation while opponent reward remains sparse until a bomb is
already able to hit or trap the target. The next controlled Stage-3 variant
should add opponent-distance potential without changing the safety mask or
shared validation protocol.

## Stage 3 opponent-distance correction

A separate hunting variant added negative nearest-opponent distance to the
potential in Stages 3–4 only. It used the same Stage-2 initialization,
entropy coefficient, 2,000 training rounds, and shared seeds as the baseline.
On the same 100 validation boards against `peaceful_agent`:

| Metric | Baseline | Hunting potential | Paired change (95% CI) |
|---|---:|---:|---:|
| Score | 1.56 | 6.69 | +5.13 [+4.07, +6.17] |
| Opponent kills | 0.09 | 0.41 | +0.32 [+0.20, +0.43] |
| Win rate | 34% | 85% | +51 pp [+40, +62] |
| Mean rank | 1.33 | 1.075 | -0.255 [-0.310, -0.200] |
| Survival | 100% | 99% | -1 pp [-3, 0] |

The variant killed the opponent in 41 games, up from 9, while scoring higher
on 76 boards. It incurred one suicide and one invalid action, so its safety is
slightly below the baseline but still high. This checkpoint is the preferred
starting point for the harder `coin_collector_agent` phase of Stage 3.

## Stage 3 coin-collector continuation

The hunting checkpoint was continued for 2,000 additional rounds against
`coin_collector_agent`, reaching 4,000 Stage-3 rounds in total. Evaluation on
the 100 validation boards produced:

| Metric | PPO | Coin collector |
|---|---:|---:|
| Score | 4.68 | 4.03 |
| Opponent kills | 0.15 | 0.04 |
| Win rate | 61% | 35% |
| Mean rank | 1.37 | 1.63 |
| Survival | 76% | 51% |
| Suicide | 21% | 34% |

PPO therefore beats the harder opponent on score, kills, wins, rank, and
survival, but its 21% suicide rate is too high for promotion to Stage 4.
Training diagnostics agree: the continuation recorded 271 kills, 551 deaths,
and 522 self-kills across 2,000 rounds.

Re-evaluation against `peaceful_agent` after the continuation found partial
forgetting. Kill rate fell from 0.41 to 0.22 (paired change -0.19, 95% CI
[-0.32, -0.05]) and score from 6.69 to 4.61, although win rate stayed nearly
unchanged at 86% and survival at 99%. The next intervention should address
temporal bomb escape and curriculum retention before full Stage-4 opponents.
