# Stages 3 and 4 tabular results

## Controlled comparison

The experiment asks whether on-policy SARSA learns safer opponent-aware
behavior than off-policy Q-learning when both use the same representation,
rewards, training budget, curriculum, and deterministic safety mask. Their
only intended algorithmic difference is the temporal-difference target:

- Q-learning: `r + gamma * max_a Q(s_next, a)`.
- SARSA: `r + gamma * Q(s_next, a_next)`, where `a_next` is sampled from the
  same epsilon-greedy policy and then executed.

Both agents append ten opponent features to the existing 24 Stage-2 features:
nearest-opponent direction and distance, whether a bomb would hit or trap that
opponent, immediate armed-opponent threat, opponent count, and nearest-opponent
bomb availability. An unseen 34-value state is initialized from the matching
24-value Stage-2 state. Stage 4 similarly falls back to the Stage-3 table
before Stage 2. Later TD updates specialize these transferred values.

Official events dominate the reward scale: coin `+10`, opponent kill `+50`,
self-kill `-40`, killed by an opponent `-30`, and survival `+5`.
Potential-based shaping adds progress toward coins, crate-bombing positions,
and opponents, plus a penalty for standing in a tile an armed opponent could
bomb immediately. A small action-dependent bonus rewards a safe bomb that
currently covers or traps an opponent.

The planned full curriculum is 2,000 Stage-3 episodes against one peaceful
agent, 3,000 against one coin collector, 4,000 Stage-4 episodes against three
rule-based agents, then 4,000 against a mixed rule/coin/peaceful lineup. The
results below are bounded pilots and must not be described as converged final
checkpoints.

## Stage 3 peaceful-agent pilot

The initial pipeline check was interrupted after 219 Q-learning and 211 SARSA
episodes against one `peaceful_agent`. Because the budgets differ and only the
first 20 validation seeds were used, this comparison is exploratory.

| Method | Score (95% CI) | Kills | Survival | Suicide | Crates | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| Tabular Q-learning | 2.85 [2.00, 3.75] | 0.05 | 100% | 0% | 45.85 | 85% |
| Tabular SARSA | 4.80 [3.10, 6.80] | 0.25 | 95% | 5% | 66.15 | 80% |

The useful finding is that the opponent state, transfer initialization,
rewards, and action pipeline can produce opponent kills while mostly retaining
Stage-2 survival. It is not evidence that SARSA is generally better.

On 300 synthetic four-player states, Q-learning had mean action latency
0.00060 seconds and maximum 0.00354 seconds. SARSA had mean 0.00097 seconds and
maximum 0.00400 seconds. Both are far below the official 0.5-second limit.

## Stage 3 equal-budget checkpoint

Both methods were subsequently brought to exactly 979 Stage-3 episodes and
evaluated against one `coin_collector_agent` on all 100 shared validation
seeds. Test seeds remained untouched.

| Method | Score (95% CI) | Kills | Survival | Suicide | Crates | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| Tabular Q-learning | 3.56 [3.10, 4.02] | 0.09 | 86% | 14% | 52.75 | 26% |
| Tabular SARSA | 3.51 [3.09, 3.97] | 0.08 | 77% | 21% | 55.25 | 26% |

The score intervals overlap almost completely, so this checkpoint does not
establish a meaningful score difference. Q-learning is safer here: survival
is nine percentage points higher and suicide seven points lower. Both still
lose most games to the harder coin collector, so Stage 3 is functional but not
converged.

## Stage 4 setup and one-versus-one diagnostic

Each method was initialized from its 979-round Stage-3 table and trained for
200 classic rounds against one `rule_based_agent`, distributed equally across
the 200 shared training seeds. Their common safety shield was extended to
treat every armed opponent's current tile as a possible next-turn bomb and to
reject actions that would then have no escape path.

| Method | Score (95% CI) | Survival | Suicide |
|---|---:|---:|---:|
| Tabular Q-learning | 2.92 [2.58, 3.24] | 53% | 41% |
| Tabular SARSA | 3.28 [2.90, 3.70] | 62% | 31% |

The opposing rule-based agents scored 6.03 and 5.88 respectively. SARSA is the
better early checkpoint in this diagnostic, but neither passes the protocol's
10% self-kill gate.

## Stage 4 held-out protocol lineups

The first three team-agreed held-out lineups were evaluated on all 100
validation seeds. The fourth remains pending because it requires the team to
identify the frozen "other agent" checkpoint; no substitute was silently
chosen.

| Method | Held-out opponents | Score | Win | Kills | Survival | Suicide |
|---|---|---:|---:|---:|---:|---:|
| Q-learning | 3 rule-based | 2.29 | 8% | 0.09 | 43% | 48% |
| SARSA | 3 rule-based | 2.11 | 7% | 0.07 | 55% | 34% |
| Q-learning | rule + coin + peaceful | 3.81 | 18% | 0.32 | 58% | 37% |
| SARSA | rule + coin + peaceful | 3.18 | 11% | 0.17 | 57% | 41% |
| Q-learning | 3 coin collectors | 1.79 | 7% | 0.06 | 81% | 19% |
| SARSA | 3 coin collectors | 1.96 | 14% | 0.12 | 77% | 23% |

Score confidence intervals and every diagnostic metric are retained in
`evaluation/results/summary/summary.csv`.

There is no universal winner after only 200 Stage-4 rounds. SARSA survives
substantially more often against three attacking rule agents, while Q-learning
scores and kills more in the mixed lineup. Against three coin collectors,
SARSA has the better score and win rate while Q-learning is slightly safer.
Both fail the self-kill gate in every lineup, so the correct decision is to
continue the fixed curriculum rather than select either as final.

The prospective-bomb shield cannot guarantee survival: it assumes armed
opponents could bomb at their current positions, while deaths can still arise
from already committed routes, moving bodies, simultaneous bombs, or the
policy fallback when every full-horizon choice is unsafe.

## Evaluation rules and remaining work

All development decisions use validation seeds and the lineups fixed in
`EXPERIMENT_PROTOCOL.md`. Reported values are official game metrics with
bootstrap confidence intervals; shaped training rewards are never reported as
performance. Test seeds must be inspected only once after all methods and
hyperparameters are frozen.

The remaining work is to finish the declared curriculum, evaluate the fourth
held-out lineup after the frozen comparison agent is agreed, and select a
checkpoint only if it passes the protocol's reliability and suicide gates.
