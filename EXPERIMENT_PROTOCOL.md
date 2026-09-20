## Evaluation protocol

We wrote this document before we started tunings. We aim that every number we report later is comparable and can be reproduced by next person. If we change a rule after seeing the results, the comparison is not same anymore, so we fix the rules here first.

### Seeds

The seed sets are in evaluation/seeds.py file. 200 training seeds, 100 validation seeds and 100 test seeds. All of them come from one master seed, so that every team member gets exactly same lists. The three sets dont overlap.

We use the training seeds for training, the validation seeds during development(for example to compare two feature sets or to pick a checkpoint), we check the test seeds only once, at the end, for the final numbers in the report. No trained model should ever be tuned against the test seeds.

The `--seed` flag only fixes the board (where the crates and coins are, and the starting corners). It does not fix the randomness in the agents. This is why one seed is not enough. We run every configuration over many seeds and we report a bootstrap 95% confidence interval(not a single score). All numbers in this document use only the official game score. Shaped or auxiliary rewards are a training tool and never appear in the reported evaluation.

The provided agents (`random_agent`, `peaceful_agent`, `coin_collector_agent`, `rule_based_agent`) do not learn, so we know that there is no risk of overfitting them. We measured their baselines on the test seeds, so that later on we can compare our own agent on the exactly same boards.

### Metrics

Main metrics (to make decision):
- mean official score
- win rate and mean rank
- survival rate and number of steps survived
- opponents killed

Diagnostic metrics (helps us to understand the behaviour):
- coins collected, crates destroyed
- suicides and invalid actions
- episode length
- mean decision time per move

Some notes so we do not read too much into these numbers:
- `survived` is an estimate. The stats file has no real flag called survived, so we compute it from step count and the suicide flag. An agent that is killed on the last step can be counted as survived by mistake. We think the error is around 1 to 2 percentage points.
- `win` only counts a single best agent. If all agents score the same (for example everyone scores 0), then we can count no win for anyone.
- `rank` and `win` are useful only when agents are different. When there are four identical agents in a match the mean rank is always 2.5.
- For now, for the decision time we only have the mean per move. The real worst-case and the p95 need a per-step timer, which we are planning to add in the compatibility phase before submission.

### How we run and aggregate matches

`benchmark.py` runs each match as a separate process, one round per seed, and writes one CSV row per agent. Then `aggregate.py` reads these rows and builds the summary table with confidence intervals. We run on two scenarios. coin-heaven is for Task 1: no crates, no bombs needed, the agent only navigates and collects coins. classic is for Tasks 2 to 4: it has crates, hidden coins and opponents.

When multiple identical agents play the same match, their rows are dependent (their coins add up to the total on the board). So `aggregate.py` first averages the identical agents in each match, and then does the bootstrap over the per-match values. The reported `n_matches` is the number of the independent matches, not the number of rows.

### Model selection rule

We score each candidate on a common evaluation mix with fixed weights: score 40%, win rate 25%, survival 20%, kills 10%, latency and reliability 5%. We decided these weights now so we cannot fit the rule to the results later.

Why these weights:
- Score gets the largest weight because the tournament ranking uses the official score.
- Win rate matters because the maps are random. An agent with a high but very noisy score that still loses most matches is worse than a steady winner.
- Survival gets a real weight because our baseline shows that even `rule_based_agent` dies in 40 to 60 percent of the four-player matches. Staying alive is both a big part of the score and something we can realistically do better than the baseline.
- Kills are worth 5 points each, so they count, but they are risky and noisy, so the weight is smaller.
- Latency and reliability get a small weight, but it is really a gate: an agent that times out or crashes loses the match.

Common evaluation line-ups (we used to call this "held-out," which overstates it -- our agents already trained against `peaceful_agent`, `coin_collector_agent` and `rule_based_agent` directly, so the opponent types are not unseen the way a real held-out set would be. What is genuinely held out is the seeds: no candidate is ever tuned against the validation or test seeds. These four line-ups are just a fixed, agreed set of matchups every candidate is measured on the same way):
- our agent against 3x `rule_based_agent`
- our agent against 1x `rule_based_agent`, 1x `coin_collector_agent`, 1x `peaceful_agent`
- our agent against 3x `coin_collector_agent`
- our agent against 1x `rule_based_agent` and 2x a frozen copy of our other agent -- "frozen" means a specific, permanently pinned checkpoint of whichever agent is playing the reference role, not whatever that agent's live model.pt happens to be that day (see `agent_code/dqn_agent_frozen_ref/` for the current pinned dqn_agent reference, and its commit hash, used for this line-up from here on)

We will run these on the validation seeds during development and on the test seeds once at the end. Our agent always starts in the same slot. The line-ups get equal weight unless we explain a reason to change that.

Penalties:
- Any match where our agent throws an exception or goes over the 0.5 second move limit is scored as 0 score, 0 survival, last rank, and is flagged.
- If the crash-or-timeout rate over the evaluation set is above 1%, the candidate is removed from selection, no matter what its weighted score is.
- If the self-kill rate is above 10%, we subtract a fixed penalty of 0.05 from the weighted score, because self-kills are the most avoidable failure and our reward shaping is meant to fix them.

### Experiment record
Every training run gets one row in the shared "Experiment log" sheet:
experiment id, git commit, model and config, training seed or seeds, training scenario and opponents, number of episodes, evaluation split, checkpoint, all metrics with confidence intervals, a one-line hypothesis, and the result and decision.

### Experiment log: q_agent, classic solo (Task 2)

Copied here in addition to the shared sheet because these four runs are the reason we deviate from a strict reading of the model selection rule above (see the note at the end).

All four rows: agent_code/q_agent, 34 features (24 core + 10 opponent), LinearQ with ALPHA=0.01, GAMMA=0.95, TD_ERROR_CLIP=5.0, EPSILON_START=1.0, EPSILON_MIN=0.05, EPSILON_DECAY=0.998. Training scenario is classic, solo (no opponents placed), TRAIN_SEEDS. Evaluation is 20 matches on the first VAL_SEEDS, scenario classic, solo, greedy policy (epsilon=0). Metrics are official score and the diagnostic metrics from `evaluation/results/summary/summary.csv`.

Caveat that applies to every row below, not just one of them: `setup_training` resets epsilon to 1.0 at the start of each of these runs rather than continuing its decay from the previous stage, so roughly the first 1500 of each stage's rounds (about 19% at 8000 rounds) are heavily exploratory regardless of what reward change is being tested. Since this is a single shared linear weight vector, not a tabular model, noisy data from that phase feeds the same weights used for the final greedy policy. None of these stage-to-stage comparisons are as clean as a single-variable change would be; we are not fixing this now given the time budget, just naming it so we do not overclaim causality from one 8000-round run.

**stage3_qagent_solo_classic** — 3000 rounds, warm start from the stage2 coin-heaven checkpoint. Reward: KILLED_SELF=-30.0, CRATE_DESTROYED=2.0, GOT_KILLED=-20.0, COIN_COLLECTED=10.0, STEP_PENALTY=-0.02, plus the coin/crate distance potential shaping.
Result: score 0.20 [0.00, 0.45], suicide rate 0.35 [0.15, 0.55], crates destroyed 5.05 [2.60, 7.55], survived 0.65 [0.45, 0.85].
Hypothesis: the agent starts bombing crates once it has opponent-aware features and enough rounds, without other agents to complicate the danger map. Confirmed partially: it does bomb crates, but one in three rounds ends in a self-kill, keeping net score low.

**stage4_qagent_solo_classic_18k** — 15000 more rounds on top of stage3 (18000 cumulative), same reward as stage3.
Result: score 0.00 [0.00, 0.00], suicide rate 0.00, crates destroyed 0.00, survived 1.00.
Hypothesis was that more training would fix the self-kill problem. It did not: it removed it by making the agent stop placing bombs entirely. This is a named negative result for the report: with KILLED_SELF=-30 fifteen times bigger than the flat step penalty and six times bigger than CRATE_DESTROYED=2, the safe local optimum for a linear value function is "never bomb, survive to 400 steps, score 0" — and 15000 rounds of TD learning found exactly that optimum. Decision: do not keep this checkpoint despite it reaching survived=1.00 and a clean run; it fails Task 2's actual requirement (destroy crates, collect the coins they hide). Rebalance the reward and continue training instead of reverting.

**stage5_qagent_solo_classic_rebalanced** — 8000 more rounds on top of stage4 (26000 cumulative). Reward changed at this point only: KILLED_SELF -30.0 to -15.0, CRATE_DESTROYED 2.0 to 5.0 (both changed together, one intervention, see caveat below).
Result: score 0.05 [0.00, 0.15], suicide rate 0.30 [0.10, 0.50], crates destroyed 2.75 [1.75, 3.80], survived 0.70 [0.50, 0.90].
Hypothesis: cheapening the self-kill penalty and rewarding crates more should push the agent back off the do-nothing optimum without recreating stage3's suicide rate. Partially confirmed: it does bomb again (crates 0 to 2.75), but suicide rate is back near stage3's level and net score is still very low. Caveat: KILLED_SELF and CRATE_DESTROYED were changed in the same run, so we cannot yet say which constant drove the change; treat this as one confounded intervention. We believe KILLED_SELF was the dominant lever, because stage4's collapse only required a very negative terminal penalty relative to a small per-step signal, not a small CRATE_DESTROYED value — but this is not proven separately. Decision: keep as the working checkpoint over stage4 (see the model-selection note below), diagnose the 30% suicide rate before tuning further.

**Diagnostic (no retraining): where the stage5 self-kills come from.** Ran 25 solo classic rounds with the stage5 checkpoint, epsilon=0, with per-step debug logging (position, safe_action_mask output, chosen action) added temporarily to `callbacks.py`. Result: 11/25 rounds ended in a self-kill, and every one of the short (13 to 22 step) losing rounds we inspected followed the same shape, reproducible byte-for-byte across rounds that spawn in the same corner (the immediate crate layout around each of the 4 classic spawn corners is fixed, so with epsilon=0 the near-spawn behaviour is deterministic): the agent bombs a nearby crate, retreats into a dead-end pocket next to the bomb instead of continuing away, comes back to the same tile once the first bomb has cleared, drops a second bomb from the same spot, and then loses a step to an unnecessary WAIT while standing in its own new blast radius before the mask reports no safe move left. This is not a mask bug — safe_action_mask correctly narrows to WAIT only once escape is already impossible; the mistake happens earlier, when moves are wasted while multiple safe options still exist. Root cause we could point to: `_goal_potential` only rewards distance to the nearest coin/crate, so while fleeing a bomb next to the crate it just destroyed, moving away from the bomb is scored the same as moving away from the goal, i.e. the shaping term actively works against fleeing instead of for it.

**stage6_qagent_solo_classic_safetyshaping** — targeted fix, not a blind hyperparameter change. Added `_safety_potential` to `training/rewards.py` (mirrored into the vendored copy): 0 when the current tile is off every active blast line, else negative and rising in magnitude as the nearest lethal countdown on that tile gets closer to zero, combined additively with the existing coin/crate potential. Falsifiable hypothesis: this should lower the self-kill rate measurably below stage5's 30% on the same 20 validation-seed matches, without pushing crates destroyed back down to stage4's 0. 8000 rounds on top of stage5 (34000 cumulative), warm start, same TRAIN_SEEDS.
Result: score 0.10 [0.00, 0.25], suicide rate 0.60 [0.40, 0.80], crates destroyed 6.15 [4.10, 8.25], survived 0.40 [0.20, 0.60]. **Hypothesis rejected**: the self-kill rate went up instead of down (the two CIs [0.10,0.50] vs [0.40,0.80] overlap at the edges, so at n=20 this alone is a weaker signal). Crates destroyed more than doubled with non-overlapping CIs (2.75 to 6.15, a real difference at this sample size), so the clearer finding is that the agent became more willing to bomb, and it did not become safer while doing so. Our reading: `_safety_potential` is 0 when fully safe and increasingly negative while a blast is imminent, so a trajectory that enters danger and then escapes back to a safe tile collects `gamma*0 - (large negative)`, a sizeable one-off positive shaping bonus, on top of whatever it already gets for the crate. That turns "survive your own bomb" into a small reward event in its own right, which looks like it made the agent more willing to keep taking that bet rather than more careful about it -- more bombing attempts, and with the per-attempt risk not clearly improved, more total deaths. Decision: reverted the code (`training/rewards.py` and the vendored copy are back to the stage5 reward function, `git diff` against the stage5 commit shows only the reward files changed and back again) and restored `agent_code/q_agent/model.npz` to the stage5 checkpoint. We are keeping this run's raw CSV and summary row as a documented negative result for the report rather than deleting it. Per the 1-day budget and the supervisor-style review we did before starting this, one bounded attempt was the agreed plan either way, win or lose; we are stopping here rather than iterating further on the reward function, and moving remaining time to Task 3 (mixed opponents), where stage5 already has real value to demonstrate (win/kills/survival mean something there).

Note on the model selection rule above: applying its 40/25/20/10/5 weights directly to these solo-classic checkpoints is a category error, not a real comparison. The rule is defined over the four held-out opponent mixtures; with zero opponents, win rate and kills are trivially fixed at 1.0 and 0 for anyone still alive, and "survival" carries no information beyond the suicide flag already in the penalty term. Plugging stage4 and stage5 into the formula anyway gives stage4 a higher weighted score (about 0.50) than stage5 (about 0.41), purely because stage4's do-nothing policy maxes out win/survival/latency while never triggering the suicide penalty — even though stage4 fails Task 2 outright and stage5 at least attempts it. We are keeping stage5 as the Task 2 checkpoint despite what the naive formula says (stage6 was tried and rejected, see above), and recording this here so the discrepancy is a documented decision and not an oversight.

### Task 3: q_agent against real opponents

We trained q_agent step by step: first against peaceful_agent (no bombs, just extra bodies on the board), then coin_collector_agent, then rule_based_agent (this one bombs back). Each stage is a warm start from the one before, 5000-6000 rounds, no fixed seed for training.

Big finding first: **training this agent is unstable**. Same code, same starting weights, same round count, different runs gave very different results. Stage7 needed 4 tries (2 bad, 1 good, 1 from a fix we later reverted). Stage9 failed hard both times we tried it. This is not bad luck on one run, it is a repeated pattern, so we treat it as a real property of training a linear agent this way, not noise to average away.

**stage7_qagent_vs_peaceful** — warm start from stage5, 5000 rounds vs 3x peaceful_agent. Try 1: score 0.30, suicide 0.65. Try 2 used a lower starting epsilon on warm start (idea: do not waste rounds on full random exploration when the weights are already trained). Result was worse, not better: suicide 0.85. We reverted that change (see below). Try 3, same plain code as try 1: score 0.00, suicide 0.00 -- the same "never bombs" collapse as stage4. Try 4: score 1.15 [0.30, 2.20], suicide 0.50, crates 4.25. We kept try 4.

We also tried a different fix during try 1's investigation: some deaths happen because another agent (even a peaceful one) walks into the exact tile our escape path needs, one step after we checked it was free. We added a safety margin in `shared/safety.py` (`escape_exists` also treats an opponent's neighboring tiles as blocked, not just their current tile). The new unit test for this passes, but re-running the same stage7 checkpoint with the fix gave the exact same benchmark numbers (score 0.30, suicide 0.65, same as try 1) -- this specific problem was not the main cause of the high suicide rate here. We reverted this too, to keep the code simple, since it did not measurably help. That re-run's raw CSV was a throwaway check and was not kept, so these two numbers only live here in this text, not in summary.csv.

**stage8_qagent_vs_coincollector** — warm start from stage7 (try 4), 5000 rounds vs 3x coin_collector_agent. Try 1: suicide 1.00, survived 0.00. Try 2: suicide 0.75, survived 0.10, crates 2.05. We kept try 2 (still bad, just less bad).

**stage9_qagent_vs_rulebased (first pass)** — warm start from stage8, 6000 rounds vs 3x rule_based_agent. Both tries collapsed: try 1 survived 0.00, suicide 1.00; try 2 survived 0.00, suicide 0.95. Against an opponent that actually bombs, our agent almost never survived at this round count.

With those numbers, the model selection score (same formula and scaling as below) came out around 0.01 -- very low. Rather than stop there, we asked: is 6000 rounds just not enough for this harder opponent, given how unstable training already was? So we trained two more candidates from the same stage8 start, 12000 rounds each (double), and benchmarked both on val seeds before picking one (the protocol reserves VAL_SEEDS for exactly this: tuning and checkpoint selection). TEST_SEEDS were not touched.

- candidate A: score 0.60, survived 0.15, suicide 0.65, crates 4.55
- candidate B: score 0.75, survived 0.00, suicide 0.90, crates 2.30

B has a higher score but never survives a match; A actually stays alive sometimes and destroys more crates. We picked survival and suicide rate as the main criteria (a checkpoint that always dies has no real value even if it scores a few points first), so we kept **candidate A** as the final stage9 checkpoint. Note: A's survival, 3 out of 20 matches, has a wide confidence interval [0.00, 0.30] that overlaps B's 0.00 -- on its own this one number is not a strong signal at n=20. The suicide rate gap is firmer ([0.45,0.85] vs [0.75,1.00], barely overlapping), and both point the same way, so we trust the choice, but survival alone would not have been enough. Doubling the rounds clearly helped either way: candidate A alone already beats both 6000-round tries on every metric.

We call these "common evaluation line-ups" now, not "held-out" (the shared rule section above still says "held-out" -- that is on purpose, renaming it there is a team decision we have not settled yet, this is just how we refer to it in our own results below). The agents already trained against these same opponent types (peaceful, coin_collector, rule_based), so "held-out" overstates it -- it is not an unseen-opponent test the way a real held-out set would be, just a fixed, agreed set of matchups every candidate is measured on the same way.

**Common evaluation line-ups (final)**, val seeds, n=20, using candidate A:
1. vs 3x rule_based_agent: score 0.60, win 0.05, survived 0.15, suicide 0.65, kills 0.05 (same run as candidate A above)
2. vs 1 rule_based + 1 coin_collector + 1 peaceful: score 0.65, win 0.05, survived 0.10, suicide 0.90, kills 0.05
3. vs 3x coin_collector_agent: score 0.90, win 0.10, survived 0.40, suicide 0.60, kills 0.10
4. vs 1 rule_based_agent + 2x frozen copy of our other agent: first attempt used two frozen q_agent copies instead of dqn_agent, because dqn_agent's checkpoint did not load at the time. That was a bad substitute for another reason too, caught in team review: since the opponent was q_agent itself, each candidate would face a different opponent (itself), so candidates would not be comparable on this line-up. Now that dqn_agent's checkpoint is fixed, we reran this with 2x frozen dqn_agent, a fixed reference for every candidate. Score 0.30, win 0.00, survived 0.30, suicide 0.50, kills 0.00. Worth noting: q_agent actually does better here than against the self-copies (survived 0.30 vs 0.08, suicide 0.50 vs 0.77) -- our guess is dqn_agent plays cleaner and creates less chaos on the board than a struggling copy of q_agent would.

Model selection score, same method as the first pass (score/5, kills/1, latency at full marks since nothing crashed or timed out, 0.05 self-kill penalty per lineup since suicide is still above 10% everywhere): about **0.11**, same as with the old line-up 4, though the number underneath changed (lower score, higher survival, roughly cancelling out). Still a real improvement over the ~0.01 first pass either way.

Decision: we keep candidate A as the current Task 3 checkpoint. This is the best result we found for q_agent against real opponents, using a systematic search (multiple candidates, picked on validation seeds) rather than accepting the first bad run. It is still clearly weaker than dqn_agent and still dies more than half the time against a real bomber, but it is a genuine, working agent now, not a coin flip between "never bombs" and "dies instantly." The instability finding from the first pass still stands and is worth keeping in the report: the same setup can land on a working or a broken agent depending on random seed, and more training data is one real way to fight that, though it does not remove it completely (candidate B, same round count as candidate A, was still much worse). Our best guess why the instability happens at all: a small linear model with epsilon-greedy exploration and a full epsilon reset on every warm start can settle into different behaviors depending on what the first ~1500 (mostly random) rounds happen to look like. This gets worse once real, moving opponents make the game less predictable. A bigger model like dqn_agent has more room to average this out; ours does not.