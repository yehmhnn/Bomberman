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

We score each candidate on a held-out opponent mix with fixed weights: score 40%, win rate 25%, survival 20%, kills 10%, latency and reliability 5%. We decided these weights now so we cannot fit the rule to the results later.

Why these weights:
- Score gets the largest weight because the tournament ranking uses the official score.
- Win rate matters because the maps are random. An agent with a high but very noisy score that still loses most matches is worse than a steady winner.
- Survival gets a real weight because our baseline shows that even `rule_based_agent` dies in 40 to 60 percent of the four-player matches. Staying alive is both a big part of the score and something we can realistically do better than the baseline.
- Kills are worth 5 points each, so they count, but they are risky and noisy, so the weight is smaller.
- Latency and reliability get a small weight, but it is really a gate: an agent that times out or crashes loses the match.

Held-out opponent mix (none of these line-ups are used for training):
- our agent against 3x `rule_based_agent`
- our agent against 1x `rule_based_agent`, 1x `coin_collector_agent`, 1x `peaceful_agent`
- our agent against 3x `coin_collector_agent`
- our agent against 1x `rule_based_agent` and 2x a frozen copy of our other agent

We will run these on the validation seeds during development and on the test seeds once at the end. Our agent always starts in the same slot. The line-ups get equal weight unless we explain a reason to change that.

Penalties:
- Any match where our agent throws an exception or goes over the 0.5 second move limit is scored as 0 score, 0 survival, last rank, and is flagged.
- If the crash-or-timeout rate over the evaluation set is above 1%, the candidate is removed from selection, no matter what its weighted score is.
- If the self-kill rate is above 10%, we subtract a fixed penalty of 0.05 from the weighted score, because self-kills are the most avoidable failure and our reward shaping is meant to fix them.

### Experiment record
Every training run gets one row in the shared "Experiment log" sheet:
experiment id, git commit, model and config, training seed or seeds, training scenario and opponents, number of episodes, evaluation split, checkpoint, all metrics with confidence intervals, a one-line hypothesis, and the result and decision.