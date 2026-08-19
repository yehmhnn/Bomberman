# Project plan

## Objective

Build at least two genuinely learned agents, select the strongest through
reproducible experiments, and submit a CPU-safe trained agent together with a
scientific report. Tournament score matters, but the primary goal is a clear
sequence of hypotheses, controlled comparisons, and evidence-based design
changes.

## Methods from the two reinforcement-learning lectures

The lectures introduce the following concepts and methods:

- fully and partially observed environments, Markov chains, hidden Markov
  models, and Markov decision processes;
- policies, finite and discounted returns, state values `V(s)`, action values
  `Q(s, a)`, and the Bellman equations;
- model-based dynamic programming with value iteration and greedy policy
  extraction;
- model-free temporal-difference control, especially off-policy Q-learning and
  on-policy SARSA;
- exploration versus exploitation using epsilon-greedy policies, decaying
  epsilon schedules, and softmax action selection;
- learning-rate schedules and stochastic-approximation conditions;
- reward shaping, especially potential-based shaping, which preserves the
  optimal policy;
- tabular Q-learning, batch/replay updates, temporal-difference targets,
  multi-step returns, and temporal-difference-lambda;
- function approximation for large state spaces, including one-hot tabular
  representations, linear models, multilayer neural networks, and action-wise
  output heads.

## Implementation stages

### 0. Reproducible foundation

- Confirm a clean headless framework run.
- Add an experiment runner that records configuration, seeds, commit hash,
  scores, survival, kills, coins, crates, invalid actions, and decision time.
- Define fixed training, validation, and test seed sets.
- Measure supplied random, peaceful, coin-collector, and rule-based baselines.

Exit criterion: repeated seeded runs produce parseable results and summary
statistics with confidence intervals.

### 1. Navigation and coins

- Implement compact state features: valid adjacent tiles, local danger, nearest
  coin direction/distance, recent-position memory, and action validity.
- Train a tabular Q-learning agent on `coin-heaven`.
- Compare raw coordinates, engineered features, and symmetry-canonical features.
- Tune alpha, gamma, epsilon schedule, and reward magnitudes.

Exit criterion: reliably collect coins faster than random on held-out seeds.

### 2. Bombing, crates, and survival

- Add an exact blast-time/danger map that respects walls, crates, bomb timers,
  and chain reactions.
- Add features for escape feasibility, useful bomb placement, and nearby crates.
- Use a curriculum from simple layouts to `classic` and potential-based shaping
  for progress without changing the intended optimum.

Exit criterion: destroy crates and reveal/collect coins while keeping self-kill
and invalid-action rates low.

### 3. Opponent hunting

- Add opponent reachability, trapping, blast coverage, and safe pursuit features.
- Train against `peaceful_agent`, then `coin_collector_agent`.
- Compare Q-learning with SARSA to test whether on-policy learning yields safer
  behavior under exploration.

Exit criterion: positive kill differential and score advantage on held-out
matches against both target agents.

### 4. Full competition and second model

- Implement `linear_q_agent` with action-wise linear value approximation,
  replay batches, target clipping/normalization where justified, and the same
  evaluation harness.
- Train against mixtures of supplied agents and snapshots of our agents; add
  controlled self-play only after stable baseline performance.
- Compare tabular Q-learning, SARSA variant, and linear approximation through
  ablations of features, shaping, symmetry augmentation, replay, and curriculum.

Exit criterion: select a statistically stronger model that approaches or beats
the supplied `rule_based_agent` without violating runtime limits.

### 5. Freeze, compatibility, and submission

- Freeze training, load parameters via paths relative to the agent directory,
  and run evaluation-only smoke tests.
- Profile worst-case `act` latency, RAM, and model size on CPU.
- Build and test in the supplied Docker environment and use the official pre-run.
- Package only the best agent directory for the code deadline.
- Produce plots/tables and write the report around hypotheses, methods,
  experiments, failures, and conclusions; identify each section's author.

## Evaluation protocol

Primary outcomes are mean official score, win rate, survival rate, kill/death
ratio, and head-to-head score difference. Diagnostic outcomes are coins, crates,
self-kills, invalid actions, episode length, and action latency. Every important
comparison should use the same held-out seeds and opponent lineups, enough rounds
to report uncertainty, and no tuning on the final test set.

## Immediate next work

1. Create the Python environment and run the headless smoke test.
2. Implement the experiment runner and baseline benchmark.
3. Implement the shared danger-map and feature tests.
4. Implement `tabular_q_agent` and begin Stage 1 training.
