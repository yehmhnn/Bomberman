## Evaluation protocol

Team decision:
We fixed the evaluation setup before starting any tuning so results stay
comparable. The seed sets live in evaluation/seeds.py: 200 training, 100
validation, 100 test seeds, all drawn from one master seed. We only look at the
test seeds once, for the final numbers in the report.

The --seed flag only fixes the board (crates, coins, start corners), not the
agents' own randomness, so we run every configuration over many seeds and report
a bootstrap 95% confidence interval instead of a single score.

benchmark.py runs each match as a separate process, one round per seed, and
writes one CSV row per agent. aggregate.py turns the raw rows into a summary
table with confidence intervals.

Model selection: we score candidates on a held-out opponent mix with fixed
weights (score 40%, win rate 25%, survival 20%, kills 10%, latency 5%), decided
now so we can't tune the rule to the results later.