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
