# Preliminary Stage-2 results

This is an engineering checkpoint, not a final comparison. Both agents were
continued for 500 training rounds on `loot-crate` after adding the 13-component
bomb-aware state, safe-bomb action mask, and crate/survival rewards. They were
then evaluated for 100 rounds with seed 88.

| Method | Coins | Crates | Bombs | Suicides |
|---|---:|---:|---:|---:|
| Tabular Q-learning | 32 | 273 | 719 | 38 |
| Tabular SARSA | 74 | 396 | 559 | 42 |

The agents learned to place bombs and destroy crates. The suicide rate remains
too high for this stage to be considered solved. The escape-feasibility mask
only establishes that a route exists at bomb-placement time; Q-learning or
SARSA must still learn the sequence of actions that follows that route.

The next experiment should add escape-direction features and reward changes in
controlled ablations, rather than hiding the problem with a rule that directly
chooses the complete escape action sequence.
