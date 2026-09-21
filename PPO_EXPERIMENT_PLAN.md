# Masked PPO experiment

## Controlled question

Can an on-policy actor-critic learn a safer and more stable policy than the
off-policy DQN when both receive the same 35-value state, action-safety shield,
event rewards, and potential shaping?

The model has a shared two-layer MLP, a six-logit policy head and a scalar value
head. Unsafe actions are assigned zero probability before sampling. Evaluation
uses a deterministic masked argmax.

## Learning equations

The temporal-difference residual is

`delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)`.

Generalized advantage estimation uses

`A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`.

For probability ratio `rho_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)`, PPO minimizes
the negative clipped surrogate

`min(rho_t A_t, clip(rho_t, 1-epsilon, 1+epsilon) A_t)`

plus value error and minus an entropy bonus. The clip prevents one rollout from
causing an excessively large policy change; GAE reduces the variance of the
policy-gradient estimate while retaining multi-step credit assignment.

## Initial configuration

- gamma: 0.95
- GAE lambda: 0.95
- clip epsilon: 0.20
- Adam learning rate: 3e-4
- rollout: 1024 transitions
- minibatch: 256
- update epochs: 4
- entropy coefficient: 0.01
- value coefficient: 0.5
- gradient norm limit: 0.5

These are an initial baseline, not claimed optimal hyperparameters. Ablate
learning rate and entropy only after the baseline learning curve is recorded.

## Curriculum

Use the same task sequence as the other agents without using test seeds:

```bash
python main.py play --agents ppo_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000
python main.py play --agents ppo_agent --train 1 --scenario loot-crate --no-gui --n-rounds 1000
python main.py play --agents ppo_agent peaceful_agent --train 1 --scenario classic --no-gui --n-rounds 2000
python main.py play --agents ppo_agent coin_collector_agent --train 1 --scenario classic --no-gui --n-rounds 2000
python main.py play --agents ppo_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 4000
```

Checkpoint evaluation must use the unchanged shared protocol. Compare learning
curves at equal environment interactions in addition to final tuned models.

## Verification completed

- Feature vector has the intended 35 values.
- Policy and value output dimensions are correct.
- Masked sampling never chooses a forbidden action.
- GAE stops bootstrapping at terminal transitions.
- The agent imports and acts after only its own directory is copied.
- A 25-round training smoke test completed multiple PPO updates and saved a
  checkpoint with 10,025 transitions.
- The checkpoint then completed 1,000 coin-heaven and 1,000 loot-crate rounds,
  for 2,025 total training rounds and 736,732 observed transitions.
- On 300 synthetic four-player states, mean action latency was 0.00057 seconds
  and maximum latency was 0.00172 seconds, below the 0.5-second limit.

This is an exploratory curriculum checkpoint rather than a final candidate:
the direct framework runs did not yet distribute training over the shared
training-seed list. Validation still uses the unchanged shared seeds and
benchmark harness. See `PPO_RESULTS.md` for the measured results.
