# dqn_agent_frozen_ref

A permanently pinned, self-contained snapshot of `dqn_agent`, for use as the
"frozen copy of our other agent" opponent in the team's shared common
evaluation line-up #4 (see `EXPERIMENT_PROTOCOL.md`).

Why this exists: `dqn_agent`'s live checkpoint and architecture keep changing
as its hyperparameters and network are tuned. Using the live `dqn_agent`
directly as a "frozen" opponent meant every candidate evaluated at a
different time faced a different, moving-target opponent -- not a fair
comparison across candidates. This directory is a frozen point-in-time copy
(code + weights together, since the two must match) that will not change
unless explicitly updated and announced to the team.

Weights pinned from: `agent_code/dqn_agent` at commit `c5c7241` (run F,
N_QUANTILES=64 -- the current best DQN checkpoint as of 2026-09-19).

Code updated 2026-09-21 to include a shield fix (escape-route planning now
accounts for opponents' possible movement, not just their current tile) --
same weights, unchanged since c5c7241, just corrected code. Verified this
is a pure improvement with no retraining needed, on the full 100 val seeds
vs the standard mixed lineup: score 9.16->9.35, win_rate 0.67 (unchanged),
kills 1.05->1.12, suicide 0.21->0.14, survival 0.78->0.82.

Deliberately self-contained (own copy of `shared/`, no dependency on
`agent_code/dqn_agent`) so it keeps working even if `dqn_agent`'s code
changes later. Inference-only: no `train.py`, never meant to occupy a
`--train` slot.

If this needs to be updated to a newer checkpoint, do it as its own
commit and tell the team -- past evaluation numbers that used the old
pinned reference stay valid for what they measured; they just won't be
comparable to results run after an update unless everyone reruns lineup #4.
