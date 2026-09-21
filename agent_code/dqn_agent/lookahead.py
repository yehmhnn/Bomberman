"""Shallow, inference-only value-lookahead on top of the trained Q-network.

The tournament gives 0.5s per decision; a single forward pass costs ~2ms, so
almost the entire budget goes unused by plain greedy action selection. This
module spends a bit of it: for each shield-allowed action, deterministically
simulate our own side of the resulting one-step-ahead game state (movement,
bomb placement, bomb countdowns, coin pickup), holding every opponent frozen
in place -- we cannot predict their move, and the shield already reacts to
their *current* turn's threat separately (see model.py's
_augment_with_opponent_threats). Re-evaluating the network on that
deterministically-computed next state grounds the value estimate in exactly
computed danger/distance features rather than relying on the network to have
implicitly learned to extrapolate movement dynamics purely from its training
distribution -- the kind of thing that generalizes worse to an unfamiliar
opponent's board state than hand-coded, rule-based features do. Similar in
spirit to model-based value expansion (Feinberg et al., 2018): a short,
exact rollout combined with a learned value function, rather than trusting
either alone.

Deliberately inference-only (never used during training): keeps this
completely isolated from exploration, the replay buffer, and PER, so it
carries zero risk of destabilizing training and needs no retraining to
evaluate. If a step needs to be reverted, deleting this file and the single
call site in callbacks.py fully undoes it.
"""

import numpy as np
import torch

from settings import BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER

from .shared.danger import blast_coords
from .shared.safety import MOVE, _occupied, tile_free

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def simulate_own_action(game_state, action, bomb_power=BOMB_POWER):
    """Best-effort, approximate game_state one real step after WE take
    `action`. Every opponent is held frozen (no movement, no new bombs from
    them) -- a simplifying assumption, not a claim about what they will
    actually do. Good enough for a shallow value-lookahead refinement; the
    shield's own safety checks do not depend on this function at all.
    """
    name, score, bombs_left, pos = game_state["self"]
    field = game_state["field"]

    new_pos = pos
    new_bombs_left = bombs_left
    bombs = list(game_state["bombs"])

    if action in MOVE:
        dx, dy = MOVE[action]
        target = (pos[0] + dx, pos[1] + dy)
        if tile_free(game_state, target, _occupied(game_state)):
            new_pos = target
    elif action == "BOMB" and bombs_left:
        bombs = bombs + [(pos, BOMB_TIMER)]
        new_bombs_left = False
    # WAIT: no change beyond the time-advance below, common to every action.

    explosion_map = game_state.get("explosion_map")
    new_explosion_map = (
        np.maximum(explosion_map.astype(int) - 1, 0) if explosion_map is not None else None
    )

    ticked_bombs = []
    for bpos, countdown in bombs:
        if countdown <= 0:
            if new_explosion_map is not None:
                for coord in blast_coords(field, bpos, bomb_power):
                    new_explosion_map[coord] = max(new_explosion_map[coord], EXPLOSION_TIMER - 1)
        else:
            ticked_bombs.append((bpos, countdown - 1))

    new_coins = [c for c in game_state["coins"] if c != new_pos]

    return dict(
        game_state,
        self=(name, score, new_bombs_left, new_pos),
        bombs=ticked_bombs,
        coins=new_coins,
        explosion_map=new_explosion_map if new_explosion_map is not None else explosion_map,
    )


def lookahead_scores(model, game_state, recent_positions, base_q, mask, beta=0.5, bomb_power=BOMB_POWER):
    """base_q blended with a one-step value-expansion term for each masked-
    allowed action; disallowed actions keep their base_q unchanged (they are
    never selectable, see select_action). beta=0 recovers plain greedy
    selection exactly; beta=1 uses only the lookahead term.
    """
    from .model import action_mask_vector, state_to_vector  # local import: avoids a model<->lookahead cycle

    scores = np.array(base_q, dtype=np.float64, copy=True)
    for i, action in enumerate(ACTIONS):
        if not mask[i]:
            continue
        next_state = simulate_own_action(game_state, action, bomb_power)
        next_vec = state_to_vector(next_state, recent_positions)
        next_mask = action_mask_vector(next_state)
        with torch.no_grad():
            next_q = model.q_values(torch.from_numpy(next_vec).unsqueeze(0)).squeeze(0).numpy()
        allowed_next_q = next_q[next_mask] if next_mask.any() else next_q
        scores[i] = (1 - beta) * base_q[i] + beta * float(np.max(allowed_next_q))
    return scores
