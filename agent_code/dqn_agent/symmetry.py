"""Dihedral (8-fold) symmetry augmentation for training transitions.

The board has 4-fold rotational + mirror symmetry that nothing in the
feature vector or training pipeline currently exploits: a bomb-dodge
learned facing one direction doesn't transfer to the same threat rotated
90 degrees, since the network sees them as unrelated states. Augmenting
each real transition with its 8 symmetric variants is close to free extra
training signal per real transition (Ng et al.-style data augmentation is
standard practice for grid-world RL) and should improve sample efficiency
without changing what the agent is trying to learn.

The scalar features (distances, boolean flags, counts) are already
rotation/reflection-invariant -- e.g. "distance to nearest coin" doesn't
care which way the board is facing. Only the five UP/RIGHT/DOWN/LEFT-
ordered direction groups in the feature vector, and the action itself
when it's a move, need permuting.
"""

import numpy as np

from .model import ACTIONS, STATE_SIZE

# Direction-ordered (UP, RIGHT, DOWN, LEFT) index groups in the 34-dim state
# vector -- see shared/features.py's and shared/opponents.py's layout
# comments. Everything else in the vector is a scalar, invariant under
# rotation/reflection.
DIRECTION_GROUPS = [
    (0, 1, 2, 3),      # move safety mask
    (6, 7, 8, 9),      # physical obstruction
    (12, 13, 14, 15),  # coin direction one-hot
    (16, 17, 18, 19),  # crate/bombable-spot direction one-hot
    (24, 25, 26, 27),  # opponent direction one-hot
]

_DIRECTION_VECTORS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # UP, RIGHT, DOWN, LEFT

# The 8 elements of the dihedral group D4 acting on (x, y).
_TRANSFORMS = {
    "identity": lambda x, y: (x, y),
    "rot90": lambda x, y: (-y, x),
    "rot180": lambda x, y: (-x, -y),
    "rot270": lambda x, y: (y, -x),
    "flip_x": lambda x, y: (-x, y),
    "flip_y": lambda x, y: (x, -y),
    "flip_diag": lambda x, y: (y, x),
    "flip_anti": lambda x, y: (-y, -x),
}


def _permutation_for(transform):
    """perm[d] = index of the direction that direction d's tile moves to
    under this transform, so new_vector[perm[d]] = old_vector[d].
    """
    perm = [None] * 4
    for d, (dx, dy) in enumerate(_DIRECTION_VECTORS):
        tx, ty = transform(dx, dy)
        perm[d] = _DIRECTION_VECTORS.index((tx, ty))
    return perm


_PERMUTATIONS = {name: _permutation_for(fn) for name, fn in _TRANSFORMS.items()}

# Precompute the full 34-index remap and the action remap for each symmetry,
# so augmenting a transition at training time is just two array indexings.
_STATE_SIZE = STATE_SIZE  # tracks model.py's real vector length; any new appended features stay untouched (invariant)
_INDEX_MAPS = {}
_ACTION_MAPS = {}
for _name, _perm in _PERMUTATIONS.items():
    _index_map = list(range(_STATE_SIZE))
    for _group in DIRECTION_GROUPS:
        for _d in range(4):
            _index_map[_group[_perm[_d]]] = _group[_d]
    _INDEX_MAPS[_name] = np.asarray(_index_map, dtype=np.int64)

    _action_map = list(ACTIONS)
    for _d in range(4):
        _action_map[_d] = ACTIONS[_perm[_d]]
    _ACTION_MAPS[_name] = tuple(_action_map)

SYMMETRY_NAMES = tuple(_TRANSFORMS)


def augment_transition(state, action, next_state):
    """All 8 symmetric variants of one (state, action, next_state), first
    of which is the untransformed original. Reward/done/n-step-count are
    unaffected by these transforms so callers reuse them unchanged.
    """
    variants = []
    for name in SYMMETRY_NAMES:
        index_map = _INDEX_MAPS[name]
        variants.append((state[index_map], _ACTION_MAPS[name][ACTIONS.index(action)], next_state[index_map]))
    return variants
