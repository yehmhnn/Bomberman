"""Combine the shared coin/crate/safety vector with opponent-awareness."""
from .shared.features import FEATURE_SIZE as _CORE_SIZE
from .shared.features import build_feature_vector
from .shared.opponents import FEATURE_SIZE as _OPPONENT_SIZE
from .shared.opponents import opponent_features

N_FEATURES = _CORE_SIZE + _OPPONENT_SIZE


def build_state_vector(game_state, recent_positions=None):
    return build_feature_vector(game_state, recent_positions) + opponent_features(game_state)
