"""One-step linear Q-learning updates."""
from pathlib import Path

from .shared.safety import safe_action_mask
from .q_learning import ACTIONS
from .rewards import reward_from_transition
from .state import build_state_vector

MODEL_FILE = Path(__file__).with_name("model.npz")
ALPHA = 0.01
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998
SAVE_EVERY_ROUNDS = 10


def setup_training(self):
    self.training_round = 0
    self.epsilon = EPSILON_START


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_transition(old_game_state, new_game_state, events)
    next_mask = safe_action_mask(new_game_state)
    next_allowed = tuple(a for a in ACTIONS if next_mask[a])
    next_features = build_state_vector(new_game_state, self.recent_positions)
    self.q.update(self.last_features, self_action, reward, next_features, next_allowed,
                  ALPHA, GAMMA, done=False)


def end_of_round(self, last_game_state, last_action, events):
    reward = reward_from_transition(last_game_state, None, events)
    self.q.update(self.last_features, last_action, reward, None, None, ALPHA, GAMMA, done=True)

    self.training_round += 1
    self.epsilon = max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** self.training_round)
    if self.training_round % SAVE_EVERY_ROUNDS == 0:
        self.q.save(MODEL_FILE)