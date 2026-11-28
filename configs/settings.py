"""Shared configuration for the Snake AI simulator."""

from dataclasses import dataclass
from enum import IntEnum


# ==================================================
# Action space
# ==================================================

class Action(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# ==================================================
# Game configuration
# ==================================================

@dataclass(frozen=True)
class GameConfig:
    grid_width: int = 5
    grid_height: int = 5

    initial_snake_length: int = 2
    apples_to_win: int = 3
    max_steps: int = 60

    obstacles: tuple = (
        (1, 1),
        (3, 3),
    )

    # Body positions ordered from head to tail.
    initial_snake: tuple = (
        (1, 2),
        (0, 2),
    )

    initial_direction: Direction = Direction.RIGHT

    food_sequence: tuple = (
        (3, 0),
        (4, 2),
        (0, 4),
    )

    # Additional start states can be added
    # after team approval.
    start_states: tuple = (
        (((1, 2), (0, 2)), Direction.RIGHT),
    )


# ==================================================
# Reward configuration
# ==================================================

@dataclass(frozen=True)
class RewardConfig:
    food: float = 10.0
    win_bonus: float = 100.0
    move: float = -1.0
    collision: float = -20.0

    @property
    def final_apple(self) -> float:
        return self.food + self.win_bonus


# ==================================================
# Reinforcement-learning configuration
# ==================================================

@dataclass(frozen=True)
class RLConfig:
    gamma: float = 0.95

    # Q-learning
    q_learning_alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995

    # REINFORCE
    reinforce_alpha: float = 0.01

    # Training
    training_episodes: int = 10_000

    # Value Iteration
    value_iteration_tolerance: float = 1e-5

    # Reproducibility
    random_seed: int = 42


# ==================================================
# Default configurations
# ==================================================

GAME_CONFIG = GameConfig()
REWARD_CONFIG = RewardConfig()
RL_CONFIG = RLConfig()