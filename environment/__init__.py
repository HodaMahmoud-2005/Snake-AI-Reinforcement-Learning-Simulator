"""Snake game and reinforcement-learning environment."""

from .game import (
    SnakeState,
    spawn_food,
    transition,
)

from .snake_env import SnakeEnv


__all__ = [
    "SnakeState",
    "SnakeEnv",
    "spawn_food",
    "transition",
]
