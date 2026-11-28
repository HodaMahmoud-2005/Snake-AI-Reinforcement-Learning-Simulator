"""Snake environment and deterministic transition model."""

from .game import (
    SnakeState,
    TransitionResult,
    spawn_food,
    transition,
    resolve_transition,
)

from .snake_env import SnakeEnv


__all__ = [
    "SnakeState",
    "TransitionResult",
    "SnakeEnv",
    "spawn_food",
    "transition",
    "resolve_transition",
]
