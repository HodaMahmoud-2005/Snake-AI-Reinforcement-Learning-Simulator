"""Tests for the live Snake environment."""

from dataclasses import replace

import pytest

from configs.settings import (
    Action,
    Direction,
    GAME_CONFIG,
)

from environment import SnakeEnv, SnakeState


@pytest.fixture
def env():
    instance = SnakeEnv(seed=42)
    instance.reset(start_index=0)
    return instance


def test_reset(env):
    state, info = env.reset(start_index=0)

    assert state.body == GAME_CONFIG.initial_snake
    assert state.direction == Direction.RIGHT
    assert state.food == GAME_CONFIG.food_sequence[0]
    assert state.length == 2

    assert env.score == 0
    assert env.steps == 0
    assert not env.done

    assert info["total_reward"] == 0


def test_get_state(env):
    state = env.get_state()

    assert isinstance(state, SnakeState)
    assert hash(state) == hash(env.get_state())


def test_normal_movement(env):
    state, reward, terminated, truncated, info = (
        env.step(Action.STRAIGHT)
    )

    assert state.body == ((2, 2), (1, 2))
    assert state.length == 2
    assert reward == -1
    assert not terminated
    assert not truncated
    assert info["steps"] == 1


def test_food_collection_and_growth(env):
    actions = [1, 0, 1, 2]

    for action in actions:
        state, reward, terminated, truncated, info = (
            env.step(action)
        )

    assert state.head == (3, 0)
    assert state.length == 3
    assert info["score"] == 1
    assert reward == 10
    assert state.food == (4, 2)
    assert not terminated
    assert not truncated


def test_obstacle_collision(env):
    state, reward, terminated, truncated, info = (
        env.step(Action.LEFT)
    )

    assert reward == -20
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "obstacle_collision"


def test_wall_collision(env):
    for action in [1, 1, 1, 1]:
        state, reward, terminated, truncated, info = (
            env.step(action)
        )

    assert reward == -20
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "wall_collision"


def test_step_limit(env):
    # Set the external counter to the last available step.
    env.steps = 59

    state, reward, terminated, truncated, info = (
        env.step(Action.STRAIGHT)
    )

    assert info["steps"] == 60
    assert reward == -1
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "step_limit"


def test_food_on_final_step(env):
    for action in [1, 0, 1]:
        env.step(action)

    env.steps = 59

    state, reward, terminated, truncated, info = (
        env.step(Action.RIGHT)
    )

    assert reward == 10
    assert info["score"] == 1
    assert truncated
    assert not terminated


def test_collision_on_final_step(env):
    env.steps = 59

    state, reward, terminated, truncated, info = (
        env.step(Action.LEFT)
    )

    assert reward == -20
    assert info["steps"] == 60
    assert terminated
    assert not truncated


@pytest.mark.parametrize(
    "action",
    [-1, 3, 100, True, 1.5],
)
def test_invalid_actions(env, action):
    with pytest.raises(ValueError):
        env.step(action)

    assert env.steps == 0
    assert env.total_reward == 0


def test_step_after_termination(env):
    env.step(Action.LEFT)

    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)


def test_step_after_truncation(env):
    env.steps = 59
    env.step(Action.STRAIGHT)

    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)


def test_reset_after_termination(env):
    env.step(Action.LEFT)

    state, info = env.reset(start_index=0)

    assert not env.done
    assert env.steps == 0
    assert env.score == 0
    assert info["total_reward"] == 0


def test_seed_reproducibility():
    env1 = SnakeEnv(seed=42)
    env2 = SnakeEnv(seed=42)

    assert env1.reset() == env2.reset()


def test_evaluation_info(env):
    _, reward, _, _, info = env.step(Action.STRAIGHT)

    assert info["score"] == 0
    assert info["steps"] == 1
    assert info["snake_length"] == 2
    assert info["total_reward"] == reward
    assert info["termination_reason"] is None
    assert info["success"] is False


WINNING_ACTIONS = [
    1, 0, 1, 2, 1, 2, 1,
    2, 1, 1, 1, 0, 1,
]


def test_full_winning_episode(env):
    for action in WINNING_ACTIONS:
        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    assert info["score"] == 3
    assert state.length == 5
    assert terminated
    assert not truncated
    assert reward == 110
    assert info["success"]
    assert info["termination_reason"] == "win"
    assert info["steps"] == 13
    assert info["total_reward"] == 120


def test_win_on_final_step(env):
    for action in WINNING_ACTIONS[:-1]:
        env.step(action)

    env.steps = 59

    state, reward, terminated, truncated, info = (
        env.step(WINNING_ACTIONS[-1])
    )

    assert reward == 110
    assert info["steps"] == 60
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "win"


def test_configurable_step_limit():
    config = replace(
        GAME_CONFIG,
        max_steps=1,
    )

    env = SnakeEnv(config=config)
    env.reset(start_index=0)

    _, reward, terminated, truncated, info = (
        env.step(Action.STRAIGHT)
    )

    assert reward == -1
    assert not terminated
    assert truncated
    assert info["steps"] == 1


def test_full_60_step_episode(env):
    # Follow the outer boundary without eating food.
    # Use a custom food sequence inside the boundary.
    config = replace(
        GAME_CONFIG,
        food_sequence=((2, 2), (4, 2), (0, 4)),
        start_states=(
            (((1, 0), (0, 0)), Direction.RIGHT),
        ),
    )

    env = SnakeEnv(config=config)
    env.reset(start_index=0)

    for _ in range(60):
        state = env.get_state()
        x, y = state.head

        at_corner = (x, y) in (
            (4, 0),
            (4, 4),
            (0, 4),
            (0, 0),
        )

        action = (
            Action.RIGHT if at_corner
            else Action.STRAIGHT
        )

        _, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert reward == -1
        assert not terminated

    assert truncated
    assert info["steps"] == 60
    assert info["total_reward"] == -60
    assert info["termination_reason"] == "step_limit"
