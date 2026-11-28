"""Automated tests for the core Snake environment."""

from dataclasses import replace

import pytest

from configs.settings import (
    Action,
    Direction,
    GAME_CONFIG,
)

from environment import (
    SnakeEnv,
    SnakeState,
    spawn_food,
    transition,
)

from environment.game import turn_direction


@pytest.fixture
def env():
    environment = SnakeEnv(seed=42)
    environment.reset(start_index=0)
    return environment


def test_reset(env):
    state, info = env.reset(start_index=0)

    assert state.body == GAME_CONFIG.initial_snake
    assert state.direction == Direction.RIGHT
    assert state.food == GAME_CONFIG.food_sequence[0]
    assert state.length == 2
    assert state.score == 0
    assert state.steps == 0
    assert not state.done
    assert info["total_reward"] == 0


def test_get_state(env):
    state = env.get_state()

    assert isinstance(state, SnakeState)
    assert hash(state) == hash(env.get_state())


@pytest.mark.parametrize(
    "direction,action,expected",
    [
        (Direction.RIGHT, Action.LEFT, Direction.UP),
        (Direction.RIGHT, Action.STRAIGHT, Direction.RIGHT),
        (Direction.RIGHT, Action.RIGHT, Direction.DOWN),
        (Direction.UP, Action.LEFT, Direction.LEFT),
        (Direction.LEFT, Action.RIGHT, Direction.UP),
        (Direction.DOWN, Action.LEFT, Direction.RIGHT),
    ],
)
def test_relative_actions(direction, action, expected):
    assert turn_direction(direction, action) == expected


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


def test_food_collection_and_growth():
    state = SnakeState(
        body=((2, 0), (1, 0)),
        direction=Direction.RIGHT,
        food=(3, 0),
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.length == 3
    assert next_state.score == 1
    assert reward == 10
    assert next_state.food == (4, 2)
    assert not next_state.done


def test_final_apple_win():
    state = SnakeState(
        body=((1, 4), (2, 4), (3, 4), (4, 4)),
        direction=Direction.LEFT,
        food=(0, 4),
        score=2,
        steps=10,
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.score == 3
    assert next_state.length == 5
    assert next_state.food is None
    assert reward == 110
    assert next_state.terminated
    assert not next_state.truncated
    assert next_state.termination_reason == "win"


def test_wall_collision():
    state = SnakeState(
        body=((4, 0), (3, 0)),
        direction=Direction.RIGHT,
        food=(0, 4),
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert reward == -20
    assert next_state.terminated
    assert not next_state.truncated
    assert next_state.termination_reason == "wall_collision"


def test_obstacle_collision(env):
    state, reward, terminated, truncated, info = (
        env.step(Action.LEFT)
    )

    assert reward == -20
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "obstacle_collision"


def test_self_collision():
    state = SnakeState(
        body=(
            (2, 2),
            (2, 3),
            (1, 3),
            (1, 2),
            (1, 1),
        ),
        direction=Direction.UP,
        food=(4, 4),
    )

    next_state, reward = transition(
        state,
        Action.LEFT,
    )

    assert reward == -20
    assert next_state.terminated
    assert next_state.termination_reason == "self_collision"


def test_move_into_vacating_tail():
    state = SnakeState(
        body=(
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
        ),
        direction=Direction.UP,
        food=(4, 4),
        score=2,
    )

    # Use a layout without an obstacle inside the test body.
    config = replace(
        GAME_CONFIG,
        obstacles=((1, 1), (0, 0)),
    )

    next_state, reward = transition(
        state,
        Action.RIGHT,
        config=config,
    )

    assert next_state.head == (3, 2)
    assert next_state.length == 4
    assert reward == -1
    assert not next_state.done


def test_fixed_food_sequence():
    body = ((1, 2), (0, 2))

    assert spawn_food(body, 0) == (3, 0)
    assert spawn_food(body, 1) == (4, 2)
    assert spawn_food(body, 2) == (0, 4)
    assert spawn_food(body, 3) is None


def test_food_fallback_when_occupied():
    body = ((3, 0), (2, 0))

    assert spawn_food(body, 0) == (0, 0)


def test_food_fallback_skips_occupied_cells():
    body = ((3, 0), (2, 0), (1, 0), (0, 0))

    assert spawn_food(body, 0) == (4, 0)


def test_food_fallback_skips_obstacles():
    config = replace(
        GAME_CONFIG,
        obstacles=((0, 0), (3, 0)),
    )

    body = ((1, 2), (0, 2))

    assert spawn_food(body, 0, config) == (1, 0)


def test_step_limit():
    state = SnakeState(
        body=((1, 2), (0, 2)),
        direction=Direction.RIGHT,
        food=(3, 0),
        steps=59,
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.steps == 60
    assert reward == -1
    assert not next_state.terminated
    assert next_state.truncated
    assert next_state.termination_reason == "step_limit"


def test_food_on_final_step():
    state = SnakeState(
        body=((2, 0), (1, 0)),
        direction=Direction.RIGHT,
        food=(3, 0),
        steps=59,
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert reward == 10
    assert next_state.score == 1
    assert next_state.truncated
    assert not next_state.terminated


def test_win_on_final_step():
    state = SnakeState(
        body=((1, 4), (2, 4), (3, 4), (4, 4)),
        direction=Direction.LEFT,
        food=(0, 4),
        score=2,
        steps=59,
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert reward == 110
    assert next_state.steps == 60
    assert next_state.terminated
    assert not next_state.truncated
    assert next_state.termination_reason == "win"


def test_collision_on_final_step():
    state = SnakeState(
        body=((4, 0), (3, 0)),
        direction=Direction.RIGHT,
        food=(0, 4),
        steps=59,
    )

    next_state, reward = transition(
        state,
        Action.STRAIGHT,
    )

    assert reward == -20
    assert next_state.steps == 60
    assert next_state.terminated
    assert not next_state.truncated


@pytest.mark.parametrize("action", [-1, 3, 100])
def test_invalid_actions(env, action):
    with pytest.raises(ValueError):
        env.step(action)


def test_step_after_termination(env):
    env.step(Action.LEFT)

    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)


def test_reset_after_termination(env):
    env.step(Action.LEFT)

    state, info = env.reset(start_index=0)

    assert not state.done
    assert state.steps == 0
    assert state.score == 0
    assert info["total_reward"] == 0


def test_deterministic_transition(env):
    state = env.get_state()

    first = transition(state, Action.STRAIGHT)
    second = transition(state, Action.STRAIGHT)

    assert first == second


def test_original_state_not_modified(env):
    state = env.get_state()

    next_state, _ = transition(
        state,
        Action.STRAIGHT,
    )

    assert state.steps == 0
    assert state.body == ((1, 2), (0, 2))
    assert next_state.steps == 1


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


def test_full_winning_episode(env):
    # A complete winning route for the default layout.
    actions = [
        Action.STRAIGHT,
        Action.LEFT,
        Action.STRAIGHT,
        Action.RIGHT,
        Action.STRAIGHT,
        Action.RIGHT,
        Action.STRAIGHT,
        Action.RIGHT,
        Action.STRAIGHT,
        Action.STRAIGHT,
        Action.STRAIGHT,
        Action.LEFT,
        Action.STRAIGHT,
    ]

    for action in actions:
        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    assert state.score == 3
    assert state.length == 5
    assert terminated
    assert not truncated
    assert reward == 110
    assert info["success"]
    assert info["termination_reason"] == "win"
    assert info["steps"] == 13
    assert info["total_reward"] == 120
