"""Tests for the deterministic planning transition interface."""

from dataclasses import (
    fields,
    FrozenInstanceError,
    replace,
)

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

from environment.game import (
    turn_direction,
    resolve_transition,
)


@pytest.fixture
def env():
    instance = SnakeEnv(seed=42)
    instance.reset(start_index=0)
    return instance


# ==================================================
# State representation
# ==================================================

def test_state_contains_only_body_and_food():
    names = {
        field.name
        for field in fields(SnakeState)
    }

    assert names == {"body", "food"}


def test_state_is_immutable(env):
    state = env.get_state()

    with pytest.raises(FrozenInstanceError):
        state.food = (0, 0)


def test_state_is_hashable(env):
    state = env.get_state()

    equivalent = SnakeState(
        body=state.body,
        food=state.food,
    )

    assert state == equivalent
    assert hash(state) == hash(equivalent)

    values = {state: 10.0}

    assert values[equivalent] == 10.0


@pytest.mark.parametrize(
    "body,expected",
    [
        (((2, 2), (2, 3)), Direction.UP),
        (((2, 2), (1, 2)), Direction.RIGHT),
        (((2, 2), (2, 1)), Direction.DOWN),
        (((2, 2), (3, 2)), Direction.LEFT),
    ],
)
def test_derived_heading(body, expected):
    state = SnakeState(
        body=body,
        food=(4, 4),
    )

    assert state.direction == expected


@pytest.mark.parametrize("length", [2, 3, 4, 5])
def test_derived_score(length):
    state = SnakeState(
        body=tuple(
            (x, 0)
            for x in reversed(range(length))
        ),
        food=(4, 4),
    )

    assert state.score == length - 2


# ==================================================
# Relative actions
# ==================================================

@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("action", list(Action))
def test_relative_actions(direction, action):
    expected = Direction(
        (int(direction) + int(action) - 1) % 4
    )

    assert turn_direction(
        direction,
        action,
    ) == expected


# ==================================================
# Determinism and immutability
# ==================================================

@pytest.mark.parametrize("action", list(Action))
def test_identical_inputs_identical_outputs(env, action):
    state = env.get_state()

    expected = transition(state, action)

    for _ in range(100):
        assert transition(state, action) == expected


def test_original_state_not_modified(env):
    state = env.get_state()

    original_body = state.body
    original_food = state.food

    next_state, _, _ = transition(
        state,
        Action.STRAIGHT,
    )

    assert state.body == original_body
    assert state.food == original_food
    assert next_state != state


def test_transition_does_not_mutate_live_game(env):
    state = env.get_state()

    original_info = env._get_info().copy()
    original_rng = env._rng.getstate()

    env.transition(state, Action.STRAIGHT)
    env.transition(state, Action.LEFT)

    assert env.get_state() == state
    assert env._get_info() == original_info
    assert env._rng.getstate() == original_rng


@pytest.mark.parametrize("action", list(Action))
def test_step_and_transition_agree(env, action):
    state = env.get_state()

    expected_state, expected_reward, expected_terminal = (
        env.transition(state, action)
    )

    next_state, reward, terminated, truncated, _ = (
        env.step(action)
    )

    assert next_state == expected_state
    assert reward == expected_reward
    assert terminated == expected_terminal
    assert not truncated


def test_transition_independent_of_elapsed_steps(env):
    state = env.get_state()

    env.steps = 0
    first = env.transition(state, Action.STRAIGHT)

    env.steps = 59
    second = env.transition(state, Action.STRAIGHT)

    assert first == second
    assert second[2] is False

    _, reward, terminated, truncated, _ = (
        env.step(Action.STRAIGHT)
    )

    assert reward == -1
    assert not terminated
    assert truncated


# ==================================================
# Movement and rewards
# ==================================================

def test_normal_movement(env):
    state = env.get_state()

    next_state, reward, terminated = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.body == ((2, 2), (1, 2))
    assert next_state.length == 2
    assert reward == -1
    assert not terminated


def test_growth():
    state = SnakeState(
        body=((2, 0), (1, 0)),
        food=(3, 0),
    )

    next_state, reward, terminated = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.body == (
        (3, 0), (2, 0), (1, 0)
    )

    assert next_state.score == 1
    assert next_state.food == (4, 2)
    assert reward == 10
    assert not terminated


def test_final_apple():
    state = SnakeState(
        body=((1, 4), (2, 4), (3, 4), (4, 4)),
        food=(0, 4),
    )

    next_state, reward, terminated = transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.length == 5
    assert next_state.score == 3
    assert next_state.food is None
    assert reward == 110
    assert terminated


def test_winning_state_is_absorbing():
    state = SnakeState(
        body=(
            (0, 4), (1, 4), (2, 4),
            (3, 4), (4, 4),
        ),
        food=None,
    )

    for action in Action:
        assert transition(state, action) == (
            state,
            0.0,
            True,
        )


# ==================================================
# Collision rules
# ==================================================

def test_wall_collision():
    state = SnakeState(
        body=((4, 0), (3, 0)),
        food=(0, 4),
    )

    result = resolve_transition(
        state,
        Action.STRAIGHT,
    )

    assert result.next_state == state
    assert result.reward == -20
    assert result.terminated
    assert result.termination_reason == "wall_collision"


def test_obstacle_collision(env):
    state = env.get_state()

    result = resolve_transition(
        state,
        Action.LEFT,
    )

    assert result.reward == -20
    assert result.terminated
    assert result.termination_reason == "obstacle_collision"


def test_self_collision():
    # A longer-snake configuration is needed to test
    # non-tail body collision before winning.
    config = replace(
        GAME_CONFIG,
        apples_to_win=4,
    )

    state = SnakeState(
        body=(
            (2, 2),
            (2, 3),
            (1, 3),
            (1, 2),
            (0, 2),
        ),
        food=(4, 4),
    )

    result = resolve_transition(
        state,
        Action.LEFT,
        config=config,
    )

    assert result.reward == -20
    assert result.terminated
    assert result.termination_reason == "self_collision"


def test_move_into_vacating_tail():
    state = SnakeState(
        body=(
            (2, 2),
            (2, 3),
            (2, 4),
            (1, 4),
        ),
        food=(4, 4),
    )

    # A four-segment loop with adjacent head and tail.
    state = SnakeState(
        body=(
            (2, 2),
            (2, 3),
            (1, 3),
            (1, 2),
        ),
        food=(4, 4),
    )

    next_state, reward, terminated = transition(
        state,
        Action.LEFT,
    )

    assert next_state.head == (1, 2)
    assert next_state.length == 4
    assert reward == -1
    assert not terminated


# ==================================================
# Deterministic food spawning
# ==================================================

def test_fixed_food_sequence():
    body = ((1, 2), (0, 2))

    assert spawn_food(body, 0) == (3, 0)
    assert spawn_food(body, 1) == (4, 2)
    assert spawn_food(body, 2) == (0, 4)
    assert spawn_food(body, 3) is None


def test_food_fallback_when_occupied():
    body = ((3, 0), (2, 0))

    assert spawn_food(body, 0) == (0, 0)


def test_food_fallback_skips_body():
    body = (
        (3, 0), (2, 0), (1, 0), (0, 0)
    )

    assert spawn_food(body, 0) == (4, 0)


def test_food_fallback_skips_obstacles():
    config = replace(
        GAME_CONFIG,
        obstacles=((0, 0), (3, 0)),
    )

    body = ((1, 2), (0, 2))

    assert spawn_food(body, 0, config) == (1, 0)


def test_food_fallback_is_deterministic():
    body = ((3, 0), (2, 0))

    results = {
        spawn_food(body, 0)
        for _ in range(100)
    }

    assert results == {(0, 0)}


def test_fallback_during_growth():
    config = replace(
        GAME_CONFIG,
        food_sequence=(
            (3, 0),
            (2, 0),
            (0, 4),
        ),
    )

    state = SnakeState(
        body=((2, 0), (1, 0)),
        food=(3, 0),
    )

    next_state, reward, terminated = transition(
        state,
        Action.STRAIGHT,
        config=config,
    )

    assert next_state.food == (0, 0)
    assert reward == 10
    assert not terminated


# ==================================================
# Planning without a live episode
# ==================================================

def test_planning_without_reset():
    env = SnakeEnv()

    state = SnakeState(
        body=((1, 2), (0, 2)),
        food=(3, 0),
    )

    next_state, reward, terminated = env.transition(
        state,
        Action.STRAIGHT,
    )

    assert next_state.head == (2, 2)
    assert reward == -1
    assert not terminated
    assert env.steps == 0

    with pytest.raises(RuntimeError):
        env.get_state()


def test_uncapped_planning():
    config = replace(
        GAME_CONFIG,
        food_sequence=((2, 2), (4, 2), (0, 4)),
    )

    state = SnakeState(
        body=((1, 0), (0, 0)),
        food=(2, 2),
    )

    total_reward = 0

    # Simulate 100 legal moves without truncation.
    for _ in range(100):
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

        state, reward, terminated = transition(
            state,
            action,
            config=config,
        )

        assert not terminated
        total_reward += reward

    assert total_reward == -100
    assert state.score == 0
