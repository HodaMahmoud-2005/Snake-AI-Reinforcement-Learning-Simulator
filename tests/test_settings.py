"""Tests for the shared project configuration."""

import pytest

from configs.settings import (
    Action,
    Direction,
    GameConfig,
    RLConfig,
    GAME_CONFIG,
    REWARD_CONFIG,
    RL_CONFIG,
)


def test_grid_size():
    assert GAME_CONFIG.grid_width == 5
    assert GAME_CONFIG.grid_height == 5


def test_game_limits():
    assert GAME_CONFIG.initial_snake_length == 2
    assert GAME_CONFIG.apples_to_win == 3
    assert GAME_CONFIG.max_steps == 60


def test_initial_snake():
    assert len(GAME_CONFIG.initial_snake) == 2
    assert len(set(GAME_CONFIG.initial_snake)) == 2
    assert GAME_CONFIG.initial_direction == Direction.RIGHT


def test_obstacles():
    assert len(GAME_CONFIG.obstacles) == 2
    assert len(set(GAME_CONFIG.obstacles)) == 2

    for obstacle in GAME_CONFIG.obstacles:
        assert obstacle not in GAME_CONFIG.initial_snake


def test_food_sequence():
    assert len(GAME_CONFIG.food_sequence) == 3
    assert len(set(GAME_CONFIG.food_sequence)) == 3

    for food in GAME_CONFIG.food_sequence:
        assert food not in GAME_CONFIG.obstacles


def test_coordinates_inside_grid():
    positions = (
        GAME_CONFIG.initial_snake
        + GAME_CONFIG.obstacles
        + GAME_CONFIG.food_sequence
    )

    for x, y in positions:
        assert 0 <= x < GAME_CONFIG.grid_width
        assert 0 <= y < GAME_CONFIG.grid_height


def test_relative_actions():
    assert Action.LEFT == 0
    assert Action.STRAIGHT == 1
    assert Action.RIGHT == 2
    assert len(Action) == 3


def test_rewards():
    assert REWARD_CONFIG.food == 10
    assert REWARD_CONFIG.win_bonus == 100
    assert REWARD_CONFIG.move == -1
    assert REWARD_CONFIG.collision == -20
    assert REWARD_CONFIG.final_apple == 110


def test_q_learning_parameters():
    assert RL_CONFIG.gamma == pytest.approx(0.95)
    assert RL_CONFIG.q_learning_alpha == pytest.approx(0.1)
    assert RL_CONFIG.epsilon_start == pytest.approx(1.0)
    assert RL_CONFIG.epsilon_min == pytest.approx(0.05)
    assert RL_CONFIG.epsilon_decay == pytest.approx(0.9995)


def test_reinforce_parameters():
    assert RL_CONFIG.reinforce_alpha == pytest.approx(0.01)


def test_training_parameters():
    assert RL_CONFIG.training_episodes == 10_000
    assert RL_CONFIG.value_iteration_tolerance == pytest.approx(1e-5)


def test_configurable_seed():
    config = RLConfig(random_seed=123)
    assert config.random_seed == 123
    assert RL_CONFIG.random_seed == 42


def test_configurable_game_settings():
    config = GameConfig(max_steps=100)
    assert config.max_steps == 100
    assert GAME_CONFIG.max_steps == 60


def test_start_states():
    assert len(GAME_CONFIG.start_states) >= 1

    for body, heading in GAME_CONFIG.start_states:
        assert len(body) == GAME_CONFIG.initial_snake_length
        assert len(set(body)) == len(body)
        assert isinstance(heading, Direction)

        for x, y in body:
            assert 0 <= x < GAME_CONFIG.grid_width
            assert 0 <= y < GAME_CONFIG.grid_height
            assert (x, y) not in GAME_CONFIG.obstacles

        head, neck = body[:2]
        vectors = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
        }

        assert (
            head[0] - neck[0],
            head[1] - neck[1],
        ) == vectors[heading]