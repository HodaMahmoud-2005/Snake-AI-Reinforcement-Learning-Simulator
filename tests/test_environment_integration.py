"""Integration tests for the deterministic Snake environment."""

from collections import deque
from dataclasses import replace

import pytest

from configs.settings import (
    Action,
    GAME_CONFIG,
    REWARD_CONFIG,
)

from environment import (
    SnakeEnv,
    SnakeState,
    transition,
)


WINNING_ACTIONS = [
    1, 0, 1, 2, 1, 2, 1,
    2, 1, 1, 1, 0, 1,
]


@pytest.fixture
def env():
    instance = SnakeEnv(seed=42)
    instance.reset(start_index=0)
    return instance


def test_complete_episode_reward_sequence(env):
    """Verify every reward in a complete winning episode."""

    rewards = []

    for action in WINNING_ACTIONS:
        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        rewards.append(reward)

        if terminated or truncated:
            break

    assert rewards == [
        -1, -1, -1, 10,
        -1, -1, 10,
        -1, -1, -1, -1, -1, 110,
    ]

    assert sum(rewards) == 120
    assert info["total_reward"] == 120
    assert info["score"] == 3
    assert info["steps"] == 13
    assert state.length == 5
    assert terminated
    assert not truncated


def test_step_transition_consistency_full_episode(env):
    """Compare planning and gameplay throughout a winning episode."""

    for action in WINNING_ACTIONS:
        current_state = env.get_state()

        expected = env.transition(
            current_state,
            action,
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert not truncated

        assert expected == (
            next_state,
            reward,
            terminated,
        )

        if terminated:
            break

    assert info["success"]


def test_transition_preserves_live_episode(env):
    """Planning must not modify gameplay or random state."""

    env.step(Action.STRAIGHT)

    original_state = env.get_state()
    original_steps = env.steps
    original_reward = env.total_reward
    original_reason = env.termination_reason
    original_flags = (env.terminated, env.truncated)
    original_rng = env._rng.getstate()

    for _ in range(100):
        for action in Action:
            env.transition(original_state, action)

    assert env.get_state() == original_state
    assert env.steps == original_steps
    assert env.total_reward == original_reward
    assert env.termination_reason == original_reason
    assert (env.terminated, env.truncated) == original_flags
    assert env._rng.getstate() == original_rng


def test_reachable_state_transition_invariants():
    """Check every reachable nonterminal state-action pair."""

    env = SnakeEnv()

    queue = deque()
    visited = set()

    # Include every configured start state.
    for index in range(len(GAME_CONFIG.start_states)):
        state, _ = env.reset(start_index=index)

        if state not in visited:
            visited.add(state)
            queue.append(state)

    valid_rewards = {
        REWARD_CONFIG.move,
        REWARD_CONFIG.food,
        REWARD_CONFIG.collision,
        REWARD_CONFIG.final_apple,
    }

    obstacles = set(GAME_CONFIG.obstacles)

    while queue:
        state = queue.popleft()

        for action in Action:
            result = env.transition(state, action)

            # Identical inputs must produce identical outputs.
            assert result == env.transition(state, action)

            next_state, reward, terminated = result

            assert reward in valid_rewards

            # The resulting body must remain valid.
            assert len(set(next_state.body)) == next_state.length

            for x, y in next_state.body:
                assert 0 <= x < GAME_CONFIG.grid_width
                assert 0 <= y < GAME_CONFIG.grid_height
                assert (x, y) not in obstacles

            # Consecutive body segments must be adjacent.
            for first, second in zip(
                next_state.body,
                next_state.body[1:],
            ):
                distance = (
                    abs(first[0] - second[0])
                    + abs(first[1] - second[1])
                )

                assert distance == 1

            # Food must occupy a valid free cell.
            if next_state.food is not None:
                x, y = next_state.food

                assert 0 <= x < GAME_CONFIG.grid_width
                assert 0 <= y < GAME_CONFIG.grid_height
                assert next_state.food not in next_state.body
                assert next_state.food not in obstacles

            if reward == REWARD_CONFIG.collision:
                assert terminated
                assert next_state == state

            elif reward == REWARD_CONFIG.move:
                assert not terminated
                assert next_state.length == state.length
                assert next_state.food == state.food

            elif reward == REWARD_CONFIG.food:
                assert not terminated
                assert next_state.length == state.length + 1

            elif reward == REWARD_CONFIG.final_apple:
                assert terminated
                assert next_state.length == state.length + 1
                assert next_state.score == GAME_CONFIG.apples_to_win
                assert next_state.food is None

            # Terminal transitions must not be expanded.
            if not terminated and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    assert len(visited) > 1

    print(f"\nReachable nonterminal states: {len(visited)}")
    print(f"State-action pairs checked: {len(visited) * len(Action)}")


def test_body_collision_with_extended_configuration():
    """Verify self-collision before the winning length."""

    config = replace(
        GAME_CONFIG,
        apples_to_win=4,
        food_sequence=(
            (3, 0),
            (4, 2),
            (0, 4),
            (4, 4),
        ),
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

    next_state, reward, terminated = transition(
        state,
        Action.LEFT,
        config=config,
    )

    assert terminated
    assert reward == REWARD_CONFIG.collision
    assert next_state == state


def test_no_extra_truncation_penalty():
    """Verify that truncation preserves the ordinary reward."""

    config = replace(
        GAME_CONFIG,
        max_steps=1,
    )

    env = SnakeEnv(config=config)
    state, _ = env.reset(start_index=0)

    expected_state, expected_reward, expected_terminal = (
        env.transition(state, Action.STRAIGHT)
    )

    next_state, reward, terminated, truncated, info = (
        env.step(Action.STRAIGHT)
    )

    assert next_state == expected_state
    assert reward == expected_reward == -1
    assert terminated == expected_terminal is False
    assert truncated
    assert info["total_reward"] == -1
    assert info["termination_reason"] == "step_limit"


def test_reset_restores_initial_state(env):
    """Verify that a completed episode can be reset."""

    initial_state = env.get_state()

    for action in WINNING_ACTIONS:
        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    assert env.done

    reset_state, info = env.reset(start_index=0)

    assert reset_state == initial_state
    assert info["score"] == 0
    assert info["steps"] == 0
    assert info["total_reward"] == 0
    assert info["termination_reason"] is None
    assert not env.done
