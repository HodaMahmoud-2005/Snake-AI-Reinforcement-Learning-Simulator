# Owner: Hoda | Backlog #10
# TODO: Test reachable state coverage, Bellman update and terminal transitions.
# TODO: Test valid policy actions, convergence residual and known real-game scenarios.

from dataclasses import replace
from importlib import import_module
from math import isfinite

import pytest

from agents.value_iteration import (
    EnumerationLimitError,
    build_transition_model,
    enumerate_reachable_states,
    extract_optimal_policy,
    plan_value_iteration,
    select_action,
    value_iteration,
)
from configs.settings import RL_CONFIG, Action, Direction, GameConfig, RewardConfig
from environment.snake_env import SnakeEnv


@pytest.fixture
def config():
    first = (((1, 1), (0, 1)), Direction.RIGHT)
    second = (((1, 1), (1, 0)), Direction.DOWN)
    return GameConfig(
        grid_width=3,
        grid_height=3,
        obstacles=(),
        apples_to_win=1,
        initial_snake=first[0],
        food_sequence=((0, 1),),
        start_states=(first, second, first),
    )


def test_fifo_starts_and_full_state_identity(config):
    env = SnakeEnv(config=config)
    first, second, duplicate = (
        env.reset(start_index=index)[0] for index in range(3)
    )
    assert first == duplicate
    assert first.head == second.head
    assert first.body != second.body

    result = enumerate_reachable_states(config)
    assert result.complete
    assert result.initial_state_count == 2
    assert result.states[:2] == (first, second)
    assert result.state_count == len(set(result.states))

    # All three moves from the first start are legal and produce new states.
    first_outcomes = tuple(env.transition(first, action) for action in Action)
    assert all(not terminated for _, _, terminated in first_outcomes)
    assert result.states[2:5] == tuple(state for state, _, _ in first_outcomes)

    # The same move destination yields the same body with different food.
    straight, _, _ = env.transition(first, Action.STRAIGHT)
    left, _, _ = env.transition(second, Action.LEFT)
    assert straight.body == left.body
    assert straight.food != left.food
    assert left in result.states
    assert enumerate_reachable_states(config).states == result.states


def test_all_nonterminal_successors_and_terminal_outcomes(config):
    env = SnakeEnv(config=config)
    result = enumerate_reachable_states(config)
    states = set(result.states)
    assert result.expanded_transitions == len(Action) * len(states)

    collisions = set()
    wins = set()
    for state in result.states:
        assert state.food is not None
        for action in Action:
            next_state, _, terminated = env.transition(state, action)
            if not terminated:
                assert next_state in states
            elif next_state == state:
                collisions.add(next_state)
            else:
                wins.add(next_state)

    assert collisions and collisions <= states
    assert wins and wins.isdisjoint(states)


def test_live_environment_and_step_limit(config):
    capped_config = replace(config, max_steps=1)
    env = SnakeEnv(config=capped_config)
    env.reset(start_index=1)
    state, _, terminated, truncated, info = env.step(Action.STRAIGHT)
    rng_state = env._rng.getstate()
    assert truncated and not terminated

    # The live episode is over, but the planning graph includes the next move.
    successor, _, terminated = env.transition(state, Action.LEFT)
    assert not terminated
    result = enumerate_reachable_states(env.config, env.rewards)
    assert successor in result.states
    assert result.states == enumerate_reachable_states(config).states
    assert env.get_state() is state
    assert env._get_info() == info
    assert env._rng.getstate() == rng_state


def test_default_policy_is_greedy_and_wins():
    plan = plan_value_iteration()
    assert plan.policy is not None
    assert plan.policy_size == plan.enumeration.state_count
    assert plan.policy.keys() == set(plan.enumeration.states)
    for state, action in plan.policy.items():
        assert isinstance(action, Action)
        returns = {}
        for candidate in Action:
            successor, reward, terminal = plan.model.transitions[state, candidate]
            returns[candidate] = (
                reward if terminal else reward + plan.solution.gamma * plan.solution.values[successor]
            )
        assert action is max(returns, key=returns.__getitem__)

    env = SnakeEnv(config=plan.enumeration.config, rewards=plan.enumeration.rewards)
    state, _ = env.reset(start_index=0)
    assert select_action(state, plan.policy) is Action.STRAIGHT
    discounted = 0.0
    steps = 0
    while not env.done:
        state, reward, terminated, truncated, info = env.step(select_action(state, plan.policy))
        discounted += plan.solution.gamma ** steps * reward
        steps += 1
    assert terminated and not truncated and info["success"]
    assert info["score"] == env.config.apples_to_win
    assert discounted == pytest.approx(plan.solution.values[plan.enumeration.states[0]])
    with pytest.raises(KeyError, match="outside the planned reachable state space"):
        select_action(state, plan.policy)


def test_exact_tie_breaking_and_close_but_lower_action(config):
    tied = plan_value_iteration(config, gamma=0)
    assert tied.policy is not None
    assert tied.policy[tied.enumeration.states[0]] is Action.LEFT

    rewards = RewardConfig(collision=-1.0 + 1e-12)
    close = plan_value_iteration(config, rewards, gamma=0)
    assert close.policy is not None
    state = next(
        state for state in close.enumeration.states
        if close.model.transitions[state, Action.LEFT][1] == rewards.move
        and close.model.transitions[state, Action.STRAIGHT][1] == rewards.collision
        and close.model.transitions[state, Action.RIGHT][1] <= rewards.collision
    )
    assert close.policy[state] is Action.STRAIGHT


def test_policy_collision_does_not_bootstrap(config):
    rewards = RewardConfig(move=-10.0, collision=-1.0, food=-20.0, win_bonus=0.0)
    plan = plan_value_iteration(config, rewards)
    assert plan.policy is not None
    for (state, action), outcome in plan.model.transitions.items():
        if outcome == (state, rewards.collision, True):
            assert plan.model.transitions[state, plan.policy[state]] == outcome


def test_policy_rejects_invalid_inputs_and_planner_reports_cap(config):
    plan = plan_value_iteration(config)
    enumeration, model, solution = plan.enumeration, plan.model, plan.solution
    capped = plan_value_iteration(config, max_iterations=1)
    assert not capped.solution.converged
    assert capped.solution.stopping_reason == "max_iterations_reached"
    assert capped.policy is None and capped.policy_size == 0
    assert capped.policy_elapsed_seconds is None
    assert capped.solution.delta_history and capped.solution.values
    with pytest.raises(ValueError, match="nonconverged"):
        extract_optimal_policy(capped.enumeration, capped.model, capped.solution)
    with pytest.raises(ValueError, match="complete enumeration"):
        extract_optimal_policy(replace(enumeration, complete=False), model, solution)
    with pytest.raises(ValueError, match="validated model"):
        extract_optimal_policy(enumeration, replace(model, validated=False), solution)
    with pytest.raises(ValueError, match="one finite value"):
        extract_optimal_policy(enumeration, model, replace(solution, values={}))
    wrong_values = dict.fromkeys(enumeration.states, 0.0)
    with pytest.raises(ValueError, match="Bellman tolerance"):
        extract_optimal_policy(enumeration, model, replace(solution, values=wrong_values))
    with pytest.raises(EnumerationLimitError, match="max_states"):
        plan_value_iteration(config, max_states=1)


def test_planning_is_repeatable_and_preserves_policy_inputs(config):
    plan = plan_value_iteration(config)
    repeated = plan_value_iteration(config)
    values = plan.solution.values.copy()
    transitions = plan.model.transitions.copy()
    extracted = extract_optimal_policy(plan.enumeration, plan.model, plan.solution)
    assert extracted == plan.policy == repeated.policy
    assert plan.solution.delta_history == repeated.solution.delta_history
    assert plan.solution.values == values
    assert plan.model.transitions == transitions
    stage_times = (
        plan.enumeration.elapsed_seconds, plan.model.elapsed_seconds,
        plan.solution.elapsed_seconds, plan.policy_elapsed_seconds,
    )
    assert all(duration is not None and duration >= 0 for duration in stage_times)
    assert plan.total_elapsed_seconds >= sum(stage_times) - 1e-9
    assert plan.enumeration.config == config


def test_total_planning_timer_spans_all_four_stages(config, monkeypatch):
    module = import_module("agents.value_iteration")
    stages = (
        "enumerate_reachable_states", "build_transition_model",
        "value_iteration", "extract_optimal_policy",
    )
    completed = []
    monkeypatch.setattr(module, "perf_counter", lambda: float(len(completed)))

    def timed(stage, original):
        def run(*args, **kwargs):
            result = original(*args, **kwargs)
            completed.append(stage)
            return result
        return run

    for stage in stages:
        monkeypatch.setattr(module, stage, timed(stage, getattr(module, stage)))
    result = plan_value_iteration(config)
    assert tuple(completed) == stages
    assert result.total_elapsed_seconds == 4.0
    assert result.policy_elapsed_seconds == 1.0


def test_state_budget(config):
    complete = enumerate_reachable_states(config)
    assert enumerate_reachable_states(config, max_states=complete.state_count).complete

    # One limit is exceeded during initialization, the other during expansion.
    for limit in (1, complete.state_count - 1):
        with pytest.raises(EnumerationLimitError, match="max_states") as caught:
            enumerate_reachable_states(config, max_states=limit)
        assert caught.value.state_count == limit
        assert caught.value.elapsed_seconds >= 0


def test_runtime_budget(config, monkeypatch):
    ticks = iter((0.0, 0.25, 1.0))
    monkeypatch.setattr("agents.value_iteration.perf_counter", lambda: next(ticks))
    with pytest.raises(EnumerationLimitError, match="max_seconds") as caught:
        enumerate_reachable_states(config, max_seconds=1)
    assert caught.value.state_count == 1
    assert caught.value.elapsed_seconds == 1


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_states", 0),
        ("max_states", -1),
        ("max_states", 2.5),
        ("max_states", True),
        ("max_seconds", 0),
        ("max_seconds", -1),
        ("max_seconds", float("inf")),
        ("max_seconds", float("nan")),
        ("max_seconds", True),
        ("max_seconds", "1"),
    ],
)
def test_invalid_limits(config, option, value):
    with pytest.raises(ValueError, match=option):
        enumerate_reachable_states(config, **{option: value})


def test_model_matches_every_real_transition(config):
    rewards = RewardConfig(food=7.0, win_bonus=20.0, move=-2.0, collision=-9.0)
    env = SnakeEnv(config=config, rewards=rewards)
    enumeration = enumerate_reachable_states(config, rewards)
    result = build_transition_model(enumeration)
    states = set(enumeration.states)
    actions = {Action.LEFT, Action.STRAIGHT, Action.RIGHT}

    assert result.validated
    assert result.elapsed_seconds >= 0
    assert result.actions_per_state == len(actions)
    assert result.state_count == len(states)
    assert result.transition_count == len(states) * len(actions)
    assert set(result.transitions) == {
        (state, action) for state in states for action in actions
    }
    assert all(isinstance(action, Action) for _, action in result.transitions)

    collisions = wins = nonterminal = 0
    for (state, action), outcome in result.transitions.items():
        assert outcome == env.transition(state, action)
        next_state, reward, terminated = outcome
        if not terminated:
            assert next_state in states
            nonterminal += 1
        elif next_state == state:
            assert reward == rewards.collision
            collisions += 1
        else:
            assert next_state not in states
            assert next_state.food is None
            assert reward == rewards.final_apple
            wins += 1

    assert collisions > 0 and wins > 0 and nonterminal > 0
    assert result.terminal_count == collisions + wins
    assert result.nonterminal_count == nonterminal


def test_model_is_deterministic_and_independent_of_live_truncation(config):
    config = replace(config, max_steps=1)
    env = SnakeEnv(config=config)
    initial, _ = env.reset(start_index=1)
    state, reward, terminated, truncated, info = env.step(Action.STRAIGHT)
    rng_state = env._rng.getstate()
    assert truncated and not terminated

    first = build_transition_model(enumerate_reachable_states(config))
    second = build_transition_model(enumerate_reachable_states(config))
    assert list(first.transitions.items()) == list(second.transitions.items())
    assert first.transitions[initial, Action.STRAIGHT] == (state, reward, False)
    assert env.get_state() is state
    assert env._get_info() == info
    assert env._rng.getstate() == rng_state


def test_model_rejects_incomplete_enumeration(config):
    enumeration = enumerate_reachable_states(config)
    with pytest.raises(ValueError, match="incomplete"):
        build_transition_model(replace(enumeration, complete=False))


def test_model_rejects_missing_nonterminal_successor(config):
    enumeration = enumerate_reachable_states(config)
    with pytest.raises(ValueError, match="Missing nonterminal successor"):
        build_transition_model(replace(enumeration, states=enumeration.states[:1]))


def test_model_rejects_duplicate_states(config):
    enumeration = enumerate_reachable_states(config)
    duplicate_states = enumeration.states + enumeration.states[:1]
    with pytest.raises(ValueError, match="model size"):
        build_transition_model(replace(enumeration, states=duplicate_states))


def test_first_two_value_sweeps_are_synchronous(config):
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    env = SnakeEnv(config=config)
    outcomes = {
        state: [env.transition(state, action) for action in Action]
        for state in enumeration.states
    }
    immediate = {
        state: max(reward for _, reward, _ in transitions)
        for state, transitions in outcomes.items()
    }
    first = value_iteration(enumeration, model, max_iterations=1)
    second = value_iteration(enumeration, model, max_iterations=2)

    assert first.gamma == RL_CONFIG.gamma
    assert first.tolerance == RL_CONFIG.value_iteration_tolerance
    assert first.values == immediate
    assert first.state_count == enumeration.state_count
    assert first.delta_history == (max(map(abs, immediate.values())),)
    assert not first.converged
    assert first.stopping_reason == "max_iterations_reached"

    expected = {
        state: max(
            reward if terminal else reward + RL_CONFIG.gamma * immediate[successor]
            for successor, reward, terminal in transitions
        )
        for state, transitions in outcomes.items()
    }
    assert expected != immediate
    assert second.values == pytest.approx(expected)
    second_delta = max(abs(expected[state] - immediate[state]) for state in immediate)
    assert second.delta_history == pytest.approx((*first.delta_history, second_delta))
    assert second.final_delta == pytest.approx(second_delta)
    assert second.iterations == 2


def test_terminal_collision_has_zero_bootstrap_and_negative_value(config):
    rewards = RewardConfig(move=-10.0, collision=-1.0, food=-20.0, win_bonus=0.0)
    enumeration = enumerate_reachable_states(config, rewards)
    model = build_transition_model(enumeration)
    state = next(
        state for (state, _), outcome in model.transitions.items()
        if outcome == (state, rewards.collision, True)
    )
    result = value_iteration(enumeration, model)
    assert result.converged
    assert result.values[state] == rewards.collision
    assert all(value < 0 for value in result.values.values())


def test_terminal_win_has_zero_bootstrap(config):
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    result = value_iteration(enumeration, model)
    wins = [
        (state, successor, reward)
        for (state, _), (successor, reward, terminal) in model.transitions.items()
        if terminal and successor.food is None
    ]
    assert wins
    for state, successor, reward in wins:
        assert successor not in result.values
        assert result.values[state] == reward


def test_zero_gamma_and_strict_tolerance(config):
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    immediate = {
        state: max(model.transitions[state, action][1] for action in Action)
        for state in enumeration.states
    }
    tolerance = max(map(abs, immediate.values()))
    result = value_iteration(enumeration, model, gamma=0, tolerance=tolerance, max_iterations=2)
    assert result.values == immediate
    assert result.delta_history == (tolerance, 0.0)
    assert result.iterations == 2
    assert result.converged
    assert result.stopping_reason == "tolerance_reached"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("gamma", -0.1), ("gamma", 1), ("gamma", float("nan")),
        ("gamma", float("inf")), ("gamma", True),
        ("tolerance", 0), ("tolerance", -1), ("tolerance", float("nan")),
        ("tolerance", float("inf")), ("tolerance", True),
        ("max_iterations", 0), ("max_iterations", -1),
        ("max_iterations", 1.5), ("max_iterations", True),
    ],
)
def test_value_iteration_rejects_invalid_parameters(config, option, value):
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    with pytest.raises(ValueError, match=option):
        value_iteration(enumeration, model, **{option: value})


def test_value_iteration_rejects_invalid_planning_inputs(config):
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    with pytest.raises(ValueError, match="complete enumeration"):
        value_iteration(replace(enumeration, complete=False), model)
    with pytest.raises(ValueError, match="validated model"):
        value_iteration(enumeration, replace(model, validated=False))
    with pytest.raises(ValueError, match="settings"):
        value_iteration(enumeration, replace(model, rewards=RewardConfig(move=-7)))

    transitions = model.transitions.copy()
    key, (successor, reward, terminal) = transitions.popitem()
    with pytest.raises(ValueError, match="model size"):
        value_iteration(enumeration, replace(model, transitions=transitions))
    transitions[key] = (successor, float("nan"), terminal)
    with pytest.raises(ValueError, match="Invalid transition outcome"):
        value_iteration(enumeration, replace(model, transitions=transitions))

    winning_key, winning_outcome = next(
        (key, outcome) for key, outcome in model.transitions.items()
        if outcome[2] and outcome[0].food is None
    )
    transitions = model.transitions.copy()
    transitions[winning_key] = (winning_outcome[0], winning_outcome[1], False)
    with pytest.raises(ValueError, match="Missing nonterminal successor"):
        value_iteration(enumeration, replace(model, transitions=transitions))


def test_convergence_residual_repeatability_and_input_preservation(config, monkeypatch):
    config = replace(config, max_steps=1)
    env = SnakeEnv(config=config)
    env.reset(start_index=1)
    state, _, _, truncated, info = env.step(Action.STRAIGHT)
    assert truncated
    enumeration = enumerate_reachable_states(config)
    model = build_transition_model(enumeration)
    states_before = enumeration.states
    transitions_before = model.transitions.copy()

    def unexpected_transition(*args):
        pytest.fail("Value iteration should use the cached transitions.")

    monkeypatch.setattr(SnakeEnv, "transition", unexpected_transition)
    result = value_iteration(enumeration, model)
    repeated = value_iteration(enumeration, model)
    assert result.converged and result.final_delta < result.tolerance
    assert result.elapsed_seconds >= 0
    assert result.values == repeated.values
    assert result.delta_history == repeated.delta_history
    assert set(result.values) == set(enumeration.states)

    for state_key, value in result.values.items():
        assert isfinite(value)
        returns = []
        for action in Action:
            successor, reward, terminal = transitions_before[state_key, action]
            returns.append(reward if terminal else reward + result.gamma * result.values[successor])
        assert abs(max(returns) - value) < result.tolerance

    assert enumeration.states == states_before
    assert model.transitions == transitions_before
    assert env.get_state() is state
    assert env._get_info() == info
