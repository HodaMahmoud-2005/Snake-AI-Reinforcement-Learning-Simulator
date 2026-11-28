"""Reachable states, cached transitions, and optimal values for Snake."""

from collections import deque
from dataclasses import dataclass
from math import isfinite
from sys import float_info
from time import perf_counter

from configs.settings import (
    GAME_CONFIG,
    REWARD_CONFIG,
    RL_CONFIG,
    Action,
    GameConfig,
    RewardConfig,
)
from environment.game import SnakeState
from environment.snake_env import SnakeEnv


DEFAULT_MAX_ITERATIONS = 10_000


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    """Complete state space in BFS discovery order, with run statistics."""

    states: tuple[SnakeState, ...]
    initial_state_count: int
    expanded_transitions: int
    elapsed_seconds: float
    complete: bool = True
    config: GameConfig = GAME_CONFIG
    rewards: RewardConfig = REWARD_CONFIG

    @property
    def state_count(self) -> int:
        return len(self.states)


class EnumerationLimitError(RuntimeError):
    """Enumeration stopped early. Only progress statistics are available."""

    def __init__(self, reason: str, state_count: int, elapsed_seconds: float):
        self.reason = reason
        self.state_count = state_count
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Incomplete enumeration ({reason}): "
            f"{state_count:,} states discovered in {elapsed_seconds:.6f} s."
        )


def enumerate_reachable_states(
    config: GameConfig = GAME_CONFIG,
    rewards: RewardConfig = REWARD_CONFIG,
    *,
    max_states: int | None = None,
    max_seconds: float | None = None,
) -> EnumerationResult:
    """Enumerate nonterminal states reachable from every configured start.

    Each optional limit must be positive and finite; None disables it.
    A state budget allows exactly that many states. Exceeding it or reaching
    the time budget raises EnumerationLimitError without returning states.
    """
    if max_states is not None and (type(max_states) is not int or max_states <= 0):
        raise ValueError("max_states must be a positive integer or None.")
    if max_seconds is not None and (
        type(max_seconds) not in (int, float)
        or max_seconds <= 0
        or (isinstance(max_seconds, float) and not isfinite(max_seconds))
    ):
        raise ValueError("max_seconds must be a positive finite number or None.")
    if not config.start_states:
        raise ValueError("At least one start state is required.")

    started = perf_counter()
    env = SnakeEnv(config=config, rewards=rewards)
    # The FIFO queue expands states in discovery order. The set prevents
    # cycles and overlapping paths from scheduling a state more than once.
    queue: deque[SnakeState] = deque()
    visited: set[SnakeState] = set()
    states: list[SnakeState] = []

    def check_time() -> float:
        elapsed = perf_counter() - started
        if max_seconds is not None and elapsed >= max_seconds:
            raise EnumerationLimitError("max_seconds reached", len(visited), elapsed)
        return elapsed

    def discover(state: SnakeState) -> None:
        if state in visited:
            return
        if max_states is not None and len(visited) >= max_states:
            raise EnumerationLimitError(
                "max_states would be exceeded", len(visited), perf_counter() - started
            )
        visited.add(state)
        queue.append(state)
        states.append(state)

    for start_index in range(len(config.start_states)):
        check_time()
        state, _ = env.reset(start_index=start_index)
        if state.apples_collected(config.initial_snake_length) < config.apples_to_win:
            discover(state)
    initial_state_count = len(visited)

    expanded_transitions = 0
    while queue:
        state = queue.popleft()
        for action in Action:
            check_time()
            next_state, _, terminated = env.transition(state, action)
            expanded_transitions += 1
            # Collisions can return the current state. The flag describes
            # the outcome, so leave any previously visited state intact.
            if not terminated:
                discover(next_state)

    ordered_states = tuple(states)
    return EnumerationResult(
        states=ordered_states,
        initial_state_count=initial_state_count,
        expanded_transitions=expanded_transitions,
        elapsed_seconds=check_time(),
        config=config,
        rewards=rewards,
    )


@dataclass(frozen=True, slots=True)
class TransitionModelResult:
    """Validated transition cache and statistics, with BFS time excluded."""

    transitions: dict[tuple[SnakeState, Action], tuple[SnakeState, float, bool]]
    state_count: int
    actions_per_state: int
    terminal_count: int
    elapsed_seconds: float
    validated: bool = True
    config: GameConfig = GAME_CONFIG
    rewards: RewardConfig = REWARD_CONFIG

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def nonterminal_count(self) -> int:
        return self.transition_count - self.terminal_count


def build_transition_model(enumeration: EnumerationResult) -> TransitionModelResult:
    """Cache all actions for a complete BFS result using its own settings.

    Raise ValueError if the enumeration is incomplete or inconsistent.
    Preserve terminal outcomes exactly as returned by the environment.
    """
    if not enumeration.complete:
        raise ValueError("Cannot build a transition model from incomplete enumeration.")

    started = perf_counter()
    env = SnakeEnv(config=enumeration.config, rewards=enumeration.rewards)
    reachable_states = set(enumeration.states)
    transitions = {}
    terminal_count = 0

    for state in enumeration.states:
        for action in Action:
            outcome = env.transition(state, action)
            next_state, _, terminated = outcome
            if terminated:
                terminal_count += 1
            elif next_state not in reachable_states:
                raise ValueError(
                    f"Missing nonterminal successor {next_state!r} "
                    f"for {state!r} and action {action.name}."
                )
            transitions[state, action] = outcome

    actions_per_state = len(Action)
    if len(transitions) != enumeration.state_count * actions_per_state:
        raise ValueError("Transition model size does not match the state-action count.")

    return TransitionModelResult(
        transitions=transitions,
        state_count=enumeration.state_count,
        actions_per_state=actions_per_state,
        terminal_count=terminal_count,
        elapsed_seconds=perf_counter() - started,
        config=enumeration.config,
        rewards=enumeration.rewards,
    )


@dataclass(frozen=True, slots=True)
class ValueIterationResult:
    """State values and sweep diagnostics, including results at the cap."""

    values: dict[SnakeState, float]
    delta_history: tuple[float, ...]
    converged: bool
    stopping_reason: str
    gamma: float
    tolerance: float
    max_iterations: int
    elapsed_seconds: float

    @property
    def state_count(self) -> int:
        return len(self.values)

    @property
    def iterations(self) -> int:
        return len(self.delta_history)

    @property
    def final_delta(self) -> float:
        return self.delta_history[-1]


def _validate_planning_inputs(
    enumeration: EnumerationResult,
    model: TransitionModelResult,
    actions: tuple[Action, ...],
) -> None:
    """Check cache coverage and outcomes before running the sweeps."""
    if not enumeration.complete or not model.validated:
        raise ValueError("Value iteration requires complete enumeration and a validated model.")
    if (model.config, model.rewards) != (enumeration.config, enumeration.rewards):
        raise ValueError("Transition model settings do not match the enumeration.")

    states = set(enumeration.states)
    if (
        len(states) != enumeration.state_count
        or model.state_count != len(states)
        or model.actions_per_state != len(actions)
        or model.transition_count != len(states) * len(actions)
    ):
        raise ValueError("Transition model size does not match the enumerated state-action pairs.")

    terminal_count = 0
    for key, outcome in model.transitions.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid transition key {key!r}.")
        state, action = key
        if state not in states or not isinstance(action, Action):
            raise ValueError(f"Unexpected state-action pair {key!r}.")
        if not isinstance(outcome, tuple) or len(outcome) != 3:
            raise ValueError(f"Invalid transition outcome for {key!r}.")
        next_state, reward, terminated = outcome
        if (
            not isinstance(next_state, SnakeState)
            or type(terminated) is not bool
            or type(reward) not in (int, float)
            or not -float_info.max <= reward <= float_info.max
        ):
            raise ValueError(f"Invalid transition outcome for {key!r}.")
        if not terminated and next_state not in states:
            raise ValueError(f"Missing nonterminal successor for {key!r}.")
        terminal_count += terminated
    if terminal_count != model.terminal_count:
        raise ValueError("Transition model terminal count does not match its cached outcomes.")


def _action_value(
    outcome: tuple[SnakeState, float, bool],
    values: dict[SnakeState, float],
    gamma: float,
) -> float:
    next_state, reward, terminated = outcome
    # A terminal outcome ends the return at the immediate reward.
    action_value = float(reward) if terminated else reward + gamma * values[next_state]
    if not isfinite(action_value):
        raise ValueError("Bellman action value exceeded finite floating-point range.")
    return action_value


def value_iteration(
    enumeration: EnumerationResult,
    model: TransitionModelResult,
    *,
    gamma: float = RL_CONFIG.gamma,
    tolerance: float = RL_CONFIG.value_iteration_tolerance,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> ValueIterationResult:
    """Compute discounted optimal values using only cached transitions.

    Stop after a sweep with delta below tolerance or after max_iterations
    sweeps. At the cap, return the approximation with converged=False
    unless the last sweep met the tolerance.
    """
    if type(gamma) not in (int, float) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and satisfy 0 <= gamma < 1.")
    if type(tolerance) not in (int, float) or not 0 < tolerance <= float_info.max:
        raise ValueError("tolerance must be positive and finite.")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")

    started = perf_counter()
    actions = tuple(Action)
    _validate_planning_inputs(enumeration, model, actions)
    values = dict.fromkeys(enumeration.states, 0.0)
    delta_history = []

    for _ in range(max_iterations):
        new_values = {}
        delta = 0.0
        for state in enumeration.states:
            action_values = []
            for action in actions:
                action_values.append(
                    _action_value(model.transitions[state, action], values, gamma)
                )

            new_values[state] = max(action_values)
            change = abs(new_values[state] - values[state])
            if not isfinite(change):
                raise ValueError("Bellman sweep delta exceeded finite floating-point range.")
            delta = max(delta, change)

        # Publish the new dictionary only after the full synchronous sweep.
        values = new_values
        delta_history.append(delta)
        if delta < tolerance:
            break

    converged = delta_history[-1] < tolerance
    return ValueIterationResult(
        values=values,
        delta_history=tuple(delta_history),
        converged=converged,
        stopping_reason="tolerance_reached" if converged else "max_iterations_reached",
        gamma=gamma,
        tolerance=tolerance,
        max_iterations=max_iterations,
        elapsed_seconds=perf_counter() - started,
    )


def extract_optimal_policy(
    enumeration: EnumerationResult,
    model: TransitionModelResult,
    solution: ValueIterationResult,
) -> dict[SnakeState, Action]:
    """Select the first exact maximizer in Action order for each state."""
    if not solution.converged:
        raise ValueError("Cannot extract an optimal policy from a nonconverged solution.")
    actions = tuple(Action)
    _validate_planning_inputs(enumeration, model, actions)
    if (
        not 0 <= solution.gamma < 1
        or not 0 < solution.tolerance <= float_info.max
        or not solution.delta_history
        or not 0 <= solution.final_delta < solution.tolerance
    ):
        raise ValueError("Invalid Value Iteration convergence diagnostics.")
    values = solution.values
    if values.keys() != set(enumeration.states) or any(
        type(value) not in (int, float) or not -float_info.max <= value <= float_info.max
        for value in values.values()
    ):
        raise ValueError("Expected one finite value per reachable nonterminal state.")

    policy = {}
    for state in enumeration.states:
        returns = {
            action: _action_value(model.transitions[state, action], values, solution.gamma)
            for action in actions
        }
        # max keeps the first key when returns are exactly tied.
        action = max(returns, key=returns.__getitem__)
        if abs(returns[action] - values[state]) > solution.tolerance:
            raise ValueError(f"Value Iteration solution exceeds Bellman tolerance at {state!r}.")
        policy[state] = action
    return policy


def select_action(state: SnakeState, policy: dict[SnakeState, Action]) -> Action:
    """Look up a planned action or raise KeyError for an unseen state."""
    try:
        return policy[state]
    except KeyError:
        raise KeyError(f"State is outside the planned reachable state space: {state!r}") from None


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Stage results and timings. Policy and extraction time are None at the cap."""

    enumeration: EnumerationResult
    model: TransitionModelResult
    solution: ValueIterationResult
    policy: dict[SnakeState, Action] | None
    policy_elapsed_seconds: float | None
    total_elapsed_seconds: float

    @property
    def policy_size(self) -> int:
        return 0 if self.policy is None else len(self.policy)


def plan_value_iteration(
    config: GameConfig = GAME_CONFIG,
    rewards: RewardConfig = REWARD_CONFIG,
    *,
    gamma: float = RL_CONFIG.gamma,
    tolerance: float = RL_CONFIG.value_iteration_tolerance,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_states: int | None = None,
    max_seconds: float | None = None,
) -> PlanningResult:
    """Run all planning stages, propagating enumeration errors to the caller."""
    started = perf_counter()
    enumeration = enumerate_reachable_states(
        config, rewards, max_states=max_states, max_seconds=max_seconds
    )
    model = build_transition_model(enumeration)
    solution = value_iteration(
        enumeration, model, gamma=gamma, tolerance=tolerance, max_iterations=max_iterations
    )
    policy = None
    policy_elapsed_seconds = None
    if solution.converged:
        policy_started = perf_counter()
        policy = extract_optimal_policy(enumeration, model, solution)
        policy_elapsed_seconds = perf_counter() - policy_started
    return PlanningResult(
        enumeration, model, solution, policy, policy_elapsed_seconds, perf_counter() - started
    )


def _demonstrate_episode(plan: PlanningResult, policy: dict[SnakeState, Action]) -> None:
    """Execute one real episode outside the planning timer."""
    env = SnakeEnv(config=plan.enumeration.config, rewards=plan.enumeration.rewards)
    state, info = env.reset(start_index=0)
    actions_taken = []
    discounted_return = 0.0
    discount = 1.0
    while not env.done:
        action = select_action(state, policy)
        actions_taken.append(action.name)
        state, reward, _, _, info = env.step(action)
        discounted_return += discount * reward
        discount *= plan.solution.gamma

    print(f"Episode actions: {', '.join(actions_taken)}")
    print(f"Episode score: {info['score']}")
    print(f"Episode steps: {info['steps']}")
    print(f"Undiscounted episode reward: {info['total_reward']:g}")
    print(f"Discounted episode return: {discounted_return:.9f}")
    print(f"Episode success: {info['success']}")
    print(f"Episode termination or truncation reason: {info['termination_reason']}")


if __name__ == "__main__":
    plan = plan_value_iteration()
    result, model, solution = plan.enumeration, plan.model, plan.solution
    print(f"Reachable nonterminal states: {result.state_count:,}")
    print(f"Unique initial states: {result.initial_state_count:,}")
    print(f"Expanded transitions: {result.expanded_transitions:,}")
    print(f"Runtime: {result.elapsed_seconds:.6f} s")
    print(f"Complete: {result.complete}")

    print(f"Transition model size: {model.transition_count:,}")
    print(f"Actions per state: {model.actions_per_state}")
    print(f"Terminal transitions: {model.terminal_count:,}")
    print(f"Nonterminal transitions: {model.nonterminal_count:,}")
    print(f"Model construction runtime: {model.elapsed_seconds:.6f} s")
    print(f"Model validated: {model.validated}")
    for state in result.states[:1]:
        print(f"Sample state: {state}")
        for action in Action:
            print(f"  {action.name}: {model.transitions[state, action]}")

    print(f"Initialized states: {solution.state_count:,}")
    print(f"Gamma: {solution.gamma}")
    print(f"Tolerance: {solution.tolerance:g}")
    print(f"Maximum iterations: {solution.max_iterations:,}")
    print(f"Completed iterations: {solution.iterations:,}")
    print(f"Initial sweep delta: {solution.delta_history[0]:.9g}")
    print(f"Final delta: {solution.final_delta:.9g}")
    print(f"Converged: {solution.converged}")
    print(f"Stopping reason: {solution.stopping_reason}")
    print(f"Value Iteration runtime: {solution.elapsed_seconds:.6f} s")
    print(f"Minimum state value: {min(solution.values.values()):.9f}")
    print(f"Maximum state value: {max(solution.values.values()):.9f}")
    print(f"Default initial state value: {solution.values[result.states[0]]:.9f}")
    print("Delta history sample (sweep, delta):")
    for index in sorted({0, 1, 2, solution.iterations - 1}):
        if index < solution.iterations:
            print(f"  {index + 1}: {solution.delta_history[index]:.9g}")
    print("Sample state values (starting with the default initial state):")
    for state in result.states[:3]:
        print(f"  {state}: {solution.values[state]:.9f}")

    print(f"Enumeration runtime: {result.elapsed_seconds:.6f} s")
    print(f"Total planning runtime: {plan.total_elapsed_seconds:.6f} s")
    if plan.policy is None:
        print("Optimal policy unavailable.")
    else:
        policy = plan.policy
        initial_state = result.states[0]
        print(f"Policy extraction runtime: {plan.policy_elapsed_seconds:.6f} s")
        print(f"Policy size: {plan.policy_size:,}")
        print(f"Policy coverage: {plan.policy_size:,}/{result.state_count:,}")
        print("Policy validated: True")
        print("Tie-breaking: first exact maximizer in LEFT, STRAIGHT, RIGHT order")
        print("Initial-state action values:")
        for action in Action:
            score = _action_value(model.transitions[initial_state, action], solution.values, solution.gamma)
            print(f"  {action.name}: {score:.9f}")
        print(f"Initial selected action: {select_action(initial_state, policy).name}")
        for action in Action:
            print(f"{action.name} decisions: {sum(choice == action for choice in policy.values()):,}")
        print("Sample policy entries:")
        for state in result.states[:3]:
            print(f"  {state}: {policy[state].name}")
        _demonstrate_episode(plan, policy)
