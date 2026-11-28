"""Headless reinforcement-learning environment for Snake."""

import random

from configs.settings import (
    Action,
    GameConfig,
    RewardConfig,
    GAME_CONFIG,
    REWARD_CONFIG,
    RL_CONFIG,
)

from .game import (
    SnakeState,
    spawn_food,
    resolve_transition,
    transition,
)


class SnakeEnv:
    """Live Snake environment with an external step limit."""

    def __init__(
        self,
        config: GameConfig = GAME_CONFIG,
        rewards: RewardConfig = REWARD_CONFIG,
        seed: int | None = RL_CONFIG.random_seed,
    ):
        self.config = config
        self.rewards = rewards

        self.seed = seed
        self._rng = random.Random(seed)

        self.action_space = tuple(Action)
        self.n_actions = len(self.action_space)

        self._state: SnakeState | None = None

        # Episode metadata, not planning-state attributes.
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.termination_reason: str | None = None

    def reset(
        self,
        seed: int | None = None,
        start_index: int | None = None,
    ) -> tuple[SnakeState, dict]:
        """Reset the live episode."""

        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)

        start_states = self.config.start_states

        if not start_states:
            raise ValueError(
                "At least one start state is required."
            )

        if start_index is None:
            start_index = self._rng.randrange(
                len(start_states)
            )

        if not 0 <= start_index < len(start_states):
            raise ValueError("Invalid start state index.")

        body, configured_direction = start_states[start_index]

        body = tuple(body)

        state = SnakeState(
            body=body,
            food=spawn_food(
                body,
                score=0,
                config=self.config,
            ),
        )

        if state.length != self.config.initial_snake_length:
            raise ValueError("Invalid initial snake length.")

        if state.direction != configured_direction:
            raise ValueError(
                "Configured heading does not match snake body."
            )

        self._state = state

        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.termination_reason = None

        return self.get_state(), self._get_info()

    def get_state(self) -> SnakeState:
        """Return the current immutable planning state."""

        if self._state is None:
            raise RuntimeError(
                "Call reset() before accessing the state."
            )

        return self._state

    def transition(
        self,
        state: SnakeState,
        action: int | Action,
    ) -> tuple[SnakeState, float, bool]:
        """Evaluate a transition without changing this environment."""

        return transition(
            state=state,
            action=action,
            config=self.config,
            rewards=self.rewards,
        )

    def step(
        self,
        action: int | Action,
    ) -> tuple[SnakeState, float, bool, bool, dict]:
        """Execute one action in the live episode."""

        state = self.get_state()

        if self.done:
            raise RuntimeError(
                "Episode has ended. Reset the environment first."
            )

        result = resolve_transition(
            state=state,
            action=action,
            config=self.config,
            rewards=self.rewards,
        )

        self._state = result.next_state

        self.steps += 1
        self.total_reward += result.reward

        self.terminated = result.terminated

        self.termination_reason = (
            result.termination_reason
        )

        # External truncation is not part of the planning MDP.
        self.truncated = (
            not self.terminated
            and self.steps >= self.config.max_steps
        )

        if self.truncated:
            self.termination_reason = "step_limit"

        return (
            self.get_state(),
            result.reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def _get_info(self) -> dict:
        """Return evaluation information."""

        state = self.get_state()

        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": state.length,
            "total_reward": self.total_reward,
            "termination_reason": self.termination_reason,
            "success": self.termination_reason == "win",
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @property
    def score(self) -> int:
        return self.get_state().apples_collected(
            self.config.initial_snake_length
        )

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def close(self) -> None:
        """No rendering resources need to be released."""
        pass
