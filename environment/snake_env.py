"""Reinforcement-learning interface for the Snake game."""

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
    transition,
)


class SnakeEnv:
    """Headless Snake environment for reinforcement learning."""

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
        self.total_reward = 0.0

    def reset(
        self,
        seed: int | None = None,
        start_index: int | None = None,
    ) -> tuple[SnakeState, dict]:
        """Start a new episode and return its initial state."""

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

        body, direction = start_states[start_index]

        body = tuple(body)

        self._state = SnakeState(
            body=body,
            direction=direction,
            food=spawn_food(
                body,
                score=0,
                config=self.config,
            ),
            score=0,
            steps=0,
        )

        self.total_reward = 0.0

        return self.get_state(), self._get_info()

    def get_state(self) -> SnakeState:
        """Return the current immutable state."""

        if self._state is None:
            raise RuntimeError(
                "Call reset() before accessing the state."
            )

        return self._state

    def step(
        self,
        action: int | Action,
    ) -> tuple[SnakeState, float, bool, bool, dict]:
        """Apply an action and return the RL transition."""

        current_state = self.get_state()

        next_state, reward = transition(
            state=current_state,
            action=action,
            config=self.config,
            rewards=self.rewards,
        )

        self._state = next_state
        self.total_reward += reward

        return (
            next_state,
            reward,
            next_state.terminated,
            next_state.truncated,
            self._get_info(),
        )

    def _get_info(self) -> dict:
        """Return additional information for evaluation."""

        state = self.get_state()

        return {
            "score": state.score,
            "steps": state.steps,
            "snake_length": state.length,
            "total_reward": self.total_reward,
            "termination_reason": state.termination_reason,
            "success": state.termination_reason == "win",
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    @property
    def score(self) -> int:
        return self.get_state().score

    @property
    def steps(self) -> int:
        return self.get_state().steps

    @property
    def termination_reason(self) -> str | None:
        return self.get_state().termination_reason

    @property
    def done(self) -> bool:
        return self.get_state().done

    def close(self) -> None:
        """Release resources; no rendering resources are used."""
        pass
