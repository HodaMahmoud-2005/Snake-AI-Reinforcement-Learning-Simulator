"""Deterministic transition model for the Snake environment."""

from dataclasses import dataclass

from configs.settings import (
    Action,
    Direction,
    GameConfig,
    RewardConfig,
    GAME_CONFIG,
    REWARD_CONFIG,
)


Position = tuple[int, int]

DIRECTION_VECTORS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}

VECTOR_DIRECTIONS = {
    vector: direction
    for direction, vector in DIRECTION_VECTORS.items()
}


# ==================================================
# Immutable planning state
# ==================================================

@dataclass(frozen=True, slots=True)
class SnakeState:
    """Planning state containing only body and food."""

    body: tuple[Position, ...]
    food: Position | None

    @property
    def head(self) -> Position:
        return self.body[0]

    @property
    def tail(self) -> Position:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def direction(self) -> Direction:
        """Derive heading from the head and neck."""

        head = self.body[0]
        neck = self.body[1]

        vector = (
            head[0] - neck[0],
            head[1] - neck[1],
        )

        return VECTOR_DIRECTIONS[vector]

    def apples_collected(
        self,
        initial_length: int = GAME_CONFIG.initial_snake_length,
    ) -> int:
        """Derive collected apples from body length."""

        return self.length - initial_length

    @property
    def score(self) -> int:
        """Score under the default game configuration."""

        return self.apples_collected()


# ==================================================
# Relative movement
# ==================================================

def turn_direction(
    direction: Direction,
    action: int | Action,
) -> Direction:
    """Convert a relative action to an absolute heading."""

    if (
        isinstance(action, bool)
        or not isinstance(action, int)
        or action not in (0, 1, 2)
    ):
        raise ValueError("Action must be 0, 1 or 2.")

    return Direction(
        (int(direction) + int(action) - 1) % 4
    )


def next_head_position(
    head: Position,
    direction: Direction,
) -> Position:
    """Calculate the next head position."""

    dx, dy = DIRECTION_VECTORS[direction]

    return head[0] + dx, head[1] + dy


def is_inside_grid(
    position: Position,
    config: GameConfig = GAME_CONFIG,
) -> bool:
    """Check grid boundaries."""

    x, y = position

    return (
        0 <= x < config.grid_width
        and 0 <= y < config.grid_height
    )


# ==================================================
# Deterministic food placement
# ==================================================

def spawn_food(
    body: tuple[Position, ...],
    score: int,
    config: GameConfig = GAME_CONFIG,
) -> Position | None:
    """Use the fixed food sequence and row-major fallback."""

    if score >= config.apples_to_win:
        return None

    occupied = set(body) | set(config.obstacles)

    preferred = config.food_sequence[score]

    if (
        is_inside_grid(preferred, config)
        and preferred not in occupied
    ):
        return preferred

    for y in range(config.grid_height):
        for x in range(config.grid_width):
            position = (x, y)

            if position not in occupied:
                return position

    raise RuntimeError(
        "No free cell is available for food spawning."
    )


# ==================================================
# Shared collision detection
# ==================================================

def get_collision_reason(
    state: SnakeState,
    new_head: Position,
    eating: bool,
    config: GameConfig = GAME_CONFIG,
) -> str | None:
    """Return the collision reason, if any."""

    if not is_inside_grid(new_head, config):
        return "wall_collision"

    if new_head in config.obstacles:
        return "obstacle_collision"

    # The tail vacates its position on non-growth moves.
    occupied_body = (
        state.body if eating else state.body[:-1]
    )

    if new_head in occupied_body:
        return "self_collision"

    return None


# ==================================================
# Detailed deterministic transition
# ==================================================

@dataclass(frozen=True, slots=True)
class TransitionResult:
    next_state: SnakeState
    reward: float
    terminated: bool
    termination_reason: str | None = None


def resolve_transition(
    state: SnakeState,
    action: int | Action,
    config: GameConfig = GAME_CONFIG,
    rewards: RewardConfig = REWARD_CONFIG,
) -> TransitionResult:
    """Shared transition implementation for planning and gameplay."""

    direction = turn_direction(
        state.direction,
        action,
    )

    score = state.apples_collected(
        config.initial_snake_length
    )

    # Winning states are absorbing.
    if score >= config.apples_to_win:
        return TransitionResult(
            next_state=state,
            reward=0.0,
            terminated=True,
            termination_reason="win",
        )

    new_head = next_head_position(
        state.head,
        direction,
    )

    eating = (
        state.food is not None
        and new_head == state.food
    )

    collision_reason = get_collision_reason(
        state,
        new_head,
        eating,
        config,
    )

    if collision_reason is not None:
        return TransitionResult(
            next_state=state,
            reward=rewards.collision,
            terminated=True,
            termination_reason=collision_reason,
        )

    if eating:
        new_body = (new_head,) + state.body
        new_score = score + 1

        if new_score >= config.apples_to_win:
            return TransitionResult(
                next_state=SnakeState(
                    body=new_body,
                    food=None,
                ),
                reward=rewards.final_apple,
                terminated=True,
                termination_reason="win",
            )

        new_food = spawn_food(
            new_body,
            new_score,
            config,
        )

        return TransitionResult(
            next_state=SnakeState(
                body=new_body,
                food=new_food,
            ),
            reward=rewards.food,
            terminated=False,
        )

    # Ordinary movement without growth.
    new_body = (new_head,) + state.body[:-1]

    return TransitionResult(
        next_state=SnakeState(
            body=new_body,
            food=state.food,
        ),
        reward=rewards.move,
        terminated=False,
    )


# ==================================================
# Public planning interface
# ==================================================

def transition(
    state: SnakeState,
    action: int | Action,
    config: GameConfig = GAME_CONFIG,
    rewards: RewardConfig = REWARD_CONFIG,
) -> tuple[SnakeState, float, bool]:
    """Return next state, reward and true-terminal flag.

    This function does not mutate a live environment,
    use randomness, or apply an episode step limit.
    """

    result = resolve_transition(
        state,
        action,
        config,
        rewards,
    )

    return (
        result.next_state,
        result.reward,
        result.terminated,
    )
