"""Core deterministic mechanics for the Snake game."""

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


@dataclass(frozen=True, slots=True)
class SnakeState:
    """Immutable and hashable representation of the game."""

    body: tuple[Position, ...]
    direction: Direction
    food: Position | None
    score: int = 0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None

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
    def done(self) -> bool:
        return self.terminated or self.truncated


def turn_direction(
    direction: Direction,
    action: int | Action,
) -> Direction:
    """Convert a relative action into an absolute direction."""

    if isinstance(action, bool) or action not in (0, 1, 2):
        raise ValueError("Action must be 0, 1 or 2.")

    action = Action(action)

    if action == Action.LEFT:
        return Direction((int(direction) - 1) % 4)

    if action == Action.RIGHT:
        return Direction((int(direction) + 1) % 4)

    return direction


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
    """Check whether a position is inside the grid."""

    x, y = position

    return (
        0 <= x < config.grid_width
        and 0 <= y < config.grid_height
    )


def spawn_food(
    body: tuple[Position, ...],
    score: int,
    config: GameConfig = GAME_CONFIG,
) -> Position | None:
    """Spawn food using the fixed sequence and deterministic fallback."""

    if score >= config.apples_to_win:
        return None

    occupied = set(body) | set(config.obstacles)

    preferred = config.food_sequence[score]

    if (
        is_inside_grid(preferred, config)
        and preferred not in occupied
    ):
        return preferred

    # Row-major fallback:
    # (0, 0), (1, 0), ..., (4, 4).
    for y in range(config.grid_height):
        for x in range(config.grid_width):
            position = (x, y)

            if position not in occupied:
                return position

    raise RuntimeError(
        "No free cell is available for food spawning."
    )


def transition(
    state: SnakeState,
    action: int | Action,
    config: GameConfig = GAME_CONFIG,
    rewards: RewardConfig = REWARD_CONFIG,
) -> tuple[SnakeState, float]:
    """Calculate a deterministic transition without mutating the state."""

    if state.done:
        raise RuntimeError(
            "Episode has ended. Reset the environment first."
        )

    new_direction = turn_direction(
        state.direction,
        action,
    )

    new_head = next_head_position(
        state.head,
        new_direction,
    )

    new_steps = state.steps + 1

    eating = (
        state.food is not None
        and new_head == state.food
    )

    collision_reason = None

    # Wall collision.
    if not is_inside_grid(new_head, config):
        collision_reason = "wall_collision"

    # Obstacle collision.
    elif new_head in config.obstacles:
        collision_reason = "obstacle_collision"

    else:
        # The tail vacates its cell during a normal move.
        # During growth, the entire body remains occupied.
        occupied_body = (
            state.body
            if eating
            else state.body[:-1]
        )

        if new_head in occupied_body:
            collision_reason = "self_collision"

    # Collision takes priority over the time limit.
    if collision_reason is not None:
        next_state = SnakeState(
            body=state.body,
            direction=state.direction,
            food=state.food,
            score=state.score,
            steps=new_steps,
            terminated=True,
            truncated=False,
            termination_reason=collision_reason,
        )

        return next_state, rewards.collision

    # Apply a legal movement.
    if eating:
        new_body = (new_head,) + state.body
        new_score = state.score + 1
        reward = rewards.food
    else:
        new_body = (new_head,) + state.body[:-1]
        new_score = state.score
        reward = rewards.move

    # Winning takes priority over the time limit.
    if new_score >= config.apples_to_win:
        next_state = SnakeState(
            body=new_body,
            direction=new_direction,
            food=None,
            score=new_score,
            steps=new_steps,
            terminated=True,
            truncated=False,
            termination_reason="win",
        )

        return next_state, rewards.final_apple

    # Spawn the next apple after growth.
    if eating:
        new_food = spawn_food(
            new_body,
            new_score,
            config,
        )
    else:
        new_food = state.food

    # Time limit does not change the movement reward.
    truncated = new_steps >= config.max_steps

    next_state = SnakeState(
        body=new_body,
        direction=new_direction,
        food=new_food,
        score=new_score,
        steps=new_steps,
        terminated=False,
        truncated=truncated,
        termination_reason=(
            "step_limit" if truncated else None
        ),
    )

    return next_state, reward
