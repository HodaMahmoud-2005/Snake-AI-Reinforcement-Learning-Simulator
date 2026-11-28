# Deterministic Transition Interface

## 1. Objective

Provide a deterministic transition model shared by the live Snake environment and model-based reinforcement-learning algorithms, particularly Value Iteration.

## 2. Planning state

The planning state is represented as:

S = (B, F)

Where:

- B is an immutable ordered tuple of snake-body coordinates, from head to tail.
- F is the current food position or None after winning.

The state is implemented using a frozen Python dataclass.

The heading is derived from the difference between the head and neck coordinates.

The number of apples collected is derived from the current body length minus the configured initial snake length.

Elapsed steps, accumulated reward, termination reason and truncation flags are not stored in the planning state.

## 3. Action space

| Action | Meaning |
|---|---|
| 0 | Relative left |
| 1 | Straight |
| 2 | Relative right |

## 4. Transition interface

The public interface is:

`transition(state, action) -> (next_state, reward, terminated)`

The transition function is deterministic, does not use randomness and does not mutate a live environment.

Identical valid states and actions produce identical outcomes when evaluated under the same configuration.

## 5. Shared transition rules

The live environment and planning interface use the same internal transition implementation.

The shared rules include:

- Relative movement.
- Wall collision.
- Obstacle collision.
- Self collision.
- Legal movement into the vacating tail cell.
- Body growth.
- Deterministic food placement.
- Shared reward values.
- Win termination.

## 6. Deterministic food placement

The next preferred food position is selected using the number of apples collected.

If the preferred cell is occupied by the snake or an obstacle, the first available cell in row-major order is selected.

No random food placement is used.

## 7. Terminal transitions

A transition is truly terminal when the snake collides or collects the final apple.

Collision transitions return the last valid board configuration, the collision reward and a true-terminal flag.

The terminal flag must be respected by planning algorithms. No future value is added following a terminal transition.

Winning states are absorbing and return zero additional reward if evaluated again.

## 8. External truncation

The live environment maintains elapsed steps outside the planning state.

Episodes are truncated at 60 steps unless a true-terminal event occurs first.

Truncation does not introduce an additional reward penalty.

## 9. Uncapped discounted MDP

Value Iteration solves the uncapped discounted MDP using gamma = 0.95.

The planning transition model does not apply the 60-step limit.

The live environment applies the external step limit during training and evaluation.

Therefore, the planning objective is not identical to an explicitly time-dependent finite-horizon objective.

This distinction must be acknowledged when comparing reinforcement-learning algorithms.

## 10. Verification

Automated tests verify:

- State immutability and hashability.
- Heading and score derivation.
- Deterministic transition outcomes.
- Absence of live-environment mutation.
- Shared collision and reward rules.
- Deterministic food spawning.
- True-terminal handling.
- External step-limit truncation.
- Complete winning episodes.