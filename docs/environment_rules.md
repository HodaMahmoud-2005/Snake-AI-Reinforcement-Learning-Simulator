# Snake AI — Environment Configuration

## 1. Game settings

| Parameter | Value |
|---|---|
| Grid size | 5 × 5 |
| Initial snake length | 2 |
| Apples required to win | 3 |
| Fixed obstacles | 2 |
| Maximum episode steps | 60 |

## 2. Coordinate system

Coordinates use the format (x, y).

The top-left cell is (0, 0), and the bottom-right cell is (4, 4).

The x-coordinate increases toward the right, and the y-coordinate increases downward.

## 3. Proposed start states

The initial start state is:

- Snake head: (1, 2)
- Snake tail: (0, 2)
- Heading: RIGHT
- Apples collected: 0
- Steps taken: 0

This start state is pending team approval. Additional start states must be validated before inclusion.

## 4. Fixed obstacles

- Obstacle 1: (1, 1)
- Obstacle 2: (3, 3)

Obstacles remain unchanged throughout each episode.

## 5. Action space

The environment uses three relative actions:

| Action | Meaning |
|---|---|
| 0 | Turn left |
| 1 | Continue straight |
| 2 | Turn right |

Direct reversal is not an available action.

## 6. Food sequence

The preferred food positions are:

1. (3, 0)
2. (4, 2)
3. (0, 4)

Only one apple is active at any time.

The next food position is selected after the current apple is collected.

## 7. Deterministic food fallback

If the preferred food position is occupied by the snake or an obstacle, scan the grid in row-major order.

Start at (0, 0), scan from left to right, then continue to the next row.

Select the first unoccupied cell.

The fallback must not use random selection.

The food sequence index advances when an apple is collected, regardless of whether the apple used its preferred or fallback position.

If no free cell exists before the required number of apples is collected, the environment must report an invalid or unsupported configuration.

## 8. Reward function

| Event | Reward |
|---|---:|
| Ordinary movement | -1 |
| Food collected | +10 |
| Collision | -20 |
| Additional win bonus | +100 |
| Final apple | +110 |

Only one event reward is applied per step, except that the final apple combines the food reward and win bonus.

## 9. Episode termination

An episode ends when:

- The snake collides with a wall.
- The snake collides with an obstacle.
- The snake collides with its own body.
- The snake collects three apples.
- The episode reaches 60 steps.

If the final apple is collected on step 60, the episode counts as a win.

Reaching the step limit does not introduce an additional reward penalty.

## 10. Reinforcement-learning hyperparameters

| Parameter | Default |
|---|---:|
| Gamma | 0.95 |
| Q-learning alpha | 0.1 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.05 |
| Epsilon decay | 0.9995 |
| REINFORCE alpha | 0.01 |
| Training episodes | 10000 |
| Value Iteration tolerance | 1e-5 |
| Default random seed | 42 |

## 11. Reproducibility

Random seeds must remain configurable.

All algorithms must use the same environment rules and reward configuration wherever technically appropriate.

The fixed obstacle layout, preferred food sequence, and deterministic food fallback support reproducible comparisons.

## 12. Approval status

The obstacle coordinates, initial start state, and preferred food sequence are proposed design choices pending team approval.