"""Generate report figures for the Snake environment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import (
    Rectangle,
    Circle,
    Patch,
    FancyBboxPatch,
)

from configs.settings import (
    Action,
    GAME_CONFIG,
)

from environment import SnakeEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEAD_COLOR = "#1D4ED8"
BODY_COLOR = "#60A5FA"
FOOD_COLOR = "#DC2626"
OBSTACLE_COLOR = "#374151"
CELL_COLOR = "#E2EDCE"


def draw_grid(ax, state, show_future_food=False):
    """Draw a labeled grid from a planning state."""

    width = GAME_CONFIG.grid_width
    height = GAME_CONFIG.grid_height

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    ax.set_aspect("equal")

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))

    ax.xaxis.tick_top()

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    ax.xaxis.set_label_position("top")

    for y in range(height):
        for x in range(width):
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=CELL_COLOR,
                    edgecolor="#94A3B8",
                    linewidth=0.8,
                )
            )

    for x, y in GAME_CONFIG.obstacles:
        ax.add_patch(
            Rectangle(
                (x - 0.43, y - 0.43),
                0.86,
                0.86,
                facecolor=OBSTACLE_COLOR,
            )
        )

        ax.text(
            x, y, "X",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )

    for index, (x, y) in enumerate(state.body):
        color = (
            HEAD_COLOR if index == 0
            else BODY_COLOR
        )

        ax.add_patch(
            FancyBboxPatch(
                (x - 0.4, y - 0.4),
                0.8,
                0.8,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
            )
        )

        label = (
            "H" if index == 0
            else "T" if index == len(state.body) - 1
            else "B"
        )

        ax.text(
            x, y, label,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )

    if state.food is not None:
        x, y = state.food

        ax.add_patch(
            Circle(
                (x, y),
                radius=0.31,
                facecolor=FOOD_COLOR,
            )
        )

        ax.text(
            x, y, "F",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )

    if show_future_food:
        for index, (x, y) in enumerate(
            GAME_CONFIG.food_sequence[1:],
            start=2,
        ):
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=0.32,
                    fill=False,
                    edgecolor=FOOD_COLOR,
                    linestyle="--",
                    linewidth=2,
                )
            )

            ax.text(
                x, y, f"F{index}",
                color=FOOD_COLOR,
                ha="center",
                va="center",
                fontweight="bold",
            )


def generate_environment_map():
    """Generate the labeled initial environment map."""

    env = SnakeEnv()
    state, _ = env.reset(start_index=0)

    fig, ax = plt.subplots(figsize=(7, 7))

    draw_grid(
        ax,
        state,
        show_future_food=True,
    )

    # Derive heading from the first two body segments.
    hx, hy = state.head
    nx, ny = state.body[1]

    dx = hx - nx
    dy = hy - ny

    ax.annotate(
        "",
        xy=(hx + dx * 0.85, hy + dy * 0.85),
        xytext=(hx + dx * 0.45, hy + dy * 0.45),
        arrowprops={
            "arrowstyle": "->",
            "color": HEAD_COLOR,
            "lw": 2,
        },
    )

    fig.suptitle(
        "Initial Snake Environment",
        fontsize=17,
        fontweight="bold",
    )

    legend = [
        Patch(color=HEAD_COLOR, label="Snake head"),
        Patch(color=BODY_COLOR, label="Snake body"),
        Patch(color=OBSTACLE_COLOR, label="Obstacle"),
        Patch(color=FOOD_COLOR, label="Active food"),
        Patch(
            facecolor="none",
            edgecolor=FOOD_COLOR,
            linestyle="--",
            label="Future preferred food",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    fig.savefig(
        OUTPUT_DIR / "environment_map.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_state_transition():
    """Generate transitions for all three relative actions."""

    env = SnakeEnv()
    state, _ = env.reset(start_index=0)

    fig = plt.figure(figsize=(12, 8))

    layout = fig.add_gridspec(2, 3)

    initial_ax = fig.add_subplot(layout[0, 1])

    draw_grid(initial_ax, state)
    initial_ax.set_title("Initial state")

    for index, action in enumerate(Action):
        ax = fig.add_subplot(layout[1, index])

        next_state, reward, terminated = env.transition(
            state,
            action,
        )

        draw_grid(ax, next_state)

        ax.set_title(
            f"{action.name.title()} ({int(action)})\n"
            f"Reward: {reward:+.0f} | "
            f"Terminal: {terminated}",
            fontsize=11,
        )

        if terminated:
            for spine in ax.spines.values():
                spine.set_color(FOOD_COLOR)
                spine.set_linewidth(2)

    fig.suptitle(
        "Deterministic State Transitions",
        fontsize=18,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95),
        h_pad=3,
    )

    fig.savefig(
        OUTPUT_DIR / "state_transition.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_environment_flowchart():
    """Generate the environment execution flowchart."""

    fig, ax = plt.subplots(figsize=(9, 12))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    def node(x, y, text, color, width=4.2, height=0.8):
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor="#475569",
                linewidth=1,
            )
        )

        ax.text(
            x, y, text,
            ha="center",
            va="center",
            fontsize=10,
        )

    def arrow(start, end):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#475569",
                "lw": 1.3,
            },
        )

    node(5, 14, "Reset environment", "#DBEAFE")
    node(5, 12.6, "Get state and select action", "#E2E8F0")
    node(5, 11.2, "Calculate deterministic transition", "#DBEAFE")
    node(5, 9.8, "Collision?", "#FEF3C7", width=3)
    node(8.3, 8.4, "Collision\nReward -20", "#FEE2E2", width=2.7)

    node(5, 8.4, "Food collected?", "#FEF3C7", width=3)
    node(1.8, 7, "Normal movement\nReward -1", "#E2E8F0", width=3)
    node(5, 7, "Grow snake\nReward +10", "#DCFCE7", width=3)

    node(5, 5.6, "Third apple?", "#FEF3C7", width=3)
    node(8.3, 4.2, "Win\nReward +110", "#DCFCE7", width=2.7)
    node(5, 4.2, "Spawn next food", "#E2E8F0", width=3)

    node(
        5, 2.8,
        "Increment live steps\nUpdate total reward",
        "#DBEAFE",
    )

    node(
        5, 1.4,
        "Return terminated if collision/win;\n"
        "otherwise truncate at step limit",
        "#FEF3C7",
        width=7,
        height=0.95,
    )

    node(
        5, 0.2,
        "Return state, reward, flags and info",
        "#DBEAFE",
        width=6.5,
        height=0.55,
    )

    arrow((5, 13.6), (5, 13))
    arrow((5, 12.2), (5, 11.6))
    arrow((5, 10.8), (5, 10.2))

    arrow((6.5, 9.8), (8.3, 8.8))
    arrow((5, 9.4), (5, 8.8))

    ax.text(7.2, 9.5, "Yes", fontsize=9)
    ax.text(5.2, 9.1, "No", fontsize=9)

    arrow((3.5, 8.4), (1.8, 7.4))
    arrow((5, 8), (5, 7.4))

    ax.text(2.8, 8, "No", fontsize=9)
    ax.text(5.2, 7.7, "Yes", fontsize=9)

    arrow((5, 6.6), (5, 6))
    arrow((6.5, 5.6), (8.3, 4.6))
    arrow((5, 5.2), (5, 4.6))

    ax.text(7.2, 5.3, "Yes", fontsize=9)
    ax.text(5.2, 4.9, "No", fontsize=9)

    arrow((5, 3.8), (5, 3.2))

    # Merge normal movement, collision and win paths.
    ax.plot(
        [1.8, 1.8, 2.9],
        [6.6, 2.8, 2.8],
        color="#475569",
        linewidth=1.3,
    )

    ax.plot(
        [9.65, 9.85, 9.85, 7.1],
        [8.4, 8.4, 2.8, 2.8],
        color="#475569",
        linewidth=1.3,
    )

    arrow((8.3, 3.8), (7.1, 2.8))

    arrow((5, 2.4), (5, 1.9))
    arrow((5, 0.9), (5, 0.5))

    ax.set_title(
        "Snake Environment Execution Flowchart",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    fig.savefig(
        OUTPUT_DIR / "environment_flowchart.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    generate_environment_map()
    generate_state_transition()
    generate_environment_flowchart()

    print("Environment figures generated successfully.")

    for filename in (
        "environment_map.png",
        "state_transition.png",
        "environment_flowchart.png",
    ):
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
