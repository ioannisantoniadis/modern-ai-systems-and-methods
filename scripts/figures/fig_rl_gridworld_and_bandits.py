import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, SEQUENTIAL_BLUE, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# Left panel: value iteration on a small gridworld.
# ---------------------------------------------------------------------------
SIZE = 5
GOAL = (0, 4)
OBSTACLES = {(1, 1), (2, 1), (3, 3)}   # blocked, cost nothing to occupy: skip
PENALTIES = {(1, 3): -5.0, (3, 1): -5.0}  # costly "pit" cells
GAMMA = 0.9
STEP_REWARD = -1.0
GOAL_REWARD = 10.0

ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def step(state, action):
    if state == GOAL:
        return state, 0.0
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if not in_bounds(nr, nc) or (nr, nc) in OBSTACLES:
        nr, nc = state  # bump into wall/obstacle: stay put
    reward = GOAL_REWARD if (nr, nc) == GOAL else PENALTIES.get((nr, nc), STEP_REWARD)
    return (nr, nc), reward

V = np.zeros((SIZE, SIZE))
for _ in range(200):
    V_new = V.copy()
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) == GOAL or (r, c) in OBSTACLES:
                continue
            action_values = []
            for a in ACTIONS:
                (nr, nc), reward = step((r, c), a)
                action_values.append(reward + GAMMA * V[nr, nc])
            V_new[r, c] = max(action_values)
    if np.max(np.abs(V_new - V)) < 1e-6:
        V = V_new
        break
    V = V_new

# Greedy policy from the converged value function.
policy = {}
for r in range(SIZE):
    for c in range(SIZE):
        if (r, c) == GOAL or (r, c) in OBSTACLES:
            continue
        best_a, best_v = None, -np.inf
        for a in ACTIONS:
            (nr, nc), reward = step((r, c), a)
            v = reward + GAMMA * V[nr, nc]
            if v > best_v:
                best_v, best_a = v, a
        policy[(r, c)] = best_a

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax1, ax2 = axes

im = ax1.imshow(V, cmap="Blues", vmin=V.min(), vmax=V.max())
for r in range(SIZE):
    for c in range(SIZE):
        if (r, c) in OBSTACLES:
            ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=MUTED))
            continue
        if (r, c) == GOAL:
            ax1.text(c, r, "GOAL", ha="center", va="center", fontsize=9,
                      color="white", fontweight="bold")
            continue
        v_norm = (V[r, c] - V.min()) / (V.max() - V.min() + 1e-12)
        text_color = "white" if v_norm > 0.55 else INK_SECONDARY
        ax1.text(c, r - 0.28, f"{V[r, c]:.1f}", ha="center", va="center",
                  fontsize=7.5, color=text_color)
        dr, dc = ACTIONS[policy[(r, c)]]
        ax1.annotate("", xy=(c + dc * 0.32, r + dr * 0.32),
                     xytext=(c, r + 0.12),
                     arrowprops=dict(arrowstyle="->", color=CATEGORICAL[7], lw=1.8))
    if (r, 3) in PENALTIES or True:
        pass
for (pr, pc) in PENALTIES:
    ax1.add_patch(plt.Rectangle((pc - 0.5, pr - 0.5), 1, 1, fill=False,
                                  edgecolor=CATEGORICAL[7], linewidth=2))
ax1.set_xticks(range(SIZE))
ax1.set_yticks(range(SIZE))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title("Value iteration: converged $V^*$ and greedy policy")
ax1.grid(False)

# ---------------------------------------------------------------------------
# Right panel: cumulative regret for epsilon-greedy, UCB1, Thompson sampling.
# ---------------------------------------------------------------------------
TRUE_PROBS = np.array([0.20, 0.35, 0.50, 0.55, 0.65])
BEST_PROB = TRUE_PROBS.max()
N_ARMS = len(TRUE_PROBS)
HORIZON = 3000
N_RUNS = 200
EPSILON = 0.1
UCB_C = 0.5  # exploration constant; smaller than the textbook c=2 so the
             # confidence-bound exploration matches this arm-gap/horizon
             # combination rather than over-exploring throughout

def run_epsilon_greedy(rng):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    regret = np.zeros(HORIZON)
    for t in range(HORIZON):
        if rng.random() < EPSILON:
            a = rng.integers(N_ARMS)
        else:
            a = int(np.argmax(values))
        reward = 1.0 if rng.random() < TRUE_PROBS[a] else 0.0
        counts[a] += 1
        values[a] += (reward - values[a]) / counts[a]
        regret[t] = BEST_PROB - TRUE_PROBS[a]
    return np.cumsum(regret)

def run_ucb1(rng):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    regret = np.zeros(HORIZON)
    for t in range(HORIZON):
        if t < N_ARMS:
            a = t
        else:
            bonus = np.sqrt(UCB_C * np.log(t + 1) / counts)
            a = int(np.argmax(values + bonus))
        reward = 1.0 if rng.random() < TRUE_PROBS[a] else 0.0
        counts[a] += 1
        values[a] += (reward - values[a]) / counts[a]
        regret[t] = BEST_PROB - TRUE_PROBS[a]
    return np.cumsum(regret)

def run_thompson(rng):
    alpha = np.ones(N_ARMS)
    beta = np.ones(N_ARMS)
    regret = np.zeros(HORIZON)
    for t in range(HORIZON):
        samples = rng.beta(alpha, beta)
        a = int(np.argmax(samples))
        reward = 1.0 if rng.random() < TRUE_PROBS[a] else 0.0
        if reward > 0:
            alpha[a] += 1
        else:
            beta[a] += 1
        regret[t] = BEST_PROB - TRUE_PROBS[a]
    return np.cumsum(regret)

algos = [
    ("epsilon-greedy (eps=0.1)", run_epsilon_greedy, CATEGORICAL[0]),
    ("UCB1", run_ucb1, CATEGORICAL[2]),
    ("Thompson sampling", run_thompson, CATEGORICAL[1]),
]

for name, fn, color in algos:
    curves = np.zeros((N_RUNS, HORIZON))
    for run in range(N_RUNS):
        run_rng = np.random.default_rng(1000 + run)
        curves[run] = fn(run_rng)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    t = np.arange(1, HORIZON + 1)
    ax2.plot(t, mean_curve, color=color, linewidth=2.0, label=name)
    ax2.fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                      color=color, alpha=0.12)

ax2.set_xlabel("time step")
ax2.set_ylabel("cumulative regret")
ax2.set_title(f"Bandit regret, {N_ARMS} Bernoulli arms, {N_RUNS} runs")
ax2.legend(loc="upper left", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "rl_gridworld_and_bandits.png"
savefig(fig, str(out))
