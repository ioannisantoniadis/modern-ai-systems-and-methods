import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# --- A small discrete HMM, defined by hand -----------------------------
# Hidden states: 0 = "calm", 1 = "volatile"
# Emissions: 0 = "small move", 1 = "medium move", 2 = "large move"
states = ["calm", "volatile"]
emissions = ["small", "medium", "large"]
n_states, n_emissions = 2, 3

pi = np.array([0.8, 0.2])                     # initial state distribution
A = np.array([[0.92, 0.08],                    # transition matrix
              [0.15, 0.85]])
B = np.array([[0.7, 0.25, 0.05],                # emission matrix P(obs | state)
              [0.1, 0.35, 0.55]])

T = 120

# --- Simulate a true hidden path and noisy observations -----------------
true_states = np.empty(T, dtype=int)
obs = np.empty(T, dtype=int)
true_states[0] = rng.choice(n_states, p=pi)
obs[0] = rng.choice(n_emissions, p=B[true_states[0]])
for t in range(1, T):
    true_states[t] = rng.choice(n_states, p=A[true_states[t - 1]])
    obs[t] = rng.choice(n_emissions, p=B[true_states[t]])


# --- Viterbi decoding (log-space DP) -------------------------------------
def viterbi(obs, pi, A, B):
    T = len(obs)
    n_states = A.shape[0]
    log_pi, log_A, log_B = np.log(pi), np.log(A), np.log(B)

    delta = np.full((T, n_states), -np.inf)
    psi = np.zeros((T, n_states), dtype=int)

    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        for s in range(n_states):
            scores = delta[t - 1] + log_A[:, s]
            psi[t, s] = np.argmax(scores)
            delta[t, s] = scores[psi[t, s]] + log_B[s, obs[t]]

    path = np.empty(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


decoded_states = viterbi(obs, pi, A, B)
accuracy = np.mean(decoded_states == true_states)
print(f"Viterbi decoding accuracy vs. true hidden path: {accuracy:.0%}")

# --- Plot: 3-row timeline -------------------------------------------------
apply_theme()
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1, 1.3]})

t_axis = np.arange(T)

ax = axes[0]
ax.step(t_axis, true_states, where="post", color=CATEGORICAL[0], linewidth=1.8)
ax.set_yticks([0, 1]); ax.set_yticklabels(states)
ax.set_ylim(-0.5, 1.5)
ax.set_title("true hidden state path")

ax = axes[1]
ax.step(t_axis, decoded_states, where="post", color=CATEGORICAL[1], linewidth=1.8)
ax.set_yticks([0, 1]); ax.set_yticklabels(states)
ax.set_ylim(-0.5, 1.5)
ax.set_title(f"Viterbi-decoded state path (accuracy {accuracy:.0%})")
mismatches = np.where(decoded_states != true_states)[0]
for m in mismatches:
    ax.axvspan(m - 0.5, m + 0.5, color=CATEGORICAL[7], alpha=0.18, linewidth=0)

ax = axes[2]
colors = [CATEGORICAL[3], CATEGORICAL[4], CATEGORICAL[7]]
for e in range(n_emissions):
    mask = obs == e
    ax.scatter(t_axis[mask], obs[mask], color=colors[e], s=16, label=emissions[e])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(emissions)
ax.set_ylim(-0.5, 2.5)
ax.set_title("noisy observed emissions")
ax.set_xlabel("time step")
ax.legend(loc="upper right", ncol=3, fontsize=8.5)

fig.suptitle("HMM Viterbi decoding: true state vs. decoded state vs. observations",
             y=1.02, fontsize=13, color=INK)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "sequence_hmm_viterbi.png"
savefig(fig, str(out))
