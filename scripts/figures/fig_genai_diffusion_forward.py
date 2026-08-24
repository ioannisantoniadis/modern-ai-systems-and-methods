"""Forward diffusion process on a 2D toy distribution.

Samples points from a two-rings distribution, then actually runs the
standard variance-preserving forward noising process
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
for a linear beta schedule, and plots snapshots at increasing t so the reader
can see the clean shape smoothly dissolve into isotropic Gaussian noise. No
learned reverse model is trained here -- the point is to make the forward
corruption process itself concrete, since the reverse (denoising) process is
defined as learning to invert exactly this.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)


def sample_two_rings(n, r1=1.0, r2=2.4, noise=0.06):
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    ring1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    ring2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
    x0 = np.concatenate([ring1, ring2], axis=0)
    x0 += rng.normal(0, noise, size=x0.shape)
    rng.shuffle(x0)
    return x0


n_points = 1200
x0 = sample_two_rings(n_points)

# Linear beta schedule, standard DDPM convention.
T = 200
beta_start, beta_end = 1e-4, 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)


def forward_noise(x0, t_index):
    """Closed-form q(x_t | x_0) sample at diffusion step t_index (0-based)."""
    a_bar = alpha_bars[t_index]
    eps = rng.normal(size=x0.shape)
    return np.sqrt(a_bar) * x0 + np.sqrt(1 - a_bar) * eps


snapshot_steps = [0, 5, 15, 40, 90, 199]

apply_theme()
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 2.9), sharex=True, sharey=True)

lim = 4.2
for ax, t in zip(axes, snapshot_steps):
    xt = x0.copy() if t == 0 else forward_noise(x0, t)
    ax.scatter(xt[:, 0], xt[:, 1], s=4, alpha=0.55, color=CATEGORICAL[0], linewidths=0)
    a_bar = alpha_bars[t]
    ax.set_title(f"$t={t}$\n" + r"$\bar\alpha_t=$" + f"{a_bar:.2f}", fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRIDLINE)

fig.suptitle(
    "Forward diffusion: $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$ "
    "turns two rings into isotropic noise",
    y=1.1, fontsize=12.5, color=INK,
)
fig.text(
    0.5, -0.06,
    "Each panel is the same 1,200 points at a later diffusion step t of T=200, under a linear "
    r"$\beta_t$ schedule. A model learns to reverse exactly this corruption, one step at a time.",
    ha="center", fontsize=9.5, color=INK_SECONDARY, wrap=True,
)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "genai_diffusion_forward.png"
savefig(fig, str(out))
