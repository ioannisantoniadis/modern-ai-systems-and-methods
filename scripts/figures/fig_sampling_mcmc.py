import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# Target: mixture of two 1D Gaussians (bimodal), unnormalized density is fine
# for Metropolis-Hastings since the acceptance ratio only needs relative
# densities.
W1, MU1, SD1 = 0.4, -2.5, 0.6
W2, MU2, SD2 = 0.6, 3.0, 1.0


def target_density(x):
    def gauss(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return W1 * gauss(x, MU1, SD1) + W2 * gauss(x, MU2, SD2)


def metropolis_hastings(n_steps, proposal_sd, x0=0.0):
    samples = np.empty(n_steps)
    x_current = x0
    p_current = target_density(x_current)
    n_accept = 0
    for t in range(n_steps):
        x_proposal = x_current + rng.normal(0, proposal_sd)
        p_proposal = target_density(x_proposal)
        # Symmetric Gaussian proposal -> acceptance ratio is just the
        # density ratio (the proposal terms cancel).
        accept_prob = min(1.0, p_proposal / p_current) if p_current > 0 else 1.0
        if rng.uniform() < accept_prob:
            x_current, p_current = x_proposal, p_proposal
            n_accept += 1
        samples[t] = x_current
    return samples, n_accept / n_steps


n_steps = 12000
burn_in = 2000
samples, accept_rate = metropolis_hastings(n_steps, proposal_sd=1.5)
post_burn = samples[burn_in:]

apply_theme()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: trace plot showing mixing between the two modes.
ax = axes[0]
ax.plot(np.arange(n_steps), samples, color=CATEGORICAL[0], linewidth=0.5, alpha=0.85)
ax.axvline(burn_in, color=MUTED, linestyle="--", linewidth=1)
ax.text(burn_in + 150, samples.max() - 0.3, "burn-in ends", color=INK_SECONDARY, fontsize=9)
ax.set_xlabel("iteration")
ax.set_ylabel("sample value $x$")
ax.set_title(f"MH trace (acceptance rate {accept_rate:.0%})")

# Right: histogram of post-burn-in samples vs. true density.
ax = axes[1]
ax.hist(post_burn, bins=60, density=True, color=CATEGORICAL[0], alpha=0.55,
         edgecolor="white", linewidth=0.3, label="MH samples")
xs = np.linspace(-6, 7, 500)
ax.plot(xs, target_density(xs), color=CATEGORICAL[1], linewidth=2.2, label="true target $p(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel("density")
ax.set_title("post burn-in samples vs. true density")
ax.legend(loc="upper left")

fig.suptitle("Metropolis-Hastings sampling a bimodal target", y=1.02, fontsize=13, color=INK)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "sampling_mcmc.png"
savefig(fig, str(out))
