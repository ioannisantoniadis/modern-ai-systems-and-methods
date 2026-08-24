import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK

import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(42)

TRUE_P = 0.7
N_TOTAL = 100
flips = rng.binomial(1, TRUE_P, size=N_TOTAL)

# Weak uniform prior: Beta(1, 1).
a0, b0 = 1.0, 1.0
checkpoints = [0, 5, 20, 100]

xs = np.linspace(0, 1, 500)

fig, ax = plt.subplots(figsize=(7.5, 4.8))

for i, n in enumerate(checkpoints):
    heads = flips[:n].sum()
    tails = n - heads
    a_post, b_post = a0 + heads, b0 + tails
    density = beta_dist.pdf(xs, a_post, b_post)
    color = CATEGORICAL[i % len(CATEGORICAL)]
    label = f"n={n} ({heads} heads)" if n > 0 else "n=0 (prior)"
    ax.plot(xs, density, color=color, linewidth=2.3, label=label)
    ax.fill_between(xs, density, color=color, alpha=0.08)

ax.axvline(TRUE_P, color=INK, linestyle=":", linewidth=1.5,
           label=f"true p = {TRUE_P}")
ax.set_xlabel(r"$\theta$ (probability of heads)")
ax.set_ylabel("posterior density")
ax.set_title("Beta posterior concentrating around the true coin bias as data arrives")
ax.set_xlim(0, 1)
ax.legend(loc="upper left", fontsize=9.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "probabilistic_posterior_update.png"
savefig(fig, str(out))
