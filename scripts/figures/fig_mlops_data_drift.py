import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

apply_theme()
rng = np.random.default_rng(11)

N_WEEKS = 16
N_PER_WEEK = 1200
N_BINS = 30

# A feature (e.g. "average session length") whose mean and variance drift
# slowly starting mid-window — a slow covariate shift, not a sudden break.
REF_MEAN, REF_STD = 10.0, 2.0
weeks = np.arange(N_WEEKS)
drift_start = 6
mean_shift = np.where(weeks < drift_start, 0.0,
                       0.35 * (weeks - drift_start))
std_shift = np.where(weeks < drift_start, 0.0,
                      0.06 * (weeks - drift_start))

batches = []
for w in weeks:
    mu = REF_MEAN + mean_shift[w]
    sigma = REF_STD + std_shift[w]
    batches.append(rng.normal(mu, sigma, size=N_PER_WEEK))

reference = batches[0]  # week 0 is the training-time reference distribution


def psi(reference, current, n_bins=10):
    """Population Stability Index between two samples, using reference-based
    quantile bin edges (the standard PSI construction)."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_frac = np.clip(cur_counts / cur_counts.sum(), 1e-4, None)
    return np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac))


psi_values = np.array([psi(reference, b) for b in batches])
ks_values = np.array([ks_2samp(reference, b).statistic for b in batches])

PSI_ALERT = 0.2  # conventional "significant shift" threshold

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

# Left: ridgeline-style stacked densities for a subset of weeks.
ax = axes[0]
show_weeks = [0, 3, 6, 9, 12, 15]
xs = np.linspace(REF_MEAN - 4 * REF_STD, REF_MEAN + 4 * REF_STD + 6, 400)
offset_step = 1.0
for i, w in enumerate(show_weeks):
    b = batches[w]
    hist, edges = np.histogram(b, bins=N_BINS, range=(xs.min(), xs.max()), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    baseline = i * offset_step
    shade = 0.35 + 0.55 * (i / (len(show_weeks) - 1))
    ax.fill_between(centers, baseline, baseline + hist * 2.2, color=CATEGORICAL[0],
                     alpha=shade, linewidth=0)
    ax.plot(centers, baseline + hist * 2.2, color=INK, linewidth=0.8, alpha=0.6)
    ax.text(xs.min() - 0.3, baseline + 0.15, f"week {w}", fontsize=9,
            color=INK_SECONDARY, ha="right", va="bottom")

ax.set_yticks([])
ax.set_xlabel("feature value (e.g. avg. session length, minutes)")
ax.set_title("Feature distribution drifting week over week\n(reference = week 0)", fontsize=11)
ax.spines["left"].set_visible(False)

# Right: drift-detection statistics over time, with alert threshold.
ax = axes[1]
ax.plot(weeks, psi_values, color=CATEGORICAL[1], marker="o", markersize=4,
         linewidth=2.0, label="PSI vs. week-0 reference")
ax2 = ax.twinx()
ax2.plot(weeks, ks_values, color=CATEGORICAL[2], marker="s", markersize=4,
          linewidth=1.6, linestyle="--", label="KS statistic vs. week-0 reference")
ax2.set_ylabel("KS statistic", color=CATEGORICAL[2])
ax2.tick_params(axis="y", colors=CATEGORICAL[2])
ax2.grid(False)

ax.axhline(PSI_ALERT, color=INK, linestyle=":", linewidth=1.6)
ax.text(weeks[-1], PSI_ALERT, " alert threshold (PSI = 0.2)", fontsize=8.5,
        color=INK_SECONDARY, va="bottom", ha="right")
ax.axvline(drift_start, color=MUTED, linestyle="-", linewidth=1.0, alpha=0.6)
ax.text(drift_start, ax.get_ylim()[1] * 0.02, " drift begins", fontsize=8.5,
        color=MUTED, rotation=90, va="bottom")

ax.set_xlabel("week")
ax.set_ylabel("PSI", color=CATEGORICAL[1])
ax.tick_params(axis="y", colors=CATEGORICAL[1])
ax.set_title("Drift statistics cross the alert threshold\nonly after real drift begins", fontsize=11)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

fig.suptitle("Simulated data drift: a slowly shifting feature and the statistics that catch it",
             fontsize=12.5, y=1.03)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "mlops_data_drift.png"
savefig(fig, str(out))
