"""Reliability (calibration) diagram for two synthetic binary classifiers.

Generates predicted probabilities and true binary outcomes for two models
that share the same underlying "true" event probability per example:
  - a well-calibrated model, whose predicted probability tracks the true
    event frequency closely, with only sampling noise;
  - an overconfident model, whose predicted probabilities are pushed toward
    0/1 relative to the true frequency (e.g. by applying a sharpening
    transform), the classic failure mode of a model optimized purely for
    accuracy or trained with label smoothing removed / temperature too low.

Both are then binned by predicted probability and the observed empirical
frequency of the positive class is plotted against the bin's mean predicted
probability -- the standard reliability diagram -- alongside the y=x
perfect-calibration reference line. Brier score and expected calibration
error (ECE) are computed and shown in the legend.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

n = 4000

# "True" underlying event probability for each example, drawn so that
# probabilities span the full range with more mass near the middle (a
# realistic mix of easy and hard cases).
true_p = rng.beta(2.0, 2.0, size=n)
y = rng.binomial(1, true_p)


def sharpen(p, power=3.0):
    """Push probabilities toward 0/1: the overconfident model's miscalibration."""
    return p**power / (p**power + (1 - p) ** power)


# Well-calibrated model: true probability plus a little independent noise,
# clipped to (0, 1) -- still tracks the true frequency on average.
well_calibrated = np.clip(true_p + rng.normal(0, 0.04, size=n), 1e-3, 1 - 1e-3)

# Overconfident model: same underlying signal, but sharpened toward the
# extremes, then a touch of noise so it isn't a deterministic function.
overconfident = np.clip(sharpen(true_p) + rng.normal(0, 0.03, size=n), 1e-3, 1 - 1e-3)


def reliability_curve(p_pred, y_true, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_pred, bin_edges[1:-1], right=True)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(p_pred[mask].mean())
        obs_freq.append(y_true[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def brier_score(p_pred, y_true):
    return np.mean((p_pred - y_true) ** 2)


def expected_calibration_error(p_pred, y_true, n_bins=10):
    mean_pred, obs_freq, counts = reliability_curve(p_pred, y_true, n_bins)
    weights = counts / counts.sum()
    return np.sum(weights * np.abs(obs_freq - mean_pred))


models = {
    "well-calibrated model": (well_calibrated, CATEGORICAL[0]),
    "overconfident model": (overconfident, CATEGORICAL[1]),
}

apply_theme()
fig, ax = plt.subplots(figsize=(6.4, 6.0))

ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.4,
        label="perfect calibration", zorder=1)

for name, (p_pred, color) in models.items():
    mean_pred, obs_freq, counts = reliability_curve(p_pred, y, n_bins=10)
    bs = brier_score(p_pred, y)
    ece = expected_calibration_error(p_pred, y, n_bins=10)
    ax.plot(mean_pred, obs_freq, marker="o", markersize=6, linewidth=2,
             color=color, zorder=3,
             label=f"{name}\n(Brier={bs:.3f}, ECE={ece:.3f})")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("observed positive frequency (bin)")
ax.set_title("Reliability diagram: predicted probability vs. observed frequency")
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal")

fig.text(
    0.5, -0.02,
    "Both models see the same 4,000 synthetic examples; the overconfident model applies the\n"
    "same underlying signal but pushes probabilities toward 0 and 1, so its curve bows away from the diagonal.",
    ha="center", fontsize=8.8, color=INK_SECONDARY,
)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "evaluation_calibration.png"
savefig(fig, str(out))
