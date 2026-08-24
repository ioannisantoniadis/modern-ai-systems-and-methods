import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

apply_theme()
rng = np.random.default_rng(3)

# ---------------------------------------------------------------------------
# Left panel: fairness/accuracy tradeoff under per-group threshold adjustment.
# ---------------------------------------------------------------------------
# Construct a biased dataset: a sensitive attribute A in {0, 1} correlates
# with the label (group 1 has a lower base rate) AND with a feature the
# model uses, so a group-blind classifier reproduces the base-rate gap.
N = 4000
A = rng.binomial(1, 0.5, size=N)
skill = rng.normal(0, 1, size=N) + np.where(A == 1, -0.6, 0.6)  # biased signal
noise = rng.normal(0, 1, size=N)
logit = 0.9 * skill - 0.3 * A  # A also affects the label directly (historical bias)
p = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, p)

X = np.column_stack([skill, noise])
clf = LogisticRegression().fit(X, y)
scores = clf.predict_proba(X)[:, 1]


def group_metrics(threshold_0, threshold_1):
    pred = np.where(A == 0, scores >= threshold_0, scores >= threshold_1).astype(int)
    acc = (pred == y).mean()
    rate0 = pred[A == 0].mean()
    rate1 = pred[A == 1].mean()
    dp_gap = abs(rate0 - rate1)  # demographic parity difference
    return acc, dp_gap


base_threshold = 0.5
# Sweep the group-1 threshold down (positive discrimination toward the
# disadvantaged group) while holding group 0 fixed, tracing out the
# accuracy/fairness tradeoff frontier. Stop just above the crossover point
# where group 1's selection rate would overshoot group 0's — past that point
# the gap direction flips and grows again, which would fold the curve back
# on itself rather than tracing a single frontier.
group1_thresholds = np.linspace(0.24, 0.5, 40)
accs, gaps = [], []
for t1 in group1_thresholds:
    acc, gap = group_metrics(base_threshold, t1)
    accs.append(acc)
    gaps.append(gap)

# ---------------------------------------------------------------------------
# Right panel: differential-privacy noise-vs-utility curve.
# ---------------------------------------------------------------------------
# A simple aggregate statistic (mean of a bounded sensitive attribute) queried
# under the Laplace mechanism at varying epsilon. Sensitivity = (max-min)/n
# for a mean query over n bounded records in [0, 1].
n_records = 2000
true_values = rng.beta(2, 5, size=n_records)  # bounded in [0, 1]
true_mean = true_values.mean()
sensitivity = 1.0 / n_records

epsilons = np.geomspace(0.01, 10, 40)
n_repeats = 300
mean_abs_error = []
for eps in epsilons:
    scale = sensitivity / eps
    noisy = true_mean + rng.laplace(0, scale, size=n_repeats)
    mean_abs_error.append(np.mean(np.abs(noisy - true_mean)))
mean_abs_error = np.array(mean_abs_error)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))

ax = axes[0]
ax.plot(gaps, accs, color=CATEGORICAL[0], linewidth=2.2, marker="o", markersize=3.5)
ax.scatter([abs(group_metrics(0.5, 0.5)[1])], [group_metrics(0.5, 0.5)[0]],
           color=INK, s=55, zorder=5, marker="D",
           label="group-blind threshold (0.5 / 0.5)")
ax.set_xlabel("demographic parity gap  |P(ŷ=1|A=0) − P(ŷ=1|A=1)|")
ax.set_ylabel("overall accuracy")
ax.set_title("Threshold adjustment trades accuracy\nfor demographic parity", fontsize=11)
ax.legend(loc="lower left", fontsize=8.5)
ax.invert_xaxis()

ax = axes[1]
ax.plot(epsilons, mean_abs_error, color=CATEGORICAL[1], linewidth=2.2, marker="o",
        markersize=3.5)
ax.set_xscale("log")
ax.set_xlabel(r"privacy budget $\varepsilon$ (log scale; smaller = more private)")
ax.set_ylabel("mean absolute error vs. true statistic")
ax.set_title("Laplace-mechanism error grows sharply\nas the privacy budget shrinks", fontsize=11)
ax.axvspan(0.01, 0.3, color=CATEGORICAL[1], alpha=0.08)
ax.text(0.033, ax.get_ylim()[1] * 0.9, "strong\nprivacy", fontsize=8.5,
        color=INK_SECONDARY, ha="left", va="top")
ax.axvspan(3, 10, color=CATEGORICAL[0], alpha=0.06)
ax.text(6.5, ax.get_ylim()[1] * 0.9, "weak\nprivacy", fontsize=8.5,
        color=INK_SECONDARY, ha="right", va="top")

fig.suptitle(
    "Responsible AI has explicit knobs: fairness costs accuracy, privacy costs utility",
    fontsize=12.5, y=1.04,
)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "responsible_fairness_and_privacy.png"
savefig(fig, str(out))
