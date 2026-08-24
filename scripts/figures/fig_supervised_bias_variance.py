import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(7)

# True underlying function: smooth, nonlinear, not a polynomial itself.
def true_fn(x):
    return np.sin(1.6 * x) + 0.35 * x

N = 25
NOISE_STD = 0.35
X_LO, X_HI = -3.0, 3.0

degrees = np.arange(1, 16)
n_resamples = 200

train_errs = np.zeros((n_resamples, len(degrees)))
test_errs = np.zeros((n_resamples, len(degrees)))

# Fixed, dense held-out set drawn from the same distribution, used to
# estimate true generalization error for every resample of the noisy
# training set.
x_test = np.linspace(X_LO, X_HI, 400)
y_test_clean = true_fn(x_test)

for r in range(n_resamples):
    x_train = rng.uniform(X_LO, X_HI, size=N)
    y_train = true_fn(x_train) + rng.normal(0, NOISE_STD, size=N)
    y_test = y_test_clean + rng.normal(0, NOISE_STD, size=x_test.shape)

    for j, d in enumerate(degrees):
        coeffs = np.polyfit(x_train, y_train, d)
        pred_train = np.polyval(coeffs, x_train)
        pred_test = np.polyval(coeffs, x_test)
        train_errs[r, j] = np.mean((pred_train - y_train) ** 2)
        test_errs[r, j] = np.mean((pred_test - y_test) ** 2)

# Median rather than mean across resamples: with only 25 points, a
# degree-14/15 polynomial occasionally fits an almost-singular system and
# produces a catastrophic (but rare) extrapolation error that would dominate
# a mean. The median shows the *typical* overfitting behavior, which is the
# point of the figure.
mean_train = np.median(train_errs, axis=0)
mean_test = np.median(test_errs, axis=0)
# Cap the axis so the informative low/mid-degree region (where the tradeoff
# actually plays out) isn't flattened by the runaway high-degree tail; the
# test curve is left to exit the top of the frame at high degree, which
# itself communicates the blow-up.
display_cap = max(mean_test[:9].max() * 4, 1.5)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
ax1, ax2 = axes

# Left panel: one example fit at low/right/high degree against the data.
x_train_ex = rng.uniform(X_LO, X_HI, size=N)
y_train_ex = true_fn(x_train_ex) + rng.normal(0, NOISE_STD, size=N)
x_plot = np.linspace(X_LO, X_HI, 400)
ax1.scatter(x_train_ex, y_train_ex, s=28, color=INK_SECONDARY, zorder=5,
            label="training points")
ax1.plot(x_plot, true_fn(x_plot), color="black", linestyle=":", linewidth=1.5,
          label="true function")
for d, color, label in [(1, CATEGORICAL[0], "degree 1 (underfit)"),
                          (4, CATEGORICAL[2], "degree 4 (good fit)"),
                          (14, CATEGORICAL[7], "degree 14 (overfit)")]:
    coeffs = np.polyfit(x_train_ex, y_train_ex, d)
    ax1.plot(x_plot, np.polyval(coeffs, x_plot), color=color, linewidth=2,
              label=label)
ax1.set_ylim(-4, 4)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Three fits on one training sample")
ax1.legend(loc="upper left", fontsize=8)

# Right panel: train vs test error vs degree, averaged over resamples.
ax2.plot(degrees, mean_train, color=CATEGORICAL[0], linewidth=2.2, marker="o",
          markersize=4, label="training error")
ax2.plot(degrees, mean_test, color=CATEGORICAL[7], linewidth=2.2, marker="o",
          markersize=4, label="test error (held-out)")
best_j = int(np.argmin(mean_test))
ax2.axvline(degrees[best_j], color=INK_SECONDARY, linestyle="--", linewidth=1,
             alpha=0.7)
ax2.annotate("sweet spot", xy=(degrees[best_j], mean_test[best_j]),
             xytext=(degrees[best_j] + 1.3, mean_test[best_j] + 0.55),
             fontsize=9, color=INK_SECONDARY,
             arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1))
ax2.set_ylim(0, display_cap)
ax2.set_xlabel("polynomial degree (model complexity)")
ax2.set_ylabel("mean squared error")
ax2.set_title(f"Bias-variance tradeoff ({n_resamples} resamples)")
ax2.legend(loc="upper center", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "supervised_bias_variance.png"
savefig(fig, str(out))
