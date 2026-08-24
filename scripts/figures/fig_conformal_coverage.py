import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

apply_theme()

TARGET_COVERAGE = 0.90
ALPHA = 1 - TARGET_COVERAGE
N_TRIALS = 400
N_TRAIN, N_CAL, N_TEST = 300, 300, 300


def make_data(rng, n):
    x = rng.uniform(-3, 3, size=n)
    noise_scale = 0.6 + 0.5 * np.abs(x)  # heteroscedastic noise
    y = np.sin(x) * 2.0 + x * 0.3 + rng.normal(0, noise_scale, size=n)
    return x.reshape(-1, 1), y


def one_trial(seed):
    rng = np.random.default_rng(seed)
    x_train, y_train = make_data(rng, N_TRAIN)
    x_cal, y_cal = make_data(rng, N_CAL)
    x_test, y_test = make_data(rng, N_TEST)

    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=int(seed))
    model.fit(x_train, y_train)

    # Split conformal: nonconformity score is the absolute residual on the
    # held-out calibration set, never seen during training.
    cal_scores = np.abs(y_cal - model.predict(x_cal))
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - ALPHA)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(cal_scores, q_level)

    preds = model.predict(x_test)
    lower, upper = preds - q_hat, preds + q_hat
    covered = (y_test >= lower) & (y_test <= upper)
    return covered.mean()

coverages = np.array([one_trial(seed) for seed in range(N_TRIALS)])

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.hist(coverages, bins=28, color=CATEGORICAL[0], alpha=0.85, edgecolor="white",
        linewidth=0.6)
ax.axvline(TARGET_COVERAGE, color=INK, linestyle="--", linewidth=2.0,
           label=f"nominal target = {TARGET_COVERAGE:.0%}")
ax.axvline(coverages.mean(), color=CATEGORICAL[1], linestyle="-", linewidth=2.0,
           label=f"empirical mean = {coverages.mean():.3f}")
ax.set_xlabel("empirical coverage on fresh test set")
ax.set_ylabel(f"trials (of {N_TRIALS})")
ax.set_title(
    "Split-conformal coverage concentrates at the nominal 90% target\n"
    "across independent calibration + evaluation draws",
    fontsize=11.5,
)
ax.legend(loc="upper left", fontsize=9.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "conformal_coverage.png"
savefig(fig, str(out))
