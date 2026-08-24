import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(7)

# Simulated setting: a new drug's dose (treatment intensity) vs. a recovery
# score, confounded by patient severity. Sicker patients both get higher
# doses (because clinicians dose up for severe cases) and recover less
# fully, no matter what dose they get. Within any single severity band, the
# dose actually helps recovery. Pooled across bands, the dose looks harmful
# because it's a proxy for who was sick to begin with.

GROUPS = [
    {"name": "Mild severity",     "severity": 0.0, "base": 82, "n": 90},
    {"name": "Moderate severity", "severity": 1.0, "base": 62, "n": 90},
    {"name": "Severe",            "severity": 2.0, "base": 40, "n": 90},
]

TRUE_DOSE_EFFECT = 6.0  # within-group: higher dose -> better recovery

rows = []
for g in GROUPS:
    # Dose (0-4) correlates with severity: sicker patients get dosed higher.
    dose = np.clip(
        rng.normal(1.0 + 0.9 * g["severity"], 0.7, size=g["n"]), 0, 4
    )
    noise = rng.normal(0, 6.0, size=g["n"])
    # g["base"] already encodes the severity-driven outcome gap (82/62/40).
    recovery = g["base"] + TRUE_DOSE_EFFECT * dose + noise
    rows.append((dose, recovery, g["name"]))

all_dose = np.concatenate([r[0] for r in rows])
all_recovery = np.concatenate([r[1] for r in rows])
group_id = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rows)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

# Left panel: pooled data, ignoring severity entirely.
ax = axes[0]
ax.scatter(all_dose, all_recovery, s=16, color=CATEGORICAL[7], alpha=0.55,
           edgecolor="none")
pooled_fit = np.polyfit(all_dose, all_recovery, 1)
xs = np.linspace(all_dose.min(), all_dose.max(), 100)
ax.plot(xs, np.polyval(pooled_fit, xs), color=INK, linewidth=2.4)
ax.set_title(
    f"Pooled: dose looks harmful\n(slope = {pooled_fit[0]:+.1f} per dose unit)",
    fontsize=11,
)
ax.set_xlabel("dose")
ax.set_ylabel("recovery score")

# Right panel: same points, colored by true severity group, with per-group fits.
ax = axes[1]
for i, (dose, recovery, name) in enumerate(rows):
    color = CATEGORICAL[i]
    ax.scatter(dose, recovery, s=16, color=color, alpha=0.65,
               edgecolor="none", label=name)
    fit = np.polyfit(dose, recovery, 1)
    xs_g = np.linspace(dose.min(), dose.max(), 50)
    ax.plot(xs_g, np.polyval(fit, xs_g), color=color, linewidth=2.4)
    ax.annotate(f"slope {fit[0]:+.1f}", xy=(xs_g[-1], np.polyval(fit, xs_g[-1])),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=color, va="center")

ax.set_title("Within each severity group: dose actually helps", fontsize=11)
ax.set_xlabel("dose")
ax.legend(loc="lower left", fontsize=8.5)

fig.suptitle(
    "Simpson's paradox: a confounder (severity) reverses the apparent effect of dose",
    fontsize=12.5, y=1.03,
)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "causal_simpsons_paradox.png"
savefig(fig, str(out))
