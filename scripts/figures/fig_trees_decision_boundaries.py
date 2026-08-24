import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

apply_theme()
rng = 13

X, y = make_moons(n_samples=260, noise=0.28, random_state=rng)

models = [
    ("Single decision tree", DecisionTreeClassifier(random_state=rng)),
    ("Random forest (200 trees)",
     RandomForestClassifier(n_estimators=200, random_state=rng)),
    ("Gradient boosting (200 rounds)",
     GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=rng)),
]

cmap_bg = ListedColormap(["#eaf2fc", "#fbe4d9"])
point_colors = [CATEGORICAL[0], CATEGORICAL[1]]

x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)

for ax, (title, model) in zip(axes, models):
    model.fit(X, y)
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5], alpha=0.9)
    ax.contour(xx, yy, zz, levels=[0.5], colors=INK, linewidths=1.0)
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=18, color=point_colors[cls],
                   edgecolor="white", linewidth=0.4, zorder=5)
    acc = model.score(X, y)
    ax.set_title(f"{title}\ntrain accuracy {acc:.2f}", fontsize=10.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRIDLINE)

fig.suptitle("Decision boundaries: one tree vs. bagging vs. boosting on the same data",
             fontsize=12.5, y=1.02)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "trees_decision_boundaries.png"
savefig(fig, str(out))
