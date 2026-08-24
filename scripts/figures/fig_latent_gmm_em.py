import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

apply_theme()
rng = 5

# Three true Gaussian clusters with different shapes/spreads, achieved by
# generating isotropic blobs and then shearing/scaling two of them so the
# fitted covariance ellipses are visually distinct (not all circles).
X, y_true = make_blobs(
    n_samples=420,
    centers=[[-4, 2], [0, -2], [4, 3]],
    cluster_std=[1.0, 1.6, 0.8],
    random_state=rng,
)
shear = np.array([[1.0, 0.6], [0.0, 1.0]])
mask = y_true == 1
X[mask] = X[mask] @ shear.T

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=rng,
                        n_init=5)
gmm.fit(X)
labels = gmm.predict(X)

fig, ax = plt.subplots(figsize=(7.5, 6))

for k in range(3):
    mask_k = labels == k
    ax.scatter(X[mask_k, 0], X[mask_k, 1], s=16, color=CATEGORICAL[k],
               alpha=0.65, edgecolor="none", label=f"cluster {k+1}")

def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(mean, width, height, angle=angle, facecolor="none",
                   edgecolor=color, linewidth=2.2, linestyle="-", zorder=6)
    ax.add_patch(ell)

for k in range(3):
    draw_ellipse(ax, gmm.means_[k], gmm.covariances_[k], color=INK)
    draw_ellipse(ax, gmm.means_[k], gmm.covariances_[k], color=CATEGORICAL[k])
    ax.scatter(*gmm.means_[k], marker="x", s=110, color=INK, linewidth=2.5,
               zorder=7)

ax.set_title("Gaussian mixture fit via EM: posterior cluster assignments\nand fitted 2-sigma covariance ellipses")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(loc="lower right", fontsize=9.5)
ax.set_aspect("equal", adjustable="datalim")

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "latent_gmm_em.png"
savefig(fig, str(out))
