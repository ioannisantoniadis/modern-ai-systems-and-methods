import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, SEQUENTIAL_BLUE, INK, INK_SECONDARY, MUTED

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)

# --- Data: 3D Swiss roll manifold -----------------------------------------
X, color = make_swiss_roll(n_samples=1500, noise=0.15, random_state=0)
# Standardize so gradient descent on the autoencoder behaves well.
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Xs = (X - X_mean) / X_std

# --- PCA projection ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xs)

# --- Tiny autoencoder: 3 -> 8 -> 2 -> 8 -> 3, tanh activations, trained with
# plain gradient descent (manual forward/backward, no autograd) on
# reconstruction MSE. Small and fast by design.
n_in, n_hid, n_lat = 3, 8, 2


def init_layer(n_out, n_in_, rng_):
    limit = np.sqrt(6 / (n_in_ + n_out))
    return rng_.uniform(-limit, limit, size=(n_in_, n_out)), np.zeros(n_out)


rng_ae = np.random.default_rng(42)
W1, b1 = init_layer(n_hid, n_in, rng_ae)      # encoder: in -> hidden
W2, b2 = init_layer(n_lat, n_hid, rng_ae)     # encoder: hidden -> latent
W3, b3 = init_layer(n_hid, n_lat, rng_ae)     # decoder: latent -> hidden
W4, b4 = init_layer(n_in, n_hid, rng_ae)      # decoder: hidden -> out


def tanh(z):
    return np.tanh(z)


def dtanh(a):
    return 1 - a ** 2


def forward(x):
    z1 = x @ W1 + b1; a1 = tanh(z1)
    z2 = a1 @ W2 + b2; a2 = tanh(z2)          # bottleneck (latent code)
    z3 = a2 @ W3 + b3; a3 = tanh(z3)
    z4 = a3 @ W4 + b4                          # linear output layer
    return a1, a2, a3, z4


lr = 0.05
n_epochs = 800
batch_size = 128
n = Xs.shape[0]

for epoch in range(n_epochs):
    idx = rng_ae.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        x = Xs[batch]
        a1, a2, a3, out = forward(x)
        m = x.shape[0]

        d_out = (out - x) * (2.0 / m)          # dMSE/dz4
        dW4 = a3.T @ d_out; db4 = d_out.sum(axis=0)
        d_a3 = d_out @ W4.T
        d_z3 = d_a3 * dtanh(a3)
        dW3 = a2.T @ d_z3; db3 = d_z3.sum(axis=0)
        d_a2 = d_z3 @ W3.T
        d_z2 = d_a2 * dtanh(a2)
        dW2 = a1.T @ d_z2; db2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ W2.T
        d_z1 = d_a1 * dtanh(a1)
        dW1 = x.T @ d_z1; db1 = d_z1.sum(axis=0)

        W4 -= lr * dW4; b4 -= lr * db4
        W3 -= lr * dW3; b3 -= lr * db3
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

_, latent_2d, _, recon = forward(Xs)
mse = np.mean((recon - Xs) ** 2)
print(f"autoencoder final reconstruction MSE: {mse:.4f}")

# --- Plot: PCA projection vs. autoencoder bottleneck, colored by position
# along the roll (the ground-truth 1D coordinate `color` from make_swiss_roll).
apply_theme()
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap=cmap, s=10, linewidths=0)
axes[0].set_title("PCA (linear projection to 2D)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")

axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=color, cmap=cmap, s=10, linewidths=0)
axes[1].set_title("autoencoder bottleneck (2D latent code)")
axes[1].set_xlabel("latent dim 1")
axes[1].set_ylabel("latent dim 2")

cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("position along the roll", color=INK_SECONDARY)
cbar.ax.tick_params(colors=MUTED)

fig.suptitle("Unrolling a Swiss roll: linear PCA vs. a small nonlinear autoencoder",
             y=1.03, fontsize=13, color=INK)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "representation_pca_vs_autoencoder.png"
savefig(fig, str(out))
