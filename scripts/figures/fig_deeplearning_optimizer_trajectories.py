import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, SEQUENTIAL_BLUE, INK

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Rosenbrock function: a classic non-convex test surface with a narrow,
# curved valley. Global minimum at (1, 1) with value 0.
def rosenbrock(x, y, a=1.0, b=100.0):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(x, y, a=1.0, b=100.0):
    dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    dy = 2 * b * (y - x ** 2)
    return np.array([dx, dy])


def run_sgd(start, lr=1e-3, n_steps=3000):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(n_steps):
        g = rosenbrock_grad(*p)
        p = p - lr * g
        path.append(p.copy())
    return np.array(path)


def run_momentum(start, lr=1e-3, beta=0.9, n_steps=3000):
    p = np.array(start, dtype=float)
    v = np.zeros(2)
    path = [p.copy()]
    for _ in range(n_steps):
        g = rosenbrock_grad(*p)
        v = beta * v + g
        p = p - lr * v
        path.append(p.copy())
    return np.array(path)


def run_adam(start, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=3000):
    p = np.array(start, dtype=float)
    m, v = np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for t in range(1, n_steps + 1):
        g = rosenbrock_grad(*p)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(p.copy())
    return np.array(path)


start = (-1.5, 2.0)
path_sgd = run_sgd(start)
path_mom = run_momentum(start)
path_adam = run_adam(start)

apply_theme()
fig, ax = plt.subplots(figsize=(7.5, 6))

xs = np.linspace(-2, 2, 400)
ys = np.linspace(-1, 3, 400)
XX, YY = np.meshgrid(xs, ys)
ZZ = rosenbrock(XX, YY)
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
ax.contourf(XX, YY, np.log1p(ZZ), levels=30, cmap=cmap)
ax.contour(XX, YY, np.log1p(ZZ), levels=30, colors="white", linewidths=0.3, alpha=0.5)

for path, label, color in [
    (path_sgd, "SGD", CATEGORICAL[1]),
    (path_mom, "SGD + momentum", CATEGORICAL[2]),
    (path_adam, "Adam", CATEGORICAL[7]),
]:
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, label=label, alpha=0.95)
    ax.plot(path[-1, 0], path[-1, 1], "o", color=color, markersize=5,
             markeredgecolor="white", markeredgewidth=0.8)

ax.plot(start[0], start[1], marker="*", color="white", markersize=16,
         markeredgecolor=INK, markeredgewidth=0.8, zorder=5, label="start")
ax.plot(1, 1, marker="X", color="white", markersize=11,
         markeredgecolor=INK, markeredgewidth=0.8, zorder=5, label="minimum (1, 1)")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Optimizer trajectories on the Rosenbrock surface")
ax.legend(loc="upper left", framealpha=0.9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "deeplearning_optimizer_trajectories.png"
savefig(fig, str(out))
