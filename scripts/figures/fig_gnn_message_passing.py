import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, SEQUENTIAL_BLUE, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

apply_theme()
rng = np.random.default_rng(3)

G = nx.karate_club_graph()
n = G.number_of_nodes()
nodes = list(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(axis=1, keepdims=True)

# Row-normalized adjacency (with self-loops) implements the "average my
# neighbors' and my own representation" update that GCN-style mean
# aggregation performs at each round of message passing.
A_self = A + np.eye(n)
D_self = A_self.sum(axis=1, keepdims=True)
A_norm = A_self / D_self

# Initial node features: nothing learned yet, just a mix of a structural
# signal (degree) and random noise, so round 0 shows no meaningful
# clustering by community.
feat_dim = 8
degree_signal = np.repeat(deg, feat_dim // 2, axis=1)
noise_signal = rng.normal(size=(n, feat_dim - feat_dim // 2))
H0 = np.concatenate([degree_signal / degree_signal.max(), noise_signal], axis=1)

H1 = A_norm @ H0
H2 = A_norm @ H1

# Ground-truth club split (Mr. Hi vs. Officer), used only to color nodes so
# the reader can see representations of same-community nodes converging —
# the GNN itself never sees this label.
club = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in nodes])
node_colors = [CATEGORICAL[0] if c == 0 else CATEGORICAL[1] for c in club]

pos = nx.spring_layout(G, seed=7)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
titles = [
    "Round 0: raw features (no propagation)",
    "Round 1: one hop of mean aggregation",
    "Round 2: two hops — communities merging",
]

for ax, H, title in zip(axes, [H0, H1, H2], titles):
    # PCA is fit fresh per round on that round's representations, purely to
    # get a 2D size/shading readout of how spread out (or collapsed) the
    # feature space is -- graph *position* stays fixed across panels so the
    # only thing changing panel to panel is color/spread, not layout.
    pca = PCA(n_components=1, random_state=0)
    score = pca.fit_transform(H).ravel()
    score_norm = (score - score.min()) / (np.ptp(score) + 1e-12)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRIDLINE, width=1.0)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, edgecolors=INK,
        linewidths=[0.4 + 1.8 * s for s in score_norm],
        node_size=220, alpha=0.95,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")

# Report a numeric over-smoothing signal: average pairwise cosine similarity
# of node representations should rise sharply round to round.
def mean_cos_sim(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    sim = Hn @ Hn.T
    iu = np.triu_indices(n, k=1)
    return sim[iu].mean()

sims = [mean_cos_sim(H0), mean_cos_sim(H1), mean_cos_sim(H2)]
fig.suptitle(
    "Mean pairwise cosine similarity of node representations: "
    f"{sims[0]:.2f} -> {sims[1]:.2f} -> {sims[2]:.2f}  (rising = over-smoothing)",
    fontsize=10, color=INK_SECONDARY, y=1.02,
)

handles = [
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORICAL[0],
               markeredgecolor=INK, markersize=9, label="Mr. Hi's faction"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORICAL[1],
               markeredgecolor=INK, markersize=9, label="Officer's faction"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=False)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "gnn_message_passing.png"
savefig(fig, str(out))
