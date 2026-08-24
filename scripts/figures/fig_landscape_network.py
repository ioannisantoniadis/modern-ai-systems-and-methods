"""Force-directed graph of how the 18 topic chapters connect to each other.

Real computation: networkx builds the graph and lays it out with a
spring/force-directed algorithm (Fruchterman-Reingold) from the edge
structure below, rather than a hand-positioned diagram. Node color marks
which of the six narrative parts a chapter belongs to (matching docs/_quarto.yml).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, MUTED, GRIDLINE

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

# node -> (label, part index)
NODES = {
    "SL":   ("Supervised\nLearning", 0),
    "TE":   ("Trees &\nEnsembles", 0),
    "PM":   ("Probabilistic\nModeling", 1),
    "GLV":  ("Graphical\nModels", 1),
    "SAI":  ("Sampling &\nInference", 1),
    "REP":  ("Representation\nLearning", 2),
    "DL":   ("Deep\nLearning", 2),
    "SEQ":  ("Sequence\nModels", 2),
    "GNN":  ("Graph Neural\nNetworks", 2),
    "RLB":  ("RL &\nBandits", 3),
    "IRR":  ("Retrieval &\nRanking", 3),
    "GEN":  ("Generative\nAI", 3),
    "AGT":  ("AI Agents\n& Tools", 3),
    "EVAL": ("Evaluation &\nBenchmarking", 4),
    "CI":   ("Causal\nInference", 4),
    "CP":   ("Conformal\nPrediction", 4),
    "MLS":  ("ML Systems\n& MLOps", 5),
    "RAI":  ("Responsible\nAI", 5),
}

EDGES = [
    ("SL", "TE"), ("SL", "PM"), ("SL", "DL"), ("SL", "IRR"), ("SL", "CI"),
    ("TE", "MLS"), ("TE", "IRR"), ("TE", "REP"),
    ("PM", "GLV"), ("PM", "SAI"), ("PM", "GEN"), ("PM", "RLB"), ("PM", "CP"),
    ("GLV", "SEQ"), ("GLV", "IRR"), ("GLV", "GEN"), ("GLV", "REP"),
    ("SAI", "RLB"), ("SAI", "GEN"), ("SAI", "REP"),
    ("REP", "DL"), ("REP", "IRR"), ("REP", "GEN"), ("REP", "GNN"),
    ("DL", "SEQ"), ("DL", "GEN"), ("DL", "IRR"), ("DL", "RLB"),
    ("DL", "MLS"), ("DL", "GNN"),
    ("SEQ", "RLB"), ("SEQ", "IRR"), ("SEQ", "GEN"),
    ("GNN", "IRR"), ("GNN", "SEQ"),
    ("RLB", "IRR"), ("RLB", "GEN"), ("RLB", "CI"),
    ("IRR", "GEN"), ("IRR", "CI"), ("IRR", "MLS"), ("IRR", "EVAL"),
    ("GEN", "MLS"), ("GEN", "RAI"), ("GEN", "AGT"), ("GEN", "EVAL"),
    ("AGT", "EVAL"), ("AGT", "MLS"), ("AGT", "RAI"),
    ("EVAL", "CI"), ("EVAL", "MLS"),
    ("CI", "MLS"), ("CI", "RAI"),
    ("CP", "MLS"), ("CP", "RAI"),
    ("MLS", "RAI"),
]

PART_NAMES = [
    "Prediction from labeled data",
    "Uncertainty and hidden structure",
    "Representations and function approximation",
    "Decisions, selection, and generation",
    "Knowing whether it works",
    "Operating it in the world",
]


def main() -> None:
    apply_theme()

    g = nx.Graph()
    for node, (label, part) in NODES.items():
        g.add_node(node, label=label, part=part)
    g.add_edges_from(EDGES)

    pos = nx.spring_layout(g, k=1.0, iterations=300, seed=7)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    fig.subplots_adjust(top=0.93, bottom=0.14)

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRIDLINE, width=1.1, alpha=0.9)

    legend_handles = []
    for part_idx, part_name in enumerate(PART_NAMES):
        nodelist = [n for n, d in g.nodes(data=True) if d["part"] == part_idx]
        nx.draw_networkx_nodes(
            g, pos, ax=ax, nodelist=nodelist,
            node_color=CATEGORICAL[part_idx % len(CATEGORICAL)],
            node_size=3400, alpha=0.95, linewidths=1.4, edgecolors="white",
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=CATEGORICAL[part_idx % len(CATEGORICAL)],
                markeredgecolor="white", markersize=11, label=part_name,
            )
        )

    labels = {n: d["label"] for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(
        g, pos, labels=labels, ax=ax, font_size=7.4, font_color="white",
        font_weight="bold", verticalalignment="center",
    )

    ax.set_title(
        "How the chapters connect — edges follow shared concepts and data flow, "
        "node position from a force-directed layout on that structure",
        fontsize=12, color=INK, pad=16,
    )
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=8.8, frameon=False, labelcolor=MUTED,
        bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
        columnspacing=1.4, handletextpad=0.6,
    )
    ax.set_axis_off()
    ax.margins(0.1)

    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "landscape-network.png"
    fig.savefig(str(out), dpi=200, facecolor="#fcfcfb", bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
