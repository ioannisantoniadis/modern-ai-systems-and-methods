"""Force-directed graph of how the 18 topic chapters connect to each other.

Real computation: networkx builds the graph and lays it out with a
spring/force-directed algorithm (Fruchterman-Reingold) from the edge
structure below, rather than a hand-positioned diagram. Node color marks
which of the six narrative parts a chapter belongs to (matching
docs/_quarto.yml); node size scales with each chapter's degree in the graph,
so heavily-connected hub chapters read as visually larger.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, MUTED, GRIDLINE, SURFACE

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

    # Roomier spacing than the default spring layout: a higher k pushes
    # loosely-connected nodes apart so the center doesn't collapse into a
    # single overlapping mass, and more iterations lets it settle cleanly.
    pos = nx.spring_layout(g, k=1.7, iterations=500, seed=7)

    fig, ax = plt.subplots(figsize=(12, 10.5))
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.03, right=0.97)

    # FancyArrowPatch edges (needed for curvature) don't feed ax.dataLim the
    # way plain node scatter does, so autoscale alone under- or off-centers
    # the view. Set explicit, evenly-padded limits from the actual node
    # positions instead of trusting autoscale/margins.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_pad = (max(xs) - min(xs)) * 0.36
    y_pad = (max(ys) - min(ys)) * 0.28
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    # Curved edges (instead of straight chords) so overlapping connections
    # separate visually into distinct arcs rather than fusing into a single
    # gray smudge at each crossing. Degree scales node size so the handful
    # of hub chapters (e.g. Generative AI, ML Systems) read as hubs at a
    # glance, and everything else recedes a little.
    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color=GRIDLINE, width=0.9, alpha=0.55,
        connectionstyle="arc3,rad=0.12", arrows=True, arrowstyle="-",
        node_size=4200,
    )

    degrees = dict(g.degree())
    min_deg, max_deg = min(degrees.values()), max(degrees.values())

    def node_size(n: str) -> float:
        if max_deg == min_deg:
            return 3800.0
        span = (degrees[n] - min_deg) / (max_deg - min_deg)
        return 3600.0 + span * 1300.0

    legend_handles = []
    for part_idx, part_name in enumerate(PART_NAMES):
        nodelist = [n for n, d in g.nodes(data=True) if d["part"] == part_idx]
        nx.draw_networkx_nodes(
            g, pos, ax=ax, nodelist=nodelist,
            node_color=CATEGORICAL[part_idx % len(CATEGORICAL)],
            node_size=[node_size(n) for n in nodelist],
            alpha=0.97, linewidths=1.6, edgecolors=SURFACE,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=CATEGORICAL[part_idx % len(CATEGORICAL)],
                markeredgecolor=SURFACE, markersize=11, label=part_name,
            )
        )

    labels = {n: d["label"] for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(
        g, pos, labels=labels, ax=ax, font_size=7.0, font_color="white",
        font_weight="bold", verticalalignment="center",
    )

    ax.set_title(
        "How the chapters connect — edges follow shared concepts and data flow\n"
        "node position and size from a force-directed layout on that structure",
        fontsize=12, color=INK, pad=16,
    )
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=8.8, frameon=False, labelcolor=MUTED,
        bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
        columnspacing=1.6, handletextpad=0.6, handlelength=1.2,
    )
    ax.set_axis_off()

    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "landscape-network.png"
    fig.savefig(str(out), dpi=200, facecolor=SURFACE, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
