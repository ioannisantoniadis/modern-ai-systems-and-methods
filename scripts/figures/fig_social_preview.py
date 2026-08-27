"""Widescreen banner: same chapter-connection graph as fig_landscape_network.py,
laid out for GitHub's social preview slot (1280x640) and link-preview crops
(LinkedIn, Slack, etc.), which fig_landscape_network.py's near-square figure
doesn't fit without cropping into the legend or nodes.

Reuses the exact same graph data, layout, and node styling as
fig_landscape_network.py (curved edges, degree-scaled node size) so the
repo has one consistent "mark" rather than two different visual identities —
this is a reflow, not a redesign. Keep this in sync by hand if
fig_landscape_network.py's styling changes again.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, CATEGORICAL, INK, MUTED, GRIDLINE, SURFACE
from fig_landscape_network import NODES, EDGES, PART_NAMES

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def main() -> None:
    apply_theme()

    g = nx.Graph()
    for node, (label, part) in NODES.items():
        g.add_node(node, label=label, part=part)
    g.add_edges_from(EDGES)

    pos = nx.spring_layout(g, k=1.7, iterations=500, seed=7)

    # 12.8 x 6.4in @ 200dpi = 2560x1280px, exactly 2x GitHub's recommended
    # 1280x640 social preview size (retina-sharp, GitHub downsamples).
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.12)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_pad = (max(xs) - min(xs)) * 0.18
    y_pad = (max(ys) - min(ys)) * 0.28
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color=GRIDLINE, width=0.8, alpha=0.55,
        connectionstyle="arc3,rad=0.12", arrows=True, arrowstyle="-",
        node_size=2700,
    )

    degrees = dict(g.degree())
    min_deg, max_deg = min(degrees.values()), max(degrees.values())

    def node_size(n: str) -> float:
        if max_deg == min_deg:
            return 2400.0
        span = (degrees[n] - min_deg) / (max_deg - min_deg)
        return 2300.0 + span * 850.0

    legend_handles = []
    for part_idx, part_name in enumerate(PART_NAMES):
        nodelist = [n for n, d in g.nodes(data=True) if d["part"] == part_idx]
        nx.draw_networkx_nodes(
            g, pos, ax=ax, nodelist=nodelist,
            node_color=CATEGORICAL[part_idx % len(CATEGORICAL)],
            node_size=[node_size(n) for n in nodelist],
            alpha=0.97, linewidths=1.3, edgecolors=SURFACE,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=CATEGORICAL[part_idx % len(CATEGORICAL)],
                markeredgecolor=SURFACE, markersize=9, label=part_name,
            )
        )

    labels = {n: d["label"] for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(
        g, pos, labels=labels, ax=ax, font_size=5.8, font_color="white",
        font_weight="bold", verticalalignment="center",
    )

    fig.text(
        0.02, 0.965, "Modern AI Systems and Methods",
        fontsize=22, fontweight="700", color=INK, ha="left", va="top",
    )
    fig.text(
        0.02, 0.905,
        "A field guide to how modern AI fits together — how the chapters connect",
        fontsize=11.5, color=MUTED, ha="left", va="top",
    )

    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6,
        fontsize=8.0, frameon=False, labelcolor=MUTED,
        bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
        columnspacing=1.1, handletextpad=0.5,
    )
    ax.set_axis_off()

    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "social-preview.png"
    fig.savefig(str(out), dpi=200, facecolor=SURFACE, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
