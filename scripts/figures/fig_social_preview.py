"""Widescreen banner for GitHub's social preview slot (1280x640) and link-preview crops.

Same chapter-connection graph as fig_landscape_network.py (same NODES, EDGES and
PART_NAMES, imported, not copied), laid out for a wide banner: one column per part
of the book, in reading order, with each part's chapters stacked in its column.
A force-directed layout (the in-book figure's) overlaps nodes at this aspect
ratio; columns cannot overlap, use the whole canvas, and read like the book. Part
names head their columns, so every node is labeled directly and no legend is
needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, CATEGORICAL, INK, MUTED, GRIDLINE, SURFACE
from fig_landscape_network import NODES, EDGES, PART_NAMES

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

W, H = 12.8, 6.4                  # figure inches = data units
COL_X0, COL_X1 = 1.15, 11.65      # centers of first and last columns
ROW_TOP, ROW_BOTTOM = 2.55, 5.95  # band the pills occupy (y grows downward)
PILL_W, PILL_H = 1.62, 0.52


def part_color(part: int) -> str:
    return CATEGORICAL[part % len(CATEGORICAL)]


def layout() -> dict[str, tuple[float, float]]:
    """Column per part; chapters of a part spread evenly down the band, centered."""
    pos = {}
    n_parts = len(PART_NAMES)
    max_rows = max(sum(1 for _, p in NODES.values() if p == k) for k in range(n_parts))
    row_gap = (ROW_BOTTOM - ROW_TOP) / (max_rows - 1)
    for part in range(n_parts):
        x = COL_X0 + part * (COL_X1 - COL_X0) / (n_parts - 1)
        members = [n for n, (_, p) in NODES.items() if p == part]
        span = (len(members) - 1) * row_gap
        y0 = (ROW_TOP + ROW_BOTTOM) / 2 - span / 2
        for i, n in enumerate(members):
            pos[n] = (x, y0 + i * row_gap)
    return pos


def edge_path(a: tuple[float, float], b: tuple[float, float]) -> MplPath:
    """Cubic curve from a's right edge to b's left edge (or a side arc within a column)."""
    (xa, ya), (xb, yb) = a, b
    if abs(xa - xb) < 1e-9:  # same column: bow out to the left
        x = xa - PILL_W / 2
        bow = 0.35 + 0.08 * abs(yb - ya)
        verts = [(x, ya), (x - bow, ya), (x - bow, yb), (x, yb)]
    else:
        if xa > xb:
            (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
        x0, x1 = xa + PILL_W / 2, xb - PILL_W / 2
        dx = (x1 - x0) * 0.45
        verts = [(x0, ya), (x0 + dx, ya), (x1 - dx, yb), (x1, yb)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)


def main() -> None:
    apply_theme()
    pos = layout()

    # 12.8 x 6.4in @ 200dpi = 2560x1280px, 2x GitHub's recommended 1280x640.
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()

    for a, b in EDGES:
        ax.add_patch(PathPatch(edge_path(pos[a], pos[b]), facecolor="none",
                               edgecolor=GRIDLINE, linewidth=1.0, alpha=0.75, zorder=1))

    for node, (label, part) in NODES.items():
        x, y = pos[node]
        ax.add_patch(FancyBboxPatch(
            (x - PILL_W / 2, y - PILL_H / 2), PILL_W, PILL_H,
            boxstyle=f"round,pad=0,rounding_size={PILL_H / 2}",
            facecolor=part_color(part), edgecolor=SURFACE, linewidth=2, zorder=2,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color="white", linespacing=1.15, zorder=3)

    # Column headers: the part names, wrapped, with a color key dot.
    for part, name in enumerate(PART_NAMES):
        x = COL_X0 + part * (COL_X1 - COL_X0) / (len(PART_NAMES) - 1)
        words, lines, line = name.split(), [], ""
        for w in words:
            if len(line) + len(w) + 1 > 25 and line:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        ax.plot([x], [1.55], "o", color=part_color(part), markersize=7, zorder=3)
        ax.text(x, 1.7, "\n".join(lines), ha="center", va="top", fontsize=8.6,
                color=MUTED, linespacing=1.2)

    ax.text(0.35, 0.42, "Modern AI Systems and Methods", fontsize=22, fontweight="bold",
            color=INK, ha="left", va="top")
    ax.text(0.35, 1.0, "A field guide to how modern AI fits together — "
            "18 chapters in six parts, and how they connect",
            fontsize=11.5, color=MUTED, ha="left", va="top")

    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "social-preview.png"
    fig.savefig(str(out), dpi=200, facecolor=SURFACE, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
