"""Optional projection figure for embeddings; needs matplotlib.

One row per model, PCA on the left and t-SNE on the right. Points are coloured by
the predeclared word group, so the eye can check whether related words gather.
"""
from pathlib import Path

import numpy as np

from .projection import PROJECTION_GROUPS


def _place_labels(ax, xs, ys, words, fontsize=8):
    """Label every point without covering another label or point.

    Greedy, in pixels: the most crowded points choose first, each from candidate
    spots right, left, above, below and on the diagonals of its point, then the
    same ring twice as far (those get a thin leader line). Call after the layout
    is final, because it reads the axes' pixel transform."""
    to_pixels = ax.transData
    points = to_pixels.transform(np.column_stack((xs, ys)))
    height = fontsize * ax.figure.dpi / 72 * 1.25
    widths = np.array([len(w) * fontsize * ax.figure.dpi / 72 * 0.6 for w in words])
    frame = ax.get_window_extent()
    marker = 7                                       # half the marker size in pixels, with a margin
    crowd = [(np.hypot(*(points - p).T) < 80).sum() for p in points]
    placed = []                                      # (center x, center y, half width, half height)

    def free(cx, cy, hw, hh):
        if not (frame.x0 <= cx - hw and cx + hw <= frame.x1 and frame.y0 <= cy - hh and cy + hh <= frame.y1):
            return False
        if any(abs(cx - x) < hw + w and abs(cy - y) < hh + h for x, y, w, h in placed):
            return False
        return not any(abs(cx - px) < hw + marker and abs(cy - py) < hh + marker for px, py in points)

    inverse = to_pixels.inverted()
    for i in sorted(range(len(words)), key=lambda i: -crowd[i]):
        hw, hh = widths[i] / 2 + 1, height / 2
        px, py = points[i]
        spots = []
        for scale in (1, 2):
            reach_x, reach_y = (hw + marker) * scale, (hh + marker) * scale
            spots += [(px + reach_x, py), (px - reach_x, py), (px, py + reach_y), (px, py - reach_y),
                      (px + reach_x, py + reach_y), (px - reach_x, py + reach_y),
                      (px + reach_x, py - reach_y), (px - reach_x, py - reach_y)]
        choice = next((k for k, (cx, cy) in enumerate(spots) if free(cx, cy, hw, hh)), 0)
        cx, cy = spots[choice]
        placed.append((cx, cy, hw, hh))
        x, y = inverse.transform((cx, cy))
        ax.text(x, y, words[i], fontsize=fontsize, ha="center", va="center", color="0.15", zorder=4)
        if choice >= 8:                                # far ring: a leader from the point to the label's edge
            edge = (np.clip(px, cx - hw, cx + hw), np.clip(py, cy - hh, cy + hh))
            ax.plot(*zip(inverse.transform((px, py)), inverse.transform(edge)), color="0.6", linewidth=0.6,
                    zorder=2)


def plot_projections(projections: dict, path) -> Path:
    """projections: model name -> (point rows, quality rows) as returned by projection.project()."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("畫圖需要 matplotlib：python -m pip install matplotlib") from None

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    colors = dict(zip(PROJECTION_GROUPS, plt.get_cmap("tab10").colors))
    figure, axes = plt.subplots(len(projections), 2, figsize=(12, 5.2 * len(projections)), squeeze=False)
    for row, (name, (points, quality)) in zip(axes, projections.items()):
        for ax, method in zip(row, ("PCA", "t-SNE")):
            subset = [p for p in points if p["method"] == method]
            score = next(q for q in quality if q["method"] == method)
            for group, color in colors.items():
                members = [p for p in subset if p["group"] == group]
                ax.scatter([p["x"] for p in members], [p["y"] for p in members], s=28, color=color,
                           label=group, zorder=3)
            ax.margins(0.18)
            detail = (f"explained variance {score['explained_variance']:.0%}" if method == "PCA"
                      else f"KL {score['kl_divergence']:.2f}, perplexity {score['perplexity']:.1f}")
            ax.set_title(f"{name} · {method}\n{detail}, trustworthiness {score['trustworthiness']:.2f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(colors),
                  frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle("Word vectors in 2-D: L2-normalized, each panel fitted separately (axes are not comparable)",
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.canvas.draw()                              # fix the layout before labels read pixel positions
    for row, (points, _) in zip(axes, projections.values()):
        for ax, method in zip(row, ("PCA", "t-SNE")):
            subset = [p for p in points if p["method"] == method]
            _place_labels(ax, [p["x"] for p in subset], [p["y"] for p in subset], [p["word"] for p in subset])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
