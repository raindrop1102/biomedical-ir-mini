"""Optional chart of the weighting experiment; needs matplotlib."""
from pathlib import Path

PANELS = ["nDCG@10", "MAP", "P@10", "MRR"]


def plot_summary(summary: list[dict], path, queries: int, highlight: str = "lnc.ltc"):
    """One panel per metric, schemes sorted by nDCG@10; the highlighted scheme in blue, the rest grey,
    values printed on the bars. `path` may be a file object."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("畫圖需要 matplotlib：python -m pip install matplotlib") from None

    rows = sorted(summary, key=lambda r: r["nDCG@10"])
    labels = [r["scheme"] for r in rows]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    figure, axes = plt.subplots(1, len(PANELS), figsize=(14, 0.42 * len(rows) + 1.6), sharey=True)
    for ax, metric in zip(axes, PANELS):
        values = [float(r[metric]) for r in rows]
        colors = ["tab:blue" if label == highlight else "0.72" for label in labels]
        ax.barh(labels, values, color=colors, height=0.68)
        for y, value in enumerate(values):
            ax.text(value + 0.01, y, f"{value:.3f}", va="center", fontsize=8, color="0.25")
        ax.set_xlim(0, 1.0)
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2)
    figure.suptitle(f"Term weighting on {queries} GLP-1 queries with MeSH relevance (mean per query; sorted by nDCG@10)",
                    fontsize=11)
    figure.tight_layout()
    if not hasattr(path, "write"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, format="png")
    plt.close(figure)
    return path
