"""Optional chart of homework 4's experiment; needs matplotlib."""
from pathlib import Path

ORDER = ["A", "B-cbow", "B-skipgram", "mean-cbow", "mean-skipgram", "A+B-cbow", "A+B-skipgram"]


def plot_summary(summary: list[dict], path, metric: str = "nDCG@10"):
    """Per method, the metric on keyword queries (grey) next to paraphrase queries (blue),
    with values on the bars. `path` may be a file object."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("畫圖需要 matplotlib：python -m pip install matplotlib") from None

    value = {(r["method"], r["queries"]): float(r[metric]) for r in summary}
    methods = [m for m in ORDER if (m, "all") in value] + sorted({r["method"] for r in summary} - set(ORDER))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    figure, ax = plt.subplots(figsize=(9, 0.62 * len(methods) + 1.6))
    height = 0.36
    for offset, (scope, color) in enumerate((("keyword", "0.7"), ("paraphrase", "tab:blue"))):
        ys = [i + (offset - 0.5) * height for i in range(len(methods))]
        values = [value[m, scope] for m in methods]
        ax.barh(ys, values, height=height * 0.92, color=color, label=f"{scope} queries")
        for y, v in zip(ys, values):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color="0.25")
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(f"mean {metric}")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    others = [m for m in methods if m != "A"]
    best = {scope: max(value[m, scope] for m in others) for scope in ("keyword", "paraphrase")}
    ax.set_title(f"{metric} vs TF-IDF (A): paraphrase {value['A', 'paraphrase']:.2f} → best {best['paraphrase']:.2f}, "
                 f"keyword {value['A', 'keyword']:.2f} → best {best['keyword']:.2f}", fontsize=10)
    figure.tight_layout()
    if not hasattr(path, "write"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, format="png")
    plt.close(figure)
    return path
