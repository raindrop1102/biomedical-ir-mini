"""The experiment's chart: time by phase, block size, and scaling in time and memory."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

COLORS = {"BSBI": "#2b6cb0", "SPIMI": "#d9894f"}


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _font():
    for name in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Heiti TC"):
        if any(name == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plain(axis) -> None:
    """Log-scale ticks as plain numbers (1,000 rather than 10³)."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot(results: Path) -> Path:
    results = Path(results)
    _font()
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    blocks, scaling = _rows(results / "blocks.csv"), _rows(results / "scaling.csv")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0][0]
    for i, method in enumerate(("BSBI", "SPIMI")):
        bottom = 0.0
        for j, (phase, seconds) in enumerate(summary["main"][method]["phases"].items()):
            ax.barh(i, seconds, left=bottom, color=COLORS[method], alpha=1 - 0.15 * j, edgecolor="white")
            if seconds > 0.3:
                ax.text(bottom + seconds / 2, i, f"{phase}\n{seconds:.2f}", ha="center", va="center", fontsize=8,
                        color="white")
            bottom += seconds
    ax.set_yticks([0, 1], ["BSBI", "SPIMI"])
    ax.set_xlabel("秒")
    ax.set_title(f"建索引時間的組成（{summary['documents']:,} 篇，其中一次執行）")

    ax = axes[0][1]
    for method in ("BSBI", "SPIMI"):
        rows = [r for r in blocks if r["method"] == method]
        ax.plot([int(r["blocks"]) for r in rows], [float(r["seconds"]) for r in rows], "o-", color=COLORS[method],
                label=f"{method} 時間")
    ax.set_xscale("log")
    _plain(ax.xaxis)
    ax.set_xlabel("block 數（block 越小越多）")
    ax.set_ylabel("秒")
    twin = ax.twinx()
    for method in ("BSBI", "SPIMI"):
        rows = [r for r in blocks if r["method"] == method]
        twin.plot([int(r["blocks"]) for r in rows], [float(r["build_peak_mb"]) for r in rows], "s--",
                  color=COLORS[method], alpha=0.6, label=f"{method} 記憶體（建 block）")
    twin.set_ylabel("MB")
    lines = ax.get_legend_handles_labels()
    more = twin.get_legend_handles_labels()
    ax.legend(lines[0] + more[0], lines[1] + more[1], fontsize=8, loc="upper center")
    ax.set_title("block 大小：時間（實線）與記憶體（虛線）")

    ax = axes[1][0]
    for method in ("BSBI", "SPIMI"):
        rows = [r for r in scaling if r["method"] == method]
        ax.plot([int(r["n"]) for r in rows], [float(r["seconds"]) for r in rows], "o-", color=COLORS[method], label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain(ax.xaxis)
    _plain(ax.yaxis)
    ax.set_xlabel("文件數")
    ax.set_ylabel("秒")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    for method in ("BSBI", "SPIMI"):
        rows = [r for r in scaling if r["method"] == method]
        x = [int(r["n"]) for r in rows]
        ax.plot(x, [float(r["build_peak_mb"]) for r in rows], "o-", color=COLORS[method], label=f"{method} 建 block")
        ax.plot(x, [float(r["merge_peak_mb"]) for r in rows], "s--", color=COLORS[method], alpha=0.6,
                label=f"{method} 合併")
    ax.set_xscale("log")
    _plain(ax.xaxis)
    ax.set_xlabel("文件數")
    ax.set_ylabel("記憶體峰值 MB")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    axes[1][0].set_title("文件數增加：建索引時間")
    axes[1][1].set_title("文件數增加：記憶體峰值（建 block 對 合併）")
    figure.tight_layout()
    path = results / "indexing.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
