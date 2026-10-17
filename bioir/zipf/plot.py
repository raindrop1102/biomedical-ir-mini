"""Optional Zipf figure; the only part of bioir that needs matplotlib.

Three panels:
1. A-D rank-frequency curves on one log-log plot, with a k = 1 reference line
   through the baseline's top frequency (f(1) / rank).
2. The baseline condition with its full-range OLS line and the same reference.
3. The baseline's OLS residuals, log10(observed) − log10(fitted).
Panels 2 and 3 mark the head / middle / tail boundaries (10% and 90% of ranks).
"""
import math
from pathlib import Path

from .stats import CONDITIONS, ConditionStats


def plot_zipf(results: dict[str, ConditionStats], path, baseline: str = "B"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("畫圖需要 matplotlib：python -m pip install matplotlib") from None

    base = results[baseline]
    if not base.fit:
        raise ValueError(f"條件 {baseline} 的詞彙不足，無法擬合")
    ranks = [t.rank for t in base.terms]
    observed = [t.cf for t in base.terms]
    fitted = [10 ** (base.fit.intercept + base.fit.slope * math.log10(r)) for r in ranks]
    reference = [observed[0] / r for r in ranks]
    boundaries = [r.rank_start for r in base.regions[1:]]

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5})
    figure, (overlay, fit_ax, residual_ax) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    blues = plt.get_cmap("Blues")

    # 1. Conditions in one hue, light (A) to dark (D): they are cumulative steps of one pipeline.
    for i, (condition, stats) in enumerate(results.items()):
        emphasis = condition == baseline
        overlay.loglog([t.rank for t in stats.terms], [t.cf for t in stats.terms],
                       color=blues(0.35 + 0.18 * i), linewidth=2.2 if emphasis else 1.2,
                       label=f"{condition}{' (baseline)' if emphasis else ''}: k={stats.fit.exponent:.2f}"
                       if stats.fit else condition)
    overlay.loglog(ranks, reference, ":", color="0.45", label="k = 1 reference")
    overlay.set(title="Preprocessing conditions A–D", xlabel="rank", ylabel="collection frequency")
    overlay.legend(frameon=False, fontsize=8.5)

    # 2. Baseline and its fit.
    r2 = "n/a" if base.fit.r2 is None else f"{base.fit.r2:.3f}"
    fit_ax.loglog(ranks, observed, color="0.35", linewidth=1.4, label="observed")
    fit_ax.loglog(ranks, fitted, "--", color=blues(0.85), label=f"OLS: k={base.fit.exponent:.3f}, R²={r2}")
    fit_ax.loglog(ranks, reference, ":", color="0.45", label="k = 1 reference")
    fit_ax.set(title=f"Condition {baseline}: {CONDITIONS[baseline]}", xlabel="rank", ylabel="collection frequency")
    fit_ax.legend(frameon=False, fontsize=8.5)

    # 3. Residuals: systematic curvature means a high R² is not proof of a power law.
    residuals = [math.log10(o) - math.log10(f) for o, f in zip(observed, fitted)]
    residual_ax.semilogx(ranks, residuals, color="0.35", linewidth=1)
    residual_ax.axhline(0, color=blues(0.85), linestyle="--", linewidth=1)
    residual_ax.set(title=f"Condition {baseline}: OLS residuals", xlabel="rank",
                    ylabel="log10(observed) − log10(fitted)")

    for ax in (fit_ax, residual_ax):
        for rank in boundaries:
            ax.axvline(rank, color="0.75", linewidth=0.8)
        top = ax.get_ylim()[1]
        for region in base.regions:
            middle = math.sqrt(region.rank_start * region.rank_end)
            ax.text(middle, top, region.name, ha="center", va="top", fontsize=8, color="0.4")
    for ax in (overlay, fit_ax, residual_ax):
        ax.grid(alpha=0.18)

    if not hasattr(path, "write"):                  # a file object (the web page's in-memory PNG) is used as is
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, format="png")
    plt.close(figure)
    return path
