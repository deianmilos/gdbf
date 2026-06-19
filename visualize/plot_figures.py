"""
plot_figures.py — High-level plot functions.

Functions
---------
plot_metric               – single-metric line plot (FER / BER / iter / failed_min/avg/max)
plot_fer_ber_combined     – FER and BER overlaid on one axes with a secondary linestyle
plot_failed_bits_summary  – three-panel (min / avg / max) line plot for failed frames
plot_failed_bits_histogram – histogram of failed-bit averages across alpha sweep points
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from plot_curves import poly_log_fit, smooth_log_interp
from plot_parse import ResultRow
from plot_style import filter_positive_fer, metric_values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_x_axis(
    ax,
    all_alphas: list,
    alpha_x_min_override: float | None,
    alpha_x_max_override: float | None,
    x_margin_frac: float,
    x_tick_step: float,
    x_tick_start: float | None,
    x_tick_end: float | None,
    x_tick_decimals: int,
) -> tuple[float, float]:
    """Set xlim and optional xticks.  Returns (alpha_lo, alpha_hi)."""
    alpha_hi = alpha_x_max_override if alpha_x_max_override is not None else max(all_alphas)
    alpha_lo = alpha_x_min_override if alpha_x_min_override is not None else min(all_alphas)
    span = max(1e-12, alpha_hi - alpha_lo)
    margin = max(0.0, x_margin_frac) * span
    ax.set_xlim(alpha_hi + margin, alpha_lo - margin)

    if x_tick_step > 0:
        ts = alpha_lo if x_tick_start is None else float(x_tick_start)
        te = alpha_hi if x_tick_end   is None else float(x_tick_end)
        if te < ts:
            ts, te = te, ts
        ticks = np.arange(ts, te + 0.5 * x_tick_step, x_tick_step)
        if len(ticks) > 0:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:.{max(0, x_tick_decimals)}f}" for v in ticks])

    return alpha_lo, alpha_hi


def _apply_grid(
    ax,
    grid_axis: str,
    show_minor_grid: bool,
    minor_x_divisions: int,
    grid_major_linestyle: str,
    grid_major_linewidth: float,
    grid_major_alpha: float,
    grid_minor_linestyle: str,
    grid_minor_linewidth: float,
    grid_minor_alpha: float,
    is_log_y: bool = False,
) -> None:
    ax.grid(True, which="major", axis=grid_axis,
            linestyle=grid_major_linestyle, linewidth=grid_major_linewidth, alpha=grid_major_alpha)
    if show_minor_grid:
        ax.grid(True, which="minor", axis=grid_axis,
                linestyle=grid_minor_linestyle, linewidth=grid_minor_linewidth, alpha=grid_minor_alpha)
    ax.minorticks_on()
    if is_log_y:
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if minor_x_divisions >= 2:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x_divisions))


def _apply_legend(
    ax,
    legend_right: bool,
    legend_right_anchor: tuple[float, float],
    legend_loc: str,
    legend_font_size,
    legend_facecolor,
    legend_edgecolor,
    legend_framealpha,
    legend_title: str | None = None,
) -> None:
    common = dict(
        fontsize=legend_font_size,
        facecolor=legend_facecolor,
        edgecolor=legend_edgecolor,
        framealpha=legend_framealpha,
        title=legend_title,
    )
    if legend_right:
        ax.legend(loc="center left", bbox_to_anchor=legend_right_anchor, **common)
    else:
        ax.legend(loc=legend_loc, **common)


def _apply_font_sizes(ax, axis_label_font_size, axis_tick_font_size) -> None:
    if axis_label_font_size is not None:
        ax.xaxis.label.set_size(axis_label_font_size)
        ax.yaxis.label.set_size(axis_label_font_size)
    if axis_tick_font_size is not None:
        ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)


def _apply_log_y_limits(ax, all_y: list, y_log_margin_decades: float,
                         y_log_min, y_log_max) -> None:
    if all_y:
        pos = [v for v in all_y if v > 0]
        if pos:
            lo = np.log10(min(pos)) - max(0.0, y_log_margin_decades)
            hi = np.log10(max(pos)) + max(0.0, y_log_margin_decades)
            ax.set_ylim(10.0 ** lo, 10.0 ** hi)
    cur_lo, cur_hi = ax.get_ylim()
    t_lo = cur_lo if y_log_min is None else float(y_log_min)
    t_hi = cur_hi if y_log_max is None else float(y_log_max)
    if t_lo > 0 and t_hi > t_lo:
        ax.set_ylim(t_lo, t_hi)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

_LOG_METRICS = {"fer", "ber"}


def plot_metric(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    metric: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    smooth_curves: bool,
    fit_mode: str,
    poly_degree: int,
    interp_points: int,
    smooth_window: int,
    show_raw_markers: bool,
    figure_size: tuple[float, float],
    show_title: bool,
    legend_loc: str,
    save_pdf: bool,
    legend_right: bool,
    legend_right_anchor: tuple[float, float],
    x_margin_frac: float,
    y_log_margin_decades: float,
    show_minor_grid: bool,
    minor_x_divisions: int,
    x_tick_step: float,
    x_tick_start: float | None,
    x_tick_end: float | None,
    x_tick_decimals: int,
    axis_label_font_size: float | None,
    axis_tick_font_size: float | None,
    legend_font_size: float | None,
    y_log_min: float | None,
    y_log_max: float | None,
    grid_axis: str,
    axis_facecolor: str | None,
    figure_facecolor: str | None,
    grid_major_linestyle: str,
    grid_major_linewidth: float,
    grid_major_alpha: float,
    grid_minor_linestyle: str,
    grid_minor_linewidth: float,
    grid_minor_alpha: float,
    axis_spine_linewidth: float,
    legend_facecolor: str | None,
    legend_edgecolor: str | None,
    legend_framealpha: float | None,
    alpha_x_min_override: float | None = None,
    alpha_x_max_override: float | None = None,
) -> None:
    is_log = metric in _LOG_METRICS
    do_smooth = smooth_curves and is_log

    fig, ax = plt.subplots(figsize=figure_size)
    if figure_facecolor:
        fig.patch.set_facecolor(figure_facecolor)
    if axis_facecolor:
        ax.set_facecolor(axis_facecolor)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_linewidth)

    all_y: List[float] = []

    for model_name, rows in model_rows.items():
        style = model_style[model_name]
        if metric == "fer":
            rows = filter_positive_fer(rows)
        x = np.array([r.alpha for r in rows], dtype=float)
        y = metric_values(rows, metric)

        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) == 0:
            continue
        if is_log:
            y = np.clip(y, 1e-15, None)

        x_plot, y_plot = x, y
        if do_smooth and len(x) >= 2:
            if fit_mode == "poly":
                x_plot, y_plot = poly_log_fit(x, y, degree=poly_degree, n_points=interp_points)
            else:
                x_plot, y_plot = smooth_log_interp(x, y, n_points=interp_points, window=smooth_window)

        all_y.extend(y_plot[np.isfinite(y_plot)].tolist())

        marker_symbol = None
        markevery_value = None
        if do_smooth and metric == "fer":
            # Keep FER curves smooth but retain sparse markers so legend shows marker identity.
            marker_symbol = style.get("marker", "o")
            markevery_value = max(1, len(x_plot) // 12)

        ax.plot(
            x_plot, y_plot,
            label=style.get("label", model_name),
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=marker_symbol if do_smooth else style.get("marker", "o"),
            markevery=markevery_value,
            linewidth=float(style.get("linewidth", 1.8)),
            markersize=float(style.get("markersize", 5.5)),
            markerfacecolor=style.get("markerfacecolor", "white"),
            markeredgewidth=float(style.get("markeredgewidth", 1.0)),
        )

        if show_raw_markers and do_smooth:
            ax.scatter(x, y, s=6,
                       marker=style.get("marker", "o"),
                       facecolors=style.get("markerfacecolor", "white"),
                       edgecolors=style.get("color", None),
                       linewidths=float(style.get("markeredgewidth", 0.8)),
                       alpha=0.75,
                       zorder=3)

    if is_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)

    _apply_font_sizes(ax, axis_label_font_size, axis_tick_font_size)
    _apply_grid(ax, grid_axis, show_minor_grid, minor_x_divisions,
                grid_major_linestyle, grid_major_linewidth, grid_major_alpha,
                grid_minor_linestyle, grid_minor_linewidth, grid_minor_alpha,
                is_log_y=is_log)

    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    _apply_x_axis(ax, all_alphas, alpha_x_min_override, alpha_x_max_override,
                  x_margin_frac, x_tick_step, x_tick_start, x_tick_end, x_tick_decimals)

    if is_log:
        _apply_log_y_limits(ax, all_y, y_log_margin_decades, y_log_min, y_log_max)

    _apply_legend(ax, legend_right, legend_right_anchor, legend_loc,
                  legend_font_size, legend_facecolor, legend_edgecolor, legend_framealpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_fer_ber_combined(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    smooth_curves: bool,
    fit_mode: str,
    poly_degree: int,
    interp_points: int,
    smooth_window: int,
    show_raw_markers: bool,
    figure_size: tuple[float, float],
    show_title: bool,
    legend_loc: str,
    save_pdf: bool,
    legend_right: bool,
    legend_right_anchor: tuple[float, float],
    x_margin_frac: float,
    y_log_margin_decades: float,
    show_minor_grid: bool,
    minor_x_divisions: int,
    x_tick_step: float,
    x_tick_start: float | None,
    x_tick_end: float | None,
    x_tick_decimals: int,
    axis_label_font_size: float | None,
    axis_tick_font_size: float | None,
    legend_font_size: float | None,
    y_log_min: float | None,
    y_log_max: float | None,
    grid_axis: str,
    axis_facecolor: str | None,
    figure_facecolor: str | None,
    grid_major_linestyle: str,
    grid_major_linewidth: float,
    grid_major_alpha: float,
    grid_minor_linestyle: str,
    grid_minor_linewidth: float,
    grid_minor_alpha: float,
    axis_spine_linewidth: float,
    legend_facecolor: str | None,
    legend_edgecolor: str | None,
    legend_framealpha: float | None,
    ber_linestyle: str,
    legend_title: str | None,
    alpha_x_min_override: float | None = None,
    alpha_x_max_override: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figure_size)
    if figure_facecolor:
        fig.patch.set_facecolor(figure_facecolor)
    if axis_facecolor:
        ax.set_facecolor(axis_facecolor)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_linewidth)

    all_y: List[float] = []

    for model_name, rows in model_rows.items():
        style = model_style[model_name]
        x = np.array([r.alpha for r in rows], dtype=float)
        y_fer = np.array([r.fer for r in rows], dtype=float)
        y_ber = np.array([r.ber for r in rows], dtype=float)

        v_fer = np.isfinite(x) & np.isfinite(y_fer) & (y_fer > 0)
        v_ber = np.isfinite(x) & np.isfinite(y_ber) & (y_ber > 0)
        x_fer, fer = x[v_fer], np.clip(y_fer[v_fer], 1e-15, None)
        x_ber, ber = x[v_ber], np.clip(y_ber[v_ber], 1e-15, None)

        if len(x_fer) == 0 and len(x_ber) == 0:
            continue

        def _smooth(xv, yv):
            if smooth_curves and len(xv) >= 2:
                if fit_mode == "poly":
                    return poly_log_fit(xv, yv, degree=poly_degree, n_points=interp_points)
                return smooth_log_interp(xv, yv, n_points=interp_points, window=smooth_window)
            return xv, yv

        xfp, ferp = _smooth(x_fer, fer)
        xbp, berp = _smooth(x_ber, ber)
        all_y.extend(ferp[np.isfinite(ferp)].tolist())
        all_y.extend(berp[np.isfinite(berp)].tolist())

        common_kw = dict(
            color=style.get("color", None),
            linewidth=float(style.get("linewidth", 1.8)),
            markersize=float(style.get("markersize", 5.5)),
            markerfacecolor=style.get("markerfacecolor", "white"),
            markeredgewidth=float(style.get("markeredgewidth", 1.0)),
        )
        no_marker = smooth_curves and len(x_fer) >= 2

        if len(xfp) > 0:
            ax.plot(xfp, ferp, label=f"{style.get('label', model_name)} FER",
                    linestyle=style.get("linestyle", "-"),
                    marker=None if no_marker else style.get("marker", "o"), **common_kw)
        if len(xbp) > 0:
            ax.plot(xbp, berp, label=f"{style.get('label', model_name)} BER",
                    linestyle=ber_linestyle,
                    marker=None if (smooth_curves and len(x_ber) >= 2) else style.get("marker", "o"),
                    alpha=0.9, **common_kw)

        if show_raw_markers and smooth_curves:
            for xr, yr in ((x_fer, fer), (x_ber, ber)):
                if len(xr) > 0:
                    ax.scatter(xr, yr, s=16,
                               marker=style.get("marker", "o"),
                               facecolors=style.get("markerfacecolor", "white"),
                               edgecolors=style.get("color", None),
                               linewidths=float(style.get("markeredgewidth", 1.0)),
                               alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)

    _apply_font_sizes(ax, axis_label_font_size, axis_tick_font_size)
    _apply_grid(ax, grid_axis, show_minor_grid, minor_x_divisions,
                grid_major_linestyle, grid_major_linewidth, grid_major_alpha,
                grid_minor_linestyle, grid_minor_linewidth, grid_minor_alpha,
                is_log_y=True)

    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    _apply_x_axis(ax, all_alphas, alpha_x_min_override, alpha_x_max_override,
                  x_margin_frac, x_tick_step, x_tick_start, x_tick_end, x_tick_decimals)
    _apply_log_y_limits(ax, all_y, y_log_margin_decades, y_log_min, y_log_max)
    _apply_legend(ax, legend_right, legend_right_anchor, legend_loc,
                  legend_font_size, legend_facecolor, legend_edgecolor, legend_framealpha,
                  legend_title=legend_title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_failed_bits_summary(
    output_path_min: Path,
    output_path_avg: Path,
    output_path_max: Path,
    title_min: str,
    title_avg: str,
    title_max: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    figure_size: tuple[float, float] = (5.2, 3.8),
    show_title: bool = True,
    save_pdf: bool = True,
    alpha_x_min_override: float | None = None,
    alpha_x_max_override: float | None = None,
) -> None:
    """
    Three separate figures — one each for min / avg / max residual bit errors
    in frames that failed to decode.  x = alpha, y = failed bits count.
    """
    panels = [
        ("failed_min", "Min residual bit errors", output_path_min, title_min),
        ("failed_avg", "Avg residual bit errors", output_path_avg, title_avg),
        ("failed_max", "Max residual bit errors", output_path_max, title_max),
    ]

    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    x_hi = alpha_x_max_override if alpha_x_max_override is not None else max(all_alphas)
    x_lo = alpha_x_min_override if alpha_x_min_override is not None else min(all_alphas)

    for metric, ylabel, out_path, title in panels:
        fig, ax = plt.subplots(figsize=figure_size)

        for model_name, rows in model_rows.items():
            x = np.array([r.alpha for r in rows], dtype=float)
            y = metric_values(rows, metric)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) == 0:
                continue
            style = model_style[model_name]
            ax.plot(
                x, y,
                label=style.get("label", model_name),
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=float(style.get("linewidth", 1.4)),
                markersize=float(style.get("markersize", 3.5)),
                markerfacecolor=style.get("markerfacecolor", "white"),
                markeredgewidth=float(style.get("markeredgewidth", 0.8)),
            )

        ax.set_xlabel(r"BSC crossover probability $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_hi, x_lo)
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.12)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8)
        if show_title:
            ax.set_title(title)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        if save_pdf:
            fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def plot_failed_bits_histogram(
    output_path: Path,
    title: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    figure_size: tuple[float, float] = (6.6, 4.0),
    show_title: bool = True,
    save_pdf: bool = True,
    bins: int = 16,
    alpha_value: float = 0.45,
    alpha_x_min_override: float | None = None,
    alpha_x_max_override: float | None = None,
) -> None:
    """
    Bar chart: x = alpha (BSC crossover probability), y = number of failed frames (NbFer).
    One grouped set of bars per model, plotted side by side per alpha point.
    """
    # Collect all unique alpha values across all models (sorted descending = high->low)
    all_alphas_set: set[float] = set()
    for rows in model_rows.values():
        for r in rows:
            all_alphas_set.add(r.alpha)
    all_alphas = sorted(all_alphas_set, reverse=True)

    # Optionally restrict x range
    if alpha_x_min_override is not None:
        all_alphas = [a for a in all_alphas if a >= alpha_x_min_override]
    if alpha_x_max_override is not None:
        all_alphas = [a for a in all_alphas if a <= alpha_x_max_override]

    n_alphas = len(all_alphas)
    n_models = len(model_rows)
    if n_alphas == 0 or n_models == 0:
        return

    bar_width = 0.8 / n_models
    x_pos = np.arange(n_alphas)

    fig, ax = plt.subplots(figsize=figure_size)

    alpha_to_idx = {a: i for i, a in enumerate(all_alphas)}

    for offset, (model_name, rows) in enumerate(model_rows.items()):
        style = model_style[model_name]
        nbfer_by_alpha = {r.alpha: r.nbfer for r in rows}
        heights = [float(nbfer_by_alpha.get(a, 0)) for a in all_alphas]
        centers = x_pos + (offset - (n_models - 1) / 2.0) * bar_width
        ax.bar(
            centers,
            heights,
            width=bar_width * 0.92,
            label=style.get("label", model_name),
            color=style.get("color", None),
            alpha=alpha_value + 0.3,
            edgecolor="white",
            linewidth=0.5,
        )

    # x ticks: show alpha values, rotated for readability
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{a:.4g}" for a in all_alphas], rotation=55, ha="right", fontsize=7)
    ax.set_xlabel(r"BSC crossover probability $\alpha$")
    ax.set_ylabel("Number of failed frames")
    if show_title:
        ax.set_title(title)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_failed_bits_bar(
    output_path: Path,
    title: str,
    metric: str,
    ylabel: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    figure_size: tuple[float, float] = (8.0, 4.0),
    show_title: bool = True,
    save_pdf: bool = True,
    bar_alpha: float = 0.75,
    alpha_x_min_override: float | None = None,
    alpha_x_max_override: float | None = None,
) -> None:
    """
    Grouped bar chart: x = alpha, y = failed_min / failed_avg / failed_max.
    One bar group per alpha point, one bar per model inside each group.
    """
    all_alphas_set: set[float] = set()
    for rows in model_rows.values():
        for r in rows:
            all_alphas_set.add(r.alpha)
    all_alphas = sorted(all_alphas_set, reverse=True)

    if alpha_x_min_override is not None:
        all_alphas = [a for a in all_alphas if a >= alpha_x_min_override]
    if alpha_x_max_override is not None:
        all_alphas = [a for a in all_alphas if a <= alpha_x_max_override]

    n_alphas = len(all_alphas)
    n_models = len(model_rows)
    if n_alphas == 0 or n_models == 0:
        return

    bar_width = 0.8 / n_models
    x_pos = np.arange(n_alphas)

    fig, ax = plt.subplots(figsize=figure_size)

    for offset, (model_name, rows) in enumerate(model_rows.items()):
        style = model_style[model_name]
        val_by_alpha = {}
        for r in rows:
            v = metric_values([r], metric)
            if len(v) > 0 and np.isfinite(v[0]):
                val_by_alpha[r.alpha] = float(v[0])
        heights = [val_by_alpha.get(a, 0.0) for a in all_alphas]
        centers = x_pos + (offset - (n_models - 1) / 2.0) * bar_width
        ax.bar(
            centers,
            heights,
            width=bar_width * 0.88,
            label=style.get("label", model_name),
            color=style.get("color", None),
            alpha=bar_alpha,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{a:.4g}" for a in all_alphas], rotation=55, ha="right", fontsize=7)
    ax.set_xlabel(r"BSC crossover probability $\alpha$")
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
