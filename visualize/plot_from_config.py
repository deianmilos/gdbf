import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


MONO_LINESTYLES = ["-", "--", "-.", ":"]
MONO_MARKERS = ["o", "s", "^", "D", "v", "x", "P", "*"]


LINE_RE = re.compile(
    r"^\s*"
    r"(?P<alpha>[0-9]*\.?[0-9]+)\s+"
    r"(?P<nber>\d+)\s*\((?P<ber>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbfer>\d+)\s*\((?P<fer>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbtested>\d+)\s+"
    r"(?P<iter_avg>[0-9]*\.?[0-9]+)\((?P<iter_max>\d+)\)\s+"
    r"(?P<failed_triplet>(?:[0-9]*\.?[0-9]+|-)/(?:[0-9]*\.?[0-9]+|-)/(?:[0-9]*\.?[0-9]+|-))"
)


@dataclass
class ResultRow:
    alpha: float
    ber: float
    fer: float
    iter_avg: float
    nbtested: int
    failed_min: float
    failed_avg: float
    failed_max: float


def parse_triplet_token(token: str) -> tuple[float, float, float]:
    parts = token.split("/")
    if len(parts) != 3:
        return (float("nan"), float("nan"), float("nan"))

    def parse_value(v: str) -> float:
        s = v.strip()
        if s == "-":
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return (parse_value(parts[0]), parse_value(parts[1]), parse_value(parts[2]))


def parse_res_file(path: Path) -> List[ResultRow]:
    """Parse simulation.res and keep the latest row for each alpha."""
    rows_by_alpha: Dict[float, ResultRow] = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.lower().startswith("alpha"):
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            alpha = float(m.group("alpha"))
            failed_min, failed_avg, failed_max = parse_triplet_token(m.group("failed_triplet"))
            rows_by_alpha[alpha] = ResultRow(
                alpha=alpha,
                ber=float(m.group("ber")),
                fer=float(m.group("fer")),
                iter_avg=float(m.group("iter_avg")),
                nbtested=int(m.group("nbtested")),
                failed_min=failed_min,
                failed_avg=failed_avg,
                failed_max=failed_max,
            )

    if not rows_by_alpha:
        raise ValueError(f"No valid rows parsed from: {path}")

    return [rows_by_alpha[a] for a in sorted(rows_by_alpha.keys(), reverse=True)]
def apply_paper_style(font_size: int = 11) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "CMU Serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#666666",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def get_figure_size(cfg: dict, key_prefix: str, default: tuple[float, float]) -> tuple[float, float]:
    w = float(cfg.get(f"{key_prefix}_fig_width", default[0]))
    h = float(cfg.get(f"{key_prefix}_fig_height", default[1]))
    return (w, h)


def apply_monochrome_model_style(model_style: Dict[str, Dict[str, str]]) -> None:
    for i, name in enumerate(model_style.keys()):
        st = model_style[name]
        st["color"] = "#000000"
        st["linestyle"] = MONO_LINESTYLES[i % len(MONO_LINESTYLES)]
        st["marker"] = MONO_MARKERS[i % len(MONO_MARKERS)]
        st["markerfacecolor"] = "white"
        st["markeredgewidth"] = str(max(1.0, float(st.get("markeredgewidth", 1.0))))


FAILED_BITS_METRIC_MAP = {
    "min": "failed_min",
    "avg": "failed_avg",
    "max": "failed_max",
    "failed_min": "failed_min",
    "failed_avg": "failed_avg",
    "failed_max": "failed_max",
}

FAILED_BITS_METRIC_LABELS = {
    "failed_min": "Minimum Uncorrected Bits",
    "failed_avg": "Average Uncorrected Bits",
    "failed_max": "Maximum Uncorrected Bits",
    "expected_failed_bits": "Failed Bits per Tested Frame",
}


def metric_values(rows: List[ResultRow], metric: str) -> np.ndarray:
    if metric == "fer":
        return np.array([r.fer for r in rows], dtype=float)
    if metric == "ber":
        return np.array([r.ber for r in rows], dtype=float)
    if metric == "iter":
        return np.array([r.iter_avg for r in rows], dtype=float)
    if metric == "failed_min":
        return np.array([r.failed_min for r in rows], dtype=float)
    if metric == "failed_avg":
        return np.array([r.failed_avg for r in rows], dtype=float)
    if metric == "failed_max":
        return np.array([r.failed_max for r in rows], dtype=float)
    if metric == "expected_failed_bits":
        return np.array([r.fer * r.failed_avg for r in rows], dtype=float)
    raise ValueError(f"Unsupported metric: {metric}")


def normalize_failed_bits_hist_metrics(raw_metrics: object) -> List[str]:
    if raw_metrics is None:
        return ["failed_avg"]

    if isinstance(raw_metrics, str):
        candidates = [raw_metrics]
    elif isinstance(raw_metrics, list):
        candidates = raw_metrics
    else:
        raise ValueError("failed_bits_hist_metrics must be a string or a list of strings")

    normalized: List[str] = []
    for item in candidates:
        if not isinstance(item, str):
            raise ValueError("Each failed_bits_hist_metrics entry must be a string")
        key = item.strip().lower()
        metric = FAILED_BITS_METRIC_MAP.get(key)
        if metric is None:
            raise ValueError(f"Unsupported failed bits histogram metric: {item}")
        if metric not in normalized:
            normalized.append(metric)

    if not normalized:
        raise ValueError("failed_bits_hist_metrics must contain at least one metric")

    return normalized


def filter_positive_fer(rows: List[ResultRow]) -> List[ResultRow]:
    return [row for row in rows if row.fer > 0]


def smooth_log_interp(x: np.ndarray, y: np.ndarray, n_points: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and lightly smooth y in log10-domain, then return descending-x arrays."""
    idx = np.argsort(x)
    xs = np.asarray(x[idx], dtype=float)
    ys = np.clip(np.asarray(y[idx], dtype=float), 1e-15, None)

    logy = np.log10(ys)
    xd = np.linspace(xs.min(), xs.max(), int(n_points))
    yd_log = np.interp(xd, xs, logy)

    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w > 1:
        pad = w // 2
        kernel = np.ones(w, dtype=float) / float(w)
        padded = np.pad(yd_log, (pad, pad), mode="edge")
        yd_log = np.convolve(padded, kernel, mode="valid")

    yd = np.power(10.0, yd_log)
    return xd[::-1], yd[::-1]


def poly_log_fit(x: np.ndarray, y: np.ndarray, degree: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial fit in log10-domain, then return descending-x arrays."""
    idx = np.argsort(x)
    xs = np.asarray(x[idx], dtype=float)
    ys = np.clip(np.asarray(y[idx], dtype=float), 1e-15, None)

    deg = max(1, int(degree))
    # Polyfit degree must be < number of samples.
    deg = min(deg, max(1, len(xs) - 1))

    logy = np.log10(ys)
    coeff = np.polyfit(xs, logy, deg)

    xd = np.linspace(xs.min(), xs.max(), int(n_points))
    yd_log = np.polyval(coeff, xd)
    yd = np.power(10.0, yd_log)
    return xd[::-1], yd[::-1]


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
) -> None:
    fig, ax = plt.subplots(figsize=figure_size)
    if figure_facecolor:
        fig.patch.set_facecolor(figure_facecolor)
    if axis_facecolor:
        ax.set_facecolor(axis_facecolor)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_linewidth)
    all_metric_y: List[float] = []

    for model_name, rows in model_rows.items():
        x = np.array([r.alpha for r in rows], dtype=float)
        y = metric_values(rows, metric)
        style = model_style[model_name]

        if metric == "fer":
            positive_rows = filter_positive_fer(rows)
            x = np.array([r.alpha for r in positive_rows], dtype=float)
            y = np.array([r.fer for r in positive_rows], dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) == 0:
            continue

        if metric in {"fer", "ber", "expected_failed_bits"}:
            y = np.clip(y, 1e-15, None)

        x_plot = x
        y_plot = y
        if smooth_curves and metric in {"fer", "ber", "expected_failed_bits"} and len(x) >= 2:
            if fit_mode == "poly":
                x_plot, y_plot = poly_log_fit(x, y, degree=poly_degree, n_points=interp_points)
            else:
                x_plot, y_plot = smooth_log_interp(x, y, n_points=interp_points, window=smooth_window)

        valid_plot_y = y_plot[np.isfinite(y_plot)]
        if len(valid_plot_y) > 0:
            all_metric_y.extend(valid_plot_y.tolist())

        ax.plot(
            x_plot,
            y_plot,
            label=style.get("label", model_name),
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=(None if (smooth_curves and metric in {"fer", "ber", "expected_failed_bits"}) else style.get("marker", "o")),
            linewidth=float(style.get("linewidth", 1.8)),
            markersize=float(style.get("markersize", 5.5)),
            markerfacecolor=style.get("markerfacecolor", "white"),
            markeredgewidth=float(style.get("markeredgewidth", 1.0)),
        )

        if show_raw_markers and smooth_curves and metric in {"fer", "ber", "expected_failed_bits"}:
            raw_x = x
            raw_y = y
            if metric == "fer":
                raw_x = np.array([r.alpha for r in filter_positive_fer(rows)], dtype=float)
                raw_y = np.array([r.fer for r in filter_positive_fer(rows)], dtype=float)
            ax.scatter(
                raw_x,
                raw_y,
                s=16,
                marker=style.get("marker", "o"),
                facecolors=style.get("markerfacecolor", "white"),
                edgecolors=style.get("color", None),
                linewidths=float(style.get("markeredgewidth", 1.0)),
                alpha=0.85,
            )

    if metric in {"fer", "ber", "expected_failed_bits"}:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if axis_label_font_size is not None:
        ax.xaxis.label.set_size(axis_label_font_size)
        ax.yaxis.label.set_size(axis_label_font_size)
    if axis_tick_font_size is not None:
        ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)
    if show_title:
        ax.set_title(title)

    ax.grid(
        True,
        which="major",
        linestyle=grid_major_linestyle,
        linewidth=grid_major_linewidth,
        alpha=grid_major_alpha,
        axis=grid_axis,
    )
    if show_minor_grid:
        ax.grid(
            True,
            which="minor",
            linestyle=grid_minor_linestyle,
            linewidth=grid_minor_linewidth,
            alpha=grid_minor_alpha,
            axis=grid_axis,
        )
    ax.minorticks_on()

    # Denser helper lines for readability in paper plots.
    if metric in {"fer", "ber", "expected_failed_bits"}:
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if minor_x_divisions >= 2:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x_divisions))

    # Higher alpha on the left, lower alpha on the right.
    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    alpha_hi = max(all_alphas)
    alpha_lo = min(all_alphas)
    alpha_span = max(1e-12, alpha_hi - alpha_lo)
    margin = max(0.0, x_margin_frac) * alpha_span
    ax.set_xlim(alpha_hi + margin, alpha_lo - margin)

    if x_tick_step > 0:
        tick_start = alpha_lo if x_tick_start is None else float(x_tick_start)
        tick_end = alpha_hi if x_tick_end is None else float(x_tick_end)
        if tick_end < tick_start:
            tick_start, tick_end = tick_end, tick_start
        xticks = np.arange(tick_start, tick_end + 0.5 * x_tick_step, x_tick_step)
        if len(xticks) > 0:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:.{max(0, x_tick_decimals)}f}" for v in xticks])

    if metric in {"fer", "ber", "expected_failed_bits"} and all_metric_y:
        y_min = min(v for v in all_metric_y if v > 0)
        y_max = max(all_metric_y)
        if y_min > 0 and y_max > 0:
            log_min = np.log10(y_min) - max(0.0, y_log_margin_decades)
            log_max = np.log10(y_max) + max(0.0, y_log_margin_decades)
            ax.set_ylim(10.0 ** log_min, 10.0 ** log_max)

    if metric in {"fer", "ber", "expected_failed_bits"}:
        current_ymin, current_ymax = ax.get_ylim()
        target_ymin = current_ymin if y_log_min is None else float(y_log_min)
        target_ymax = current_ymax if y_log_max is None else float(y_log_max)
        if target_ymin > 0 and target_ymax > target_ymin:
            ax.set_ylim(target_ymin, target_ymax)

    if legend_right:
        ax.legend(
            loc="center left",
            bbox_to_anchor=legend_right_anchor,
            fontsize=legend_font_size,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            framealpha=legend_framealpha,
        )
    else:
        ax.legend(
            loc=legend_loc,
            fontsize=legend_font_size,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            framealpha=legend_framealpha,
        )

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
) -> None:
    fig, ax = plt.subplots(figsize=figure_size)
    if figure_facecolor:
        fig.patch.set_facecolor(figure_facecolor)
    if axis_facecolor:
        ax.set_facecolor(axis_facecolor)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_linewidth)

    all_metric_y: List[float] = []

    for model_name, rows in model_rows.items():
        style = model_style[model_name]
        x = np.array([r.alpha for r in rows], dtype=float)
        y_fer = np.array([r.fer for r in rows], dtype=float)
        y_ber = np.array([r.ber for r in rows], dtype=float)

        valid_fer = np.isfinite(x) & np.isfinite(y_fer) & (y_fer > 0)
        valid_ber = np.isfinite(x) & np.isfinite(y_ber) & (y_ber > 0)

        x_fer = x[valid_fer]
        fer = np.clip(y_fer[valid_fer], 1e-15, None)
        x_ber = x[valid_ber]
        ber = np.clip(y_ber[valid_ber], 1e-15, None)

        if len(x_fer) == 0 and len(x_ber) == 0:
            continue

        x_fer_plot = x_fer
        fer_plot = fer
        if smooth_curves and len(x_fer) >= 2:
            if fit_mode == "poly":
                x_fer_plot, fer_plot = poly_log_fit(x_fer, fer, degree=poly_degree, n_points=interp_points)
            else:
                x_fer_plot, fer_plot = smooth_log_interp(x_fer, fer, n_points=interp_points, window=smooth_window)

        x_ber_plot = x_ber
        ber_plot = ber
        if smooth_curves and len(x_ber) >= 2:
            if fit_mode == "poly":
                x_ber_plot, ber_plot = poly_log_fit(x_ber, ber, degree=poly_degree, n_points=interp_points)
            else:
                x_ber_plot, ber_plot = smooth_log_interp(x_ber, ber, n_points=interp_points, window=smooth_window)

        fer_valid_plot = fer_plot[np.isfinite(fer_plot)]
        ber_valid_plot = ber_plot[np.isfinite(ber_plot)]
        if len(fer_valid_plot) > 0:
            all_metric_y.extend(fer_valid_plot.tolist())
        if len(ber_valid_plot) > 0:
            all_metric_y.extend(ber_valid_plot.tolist())

        if len(x_fer_plot) > 0:
            ax.plot(
                x_fer_plot,
                fer_plot,
                label=f"{style.get('label', model_name)} FER",
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                marker=(None if (smooth_curves and len(x_fer) >= 2) else style.get("marker", "o")),
                linewidth=float(style.get("linewidth", 1.8)),
                markersize=float(style.get("markersize", 5.5)),
                markerfacecolor=style.get("markerfacecolor", "white"),
                markeredgewidth=float(style.get("markeredgewidth", 1.0)),
            )

        if len(x_ber_plot) > 0:
            ax.plot(
                x_ber_plot,
                ber_plot,
                label=f"{style.get('label', model_name)} BER",
                color=style.get("color", None),
                linestyle=ber_linestyle,
                marker=(None if (smooth_curves and len(x_ber) >= 2) else style.get("marker", "o")),
                linewidth=float(style.get("linewidth", 1.8)),
                markersize=float(style.get("markersize", 5.5)),
                markerfacecolor=style.get("markerfacecolor", "white"),
                markeredgewidth=float(style.get("markeredgewidth", 1.0)),
                alpha=0.9,
            )

        if show_raw_markers and smooth_curves:
            if len(x_fer) > 0:
                ax.scatter(
                    x_fer,
                    fer,
                    s=16,
                    marker=style.get("marker", "o"),
                    facecolors=style.get("markerfacecolor", "white"),
                    edgecolors=style.get("color", None),
                    linewidths=float(style.get("markeredgewidth", 1.0)),
                    alpha=0.85,
                )
            if len(x_ber) > 0:
                ax.scatter(
                    x_ber,
                    ber,
                    s=12,
                    marker=style.get("marker", "o"),
                    facecolors="none",
                    edgecolors=style.get("color", None),
                    linewidths=float(style.get("markeredgewidth", 1.0)),
                    alpha=0.75,
                )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if axis_label_font_size is not None:
        ax.xaxis.label.set_size(axis_label_font_size)
        ax.yaxis.label.set_size(axis_label_font_size)
    if axis_tick_font_size is not None:
        ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)
    if show_title:
        ax.set_title(title)

    ax.grid(
        True,
        which="major",
        linestyle=grid_major_linestyle,
        linewidth=grid_major_linewidth,
        alpha=grid_major_alpha,
        axis=grid_axis,
    )
    if show_minor_grid:
        ax.grid(
            True,
            which="minor",
            linestyle=grid_minor_linestyle,
            linewidth=grid_minor_linewidth,
            alpha=grid_minor_alpha,
            axis=grid_axis,
        )
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if minor_x_divisions >= 2:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x_divisions))

    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    alpha_hi = max(all_alphas)
    alpha_lo = min(all_alphas)
    alpha_span = max(1e-12, alpha_hi - alpha_lo)
    margin = max(0.0, x_margin_frac) * alpha_span
    ax.set_xlim(alpha_hi + margin, alpha_lo - margin)

    if x_tick_step > 0:
        tick_start = alpha_lo if x_tick_start is None else float(x_tick_start)
        tick_end = alpha_hi if x_tick_end is None else float(x_tick_end)
        if tick_end < tick_start:
            tick_start, tick_end = tick_end, tick_start
        xticks = np.arange(tick_start, tick_end + 0.5 * x_tick_step, x_tick_step)
        if len(xticks) > 0:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:.{max(0, x_tick_decimals)}f}" for v in xticks])

    if all_metric_y:
        y_min = min(v for v in all_metric_y if v > 0)
        y_max = max(all_metric_y)
        if y_min > 0 and y_max > 0:
            log_min = np.log10(y_min) - max(0.0, y_log_margin_decades)
            log_max = np.log10(y_max) + max(0.0, y_log_margin_decades)
            ax.set_ylim(10.0 ** log_min, 10.0 ** log_max)

    current_ymin, current_ymax = ax.get_ylim()
    target_ymin = current_ymin if y_log_min is None else float(y_log_min)
    target_ymax = current_ymax if y_log_max is None else float(y_log_max)
    if target_ymin > 0 and target_ymax > target_ymin:
        ax.set_ylim(target_ymin, target_ymax)

    if legend_right:
        ax.legend(
            loc="center left",
            bbox_to_anchor=legend_right_anchor,
            fontsize=legend_font_size,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            framealpha=legend_framealpha,
            title=legend_title,
        )
    else:
        ax.legend(
            loc=legend_loc,
            fontsize=legend_font_size,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            framealpha=legend_framealpha,
            title=legend_title,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_failed_bits_summary(
    output_path: Path,
    title: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    failed_bits_x_max: float | None = None,
    figure_size: tuple[float, float] = (10.5, 3.8),
    show_title: bool = True,
    save_pdf: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figure_size, sharex=True)
    metrics = [
        ("failed_min", "Min Uncorrected Bits"),
        ("failed_avg", "Avg Uncorrected Bits"),
        ("failed_max", "Max Uncorrected Bits"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for model_name, rows in model_rows.items():
            x = np.array([r.alpha for r in rows], dtype=float)
            y = metric_values(rows, metric)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) == 0:
                continue

            style = model_style[model_name]
            ax.plot(
                x,
                y,
                label=style.get("label", model_name),
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=float(style.get("linewidth", 1.8)),
                markersize=float(style.get("markersize", 5.0)),
                markerfacecolor=style.get("markerfacecolor", "white"),
                markeredgewidth=float(style.get("markeredgewidth", 1.0)),
            )

        ax.set_xlabel("Alpha")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linestyle="--")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6)

        all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
        x_max = failed_bits_x_max if failed_bits_x_max is not None else min(all_alphas)
        ax.set_xlim(max(all_alphas), x_max)

    if show_title:
        fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, 1.04))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_failed_bits_histogram(
    output_path: Path,
    title: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
    hist_metrics: List[str],
    failed_bits_x_max: float | None = None,
    show_title: bool = True,
    save_pdf: bool = True,
) -> None:
    model_names = list(model_rows.keys())
    if not model_names:
        return

    metric_to_alphas: Dict[str, List[float]] = {}
    for metric in hist_metrics:
        alpha_set = set()
        for rows in model_rows.values():
            for r in rows:
                value = metric_values([r], metric)[0]
                if np.isfinite(value) and value > 0:
                    # Filter by x_max if specified
                    if failed_bits_x_max is None or r.alpha >= failed_bits_x_max:
                        alpha_set.add(r.alpha)
        metric_to_alphas[metric] = sorted(alpha_set, reverse=True)

    if not any(metric_to_alphas.values()):
        return

    n_metrics = len(hist_metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(max(9.0, 0.55 * max((len(v) for v in metric_to_alphas.values()), default=1) + 4.0, 6.2 * n_metrics), 5.2),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    def bright_color(color_value: str | None) -> tuple[float, float, float]:
        if color_value is None:
            return to_rgb("#2E8BFF")
        return to_rgb(color_value)

    for ax, metric in zip(axes, hist_metrics):
        alphas = metric_to_alphas[metric]
        if not alphas:
            ax.set_visible(False)
            continue

        x = np.arange(len(alphas), dtype=float)
        group_width = 0.82
        width = group_width / max(1, len(model_names))
        offset_left = -0.5 * group_width + 0.5 * width

        ax.set_facecolor("#fbfcfe")

        model_y_values: Dict[str, np.ndarray] = {}
        model_containers = {}

        for m_idx, model_name in enumerate(model_names):
            rows = model_rows[model_name]
            style = model_style[model_name]
            y_by_alpha = {}

            for r in rows:
                v = metric_values([r], metric)[0]
                if np.isfinite(v) and v > 0:
                    y_by_alpha[r.alpha] = v

            y = np.array([y_by_alpha.get(a, np.nan) for a in alphas], dtype=float)
            model_y_values[model_name] = y
            positions = x + offset_left + m_idx * width

            container = ax.bar(
                positions,
                y,
                width,
                label=style.get("label", model_name),
                color=bright_color(style.get("color", None)),
                edgecolor="#1f1f1f",
                linewidth=0.95,
                alpha=0.96,
            )
            model_containers[model_name] = container

        # For each alpha, find the best model and highlight discretely
        for alpha_idx in range(len(alphas)):
            best_value = float('inf')
            best_model = None
            
            # Find best value at this alpha
            for model_name in model_names:
                y = model_y_values[model_name]
                if alpha_idx < len(y) and np.isfinite(y[alpha_idx]):
                    if y[alpha_idx] < best_value:
                        best_value = y[alpha_idx]
                        best_model = model_name
            
            # Highlight best bar by keeping it fully opaque, dim others
            for model_name in model_names:
                container = model_containers[model_name]
                bar = container[alpha_idx]
                
                if model_name == best_model:
                    # Best bar: keep full opacity and slightly thicker border
                    bar.set_alpha(0.96)
                    bar.set_linewidth(1.3)
                else:
                    # Non-best bars: reduce opacity to make them less visible
                    bar.set_alpha(0.50)

        ax.set_title(FAILED_BITS_METRIC_LABELS.get(metric, metric))
        ax.set_ylabel(FAILED_BITS_METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Alpha")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.3f}" for a in alphas], rotation=45, ha="right")
        ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.7)

    if show_title:
        if n_metrics == 1:
            axes[0].set_title(title)
        else:
            fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style FER/BER/iteration plots from simulation.res files via JSON config"
    )
    parser.add_argument("--config", required=True, help="Path to JSON plotting config")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    apply_paper_style(font_size=int(cfg.get("font_size", 11)))

    output_dir = Path(cfg.get("output_dir", "visualize/plots/paper"))
    dpi = int(cfg.get("dpi", 400))
    save_pdf = bool(cfg.get("save_pdf", True))
    show_title = bool(cfg.get("show_title", False))
    legend_loc = str(cfg.get("legend_loc", "best"))
    legend_right = bool(cfg.get("legend_right", False))
    legend_right_anchor = tuple(cfg.get("legend_right_anchor", [1.02, 0.5]))
    paper_monochrome = bool(cfg.get("paper_monochrome", True))
    x_margin_frac = float(cfg.get("x_margin_frac", 0.06))
    y_log_margin_decades = float(cfg.get("y_log_margin_decades", 0.12))
    show_minor_grid = bool(cfg.get("show_minor_grid", True))
    minor_x_divisions = int(cfg.get("minor_x_divisions", 4))
    x_tick_step = float(cfg.get("x_tick_step", 0.0))
    x_tick_start = cfg.get("x_tick_start", None)
    x_tick_end = cfg.get("x_tick_end", None)
    x_tick_decimals = int(cfg.get("x_tick_decimals", 2))
    axis_label_font_size = cfg.get("axis_label_font_size", None)
    axis_tick_font_size = cfg.get("axis_tick_font_size", None)
    legend_font_size = cfg.get("legend_font_size", None)
    y_log_min = cfg.get("y_log_min", None)
    y_log_max = cfg.get("y_log_max", None)
    grid_axis = str(cfg.get("grid_axis", "both")).strip().lower()
    if grid_axis not in {"both", "x", "y"}:
        grid_axis = "both"
    axis_facecolor = cfg.get("axis_facecolor", None)
    figure_facecolor = cfg.get("figure_facecolor", None)
    grid_major_linestyle = str(cfg.get("grid_major_linestyle", "--"))
    grid_major_linewidth = float(cfg.get("grid_major_linewidth", 0.6))
    grid_major_alpha = float(cfg.get("grid_major_alpha", 0.35))
    grid_minor_linestyle = str(cfg.get("grid_minor_linestyle", ":"))
    grid_minor_linewidth = float(cfg.get("grid_minor_linewidth", 0.45))
    grid_minor_alpha = float(cfg.get("grid_minor_alpha", 0.45))
    axis_spine_linewidth = float(cfg.get("axis_spine_linewidth", 1.0))
    legend_facecolor = cfg.get("legend_facecolor", None)
    legend_edgecolor = cfg.get("legend_edgecolor", None)
    legend_framealpha = cfg.get("legend_framealpha", None)
    smooth_curves = bool(cfg.get("smooth_curves", True))
    fit_mode = str(cfg.get("fit_mode", "interp")).strip().lower()
    if fit_mode not in {"interp", "poly"}:
        raise ValueError("fit_mode must be 'interp' or 'poly'")
    poly_degree = int(cfg.get("poly_degree", 2))
    interp_points = int(cfg.get("interp_points", 280))
    smooth_window = int(cfg.get("smooth_window", 9))
    show_raw_markers = bool(cfg.get("show_raw_markers", True))
    plot_combined_fer_ber = bool(cfg.get("plot_combined_fer_ber", False))
    ber_linestyle = str(cfg.get("ber_linestyle", ":"))
    fer_ber_legend_title = cfg.get("fer_ber_legend_title", "Solid = FER, Dotted = BER")
    failed_bits_hist_metrics = normalize_failed_bits_hist_metrics(
        cfg.get("failed_bits_hist_metrics", ["min", "avg", "max"])
    )
    failed_bits_x_max = cfg.get("failed_bits_x_max", None)

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("Config must contain a non-empty 'models' list")

    model_rows: Dict[str, List[ResultRow]] = {}
    model_style: Dict[str, Dict[str, str]] = {}

    for m in models_cfg:
        name = m["name"]
        path = Path(m["path"])
        rows = parse_res_file(path)

        model_rows[name] = rows
        model_style[name] = {
            "label": m.get("label", name),
            "color": m.get("color", None),
            "linestyle": m.get("linestyle", "-"),
            "marker": m.get("marker", "o"),
            "linewidth": str(m.get("linewidth", 1.8)),
            "markersize": str(m.get("markersize", 5.5)),
            "markerfacecolor": m.get("markerfacecolor", "white"),
            "markeredgewidth": str(m.get("markeredgewidth", 1.0)),
        }

    if paper_monochrome:
        apply_monochrome_model_style(model_style)

    titles = cfg.get("titles", {})
    axis_labels = cfg.get("axis_labels", {})

    plot_metric(
        output_path=output_dir / cfg.get("fer_filename", "fer_paper.png"),
        title=titles.get("fer", "FER Comparison"),
        xlabel=axis_labels.get("fer_x", "Alpha"),
        ylabel=axis_labels.get("fer_y", "FER"),
        metric="fer",
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        smooth_curves=smooth_curves,
        fit_mode=fit_mode,
        poly_degree=poly_degree,
        interp_points=interp_points,
        smooth_window=smooth_window,
        show_raw_markers=show_raw_markers,
        figure_size=get_figure_size(cfg, "fer", (3.5, 2.7)),
        show_title=show_title,
        legend_loc=legend_loc,
        save_pdf=save_pdf,
        legend_right=legend_right,
        legend_right_anchor=legend_right_anchor,
        x_margin_frac=x_margin_frac,
        y_log_margin_decades=y_log_margin_decades,
        show_minor_grid=show_minor_grid,
        minor_x_divisions=minor_x_divisions,
        x_tick_step=x_tick_step,
        x_tick_start=x_tick_start,
        x_tick_end=x_tick_end,
        x_tick_decimals=x_tick_decimals,
        axis_label_font_size=axis_label_font_size,
        axis_tick_font_size=axis_tick_font_size,
        legend_font_size=legend_font_size,
        y_log_min=y_log_min,
        y_log_max=y_log_max,
        grid_axis=grid_axis,
        axis_facecolor=axis_facecolor,
        figure_facecolor=figure_facecolor,
        grid_major_linestyle=grid_major_linestyle,
        grid_major_linewidth=grid_major_linewidth,
        grid_major_alpha=grid_major_alpha,
        grid_minor_linestyle=grid_minor_linestyle,
        grid_minor_linewidth=grid_minor_linewidth,
        grid_minor_alpha=grid_minor_alpha,
        axis_spine_linewidth=axis_spine_linewidth,
        legend_facecolor=legend_facecolor,
        legend_edgecolor=legend_edgecolor,
        legend_framealpha=legend_framealpha,
    )

    if plot_combined_fer_ber:
        plot_fer_ber_combined(
            output_path=output_dir / cfg.get("fer_ber_filename", "fer_ber_combined.png"),
            title=titles.get("fer_ber", "FER/BER Comparison"),
            xlabel=axis_labels.get("fer_ber_x", axis_labels.get("fer_x", "Alpha")),
            ylabel=axis_labels.get("fer_ber_y", "Error Rate"),
            model_rows=model_rows,
            model_style=model_style,
            dpi=dpi,
            smooth_curves=smooth_curves,
            fit_mode=fit_mode,
            poly_degree=poly_degree,
            interp_points=interp_points,
            smooth_window=smooth_window,
            show_raw_markers=show_raw_markers,
            figure_size=get_figure_size(cfg, "fer_ber", (4.8, 3.6)),
            show_title=show_title,
            legend_loc=legend_loc,
            save_pdf=save_pdf,
            legend_right=legend_right,
            legend_right_anchor=legend_right_anchor,
            x_margin_frac=x_margin_frac,
            y_log_margin_decades=y_log_margin_decades,
            show_minor_grid=show_minor_grid,
            minor_x_divisions=minor_x_divisions,
            x_tick_step=x_tick_step,
            x_tick_start=x_tick_start,
            x_tick_end=x_tick_end,
            x_tick_decimals=x_tick_decimals,
            axis_label_font_size=axis_label_font_size,
            axis_tick_font_size=axis_tick_font_size,
            legend_font_size=legend_font_size,
            y_log_min=y_log_min,
            y_log_max=y_log_max,
            grid_axis=grid_axis,
            axis_facecolor=axis_facecolor,
            figure_facecolor=figure_facecolor,
            grid_major_linestyle=grid_major_linestyle,
            grid_major_linewidth=grid_major_linewidth,
            grid_major_alpha=grid_major_alpha,
            grid_minor_linestyle=grid_minor_linestyle,
            grid_minor_linewidth=grid_minor_linewidth,
            grid_minor_alpha=grid_minor_alpha,
            axis_spine_linewidth=axis_spine_linewidth,
            legend_facecolor=legend_facecolor,
            legend_edgecolor=legend_edgecolor,
            legend_framealpha=legend_framealpha,
            ber_linestyle=ber_linestyle,
            legend_title=fer_ber_legend_title,
        )

    plot_metric(
        output_path=output_dir / cfg.get("ber_filename", "ber_paper.png"),
        title=titles.get("ber", "BER Comparison"),
        xlabel=axis_labels.get("ber_x", "Alpha"),
        ylabel=axis_labels.get("ber_y", "BER"),
        metric="ber",
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        smooth_curves=smooth_curves,
        fit_mode=fit_mode,
        poly_degree=poly_degree,
        interp_points=interp_points,
        smooth_window=smooth_window,
        show_raw_markers=show_raw_markers,
        figure_size=get_figure_size(cfg, "ber", (3.5, 2.7)),
        show_title=show_title,
        legend_loc=legend_loc,
        save_pdf=save_pdf,
        legend_right=legend_right,
        legend_right_anchor=legend_right_anchor,
        x_margin_frac=x_margin_frac,
        y_log_margin_decades=y_log_margin_decades,
        show_minor_grid=show_minor_grid,
        minor_x_divisions=minor_x_divisions,
        x_tick_step=x_tick_step,
        x_tick_start=x_tick_start,
        x_tick_end=x_tick_end,
        x_tick_decimals=x_tick_decimals,
        axis_label_font_size=axis_label_font_size,
        axis_tick_font_size=axis_tick_font_size,
        legend_font_size=legend_font_size,
        y_log_min=y_log_min,
        y_log_max=y_log_max,
        grid_axis=grid_axis,
        axis_facecolor=axis_facecolor,
        figure_facecolor=figure_facecolor,
        grid_major_linestyle=grid_major_linestyle,
        grid_major_linewidth=grid_major_linewidth,
        grid_major_alpha=grid_major_alpha,
        grid_minor_linestyle=grid_minor_linestyle,
        grid_minor_linewidth=grid_minor_linewidth,
        grid_minor_alpha=grid_minor_alpha,
        axis_spine_linewidth=axis_spine_linewidth,
        legend_facecolor=legend_facecolor,
        legend_edgecolor=legend_edgecolor,
        legend_framealpha=legend_framealpha,
    )

    plot_metric(
        output_path=output_dir / cfg.get("iter_filename", "iter_paper.png"),
        title=titles.get("iter", "Average Iteration Comparison"),
        xlabel=axis_labels.get("iter_x", "Alpha"),
        ylabel=axis_labels.get("iter_y", "Average Iterations"),
        metric="iter",
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        smooth_curves=False,
        fit_mode=fit_mode,
        poly_degree=poly_degree,
        interp_points=interp_points,
        smooth_window=smooth_window,
        show_raw_markers=False,
        figure_size=get_figure_size(cfg, "iter", (3.5, 2.7)),
        show_title=show_title,
        legend_loc=legend_loc,
        save_pdf=save_pdf,
        legend_right=legend_right,
        legend_right_anchor=legend_right_anchor,
        x_margin_frac=x_margin_frac,
        y_log_margin_decades=y_log_margin_decades,
        show_minor_grid=show_minor_grid,
        minor_x_divisions=minor_x_divisions,
        x_tick_step=x_tick_step,
        x_tick_start=x_tick_start,
        x_tick_end=x_tick_end,
        x_tick_decimals=x_tick_decimals,
        axis_label_font_size=axis_label_font_size,
        axis_tick_font_size=axis_tick_font_size,
        legend_font_size=legend_font_size,
        y_log_min=y_log_min,
        y_log_max=y_log_max,
        grid_axis=grid_axis,
        axis_facecolor=axis_facecolor,
        figure_facecolor=figure_facecolor,
        grid_major_linestyle=grid_major_linestyle,
        grid_major_linewidth=grid_major_linewidth,
        grid_major_alpha=grid_major_alpha,
        grid_minor_linestyle=grid_minor_linestyle,
        grid_minor_linewidth=grid_minor_linewidth,
        grid_minor_alpha=grid_minor_alpha,
        axis_spine_linewidth=axis_spine_linewidth,
        legend_facecolor=legend_facecolor,
        legend_edgecolor=legend_edgecolor,
        legend_framealpha=legend_framealpha,
    )

    plot_failed_bits_histogram(
        output_path=output_dir / cfg.get("expected_failed_bits_filename", "expected_failed_bits_all_models.png"),
        title=titles.get("expected_failed_bits", "Failed Bits per Tested Frame (FER x FailedAvg)"),
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        hist_metrics=["expected_failed_bits"],
        show_title=show_title,
        save_pdf=save_pdf,
    )

    plot_failed_bits_summary(
        output_path=output_dir / cfg.get("failed_bits_filename", "failed_bits_paper.png"),
        title=titles.get("failed_bits", "Remaining Uncorrected Bits (Min/Avg/Max)"),
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        failed_bits_x_max=failed_bits_x_max,
        figure_size=get_figure_size(cfg, "failed_bits", (7.2, 2.8)),
        show_title=show_title,
        save_pdf=save_pdf,
    )

    plot_failed_bits_histogram(
        output_path=output_dir / cfg.get("failed_bits_hist_filename", "failed_bits_histogram.png"),
        title=titles.get("failed_bits_hist", "Failed Bits by Alpha and Model"),
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        hist_metrics=failed_bits_hist_metrics,
        failed_bits_x_max=failed_bits_x_max,
        show_title=show_title,
        save_pdf=save_pdf,
    )

    print(f"Saved FER/BER/Iter/ExpectedFailedBits/FailedBits plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
