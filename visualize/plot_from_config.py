import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


LINE_RE = re.compile(
    r"^\s*"
    r"(?P<alpha>[0-9]*\.?[0-9]+)\s+"
    r"(?P<nber>\d+)\s*\((?P<ber>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbfer>\d+)\s*\((?P<fer>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbtested>\d+)\s+"
    r"(?P<iter_avg>[0-9]*\.?[0-9]+)\((?P<iter_max>\d+)\)"
)

FAILED_BITS_RE = re.compile(
    r"\s+\d+\(\d+\)\s+"
    r"(?P<failed_min>[0-9]*\.?[0-9]+)\/(?P<failed_avg>[0-9]*\.?[0-9]+)\/(?P<failed_max>[0-9]*\.?[0-9]+)"
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
            failed_match = FAILED_BITS_RE.search(line)
            rows_by_alpha[alpha] = ResultRow(
                alpha=alpha,
                ber=float(m.group("ber")),
                fer=float(m.group("fer")),
                iter_avg=float(m.group("iter_avg")),
                nbtested=int(m.group("nbtested")),
                failed_min=float(failed_match.group("failed_min")) if failed_match else float("nan"),
                failed_avg=float(failed_match.group("failed_avg")) if failed_match else float("nan"),
                failed_max=float(failed_match.group("failed_max")) if failed_match else float("nan"),
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
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#666666",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "savefig.bbox": "tight",
        }
    )


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
    raise ValueError(f"Unsupported metric: {metric}")


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
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

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

        if metric in {"fer", "ber"}:
            y = np.clip(y, 1e-15, None)

        x_plot = x
        y_plot = y
        if smooth_curves and metric in {"fer", "ber"} and len(x) >= 2:
            if fit_mode == "poly":
                x_plot, y_plot = poly_log_fit(x, y, degree=poly_degree, n_points=interp_points)
            else:
                x_plot, y_plot = smooth_log_interp(x, y, n_points=interp_points, window=smooth_window)

        ax.plot(
            x_plot,
            y_plot,
            label=style.get("label", model_name),
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=(None if (smooth_curves and metric in {"fer", "ber"}) else style.get("marker", "o")),
            linewidth=float(style.get("linewidth", 1.8)),
            markersize=float(style.get("markersize", 5.5)),
            markerfacecolor=style.get("markerfacecolor", "white"),
            markeredgewidth=float(style.get("markeredgewidth", 1.0)),
        )

        if show_raw_markers and smooth_curves and metric in {"fer", "ber"}:
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

    if metric in {"fer", "ber"}:
        ax.set_yscale("log")

    ax.set_xlabel("Alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, which="major", linestyle="--")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6)
    ax.minorticks_on()

    # Higher alpha on the left, lower alpha on the right.
    all_alphas = [r.alpha for rows in model_rows.values() for r in rows]
    ax.set_xlim(max(all_alphas), min(all_alphas))

    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_failed_bits_summary(
    output_path: Path,
    title: str,
    model_rows: Dict[str, List[ResultRow]],
    model_style: Dict[str, Dict[str, str]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
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
        ax.set_xlim(max(all_alphas), min(all_alphas))

    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, 1.04))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
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
    smooth_curves = bool(cfg.get("smooth_curves", True))
    fit_mode = str(cfg.get("fit_mode", "interp")).strip().lower()
    if fit_mode not in {"interp", "poly"}:
        raise ValueError("fit_mode must be 'interp' or 'poly'")
    poly_degree = int(cfg.get("poly_degree", 2))
    interp_points = int(cfg.get("interp_points", 280))
    smooth_window = int(cfg.get("smooth_window", 9))
    show_raw_markers = bool(cfg.get("show_raw_markers", True))

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

    titles = cfg.get("titles", {})

    plot_metric(
        output_path=output_dir / cfg.get("fer_filename", "fer_paper.png"),
        title=titles.get("fer", "FER Comparison"),
        ylabel="FER",
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
    )

    plot_metric(
        output_path=output_dir / cfg.get("ber_filename", "ber_paper.png"),
        title=titles.get("ber", "BER Comparison"),
        ylabel="BER",
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
    )

    plot_metric(
        output_path=output_dir / cfg.get("iter_filename", "iter_paper.png"),
        title=titles.get("iter", "Average Iteration Comparison"),
        ylabel="Average Iterations",
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
    )

    plot_failed_bits_summary(
        output_path=output_dir / cfg.get("failed_bits_filename", "failed_bits_paper.png"),
        title=titles.get("failed_bits", "Remaining Uncorrected Bits (Min/Avg/Max)"),
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
    )

    print(f"Saved FER/BER/Iter/FailedBits plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
