import argparse
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization.\n"
        "If you use conda: conda install -n gdbf-ml matplotlib\n"
        "Or with pip: pip install matplotlib"
    ) from exc

import numpy as np


LINE_RE = re.compile(
    r"^\s*"
    r"(?P<alpha>[0-9]*\.?[0-9]+)\s+"
    r"(?P<nber>\d+)\s*\((?P<ber>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbfer>\d+)\s*\((?P<fer>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbtested>\d+)\s+"
    r"(?P<iter_avg>[0-9]*\.?[0-9]+)\((?P<iter_max>\d+)\)"
)


@dataclass
class ResultRow:
    alpha: float
    fer: float


def parse_res_file(path: Path):
    rows_by_alpha = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("alpha"):
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            alpha = float(m.group("alpha"))
            rows_by_alpha[alpha] = ResultRow(
                alpha=alpha,
                fer=float(m.group("fer")),
            )

    if not rows_by_alpha:
        raise ValueError(f"No valid result rows parsed from {path}")

    return [rows_by_alpha[a] for a in sorted(rows_by_alpha.keys(), reverse=True)]


def rows_to_series(rows):
    return {
        "alpha": [r.alpha for r in rows],
        "fer": [r.fer for r in rows],
    }


def filter_positive_fer(alphas, fers):
    filtered_alpha = []
    filtered_fer = []

    for alpha, fer in zip(alphas, fers):
        if fer > 0:
            filtered_alpha.append(alpha)
            filtered_fer.append(fer)

    return filtered_alpha, filtered_fer


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#666666",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
        }
    )


def smooth_fer_curve(alphas, fers, interp_points=350, smooth_window=11):
    x, y = filter_positive_fer(alphas, fers)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0:
        return x, y

    if x.size < 2:
        return x, np.clip(y, 1e-15, None)

    # Interpolate and smooth in log-domain for a cleaner FER trend.
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = np.clip(y[sort_idx], 1e-15, None)
    log_y = np.log10(y_sorted)

    dense_x = np.linspace(x_sorted.min(), x_sorted.max(), int(interp_points))
    dense_log_y = np.interp(dense_x, x_sorted, log_y)

    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    if smooth_window > 1:
        pad = smooth_window // 2
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        padded = np.pad(dense_log_y, (pad, pad), mode="edge")
        dense_log_y = np.convolve(padded, kernel, mode="valid")

    dense_y = np.power(10.0, dense_log_y)

    # Plot from larger alpha to smaller alpha.
    return dense_x[::-1], dense_y[::-1]


def make_plot(
    baseline_rows,
    ml_rows,
    title,
    output_path,
    show_plot,
    interp_points,
    smooth_window,
    show_raw,
):
    baseline = rows_to_series(baseline_rows)
    ml = rows_to_series(ml_rows)

    baseline_x, baseline_y = filter_positive_fer(baseline["alpha"], baseline["fer"])
    ml_x, ml_y = filter_positive_fer(ml["alpha"], ml["fer"])

    base_x_s, base_y_s = smooth_fer_curve(
        baseline_x, baseline_y, interp_points=interp_points, smooth_window=smooth_window
    )
    ml_x_s, ml_y_s = smooth_fer_curve(
        ml_x, ml_y, interp_points=interp_points, smooth_window=smooth_window
    )

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    ax.plot(
        base_x_s,
        base_y_s,
        linestyle="-",
        linewidth=1.8,
        marker="x",
        markevery=max(1, len(base_x_s) // 9),
        markersize=7,
        markeredgewidth=1.2,
        label="Baseline GDBF",
        color="black",
    )
    ax.plot(
        ml_x_s,
        ml_y_s,
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markevery=max(1, len(ml_x_s) // 9),
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="ML-assisted GDBF",
        color="#b22222",
    )

    if show_raw:
        ax.scatter(
            baseline_x,
            baseline_y,
            s=18,
            marker="x",
            alpha=0.7,
            color="black",
            label="Baseline raw",
        )
        ax.scatter(
            ml_x,
            ml_y,
            s=18,
            marker="o",
            alpha=0.7,
            facecolors="white",
            edgecolors="#b22222",
            label="ML raw",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("FER")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="major", linestyle="--")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.7)
    ax.minorticks_on()

    # Force x-axis from larger alpha to smaller alpha.
    all_alphas = baseline["alpha"] + ml["alpha"]
    ax.set_xlim(max(all_alphas), min(all_alphas))
    # ax.set_xlim(0.01, min(all_alphas))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description="Plot FER comparison from .res files")
    p.add_argument(
        "--baseline",
        default="results/wifin_r_1_2/baseline/simulation.res",
        help="Path to baseline .res file",
    )
    p.add_argument(
        "--ml",
        default="results/wifin_r_1_2/ml/simulation.res",
        help="Path to ML .res file",
    )
    p.add_argument(
        "--output",
        default="visualize/plots/fer_comparison.png",
        help="Output image path",
    )
    p.add_argument(
        "--title",
        default="GDBF FER Comparison",
        help="Plot title",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window",
    )
    p.add_argument(
        "--interp-points",
        type=int,
        default=6,
        help="Number of interpolated points used for smooth plotting",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=6,
        help="Odd moving-average window (in interpolated points) applied in log-domain",
    )
    p.add_argument(
        "--show-raw",
        action="store_true",
        help="Overlay raw FER markers on top of smooth curves",
    )
    return p


def main():
    args = build_parser().parse_args()

    baseline_path = Path(args.baseline)
    ml_path = Path(args.ml)
    output_path = Path(args.output)

    baseline_rows = parse_res_file(baseline_path)
    ml_rows = parse_res_file(ml_path)

    make_plot(
        baseline_rows,
        ml_rows,
        title=args.title,
        output_path=output_path,
        show_plot=args.show,
        interp_points=args.interp_points,
        smooth_window=args.smooth_window,
        show_raw=args.show_raw,
    )


if __name__ == "__main__":
    main()
