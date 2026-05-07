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


def make_plot(baseline_rows, ml_rows, title, output_path, show_plot):
    baseline = rows_to_series(baseline_rows)
    ml = rows_to_series(ml_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        baseline["alpha"],
        baseline["fer"],
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=6,
        label="Baseline GDBF",
        color="#d1495b",
    )
    ax.plot(
        ml["alpha"],
        ml["fer"],
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=6,
        label="ML-assisted GDBF",
        color="#00798c",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Alpha (crossover probability)")
    ax.set_ylabel("FER (log scale)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.8)

    # Force x-axis from larger alpha to smaller alpha.
    all_alphas = baseline["alpha"] + ml["alpha"]
    ax.set_xlim(max(all_alphas), min(all_alphas))

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
        default="results/Res_baseline.res",
        help="Path to baseline .res file",
    )
    p.add_argument(
        "--ml",
        default="results/Res_ml_v1.res",
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
    )


if __name__ == "__main__":
    main()
