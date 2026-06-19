"""
plot_style.py — Matplotlib style helpers, model style application, and metric accessors.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot_parse import ResultRow

MONO_LINESTYLES = ["-", "--", "-.", ":"]
MONO_MARKERS = ["o", "s", "^", "D", "v", "x", "P", "*"]


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
    """Read <prefix>_fig_width and <prefix>_fig_height from cfg, falling back to default."""
    w = float(cfg.get(f"{key_prefix}_fig_width", default[0]))
    h = float(cfg.get(f"{key_prefix}_fig_height", default[1]))
    return (w, h)


def apply_monochrome_model_style(model_style: Dict[str, Dict[str, str]]) -> None:
    """Override all model colours to black, assigning distinct linestyles and markers."""
    for i, name in enumerate(model_style.keys()):
        st = model_style[name]
        st["color"] = "#000000"
        st["linestyle"] = MONO_LINESTYLES[i % len(MONO_LINESTYLES)]
        st["marker"] = MONO_MARKERS[i % len(MONO_MARKERS)]
        st["markerfacecolor"] = "white"
        st["markeredgewidth"] = str(max(1.0, float(st.get("markeredgewidth", 1.0))))


def metric_values(rows: List[ResultRow], metric: str) -> np.ndarray:
    """Return a numpy array of the requested metric values from a list of ResultRow."""
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
    raise ValueError(f"Unsupported metric: {metric!r}")


def filter_positive_fer(rows: List[ResultRow]) -> List[ResultRow]:
    """Return only rows where FER > 0 (i.e. at least one frame failed)."""
    return [r for r in rows if r.fer > 0]
