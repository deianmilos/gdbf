"""
plot_curves.py — Curve smoothing / fitting helpers for error-rate plots.

Both functions operate in log10 domain and return arrays sorted descending in x
(high alpha on the left, low alpha on the right, matching the standard BER/FER convention).
"""

import numpy as np


def smooth_log_interp(
    x: np.ndarray, y: np.ndarray, n_points: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate y onto a dense x grid and apply a box-car smoothing kernel,
    both in log10 domain.  Returns (x_dense_desc, y_dense_desc).
    """
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

    return xd[::-1], np.power(10.0, yd_log)[::-1]


def poly_log_fit(
    x: np.ndarray, y: np.ndarray, degree: int, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a polynomial of the given degree to log10(y) as a function of x,
    then evaluate on a dense grid.  Returns (x_dense_desc, y_dense_desc).
    """
    idx = np.argsort(x)
    xs = np.asarray(x[idx], dtype=float)
    ys = np.clip(np.asarray(y[idx], dtype=float), 1e-15, None)

    deg = max(1, min(int(degree), len(xs) - 1))
    coeff = np.polyfit(xs, np.log10(ys), deg)

    xd = np.linspace(xs.min(), xs.max(), int(n_points))
    return xd[::-1], np.power(10.0, np.polyval(coeff, xd))[::-1]
