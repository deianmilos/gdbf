"""
plot_from_config.py — CLI entry point for generating paper-style plots.

Usage
-----
    python visualize/plot_from_config.py --config visualize/plot_config_irisc_all_models_full.json

Modules
-------
    plot_parse   — simulation.res parser  (ResultRow, parse_res_file)
    plot_style   — style helpers          (apply_paper_style, metric_values, ...)
    plot_curves  — curve fitting          (smooth_log_interp, poly_log_fit)
    plot_figures — plot functions         (plot_metric, plot_fer_ber_combined, plot_failed_bits_summary)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

from plot_figures import (
    plot_failed_bits_bar,
    plot_failed_bits_histogram,
    plot_failed_bits_summary,
    plot_fer_ber_combined,
    plot_metric,
)
from plot_parse import ResultRow, parse_res_file
from plot_style import apply_monochrome_model_style, apply_paper_style, get_figure_size


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _build_model_data(
    models_cfg: list,
) -> tuple[Dict[str, List[ResultRow]], Dict[str, Dict[str, str]]]:
    model_rows: Dict[str, List[ResultRow]] = {}
    model_style: Dict[str, Dict[str, str]] = {}
    for m in models_cfg:
        name = m["name"]
        rows = parse_res_file(Path(m["path"]))
        model_rows[name] = rows
        model_style[name] = {
            "label":           m.get("label", name),
            "color":           m.get("color", None),
            "linestyle":       m.get("linestyle", "-"),
            "marker":          m.get("marker", "o"),
            "linewidth":       str(m.get("linewidth", 1.8)),
            "markersize":      str(m.get("markersize", 5.5)),
            "markerfacecolor": m.get("markerfacecolor", "white"),
            "markeredgewidth": str(m.get("markeredgewidth", 1.0)),
        }
    return model_rows, model_style


def _parse_alpha_values(cfg: dict) -> Set[float] | None:
    values = cfg.get("alpha_values", None)
    if values is None:
        return None
    if not isinstance(values, list):
        sys.exit("alpha_values must be a JSON list of numbers, e.g. [0.07, 0.06, 0.05]")

    parsed: Set[float] = set()
    for v in values:
        try:
            parsed.add(float(v))
        except (TypeError, ValueError):
            sys.exit(f"Invalid alpha value in alpha_values: {v!r}")

    if len(parsed) == 0:
        sys.exit("alpha_values was provided but empty; include at least one alpha.")
    return parsed


def _filter_model_rows_by_alpha_values(
    model_rows: Dict[str, List[ResultRow]],
    alpha_values: Set[float] | None,
) -> Dict[str, List[ResultRow]]:
    if alpha_values is None:
        return model_rows

    filtered: Dict[str, List[ResultRow]] = {}
    for name, rows in model_rows.items():
        kept = [r for r in rows if r.alpha in alpha_values]
        if len(kept) == 0:
            sys.exit(
                f"No rows left for model '{name}' after alpha_values filtering. "
                f"Check that selected alphas exist in {name} result file."
            )
        filtered[name] = kept
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style FER/BER/iteration/failed-bits plots from JSON config."
    )
    parser.add_argument("--config", required=True, help="Path to JSON plotting config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    apply_paper_style(font_size=int(cfg.get("font_size", 11)))

    output_dir   = Path(cfg.get("output_dir", "visualize/plots/paper"))
    dpi          = int(cfg.get("dpi", 400))
    save_pdf     = bool(cfg.get("save_pdf", True))
    show_title   = bool(cfg.get("show_title", False))
    paper_mono   = bool(cfg.get("paper_monochrome", True))

    legend_loc          = str(cfg.get("legend_loc", "best"))
    legend_right        = bool(cfg.get("legend_right", False))
    legend_right_anchor = tuple(cfg.get("legend_right_anchor", [1.02, 0.5]))
    legend_font_size    = cfg.get("legend_font_size", None)
    legend_facecolor    = cfg.get("legend_facecolor", None)
    legend_edgecolor    = cfg.get("legend_edgecolor", None)
    legend_framealpha   = cfg.get("legend_framealpha", None)

    smooth_curves    = bool(cfg.get("smooth_curves", True))
    fit_mode         = str(cfg.get("fit_mode", "interp")).strip().lower()
    if fit_mode not in {"interp", "poly"}:
        sys.exit("fit_mode must be 'interp' or 'poly'")
    poly_degree      = int(cfg.get("poly_degree", 2))
    interp_points    = int(cfg.get("interp_points", 280))
    smooth_window    = int(cfg.get("smooth_window", 9))
    show_raw_markers = bool(cfg.get("show_raw_markers", True))

    x_margin_frac  = float(cfg.get("x_margin_frac", 0.06))
    minor_x_divs   = int(cfg.get("minor_x_divisions", 4))
    x_tick_step    = float(cfg.get("x_tick_step", 0.0))
    x_tick_start   = cfg.get("x_tick_start", None)
    x_tick_end     = cfg.get("x_tick_end", None)
    x_tick_dec     = int(cfg.get("x_tick_decimals", 2))
    alpha_x_min    = float(cfg["alpha_x_min"]) if "alpha_x_min" in cfg else None
    alpha_x_max    = float(cfg["alpha_x_max"]) if "alpha_x_max" in cfg else None

    y_log_margin  = float(cfg.get("y_log_margin_decades", 0.12))
    y_log_min     = cfg.get("y_log_min", None)
    y_log_max     = cfg.get("y_log_max", None)

    show_minor_grid  = bool(cfg.get("show_minor_grid", True))
    grid_axis        = str(cfg.get("grid_axis", "both")).strip().lower()
    if grid_axis not in {"both", "x", "y"}:
        grid_axis = "both"
    grid_major_ls    = str(cfg.get("grid_major_linestyle", "--"))
    grid_major_lw    = float(cfg.get("grid_major_linewidth", 0.6))
    grid_major_alpha = float(cfg.get("grid_major_alpha", 0.35))
    grid_minor_ls    = str(cfg.get("grid_minor_linestyle", ":"))
    grid_minor_lw    = float(cfg.get("grid_minor_linewidth", 0.45))
    grid_minor_alpha = float(cfg.get("grid_minor_alpha", 0.45))

    axis_label_fs    = cfg.get("axis_label_font_size", None)
    axis_tick_fs     = cfg.get("axis_tick_font_size", None)
    axis_facecolor   = cfg.get("axis_facecolor", None)
    figure_facecolor = cfg.get("figure_facecolor", None)
    axis_spine_lw    = float(cfg.get("axis_spine_linewidth", 1.0))

    plot_ber_only       = bool(cfg.get("plot_ber", True))
    plot_combined       = bool(cfg.get("plot_combined_fer_ber", False))
    plot_fer_poly       = bool(cfg.get("plot_fer_poly", False))
    plot_failed_hist    = bool(cfg.get("plot_failed_bits_histogram", False))
    plot_failed_summary = bool(cfg.get("plot_failed_bits_summary", False))
    failed_hist_bins    = int(cfg.get("failed_bits_histogram_bins", 16))
    failed_hist_alpha   = float(cfg.get("failed_bits_histogram_alpha", 0.45))
    ber_linestyle       = str(cfg.get("ber_linestyle", ":"))
    ferber_legend_title = cfg.get("fer_ber_legend_title", "")

    fer_poly_degree        = int(cfg.get("fer_poly_degree", poly_degree))
    fer_poly_interp_points = int(cfg.get("fer_poly_interp_points", interp_points))
    fer_poly_smooth_window = int(cfg.get("fer_poly_smooth_window", smooth_window))
    fer_poly_show_raw      = bool(cfg.get("fer_poly_show_raw_markers", show_raw_markers))

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        sys.exit("Config must contain a non-empty 'models' list")

    model_rows, model_style = _build_model_data(models_cfg)
    alpha_values = _parse_alpha_values(cfg)
    model_rows = _filter_model_rows_by_alpha_values(model_rows, alpha_values)

    if paper_mono:
        apply_monochrome_model_style(model_style)

    titles      = cfg.get("titles", {})
    axis_labels = cfg.get("axis_labels", {})

    shared = dict(
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        smooth_curves=smooth_curves,
        fit_mode=fit_mode,
        poly_degree=poly_degree,
        interp_points=interp_points,
        smooth_window=smooth_window,
        show_raw_markers=show_raw_markers,
        show_title=show_title,
        legend_loc=legend_loc,
        save_pdf=save_pdf,
        legend_right=legend_right,
        legend_right_anchor=legend_right_anchor,
        x_margin_frac=x_margin_frac,
        y_log_margin_decades=y_log_margin,
        show_minor_grid=show_minor_grid,
        minor_x_divisions=minor_x_divs,
        x_tick_step=x_tick_step,
        x_tick_start=x_tick_start,
        x_tick_end=x_tick_end,
        x_tick_decimals=x_tick_dec,
        axis_label_font_size=axis_label_fs,
        axis_tick_font_size=axis_tick_fs,
        legend_font_size=legend_font_size,
        y_log_min=y_log_min,
        y_log_max=y_log_max,
        grid_axis=grid_axis,
        axis_facecolor=axis_facecolor,
        figure_facecolor=figure_facecolor,
        grid_major_linestyle=grid_major_ls,
        grid_major_linewidth=grid_major_lw,
        grid_major_alpha=grid_major_alpha,
        grid_minor_linestyle=grid_minor_ls,
        grid_minor_linewidth=grid_minor_lw,
        grid_minor_alpha=grid_minor_alpha,
        axis_spine_linewidth=axis_spine_lw,
        legend_facecolor=legend_facecolor,
        legend_edgecolor=legend_edgecolor,
        legend_framealpha=legend_framealpha,
        alpha_x_min_override=alpha_x_min,
        alpha_x_max_override=alpha_x_max,
    )

    plot_metric(
        output_path=output_dir / cfg.get("fer_filename", "fer_paper.png"),
        title=titles.get("fer", "FER Comparison"),
        xlabel=axis_labels.get("fer_x", "Alpha"),
        ylabel=axis_labels.get("fer_y", "FER"),
        metric="fer",
        figure_size=get_figure_size(cfg, "fer", (3.5, 2.7)),
        **shared,
    )

    if plot_fer_poly:
        plot_metric(
            output_path=output_dir / cfg.get("fer_poly_filename", "fer_poly_paper.png"),
            title=titles.get("fer_poly", titles.get("fer", "FER Comparison") + " (poly fit)"),
            xlabel=axis_labels.get("fer_x", "Alpha"),
            ylabel=axis_labels.get("fer_y", "FER"),
            metric="fer",
            figure_size=get_figure_size(cfg, "fer", (3.5, 2.7)),
            **{
                **shared,
                "fit_mode": "poly",
                "poly_degree": fer_poly_degree,
                "interp_points": fer_poly_interp_points,
                "smooth_window": fer_poly_smooth_window,
                "show_raw_markers": fer_poly_show_raw,
            },
        )

    if plot_ber_only:
        plot_metric(
            output_path=output_dir / cfg.get("ber_filename", "ber_paper.png"),
            title=titles.get("ber", "BER Comparison"),
            xlabel=axis_labels.get("ber_x", "Alpha"),
            ylabel=axis_labels.get("ber_y", "BER"),
            metric="ber",
            figure_size=get_figure_size(cfg, "ber", (3.5, 2.7)),
            **shared,
        )

    if plot_combined:
        plot_fer_ber_combined(
            output_path=output_dir / cfg.get("fer_ber_filename", "fer_ber_combined.png"),
            title=titles.get("fer_ber", "FER/BER Comparison"),
            xlabel=axis_labels.get("fer_ber_x", axis_labels.get("fer_x", "Alpha")),
            ylabel=axis_labels.get("fer_ber_y", "Error Rate"),
            figure_size=get_figure_size(cfg, "fer_ber", (4.8, 3.6)),
            ber_linestyle=ber_linestyle,
            legend_title=ferber_legend_title,
            **shared,
        )

    plot_metric(
        output_path=output_dir / cfg.get("iter_filename", "iter_paper.png"),
        title=titles.get("iter", "Average Iteration Comparison"),
        xlabel=axis_labels.get("iter_x", "Alpha"),
        ylabel=axis_labels.get("iter_y", "Average Iterations"),
        metric="iter",
        figure_size=get_figure_size(cfg, "iter", (3.5, 2.7)),
        **{
            **shared,
            "smooth_curves": bool(cfg.get("iter_smooth_curves", True)),
            "show_raw_markers": bool(cfg.get("iter_show_raw_markers", True)),
            "fit_mode": str(cfg.get("iter_fit_mode", "poly")).strip().lower(),
            "poly_degree": int(cfg.get("iter_poly_degree", 2)),
            "interp_points": int(cfg.get("iter_interp_points", cfg.get("interp_points", 200))),
            "smooth_window": int(cfg.get("iter_smooth_window", cfg.get("smooth_window", 3))),
        },
    )

    if plot_failed_summary:
        plot_failed_bits_summary(
            output_path_min=output_dir / cfg.get("failed_bits_min_filename", "failed_bits_min.png"),
            output_path_avg=output_dir / cfg.get("failed_bits_avg_filename", "failed_bits_avg.png"),
            output_path_max=output_dir / cfg.get("failed_bits_max_filename", "failed_bits_max.png"),
            title_min=titles.get("failed_bits_min", "Min residual bit errors in failed frames"),
            title_avg=titles.get("failed_bits_avg", "Avg residual bit errors in failed frames"),
            title_max=titles.get("failed_bits_max", "Max residual bit errors in failed frames"),
            model_rows=model_rows,
            model_style=model_style,
            dpi=dpi,
            figure_size=get_figure_size(cfg, "failed_bits_per", (5.2, 3.8)),
            show_title=show_title,
            save_pdf=save_pdf,
            alpha_x_min_override=alpha_x_min,
            alpha_x_max_override=alpha_x_max,
        )

    _bar_kw = dict(
        model_rows=model_rows,
        model_style=model_style,
        dpi=dpi,
        figure_size=get_figure_size(cfg, "failed_bits_bar", (8.0, 4.0)),
        show_title=show_title,
        save_pdf=save_pdf,
        bar_alpha=float(cfg.get("failed_bits_bar_alpha", 0.75)),
        alpha_x_min_override=alpha_x_min,
        alpha_x_max_override=alpha_x_max,
    )
    plot_failed_bits_bar(
        output_path=output_dir / cfg.get("failed_bits_bar_min_filename", "failed_bits_bar_min.png"),
        title=titles.get("failed_bits_bar_min", "Min residual bit errors per alpha (bar)"),
        metric="failed_min",
        ylabel="Min residual bit errors",
        **_bar_kw,
    )
    plot_failed_bits_bar(
        output_path=output_dir / cfg.get("failed_bits_bar_avg_filename", "failed_bits_bar_avg.png"),
        title=titles.get("failed_bits_bar_avg", "Avg residual bit errors per alpha (bar)"),
        metric="failed_avg",
        ylabel="Avg residual bit errors",
        **_bar_kw,
    )
    plot_failed_bits_bar(
        output_path=output_dir / cfg.get("failed_bits_bar_max_filename", "failed_bits_bar_max.png"),
        title=titles.get("failed_bits_bar_max", "Max residual bit errors per alpha (bar)"),
        metric="failed_max",
        ylabel="Max residual bit errors",
        **_bar_kw,
    )

    if plot_failed_hist:
        plot_failed_bits_histogram(
            output_path=output_dir / cfg.get("failed_bits_histogram_filename", "failed_bits_histogram.png"),
            title=titles.get("failed_bits_histogram", "Number of failed frames per alpha"),
            model_rows=model_rows,
            model_style=model_style,
            dpi=dpi,
            figure_size=get_figure_size(cfg, "failed_bits_histogram", (6.6, 4.0)),
            show_title=show_title,
            save_pdf=save_pdf,
            bins=failed_hist_bins,
            alpha_value=failed_hist_alpha,
            alpha_x_min_override=alpha_x_min,
            alpha_x_max_override=alpha_x_max,
        )

    print(f"Plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
