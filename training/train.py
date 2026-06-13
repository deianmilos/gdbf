"""
Train absorbing-set escape models and export C headers.

Supports multiple variants in one run:
- budgeted (budget head + top-k)
- mask_only (direct mask prediction)

Usage:
    python training/train.py [--config training/configs/default.json] [--code CODE_NAME]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_CODE_NAME,
    MIN_TRAIN_SAMPLES_WARNING,
    dataset_path,
)
from data import load_data
from export import export_quantized_header, export_ref_header
from model import extract_weights_pytorch, require_torch, train_pytorch_variant


DEFAULT_CONFIG_PATH = Path("training/configs/default.json")


def _default_runtime_config():
    return {
        "code_name": DEFAULT_CODE_NAME,
        "dataset": {
            "path_template": "datasets/{code_name}/dataset.csv",
            "first_n_samples": None,
        },
        "training": {
            "batch_size": 32,
            "epochs": 500,
            "learning_rate": 0.001,
            "test_split": 0.2,
            "hidden1": 32,
            "hidden2": 16,
            "budget_loss_weight": 0.35,
            "valid_budgets": [0, 2, 4],
            "filter_invalid_budget_rows": True,
            "min_positive_count_for_output": 50,
            "max_pos_weight": 8.0,
            "mask_logit_threshold": 0.0,
        },
        "variants": ["budgeted", "mask_only"],
        "export": {
            "output_dir": "model",
            "active_variant": "budgeted",
            "write_active_alias": True,
            "write_legacy_active_name": True,
            "remove_legacy_headers": True,
        },
    }


def _load_json_config(path: Path):
    cfg = _default_runtime_config()

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)

        # shallow merge for top-level, then explicit nested merges
        cfg.update({k: v for k, v in user_cfg.items() if k not in ("dataset", "training", "export")})

        if "dataset" in user_cfg:
            cfg["dataset"].update(user_cfg["dataset"])
        if "training" in user_cfg:
            cfg["training"].update(user_cfg["training"])
        if "export" in user_cfg:
            cfg["export"].update(user_cfg["export"])

    return cfg


def _resolve_dataset_path(cfg, code_name):
    template = cfg["dataset"]["path_template"]
    return Path(template.format(code_name=code_name))


def _variant_output_paths(output_dir: Path, code_name: str, variant: str):
    suffix = f"_{variant}" if variant else ""
    return {
        "ref": output_dir / f"as_model_{code_name}{suffix}_ref.h",
        "quantized": output_dir / f"as_model_{code_name}{suffix}_quantized.h",
    }


def _canonical_output_paths(output_dir: Path, code_name: str):
    return {
        "ref": output_dir / f"as_model_{code_name}_ref.h",
        "quantized": output_dir / f"as_model_{code_name}_quantized.h",
    }


def _write_active_model_alias(code_name: str, output_dir: Path, active_variant: str, write_legacy_active_name: bool):
    if write_legacy_active_name:
        include_name = f"as_model_{code_name}_quantized.h"
    else:
        include_name = f"as_model_{code_name}_{active_variant}_quantized.h"

    alias_path = output_dir / "as_model_active_quantized.h"
    with alias_path.open("w", encoding="utf-8") as f:
        f.write("/* Auto-generated active quantized model alias. */\n")
        f.write("#ifndef AS_MODEL_ACTIVE_QUANTIZED_H\n")
        f.write("#define AS_MODEL_ACTIVE_QUANTIZED_H\n\n")
        f.write(f"#include \"{include_name}\"\n\n")
        f.write("#endif /* AS_MODEL_ACTIVE_QUANTIZED_H */\n")

    print(f"Updated active model alias -> {alias_path}")


def _remove_legacy_headers(output_dir: Path):
    legacy_headers = [
        output_dir / "as_model_weights.h",
        output_dir / "as_model_ref.h",
        output_dir / "as_model_quantized.h",
    ]
    for header in legacy_headers:
        if header.exists():
            header.unlink()
            print(f"Removed legacy header -> {header}")


def _normalize_export_params(params, metadata):
    out = {}
    for key, value in params.items():
        if value is None:
            out[key] = None
        elif isinstance(value, np.ndarray):
            out[key] = np.asarray(value, dtype=np.float64)
        else:
            out[key] = value

    out["variant"] = metadata["variant"]
    out["valid_budgets"] = metadata.get("valid_budgets", [0, 2, 4])
    out["mask_logit_threshold"] = metadata.get("mask_logit_threshold", 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train ML escape model variants and export C headers.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Optional code override (replaces config code_name)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_json_config(cfg_path)

    code_name = args.code if args.code else cfg.get("code_name", DEFAULT_CODE_NAME)

    dataset_file = _resolve_dataset_path(cfg, code_name)
    if not dataset_file.exists():
        fallback = dataset_path(code_name)
        if fallback.exists():
            dataset_file = fallback

    if not dataset_file.exists():
        print(f"ERROR: {dataset_file} not found.")
        print("Run the GDBF simulator first to generate training data.")
        raise SystemExit(1)

    first_n_samples = cfg.get("dataset", {}).get("first_n_samples", None)
    dataset_options = cfg.get("dataset", {}).get("subset", None)
    X, y, _, _ = load_data(
        dataset_file,
        first_n_samples=first_n_samples,
        dataset_options=dataset_options,
    )

    output_dir = Path(cfg["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in cfg.get("variants", []) if v.strip()]
    if not variants:
        print("ERROR: No variants configured. Set variants in JSON config.")
        raise SystemExit(1)

    active_variant = cfg["export"].get("active_variant", variants[0])
    if active_variant not in variants:
        print(f"WARNING: active_variant '{active_variant}' not in variants list. Using '{variants[0]}'.")
        active_variant = variants[0]

    print(f"Using code: {code_name}")
    print(f"Config: {cfg_path}")
    print(f"Dataset: {dataset_file}")
    print(f"Variants: {variants}")

    if len(X) < MIN_TRAIN_SAMPLES_WARNING:
        print(f"WARNING: Only {len(X)} samples. Need more data for reliable training.")
        print("Run simulator with more Monte Carlo iterations or lower SNR.")

    require_torch()

    training_settings = cfg.get("training", {})

    for variant in variants:
        print(f"\n=== Training variant: {variant} ===\n")

        model, mean, std, output_bits, metadata = train_pytorch_variant(
            X,
            y,
            variant=variant,
            settings=training_settings,
        )

        raw_params = extract_weights_pytorch(model)
        params = _normalize_export_params(raw_params, metadata)

        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        variant_paths = _variant_output_paths(output_dir, code_name, variant)

        print("--- Exporting variant headers ---")
        export_ref_header(params, mean, std, output_bits, str(variant_paths["ref"]))
        export_quantized_header(params, mean, std, output_bits, str(variant_paths["quantized"]))

        if cfg["export"].get("write_legacy_active_name", True) and variant == active_variant:
            canonical_paths = _canonical_output_paths(output_dir, code_name)
            print("--- Exporting active canonical headers ---")
            export_ref_header(params, mean, std, output_bits, str(canonical_paths["ref"]))
            export_quantized_header(params, mean, std, output_bits, str(canonical_paths["quantized"]))

    if cfg["export"].get("write_active_alias", True):
        _write_active_model_alias(
            code_name,
            output_dir,
            active_variant,
            cfg["export"].get("write_legacy_active_name", True),
        )

    if cfg["export"].get("remove_legacy_headers", True):
        _remove_legacy_headers(output_dir)


if __name__ == "__main__":
    main()
