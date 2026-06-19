"""
Train corrective-mask (mask-only) escape model and export C headers.

Usage:
    python training/train.py [--config configs/ml/default.json] [--code CODE_NAME]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from data import load_data
from export import export_quantized_header, export_ref_header
from model import extract_weights_pytorch, require_torch, train_pytorch_mask_only


DEFAULT_CONFIG_PATH = Path("configs/ml/default.json")


def _default_runtime_config():
    return {
        "code_name": "wifin_r_1_2",
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
            "min_positive_count_for_output": 50,
            "max_pos_weight": 8.0,
            "mask_logit_threshold": 0.0,
        },
        "export": {
            "output_dir": "model",
            "write_active_alias": True,
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


def _legacy_dataset_path(code_name):
    return Path(f"datasets/{code_name}/dataset.csv")


def _canonical_output_paths(output_dir: Path, code_name: str):
    return {
        "ref": output_dir / f"as_model_{code_name}_ref.h",
        "quantized": output_dir / f"as_model_{code_name}_quantized.h",
    }


def _write_active_model_alias(code_name: str, output_dir: Path):
    include_name = f"as_model_{code_name}_quantized.h"
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


def _normalize_export_params(params, metadata, training_settings):
    out = {}
    for key, value in params.items():
        if value is None:
            out[key] = None
        elif isinstance(value, np.ndarray):
            out[key] = np.asarray(value, dtype=np.float64)
        else:
            out[key] = value

    out["mask_logit_threshold"] = metadata.get("mask_logit_threshold", 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train ML escape model (mask-only) and export C headers.")
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

    code_name = args.code if args.code else cfg.get("code_name", "wifin_r_1_2")

    dataset_file = _resolve_dataset_path(cfg, code_name)
    if not dataset_file.exists():
        fallback = _legacy_dataset_path(code_name)
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

    print(f"Using code: {code_name}")
    print(f"Config: {cfg_path}")
    print(f"Dataset: {dataset_file}")
    print("Variant: mask_only")

    require_torch()

    training_settings = cfg.get("training", {})

    print("\n=== Training variant: mask_only ===\n")
    model, mean, std, output_bits, metadata = train_pytorch_mask_only(
        X,
        y,
        settings=training_settings,
    )

    raw_params = extract_weights_pytorch(model)
    params = _normalize_export_params(raw_params, metadata, training_settings)

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    canonical_paths = _canonical_output_paths(output_dir, code_name)
    print("--- Exporting headers ---")
    export_ref_header(params, mean, std, output_bits, str(canonical_paths["ref"]))
    export_quantized_header(params, mean, std, output_bits, str(canonical_paths["quantized"]))

    if cfg["export"].get("write_active_alias", True):
        _write_active_model_alias(code_name, output_dir)

    if cfg["export"].get("remove_legacy_headers", True):
        _remove_legacy_headers(output_dir)


if __name__ == "__main__":
    main()
