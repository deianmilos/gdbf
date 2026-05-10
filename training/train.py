"""
Train absorbing-set escape model and export C headers.

Usage:
    python training/train.py [--code CODE_NAME]
    
Example:
    python training/train.py --code wifin_r_1_2
"""

import argparse
import numpy as np
from pathlib import Path

from config import (
    DEFAULT_CODE_NAME,
    dataset_path,
    infer_code_name,
    model_output_paths,
    MIN_TRAIN_SAMPLES_WARNING,
)
from data import load_data
from export import export_quantized_header, export_ref_header
from model import extract_weights_pytorch, require_torch, train_pytorch


def _remove_legacy_headers():
    legacy_headers = [
        Path("model/as_model_weights.h"),
        Path("model/as_model_ref.h"),
        Path("model/as_model_quantized.h"),
    ]
    for header in legacy_headers:
        if header.exists():
            header.unlink()
            print(f"Removed legacy header -> {header}")


def main():
    parser = argparse.ArgumentParser(description="Train ML escape model and export C headers.")
    parser.add_argument(
        "--code",
        type=str,
        default=DEFAULT_CODE_NAME,
        help=f"LDPC code name (default: {DEFAULT_CODE_NAME})"
    )
    args = parser.parse_args()
    
    code_name = args.code
    dataset_file = dataset_path(code_name)
    
    if not dataset_file.exists():
        print(f"ERROR: {dataset_file} not found.")
        print(f"Run the GDBF simulator first to generate training data:")
        print(f"  ./GDBF_COLLECT 1000000 100 {code_name} 0.035")
        raise SystemExit(1)

    X, y, _, _ = load_data(dataset_file)
    output_paths = model_output_paths(code_name)
    output_paths["ref_code"].parent.mkdir(parents=True, exist_ok=True)

    print(f"Using code: {code_name}")
    print(f"Dataset: {dataset_file}")

    if len(X) < MIN_TRAIN_SAMPLES_WARNING:
        print(f"WARNING: Only {len(X)} samples. Need more data for reliable training.")
        print("Run simulator with more Monte Carlo iterations or lower SNR.")

    require_torch()

    print("\n=== Training with PyTorch ===\n")
    model, mean, std = train_pytorch(X, y)
    w1, b1, w2, b2, w3, b3 = extract_weights_pytorch(model)

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    w3 = np.asarray(w3, dtype=np.float64)
    b3 = np.asarray(b3, dtype=np.float64)

    print("\n--- Exporting headers ---")
    export_ref_header(w1, b1, w2, b2, w3, b3, mean, std, str(output_paths["ref_code"]))
    export_quantized_header(w1, b1, w2, b2, w3, b3, mean, std, str(output_paths["quantized_code"]))
    _remove_legacy_headers()


if __name__ == "__main__":
    main()
