"""
Train absorbing-set escape model and export C headers.

Usage:
    python training/train.py
"""

import numpy as np

from config import (
    DATASET_PATH,
    QUANTIZED_OUTPUT_PATH,
    MIN_TRAIN_SAMPLES_WARNING,
    MODEL_OUTPUT_PATH,
    REF_OUTPUT_PATH,
)
from data import load_data
from export import export_quantized_header, export_ref_header, export_to_c_header
from model import extract_weights_pytorch, require_torch, train_pytorch


def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run the GDBF simulator first to generate training data:")
        print("  ./GDBF_TRAIN 1000000 100 codes/wifin_r_1_2/wifin_r_1_2_Dform codes/wifin_r_1_2/wifin_r_1_2_Base 0.035")
        raise SystemExit(1)

    X, y, _, _ = load_data(DATASET_PATH)

    if len(X) < MIN_TRAIN_SAMPLES_WARNING:
        print(f"WARNING: Only {len(X)} samples. Need more data for reliable training.")
        print("Run simulator with more Monte Carlo iterations or lower SNR.")

    require_torch()

    print("\n=== Training with PyTorch ===\n")
    model, mean, std = train_pytorch(X, y)
    w1, b1, w2, b2, w3, b3 = extract_weights_pytorch(model)

    export_to_c_header(model, mean, std, str(MODEL_OUTPUT_PATH))

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    w3 = np.asarray(w3, dtype=np.float64)
    b3 = np.asarray(b3, dtype=np.float64)

    print("\n--- Exporting headers ---")
    export_ref_header(w1, b1, w2, b2, w3, b3, mean, std, str(REF_OUTPUT_PATH))
    export_quantized_header(w1, b1, w2, b2, w3, b3, mean, std, str(QUANTIZED_OUTPUT_PATH))


if __name__ == "__main__":
    main()
