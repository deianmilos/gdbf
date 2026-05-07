import numpy as np
import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    print(f"Loaded {after} samples from {path}")
    print(f"Columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if not c.startswith("L")]
    label_cols = [c for c in df.columns if c.startswith("L")]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_cols].values.astype(np.float32)

    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print("Label distribution per bit:")
    for i, col in enumerate(label_cols):
        print(f"  {col}: {y[:, i].mean():.3f} positive rate")

    return X, y, feature_cols, label_cols
