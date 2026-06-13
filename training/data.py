import numpy as np
import pandas as pd
import re


def _parse_candidate_feature_columns(columns):
    parsed = {}
    pat = re.compile(r"^C(\d+)_(.+)$")
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        ci = int(m.group(1))
        fn = m.group(2)
        if ci not in parsed:
            parsed[ci] = {}
        parsed[ci][fn] = c
    return parsed


def _parse_label_columns(columns):
    parsed = {}
    pat = re.compile(r"^[LY](\d+)$")
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        li = int(m.group(1))
        parsed[li] = c
    return parsed


def _resolve_subset_columns(df, dataset_options):
    columns = list(df.columns)
    cand_map = _parse_candidate_feature_columns(columns)
    label_map = _parse_label_columns(columns)

    if not cand_map or not label_map:
        return None, None

    options = dict(dataset_options or {})
    anchor_idx = int(options.get("anchor_candidate_index", 0))
    exclude_anchor = bool(options.get("exclude_anchor_candidate", False))
    candidate_indices = options.get("candidate_indices")
    feature_names = options.get("feature_names")
    label_encoding = str(options.get("label_encoding", "mask"))

    if label_encoding != "mask":
        raise ValueError(
            "Unsupported label_encoding. This training pipeline expects mask labels aligned "
            "with the selected input candidates. Regenerate the dataset with mask labels."
        )

    all_candidates = sorted(cand_map.keys())
    if candidate_indices is None:
        chosen_candidates = all_candidates
    else:
        chosen_candidates = [int(v) for v in candidate_indices]

    if exclude_anchor:
        chosen_candidates = [c for c in chosen_candidates if c != anchor_idx]

    if feature_names is None:
        # Keep dataset order for per-candidate feature names.
        sample_candidate = chosen_candidates[0] if chosen_candidates else all_candidates[0]
        ordered_features = []
        for c in columns:
            m = re.match(r"^C(\d+)_(.+)$", c)
            if m and int(m.group(1)) == sample_candidate:
                ordered_features.append(m.group(2))
    else:
        ordered_features = [str(f) for f in feature_names]

    feature_cols = []
    for ci in chosen_candidates:
        fdict = cand_map.get(ci, {})
        for fn in ordered_features:
            if fn in fdict:
                feature_cols.append(fdict[fn])

    label_cols = []
    for ci in chosen_candidates:
        if ci in label_map:
            label_cols.append(label_map[ci])

    if not feature_cols or not label_cols:
        raise ValueError(
            "Dataset subset selection produced no columns. "
            "Check dataset.candidate_indices / feature_names / exclude_anchor_candidate settings."
        )

    return feature_cols, label_cols


def load_data(path, first_n_samples=None, dataset_options=None):
    df = pd.read_csv(path)

    if first_n_samples is not None:
        first_n_samples = int(first_n_samples)
        if first_n_samples <= 0:
            raise ValueError("first_n_samples must be > 0 when provided")
        before_limit = len(df)
        df = df.head(first_n_samples)
        print(f"Applied dataset limit: first {len(df)} of {before_limit} rows")

    before = len(df)
    # df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    print(f"Loaded {after} samples from {path}")
    print(f"Columns: {list(df.columns)}")

    subset_feature_cols, subset_label_cols = _resolve_subset_columns(df, dataset_options)

    if subset_feature_cols is not None and subset_label_cols is not None:
        feature_cols = subset_feature_cols
        label_cols = subset_label_cols
        print(
            f"Applied dataset subset: {len(feature_cols)} feature cols, "
            f"{len(label_cols)} label cols"
        )
    else:
        label_cols = [c for c in df.columns if c.startswith("L") or c.startswith("Y")]
        feature_cols = [c for c in df.columns if c not in label_cols]

    if not label_cols:
        raise ValueError(
            "No label columns found. Expected columns prefixed by 'Y' (current) or 'L' (legacy)."
        )

    selected = df[feature_cols + label_cols]
    nan_mask = selected.isna().any(axis=1)
    if nan_mask.any():
        bad_rows = int(nan_mask.sum())
        print(
            f"Dropping {bad_rows} malformed rows with NaN values in selected columns "
            f"({len(df)} -> {len(df) - bad_rows})"
        )
        df = df.loc[~nan_mask].copy()

    X = df[feature_cols].values.astype(np.float32)

    y = df[label_cols].values.astype(np.float32)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values remain after row filtering; dataset needs regeneration.")

    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError(
            "Dataset labels are not binary masks. Delete or rename dataset.csv and recollect it "
            "with the current collector before training."
        )

    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print("Label distribution per bit:")
    for i, col in enumerate(label_cols):
        print(f"  {col}: {y[:, i].mean():.3f} positive rate")

    return X, y, feature_cols, label_cols
