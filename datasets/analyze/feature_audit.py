#!/usr/bin/env python3
# Outdated. Can be addapted if needed
"""
Audit one dataset row and print a human-readable explanation for every candidate feature.

Usage:
  python datasets/analyze/feature_audit.py \
      --dataset datasets/IRISC_P_dv3_R050_L54_N1296/dataset.csv \
      --row 0

Notes:
- --row is 0-based on data rows (header excluded).
- This script is schema-aware for the current compact format:
  C{k}_{E,M,U,S,I,P,SP,D1,D1A,RE,RU,RI}, Y{k}
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

FEATURES = ["E", "M", "U", "S", "I", "P", "SP", "D1", "D1A", "RE", "RU", "RI"]
FEATURE_DESCRIPTIONS = {
    "E": "Bit energy for this candidate (higher tends to be less reliable in this setup).",
    "M": "Mismatch flag: decoded bit XOR received hard bit (0 or 1).",
    "U": "Number of unsatisfied checks connected to this candidate bit.",
    "S": "Number of satisfied checks connected to this candidate bit.",
    "I": "Suspicious-satisfied pressure accumulated from suspicious satisfied checks.",
    "P": "Participation count in seed-local check neighborhood.",
    "SP": "Participation count in suspicious seed-local checks.",
    "D1": "Single-flip syndrome gain: syndrome_weight_before - syndrome_weight_after_flip.",
    "D1A": "Accept flag derived from D1 (1 if D1 >= 0 else 0).",
    "RE": "Local rank by energy in candidate group (larger means higher rank).",
    "RU": "Local rank by unsatisfied-count in candidate group.",
    "RI": "Local rank by suspicious-satisfied count in candidate group.",
}


def parse_int(value: str, name: str) -> int:
    try:
        # The dataset is expected to be integer-valued; tolerate float-like tokens.
        return int(float(value.strip()))
    except Exception as exc:
        raise ValueError(f"Cannot parse integer for column '{name}': {value!r}") from exc


def detect_candidate_count(header: List[str]) -> int:
    max_idx = -1
    for col in header:
        if col.startswith("C") and "_" in col:
            left, _ = col.split("_", 1)
            suffix = left[1:]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    if max_idx < 0:
        raise ValueError("No candidate feature columns found (expected C0_* pattern).")
    return max_idx + 1


def get_expected_columns(candidate_count: int) -> Tuple[List[str], List[str]]:
    feature_cols = [f"C{k}_{feat}" for k in range(candidate_count) for feat in FEATURES]
    label_cols = [f"Y{k}" for k in range(candidate_count)]
    return feature_cols, label_cols


def load_row(dataset: Path, row_index: int) -> Tuple[Dict[str, int], List[str]]:
    with dataset.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("Dataset appears empty or header could not be read.")

        for i, row in enumerate(reader):
            if i == row_index:
                parsed = {name: parse_int(value, name) for name, value in row.items() if name is not None}
                return parsed, header

    raise IndexError(f"Requested row {row_index} but dataset has fewer rows.")


def iter_rows(dataset: Path) -> Tuple[List[str], Iterable[Tuple[int, Dict[str, int]]]]:
    with dataset.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("Dataset appears empty or header could not be read.")

        rows: List[Tuple[int, Dict[str, int]]] = []
        for i, row in enumerate(reader):
            parsed = {name: parse_int(value, name) for name, value in row.items() if name is not None}
            rows.append((i, parsed))

    return header, rows


def schema_checks(header: List[str], candidate_count: int) -> List[str]:
    issues: List[str] = []
    feature_cols, label_cols = get_expected_columns(candidate_count)

    missing_features = [c for c in feature_cols if c not in header]
    missing_labels = [c for c in label_cols if c not in header]

    if missing_features:
        issues.append(f"Missing feature columns: {missing_features[:8]}{' ...' if len(missing_features) > 8 else ''}")
    if missing_labels:
        issues.append(f"Missing label columns: {missing_labels[:8]}{' ...' if len(missing_labels) > 8 else ''}")

    return issues


def row_checks(row: Dict[str, int], candidate_count: int) -> List[str]:
    issues: List[str] = []

    # Binary checks
    binary_cols = [f"C{k}_M" for k in range(candidate_count)]
    binary_cols += [f"C{k}_D1A" for k in range(candidate_count)]
    binary_cols += [f"Y{k}" for k in range(candidate_count)]

    bad_binary = [c for c in binary_cols if c in row and row[c] not in (0, 1)]
    if bad_binary:
        issues.append(f"Binary columns with non-binary values: {bad_binary[:10]}{' ...' if len(bad_binary) > 10 else ''}")

    # D1A consistency with D1
    d1a_mismatch = []
    for k in range(candidate_count):
        d1 = row.get(f"C{k}_D1")
        d1a = row.get(f"C{k}_D1A")
        p = row.get(f"C{k}_P", 0)
        sp = row.get(f"C{k}_SP", 0)
        re_rank = row.get(f"C{k}_RE", 0)
        ru_rank = row.get(f"C{k}_RU", 0)
        ri_rank = row.get(f"C{k}_RI", 0)
        if d1 is None or d1a is None:
            continue
        # Tail slots are padded in the C pipeline; keep their defaults out of strict D1A checks.
        is_padded_slot = (p == 0 and sp == 0 and re_rank == 0 and ru_rank == 0 and ri_rank == 0)
        if is_padded_slot:
            continue
        expected = 1 if d1 >= 0 else 0
        if d1a != expected:
            d1a_mismatch.append(k)
    if d1a_mismatch:
        issues.append(f"D1A mismatch at candidates: {d1a_mismatch}")

    # Rank range checks
    for rank_name in ("RE", "RU", "RI"):
        values = [row.get(f"C{k}_{rank_name}", 0) for k in range(candidate_count)]
        for k, v in enumerate(values):
            if v != 0 and not (1 <= v <= candidate_count):
                issues.append(f"C{k}_{rank_name}={v} out of expected range [1,{candidate_count}] (or 0 for padded).")
                break

    # Label cardinality expectation in current policy: even 2 or 4; sometimes zero if data from older policy.
    ys = [row.get(f"Y{k}", 0) for k in range(candidate_count)]
    label_weight = sum(ys)
    if label_weight not in (0, 2, 4):
        issues.append(f"Label mask weight is {label_weight}; expected one of 0,2,4 under current rollout policy.")

    return issues


def print_row_explanation(row: Dict[str, int], candidate_count: int) -> None:
    print("=== Feature Definitions ===")
    for feat in FEATURES:
        print(f"{feat:>3}: {FEATURE_DESCRIPTIONS[feat]}")

    ys = [row.get(f"Y{k}", 0) for k in range(candidate_count)]
    selected = [k for k, v in enumerate(ys) if v == 1]

    print("\n=== Row Label Summary ===")
    print(f"Active label slots: {selected if selected else 'none'}")
    print(f"Mask cardinality: {len(selected)}")

    print("\n=== Per-Candidate Breakdown ===")
    for k in range(candidate_count):
        values = {feat: row.get(f"C{k}_{feat}", 0) for feat in FEATURES}
        label = row.get(f"Y{k}", 0)
        marker = "*" if label == 1 else " "

        print(f"\n[{marker}] Candidate C{k}")
        print(
            "    "
            + " ".join(
                [
                    f"E={values['E']}",
                    f"M={values['M']}",
                    f"U={values['U']}",
                    f"S={values['S']}",
                    f"I={values['I']}",
                    f"P={values['P']}",
                    f"SP={values['SP']}",
                    f"D1={values['D1']}",
                    f"D1A={values['D1A']}",
                    f"RE={values['RE']}",
                    f"RU={values['RU']}",
                    f"RI={values['RI']}",
                    f"Y={label}",
                ]
            )
        )

        # Short interpretation helpers
        notes: List[str] = []
        if values["M"] == 1:
            notes.append("channel mismatch")
        if values["U"] > values["S"]:
            notes.append("more unsatisfied than satisfied checks")
        if values["SP"] > 0:
            notes.append("in suspicious seed-local checks")
        if values["D1"] > 0:
            notes.append("single flip reduces syndrome weight")
        elif values["D1"] < 0:
            notes.append("single flip increases syndrome weight")

        if notes:
            print("    notes: " + "; ".join(notes))


def print_dataset_summary(dataset: Path, max_issue_rows: int = 20) -> None:
    header, rows = iter_rows(dataset)
    candidate_count = detect_candidate_count(header)

    print(f"Dataset: {dataset}")
    print(f"Detected candidates: {candidate_count}")
    print(f"Total rows: {len(rows)}")

    s_issues = schema_checks(header, candidate_count)

    issue_counter: Counter[str] = Counter()
    issue_rows: List[Tuple[int, List[str]]] = []
    label_cardinality_counter: Counter[int] = Counter()
    slot_label_ones = [0 for _ in range(candidate_count)]
    padded_slot_hits = [0 for _ in range(candidate_count)]

    for row_idx, row in rows:
        r_issues = row_checks(row, candidate_count)
        if r_issues:
            issue_rows.append((row_idx, r_issues))
            for msg in r_issues:
                issue_counter[msg] += 1

        ys = [row.get(f"Y{k}", 0) for k in range(candidate_count)]
        label_weight = sum(ys)
        label_cardinality_counter[label_weight] += 1
        for k, y in enumerate(ys):
            slot_label_ones[k] += y

        for k in range(candidate_count):
            p = row.get(f"C{k}_P", 0)
            sp = row.get(f"C{k}_SP", 0)
            re_rank = row.get(f"C{k}_RE", 0)
            ru_rank = row.get(f"C{k}_RU", 0)
            ri_rank = row.get(f"C{k}_RI", 0)
            is_padded_slot = (p == 0 and sp == 0 and re_rank == 0 and ru_rank == 0 and ri_rank == 0)
            if is_padded_slot:
                padded_slot_hits[k] += 1

    print("\n=== Schema Checks ===")
    if not s_issues:
        print("All checks passed.")
    else:
        for issue in s_issues:
            print(f"- {issue}")

    print("\n=== Row Issue Summary ===")
    bad_rows = len(issue_rows)
    if bad_rows == 0:
        print("All rows passed row-level checks.")
    else:
        print(f"Rows with one or more issues: {bad_rows}/{len(rows)}")
        print("Issue counts:")
        for msg, cnt in issue_counter.most_common():
            print(f"- {cnt}: {msg}")

        print(f"\nFirst {min(max_issue_rows, len(issue_rows))} problematic rows:")
        for row_idx, messages in issue_rows[:max_issue_rows]:
            joined = " | ".join(messages)
            print(f"- row {row_idx}: {joined}")

    print("\n=== Label Cardinality Distribution ===")
    for weight in sorted(label_cardinality_counter):
        cnt = label_cardinality_counter[weight]
        pct = (100.0 * cnt / len(rows)) if rows else 0.0
        print(f"- weight={weight}: {cnt} ({pct:.2f}%)")

    print("\n=== Positive Label Frequency Per Slot ===")
    for k, cnt in enumerate(slot_label_ones):
        pct = (100.0 * cnt / len(rows)) if rows else 0.0
        print(f"- Y{k}: {cnt} ({pct:.2f}%)")

    print("\n=== Padded Slot Frequency Per Slot ===")
    for k, cnt in enumerate(padded_slot_hits):
        pct = (100.0 * cnt / len(rows)) if rows else 0.0
        print(f"- C{k}: {cnt} ({pct:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one row or full dataset for compact AS features")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset.csv",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="0-based data row index (header excluded)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all rows and print aggregate dataset summary.",
    )
    parser.add_argument(
        "--max-issue-rows",
        type=int,
        default=20,
        help="Maximum number of problematic row indices to print in --all mode.",
    )
    args = parser.parse_args()

    if args.row < 0:
        raise ValueError("--row must be >= 0")
    if args.max_issue_rows < 0:
        raise ValueError("--max-issue-rows must be >= 0")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.all:
        print_dataset_summary(args.dataset, max_issue_rows=args.max_issue_rows)
        return

    row, header = load_row(args.dataset, args.row)
    candidate_count = detect_candidate_count(header)

    print(f"Dataset: {args.dataset}")
    print(f"Row index: {args.row}")
    print(f"Detected candidates: {candidate_count}")

    s_issues = schema_checks(header, candidate_count)
    r_issues = row_checks(row, candidate_count)

    print("\n=== Sanity Checks ===")
    if not s_issues and not r_issues:
        print("All checks passed.")
    else:
        for issue in s_issues + r_issues:
            print(f"- {issue}")

    print_row_explanation(row, candidate_count)


if __name__ == "__main__":
    main()
