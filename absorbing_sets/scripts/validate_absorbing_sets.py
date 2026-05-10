"""
validate_absorbing_sets.py

Reads candidate absorbing sets produced by AS_ENUM.exe and independently
validates each one against the Tanner graph built from the QC-LDPC base matrix.

Steps per candidate set S:
  1. Find all check nodes neighbouring S.
  2. For each such check node c, compute deg_S(c) = |N(c) ∩ S|.
  3. Classify:  odd  deg_S(c)  → unsatisfied check
               even deg_S(c)  → satisfied   check
  4. b = number of unsatisfied checks.
  5. Absorbing-set condition: for every v ∈ S,
       |satisfied neighbours of v| > |unsatisfied neighbours of v|
  6. If condition holds → valid (6, b) absorbing set; otherwise "invalid".

Usage:
    python validate_absorbing_sets.py <BaseMatrixPrefix> <CandidateFile> [OutputFile]

Example:
    python validate_absorbing_sets.py \
        codes/wifin_r_1_2/wifin_r_1_2_Base \
        absorbing_sets/validated/absorbing_sets_6.txt \
        absorbing_sets/validated/absorbing_sets_6_validated.txt
"""

import sys
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1.  Tanner-graph construction from QC-LDPC base matrix
# ---------------------------------------------------------------------------

def load_base_matrix(prefix):
    """
    Returns (mb, nb, Z, shift_matrix)
      shift_matrix[r][c] = shift value (-1 means no edge for that block)
    """
    size_file = prefix + "_size"
    mat_file  = prefix + "_mat"

    with open(size_file) as f:
        tokens = f.read().split()
    mb, nb, Z = int(tokens[0]), int(tokens[1]), int(tokens[2])

    shift_matrix = [[0]*nb for _ in range(mb)]
    with open(mat_file) as f:
        tokens = f.read().split()
    idx = 0
    for r in range(mb):
        for c in range(nb):
            shift_matrix[r][c] = int(tokens[idx])
            idx += 1

    return mb, nb, Z, shift_matrix


def build_tanner_graph(mb, nb, Z, shift_matrix):
    """
    Returns:
      var_to_checks : list of length N  – var_to_checks[v] = list of check indices
      check_to_vars : list of length M  – check_to_vars[c] = list of var   indices
    where M = mb*Z, N = nb*Z.
    """
    M = mb * Z
    N = nb * Z

    var_to_checks  = [[] for _ in range(N)]
    check_to_vars  = [[] for _ in range(M)]

    for r in range(mb):
        for c in range(nb):
            s = shift_matrix[r][c]
            if s < 0:
                continue
            for i in range(Z):
                chk = r * Z + i
                var = c * Z + (i + s) % Z
                check_to_vars[chk].append(var)
                var_to_checks[var].append(chk)

    return var_to_checks, check_to_vars


# ---------------------------------------------------------------------------
# 2.  Absorbing-set validation
# ---------------------------------------------------------------------------

def validate_candidate(S, var_to_checks, check_to_vars):
    """
    Parameters
    ----------
    S : list/set of variable-node indices (0-based)
    var_to_checks  : adjacency lists variable → checks
    check_to_vars  : adjacency lists check    → variables  (not needed here but kept for clarity)

    Returns
    -------
    is_valid  : bool    – True iff S is a valid absorbing set
    b         : int     – number of unsatisfied (odd-degree) check nodes
    details   : dict    – per-variable satisfied/unsatisfied neighbour counts
    """
    S_set = set(S)

    # Step 1-3: for every check neighbouring S, compute deg_S(c)
    deg_in_S = defaultdict(int)
    for v in S:
        for c in var_to_checks[v]:
            deg_in_S[c] += 1

    satisfied_checks   = {c for c, d in deg_in_S.items() if d % 2 == 0}
    unsatisfied_checks = {c for c, d in deg_in_S.items() if d % 2 == 1}

    # Step 4
    b = len(unsatisfied_checks)

    # Step 5: for every v ∈ S count satisfied / unsatisfied neighbours
    details = {}
    all_ok  = True
    for v in S:
        sat = sum(1 for c in var_to_checks[v] if c in satisfied_checks)
        uns = sum(1 for c in var_to_checks[v] if c in unsatisfied_checks)
        details[v] = {"satisfied": sat, "unsatisfied": uns}
        if sat <= uns:
            all_ok = False

    return all_ok, b, details


# ---------------------------------------------------------------------------
# 3.  File I/O helpers
# ---------------------------------------------------------------------------

def parse_candidates(path):
    """
    Yields (raw_line, size, second_number, nodes_list) for every data line.
    Comment lines (starting with #) are yielded as (raw_line, None, None, None).
    """
    with open(path) as f:
        for raw in f:
            stripped = raw.rstrip("\n")
            if stripped.startswith("#") or stripped.strip() == "":
                yield stripped, None, None, None
                continue
            parts = stripped.split()
            size   = int(parts[0])
            second = int(parts[1])
            nodes  = [int(x) for x in parts[2:]]
            yield stripped, size, second, nodes


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_absorbing_sets.py "
              "<BaseMatrixPrefix> <CandidateFile> [OutputFile]")
        sys.exit(1)

    prefix         = sys.argv[1]
    candidate_file = sys.argv[2]
    output_file    = sys.argv[3] if len(sys.argv) > 3 else None

    # --- Build Tanner graph --------------------------------------------------
    print(f"Loading base matrix from '{prefix}' ...", flush=True)
    mb, nb, Z, shift_matrix = load_base_matrix(prefix)
    M, N = mb * Z, nb * Z
    print(f"  Base matrix: {mb} x {nb}, Z={Z}  →  M={M} checks, N={N} vars")

    print("Building Tanner graph ...", flush=True)
    var_to_checks, check_to_vars = build_tanner_graph(mb, nb, Z, shift_matrix)
    print("  Done.", flush=True)

    # --- Validate candidates -------------------------------------------------
    out_lines  = []
    total      = 0
    valid      = 0
    invalid    = 0
    b_counts   = defaultdict(int)   # valid sets by b value

    for raw, size, second, nodes in parse_candidates(candidate_file):
        if size is None:
            # comment / blank – pass through unchanged
            out_lines.append(raw)
            continue

        total += 1
        is_valid, b, details = validate_candidate(nodes, var_to_checks, check_to_vars)

        if is_valid:
            valid += 1
            b_counts[b] += 1
            label = f"VALID ({size},{b})-AS"
        else:
            invalid += 1
            label = f"INVALID"

        # Build output line:
        # keep original candidate line, append validation result as comment
        node_str = " ".join(str(v) for v in nodes)
        out_line = f"{size} {b} {node_str}  # {label}  [orig_2nd={second}]"
        out_lines.append(out_line)

    # --- Summary -------------------------------------------------------------
    summary_lines = [
        "",
        f"# === Validation summary ===",
        f"# Total candidates : {total}",
        f"# Valid AS         : {valid}",
        f"# Invalid          : {invalid}",
        f"# Valid by b value :",
    ]
    for bv in sorted(b_counts):
        summary_lines.append(f"#   (6,{bv})-AS : {b_counts[bv]}")

    # Print summary to stderr
    for ln in summary_lines:
        print(ln.lstrip("# "), file=sys.stderr)

    # --- Output --------------------------------------------------------------
    all_output = "\n".join(out_lines + summary_lines) + "\n"

    if output_file:
        with open(output_file, "w") as f:
            f.write(all_output)
        print(f"\nResults written to '{output_file}'", flush=True)
    else:
        print(all_output)


if __name__ == "__main__":
    main()
