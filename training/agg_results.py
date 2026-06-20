import csv
from pathlib import Path

base = Path(r"c:/Users/dmilos/git/gdbf/results/IRISC_P_dv3_R050_L54_N1296/ml")
files = {
    "2it": base / "ml_outcome_summary_2it.csv",
    "3it": base / "ml_outcome_summary.csv",
    "4it": base / "ml_outcome_summary_4it.csv",
}

def pct(num, den):
    return 100.0 * num / den if den else 0.0

all_global = []

for tag, fp in files.items():
    rows = list(csv.DictReader(fp.open(newline="", encoding="utf-8")))

    # Keep one row per alpha (2it file has duplicated alphas in your data)
    by_alpha = {}
    for r in rows:
        by_alpha[float(r["alpha"])] = r
    rows = [by_alpha[a] for a in sorted(by_alpha)]

    out_rows = []
    g = {
        "tested": 0.0,
        "baseline_clean": 0.0,
        "stuck_frames": 0.0,
        "recovered_after_ml": 0.0,
        "total_clean": 0.0,
    }

    for r in rows:
        tested = float(r["frames_tested"])
        baseline_clean = float(r["frames_decoded_clean_without_ml_invocation"])
        stuck_frames = float(r["frames_with_ml_invocation"])
        recovered_after_ml = float(r["frames_with_ml_invocation_decoded_clean"])
        total_clean = float(r["frames_decoded_clean"])
        not_recovered_after_ml = stuck_frames - recovered_after_ml

        out_rows.append({
            "trigger": tag,
            "alpha": float(r["alpha"]),
            "tested": int(tested),
            "baseline_clean": int(baseline_clean),
            "stuck_frames": int(stuck_frames),
            "recovered_after_ml": int(recovered_after_ml),
            "not_recovered_after_ml": int(not_recovered_after_ml),
            "baseline_clean_rate_pct": round(pct(baseline_clean, tested), 4),
            "stuck_rate_pct": round(pct(stuck_frames, tested), 4),
            "ml_recovery_given_stuck_pct": round(pct(recovered_after_ml, stuck_frames), 4),
            "ml_failure_given_stuck_pct": round(pct(not_recovered_after_ml, stuck_frames), 4),
            "ml_contribution_to_total_clean_pct": round(pct(recovered_after_ml, total_clean), 4),
        })

        g["tested"] += tested
        g["baseline_clean"] += baseline_clean
        g["stuck_frames"] += stuck_frames
        g["recovered_after_ml"] += recovered_after_ml
        g["total_clean"] += total_clean

    # Write per-alpha detailed file
    detail_fp = base / f"ml_effectiveness_breakdown_{tag}.csv"
    fieldnames = list(out_rows[0].keys())
    with detail_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Global summary for this trigger
    not_recovered = g["stuck_frames"] - g["recovered_after_ml"]
    all_global.append({
        "trigger": tag,
        "tested": int(g["tested"]),
        "baseline_clean": int(g["baseline_clean"]),
        "stuck_frames": int(g["stuck_frames"]),
        "recovered_after_ml": int(g["recovered_after_ml"]),
        "not_recovered_after_ml": int(not_recovered),
        "baseline_clean_rate_pct": round(pct(g["baseline_clean"], g["tested"]), 4),
        "stuck_rate_pct": round(pct(g["stuck_frames"], g["tested"]), 4),
        "ml_recovery_given_stuck_pct": round(pct(g["recovered_after_ml"], g["stuck_frames"]), 4),
        "ml_failure_given_stuck_pct": round(pct(not_recovered, g["stuck_frames"]), 4),
        "ml_contribution_to_total_clean_pct": round(pct(g["recovered_after_ml"], g["total_clean"]), 4),
    })

# Write global summary
global_fp = base / "ml_effectiveness_global_summary.csv"
with global_fp.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_global[0].keys()))
    w.writeheader()
    w.writerows(all_global)

print("Wrote:")
for tag in files:
    print(base / f"ml_effectiveness_breakdown_{tag}.csv")
print(global_fp)