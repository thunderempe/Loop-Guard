"""Phase 3 Step 2 / Part A: Evaluate unchanged Phase 2 diagnostics on REPORT split (runs 51..100).
Loop: OP = xmv_10, PV = xmeas_9, SP = 120.40.
Samples: 21..500.
Classes: healthy, Fault 14, Fault 11, Fault 4.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics import (
    compute_setpoint_tracking,
    compute_oscillation_detection,
    compute_stiction_detection_v2,
    compute_harris_index,
)

# 1. Load fault-free data
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
# Filter to REPORT split (runs 51..100) and post-onset samples (21..500)
ff_report = df_ff[(df_ff.simulationRun >= 51) & (df_ff.simulationRun <= 100) & (df_ff["sample"] >= 21)].copy()

# 2. Load faulty subset
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)
faulty_report = df_faulty[(df_faulty.simulationRun >= 51) & (df_faulty.simulationRun <= 100) & (df_faulty["sample"] >= 21)].copy()

CLASSES = {
    "Healthy": ff_report,
    "Fault 14 (Valve Sticking)": faulty_report[faulty_report.faultNumber == 14],
    "Fault 11 (Cooling Water Temp Osc)": faulty_report[faulty_report.faultNumber == 11],
    "Fault 4 (Cooling Water Temp Step)": faulty_report[faulty_report.faultNumber == 4],
}

SP_NOMINAL = 120.40
DELAYS = [1, 2, 3, 5]

results = []

for cname, cdf in CLASSES.items():
    d1_drift_50 = 0
    d1_drift_65 = 0
    d2_noise_flags = 0
    d2_osc_flags = 0
    d3_stiction_flags = 0
    d4_vals = {d: [] for d in DELAYS}
    
    runs = sorted(cdf.simulationRun.unique().tolist())
    assert len(runs) == 50, f"Expected 50 runs for {cname}, got {len(runs)}"
    
    for r in runs:
        sub = cdf[cdf.simulationRun == r].sort_values("sample")
        pv = sub["xmeas_9"].to_numpy(dtype=float)
        op = sub["xmv_10"].to_numpy(dtype=float)
        sp = np.full_like(pv, SP_NOMINAL)
        t = sub["sample"].to_numpy(dtype=float)
        ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
        
        # D#1
        d1 = compute_setpoint_tracking(ldf)
        if d1["r2_linear"] >= 0.50:
            d1_drift_50 += 1
        if d1["r2_linear"] >= 0.65:
            d1_drift_65 += 1
            
        # D#2
        d2 = compute_oscillation_detection(ldf)
        if d2["zero_crossing_rate"] > 30.0:
            d2_noise_flags += 1
        if d2["acf_osc_score"] >= 0.34:
            d2_osc_flags += 1
            
        # D#3 v2
        d3 = compute_stiction_detection_v2(ldf)
        if d3["stiction_flag"]:
            d3_stiction_flags += 1
            
        # D#4
        for d in DELAYS:
            d4 = compute_harris_index(ldf, delay=d, settle_startup=30, settle_sp=0)
            d4_vals[d].append(d4["harris_index"])
            
    row = {
        "Class": cname,
        "N_runs": len(runs),
        "D1_drift_ge_0.50": f"{d1_drift_50}/{len(runs)} ({d1_drift_50/len(runs):.1%})",
        "D1_drift_ge_0.65": f"{d1_drift_65}/{len(runs)} ({d1_drift_65/len(runs):.1%})",
        "D2_noise_ZCR_gt_30": f"{d2_noise_flags}/{len(runs)} ({d2_noise_flags/len(runs):.1%})",
        "D2_osc_ACF_ge_0.34": f"{d2_osc_flags}/{len(runs)} ({d2_osc_flags/len(runs):.1%})",
        "D3_stiction_flag": f"{d3_stiction_flags}/{len(runs)} ({d3_stiction_flags/len(runs):.1%})",
    }
    for d in DELAYS:
        arr = np.array(d4_vals[d])
        row[f"D4_d{d}_mean"] = f"{arr.mean():.4f}"
        row[f"D4_d{d}_range"] = f"[{arr.min():.4f}, {arr.max():.4f}]"
        
    results.append(row)

res_df = pd.DataFrame(results)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("=== PART A: UNCHANGED PHASE 2 DIAGNOSTICS ON REPORT SPLIT (RUNS 51..100) ===")
print(res_df.to_string(index=False))
