"""Phase 3 Step 2 / Part C: One-time evaluation of frozen adapted rules on REPORT split (runs 51..100).
Rules calibrated on DEV split (runs 1..50) only.
Evaluated strictly once without tuning.
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

# -----------------------------------------------------------------------------
# Adapted Diagnostic Functions (Scratch Only - not modifying src/)
# -----------------------------------------------------------------------------
def eval_adapted_rules(ldf):
    """Evaluate D#1, D#2 adapted, D#3 v2 unchanged, Chatter period-2, and D#4 adapted."""
    # 1. D#1 (unchanged drift rule)
    d1 = compute_setpoint_tracking(ldf)
    d1_drift = bool(d1["r2_linear"] >= 0.50)
    
    # 2. D#2 (adapted oscillation rule)
    d2 = compute_oscillation_detection(ldf)
    zcr = d2["zero_crossing_rate"]
    acf = d2["acf_osc_score"]
    period = d2["dominant_period"]
    d2_osc_adapted = bool((acf >= 0.55) or (zcr <= 30.0 and acf >= 0.35))
    
    # 3. D#3 v2 (unmodified Phase 2 stiction rule)
    d3 = compute_stiction_detection_v2(ldf)
    d3_v2_flag = bool(d3["stiction_flag"])
    
    # 3b. Period-2 Chatter detector (TEP Fault 14 specific)
    chatter_period2 = bool(zcr >= 80.0 and period == 2.0 and acf >= 0.80)
    
    # 4. D#4 (adapted Harris Index degradation)
    d4_d1 = compute_harris_index(ldf, delay=1, settle_startup=30, settle_sp=0)["harris_index"]
    d4_d3 = compute_harris_index(ldf, delay=3, settle_startup=30, settle_sp=0)["harris_index"]
    d4_degraded_mod = bool(d4_d1 < 0.50)
    d4_degraded_sev = bool(d4_d3 < 0.20)
    
    # 5. Composite flag
    any_flag = bool(d1_drift or d2_osc_adapted or chatter_period2 or d4_degraded_mod)
    
    return {
        "d1_drift": d1_drift,
        "d2_osc_adapted": d2_osc_adapted,
        "d3_v2_unchanged": d3_v2_flag,
        "chatter_period2": chatter_period2,
        "d4_degraded_mod": d4_degraded_mod,
        "d4_degraded_sev": d4_degraded_sev,
        "any_flagged": any_flag,
    }

# -----------------------------------------------------------------------------
# Load REPORT Split Data (runs 51..100, samples 21..500)
# -----------------------------------------------------------------------------
SP_NOMINAL = 120.40

df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
ff_rep = df_ff[(df_ff.simulationRun >= 51) & (df_ff.simulationRun <= 100) & (df_ff["sample"] >= 21)].copy()

df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)
faulty_rep = df_faulty[(df_faulty.simulationRun >= 51) & (df_faulty.simulationRun <= 100) & (df_faulty["sample"] >= 21)].copy()

CLASSES = {
    "Healthy": ff_rep,
    "Fault 14 (Valve Sticking)": faulty_rep[faulty_rep.faultNumber == 14],
    "Fault 11 (Cooling Water Temp Osc)": faulty_rep[faulty_rep.faultNumber == 11],
    "Fault 4 (Cooling Water Temp Step)": faulty_rep[faulty_rep.faultNumber == 4],
}

summary_counts = {}

for cname, cdf in CLASSES.items():
    runs = sorted(cdf.simulationRun.unique().tolist())
    assert len(runs) == 50, f"Expected 50 runs for {cname}, got {len(runs)}"
    
    counts = {
        "d1_drift": 0,
        "d2_osc_adapted": 0,
        "d3_v2_unchanged": 0,
        "chatter_period2": 0,
        "d4_degraded_mod": 0,
        "d4_degraded_sev": 0,
        "any_flagged": 0,
    }
    
    for r in runs:
        sub = cdf[cdf.simulationRun == r].sort_values("sample")
        pv = sub["xmeas_9"].to_numpy(dtype=float)
        op = sub["xmv_10"].to_numpy(dtype=float)
        sp = np.full_like(pv, SP_NOMINAL)
        t = sub["sample"].to_numpy(dtype=float)
        ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
        
        res = eval_adapted_rules(ldf)
        for k, v in res.items():
            if v:
                counts[k] += 1
                
    summary_counts[cname] = counts

print("=== PART C: ONE-TIME EVALUATION ON REPORT SPLIT (RUNS 51..100) ===")
print(f"{'Class':35s} | {'D1 Drift':10s} | {'D2 Osc':10s} | {'D3v2 Unmod':12s} | {'Chatter P2':10s} | {'D4 Degraded':12s} | {'Any Flagged':12s}")
print("-" * 115)
for cname, c in summary_counts.items():
    print(f"{cname:35s} | {c['d1_drift']:2d}/50 ({c['d1_drift']/50:5.1%}) | {c['d2_osc_adapted']:2d}/50 ({c['d2_osc_adapted']/50:5.1%}) | {c['d3_v2_unchanged']:2d}/50 ({c['d3_v2_unchanged']/50:5.1%}) | {c['chatter_period2']:2d}/50 ({c['chatter_period2']/50:5.1%}) | {c['d4_degraded_mod']:2d}/50 ({c['d4_degraded_mod']/50:5.1%}) | {c['any_flagged']:2d}/50 ({c['any_flagged']/50:5.1%})")
