"""TEP Step 1 / script 07: Detailed evaluation of candidate loops on fault-free baseline.
Tests D#1, D#2, D#3 v2, D#4 baseline outputs on runs 1..100 of fault-free data.
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

df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df_ff = df_ff[(df_ff["simulationRun"] >= 1) & (df_ff["simulationRun"] <= 100)].sort_values(["simulationRun", "sample"])

# Candidate loops: (OP, PV, Name)
CANDIDATE_LOOPS = [
    ("xmv_10", "xmeas_9",  "Reactor Temperature Loop"),
    ("xmv_7",  "xmeas_12", "Separator Level Loop"),
    ("xmv_8",  "xmeas_15", "Stripper Level Loop"),
    ("xmv_11", "xmeas_17", "Stripper Temperature Loop"),
    ("xmv_3",  "xmeas_23", "Reactor Feed Comp A (Analyzer) Loop"),
    ("xmv_6",  "xmeas_30", "Purge Comp A (Analyzer) Loop"),
    ("xmv_1",  "xmeas_26", "Reactor Feed Comp D (Analyzer) Loop"),
    ("xmv_2",  "xmeas_27", "Reactor Feed Comp E (Analyzer) Loop"),
    ("xmv_9",  "xmeas_38", "Product Comp E (Analyzer) Loop"),
]

results = []

for op_col, pv_col, loop_name in CANDIDATE_LOOPS:
    # 1. Fault-free baseline stats across runs 1..100
    pv_all = df_ff[pv_col].to_numpy(dtype=float)
    op_all = df_ff[op_col].to_numpy(dtype=float)
    
    sp_nominal = float(np.mean(pv_all))
    
    # Run diagnostics per run (1..100) using SP = sp_nominal
    d1_slopes, d1_r2s = [], []
    d2_acfs, d2_zcrs, d2_periods = [], [], []
    d3_flags, d3_corrs, d3_acfs = [], [], []
    d4_ihs = []
    
    # Check holding/quantization: fraction of samples where PV does not change
    holds = []
    
    for r in range(1, 101):
        sub = df_ff[df_ff["simulationRun"] == r].sort_values("sample")
        pv = sub[pv_col].to_numpy(dtype=float)
        op = sub[op_col].to_numpy(dtype=float)
        sp = np.full_like(pv, sp_nominal)
        t = sub["sample"].to_numpy(dtype=float)
        
        loop_df = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
        
        # Check hold
        holds.append(float(np.mean(np.diff(pv) == 0)))
        
        # D#1
        d1 = compute_setpoint_tracking(loop_df)
        d1_slopes.append(d1["error_slope"])
        d1_r2s.append(d1["r2_linear"])
        
        # D#2
        d2 = compute_oscillation_detection(loop_df)
        d2_acfs.append(d2["acf_osc_score"])
        d2_zcrs.append(d2["zero_crossing_rate"])
        d2_periods.append(d2["dominant_period"])
        
        # D#3 v2
        d3 = compute_stiction_detection_v2(loop_df)
        d3_flags.append(d3["stiction_flag"])
        d3_corrs.append(d3["op_pv_corr_filt"])
        d3_acfs.append(d3["acf_osc_score"])
        
        # D#4 (use default delay=3, settle_startup=30, settle_sp=0 since constant SP)
        d4 = compute_harris_index(loop_df, delay=3, settle_startup=30, settle_sp=0)
        d4_ihs.append(d4["harris_index"])
        
    results.append({
        "Loop": loop_name,
        "OP": op_col,
        "PV": pv_col,
        "SP_val": sp_nominal,
        "PV_mean": float(np.mean(pv_all)),
        "PV_std": float(np.std(pv_all)),
        "OP_mean": float(np.mean(op_all)),
        "OP_std": float(np.std(op_all)),
        "Hold_frac": float(np.mean(holds)),
        "D1_r2_mean": float(np.mean(d1_r2s)),
        "D1_r2_max": float(np.max(d1_r2s)),
        "D2_acf_mean": float(np.mean(d2_acfs)),
        "D2_zcr_mean": float(np.mean(d2_zcrs)),
        "D3_FP_count": int(np.sum(d3_flags)),
        "D3_corr_mean": float(np.mean(d3_corrs)),
        "D4_IH_mean": float(np.mean(d4_ihs)),
        "D4_IH_std": float(np.std(d4_ihs)),
    })

res_df = pd.DataFrame(results)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("=== BASELINE FAULT-FREE PERFORMANCE ON CANDIDATE LOOPS (Runs 1..100) ===")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
