"""TEP Step 1 / script 08: Fault relevance across all 20 faults on candidate loops.
Evaluates post-onset window (samples 21..500) for runs 1..100 of each fault.
Compares to fault-free baseline runs 1..100 (samples 21..500).
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

# Load fault-free baseline
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df_ff = df_ff[(df_ff["simulationRun"] >= 1) & (df_ff["simulationRun"] <= 100)]
df_ff = df_ff[df_ff["sample"] >= 21].sort_values(["simulationRun", "sample"])

# Load faulty training subset
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)
df_faulty = df_faulty[(df_faulty["simulationRun"] >= 1) & (df_faulty["simulationRun"] <= 100)]
df_faulty = df_faulty[df_faulty["sample"] >= 21].sort_values(["faultNumber", "simulationRun", "sample"])

# Focus on primary loops:
# Loop 1: Reactor Temperature (xmv_10, xmeas_9)
# Loop 2: Separator Level (xmv_7, xmeas_12)
# Loop 3: Stripper Level (xmv_8, xmeas_15)
# Loop 4: Stripper Temperature (xmv_11, xmeas_17)
# Loop 5: Reactor Feed Comp A (xmv_3, xmeas_23)
# Loop 6: Purge Comp A (xmv_6, xmeas_30)

LOOPS = [
    ("Reactor_Temp", "xmv_10", "xmeas_9"),
    ("Sep_Level",    "xmv_7",  "xmeas_12"),
    ("Strip_Level",  "xmv_8",  "xmeas_15"),
    ("Strip_Temp",   "xmv_11", "xmeas_17"),
    ("Feed_CompA",   "xmv_3",  "xmeas_23"),
    ("Purge_CompA",  "xmv_6",  "xmeas_30"),
]

# Baseline statistics
base_stats = {}
for name, op_col, pv_col in LOOPS:
    pv_b = df_ff[pv_col].to_numpy(dtype=float)
    op_b = df_ff[op_col].to_numpy(dtype=float)
    base_stats[name] = {
        "pv_mean": float(np.mean(pv_b)),
        "pv_std": float(np.std(pv_b)),
        "op_mean": float(np.mean(op_b)),
        "op_std": float(np.std(op_b)),
    }

fault_records = []

for fn in range(1, 21):
    sub_fn = df_faulty[df_faulty["faultNumber"] == fn]
    
    for name, op_col, pv_col in LOOPS:
        bs = base_stats[name]
        sp_nominal = bs["pv_mean"]
        
        pv_f = sub_fn[pv_col].to_numpy(dtype=float)
        op_f = sub_fn[op_col].to_numpy(dtype=float)
        
        # Aggregate shift
        pv_z_shift = abs(float(np.mean(pv_f)) - bs["pv_mean"]) / bs["pv_std"]
        pv_var_ratio = float(np.var(pv_f)) / (bs["pv_std"] ** 2)
        op_z_shift = abs(float(np.mean(op_f)) - bs["op_mean"]) / bs["op_std"]
        op_var_ratio = float(np.var(op_f)) / (bs["op_std"] ** 2)
        
        # Run diagnostics over 20 representative runs (runs 1..20) for speed
        d1_r2_list, d2_acf_list, d2_period_list, d3_flag_list, d4_ih_list = [], [], [], [], []
        
        for r in range(1, 21):
            run_df = sub_fn[sub_fn["simulationRun"] == r].sort_values("sample")
            pv_r = run_df[pv_col].to_numpy(dtype=float)
            op_r = run_df[op_col].to_numpy(dtype=float)
            sp_r = np.full_like(pv_r, sp_nominal)
            t_r = run_df["sample"].to_numpy(dtype=float)
            ldf = pd.DataFrame({"timestep": t_r, "SP": sp_r, "PV": pv_r, "OP": op_r})
            
            d1 = compute_setpoint_tracking(ldf)
            d1_r2_list.append(d1["r2_linear"])
            
            d2 = compute_oscillation_detection(ldf)
            d2_acf_list.append(d2["acf_osc_score"])
            d2_period_list.append(d2["dominant_period"])
            
            d3 = compute_stiction_detection_v2(ldf)
            d3_flag_list.append(d3["stiction_flag"])
            
            d4 = compute_harris_index(ldf, delay=3, settle_startup=10, settle_sp=0)
            d4_ih_list.append(d4["harris_index"])
            
        fault_records.append({
            "fault": fn,
            "loop": name,
            "pv_z_shift": pv_z_shift,
            "pv_var_ratio": pv_var_ratio,
            "op_z_shift": op_z_shift,
            "op_var_ratio": op_var_ratio,
            "d1_r2_mean": float(np.mean(d1_r2_list)),
            "d1_r2_max": float(np.max(d1_r2_list)),
            "d2_acf_mean": float(np.mean(d2_acf_list)),
            "d2_period_med": float(np.median(d2_period_list)),
            "d3_flag_frac": float(np.mean(d3_flag_list)),
            "d4_ih_mean": float(np.mean(d4_ih_list)),
        })

f_df = pd.DataFrame(fault_records)
f_df.to_pickle(ROOT / "scratch" / "tep_fault_effects.pkl")

# Summarize per fault which loops are visibly affected (e.g. pv_z_shift > 2.0 or pv_var_ratio > 2.0 or pv_var_ratio < 0.5)
print("=== FAULT RELEVANCE MATRIX (Faults 1 to 20) ===")
for fn in range(1, 21):
    sub = f_df[f_df["fault"] == fn]
    affected = []
    for _, row in sub.iterrows():
        # criterion for visibly affected
        if row["pv_z_shift"] > 1.5 or row["pv_var_ratio"] > 2.0 or row["op_z_shift"] > 1.5 or row["op_var_ratio"] > 2.0:
            affected.append(f"{row['loop']}(pv_z={row['pv_z_shift']:.1f},pv_var={row['pv_var_ratio']:.1f}x,op_z={row['op_z_shift']:.1f})")
    aff_str = ", ".join(affected) if affected else "None of monitored loops visibly affected"
    print(f"Fault {fn:2d}: {aff_str}")
