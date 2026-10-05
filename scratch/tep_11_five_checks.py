"""TEP Step 1 / script 11: Five pre-Step-2 checks.
1. Label check: xmeas_17 vs xmeas_18 (mean, std, lag-1 acf, relation to xmv_11).
2. D#2 verdict on 100 healthy Reactor Temp runs: count ZCR > 30 and ACF >= 0.34.
3. D#3 raw metrics on 5 Fault 14 runs vs 5 Healthy runs.
4. Harris delay on healthy Reactor Temp for d = 1, 2, 3, 5.
5. Fault 15 analysis on the 18 subset columns, identifying what extra columns are needed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics import (
    compute_oscillation_detection,
    compute_stiction_detection_v2,
    compute_harris_index,
)

# Load fault-free training (has all 55 columns for runs 1..500)
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")

# Load faulty subset (runs 1..100, 20 faults, 18 loop columns)
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)

print("=" * 80)
print("CHECK 1: LABEL CHECK FOR xmeas_17 AND xmeas_18")
print("=" * 80)
# Use runs 1..100 of fault-free
ff_100 = df_ff[(df_ff.simulationRun >= 1) & (df_ff.simulationRun <= 100)].sort_values(["simulationRun", "sample"])

for col in ["xmeas_17", "xmeas_18", "xmv_11", "xmeas_11", "xmeas_19"]:
    vals = ff_100[col].to_numpy(dtype=float)
    # lag-1 ACF computed pooled across runs (demeaned within run)
    R, S = 100, 500
    arr = vals.reshape(R, S)
    dm = arr - arr.mean(axis=1, keepdims=True)
    c0 = np.sum(dm * dm)
    c1 = np.sum(dm[:, 1:] * dm[:, :-1])
    acf1 = c1 / c0
    print(f"  {col:9s}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}, lag1_acf={acf1:.4f}")

# Cross correlation xmv_11 vs xmeas_17, xmeas_18, xmeas_11
u11 = ff_100["xmv_11"].to_numpy(dtype=float)
y17 = ff_100["xmeas_17"].to_numpy(dtype=float)
y18 = ff_100["xmeas_18"].to_numpy(dtype=float)
y11 = ff_100["xmeas_11"].to_numpy(dtype=float)

print(f"\n  corr(xmv_11, xmeas_17): {np.corrcoef(u11, y17)[0,1]:+.6f}")
print(f"  corr(xmv_11, xmeas_18): {np.corrcoef(u11, y18)[0,1]:+.6f}")
print(f"  corr(xmv_11, xmeas_11): {np.corrcoef(u11, y11)[0,1]:+.6f}")

# Check linear regression xmv_11 vs xmeas_17
poly17 = np.polyfit(y17, u11, 1)
res17 = u11 - (poly17[0] * y17 + poly17[1])
print(f"  Affine fit xmv_11 = {poly17[0]:.4f} * xmeas_17 + {poly17[1]:.4f}, residual std = {res17.std():.6f}")

print("\n" + "=" * 80)
print("CHECK 2: D#2 VERDICT ON 100 HEALTHY REACTOR TEMP RUNS")
print("=" * 80)
d2_zcrs = []
d2_acfs = []
sp_nominal = 120.40

for r in range(1, 101):
    sub = ff_100[ff_100.simulationRun == r].sort_values("sample")
    pv = sub["xmeas_9"].to_numpy(dtype=float)
    op = sub["xmv_10"].to_numpy(dtype=float)
    sp = np.full_like(pv, sp_nominal)
    t = sub["sample"].to_numpy(dtype=float)
    ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
    d2 = compute_oscillation_detection(ldf)
    d2_zcrs.append(d2["zero_crossing_rate"])
    d2_acfs.append(d2["acf_osc_score"])

d2_zcrs = np.array(d2_zcrs)
d2_acfs = np.array(d2_acfs)

n_noise = int(np.sum(d2_zcrs > 30.0))
n_osc = int(np.sum(d2_acfs >= 0.34))
n_both = int(np.sum((d2_zcrs > 30.0) & (d2_acfs >= 0.34)))

print(f"  Total healthy runs evaluated: {len(d2_zcrs)}")
print(f"  Classed as sensor noise (ZCR > 30.0): {n_noise} / 100 ({n_noise}%)")
print(f"    ZCR stats: min={d2_zcrs.min():.2f}, mean={d2_zcrs.mean():.2f}, max={d2_zcrs.max():.2f}")
print(f"  Classed as oscillatory (ACF >= 0.34): {n_osc} / 100 ({n_osc}%)")
print(f"    ACF stats: min={d2_acfs.min():.4f}, mean={d2_acfs.mean():.4f}, max={d2_acfs.max():.4f}")
print(f"  Meeting both criteria: {n_both} / 100 ({n_both}%)")

print("\n" + "=" * 80)
print("CHECK 3: D#3 ON 5 FAULT 14 RUNS VS 5 HEALTHY RUNS (RAW METRICS)")
print("=" * 80)
rows_c3 = []

# 5 Healthy runs (runs 1..5)
for r in range(1, 6):
    sub = ff_100[ff_100.simulationRun == r].sort_values("sample")
    # post sample 20 to match faulty post-onset window (samples 21..500)
    sub = sub[sub["sample"] >= 21]
    pv = sub["xmeas_9"].to_numpy(dtype=float)
    op = sub["xmv_10"].to_numpy(dtype=float)
    sp = np.full_like(pv, sp_nominal)
    t = sub["sample"].to_numpy(dtype=float)
    ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
    d3 = compute_stiction_detection_v2(ldf)
    rows_c3.append({
        "Condition": f"Healthy_Run_{r}",
        "zero_crossing_rate": d3["zero_crossing_rate"],
        "dominant_period": d3["dominant_period"],
        "acf_osc_score": d3["acf_osc_score"],
        "op_pv_corr_filt": d3["op_pv_corr_filt"],
        "r2_linear": d3["r2_linear"],
        "stiction_flag": d3["stiction_flag"],
        "noise_excluded": d3["noise_excluded"],
        "oscillating": d3["oscillating"],
        "corr_ok": d3["corr_ok"],
    })

# 5 Fault 14 runs (runs 1..5)
f14_sub = df_faulty[df_faulty.faultNumber == 14]
for r in range(1, 6):
    sub = f14_sub[f14_sub.simulationRun == r].sort_values("sample")
    sub = sub[sub["sample"] >= 21]
    pv = sub["xmeas_9"].to_numpy(dtype=float)
    op = sub["xmv_10"].to_numpy(dtype=float)
    sp = np.full_like(pv, sp_nominal)
    t = sub["sample"].to_numpy(dtype=float)
    ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
    d3 = compute_stiction_detection_v2(ldf)
    rows_c3.append({
        "Condition": f"Fault14_Run_{r}",
        "zero_crossing_rate": d3["zero_crossing_rate"],
        "dominant_period": d3["dominant_period"],
        "acf_osc_score": d3["acf_osc_score"],
        "op_pv_corr_filt": d3["op_pv_corr_filt"],
        "r2_linear": d3["r2_linear"],
        "stiction_flag": d3["stiction_flag"],
        "noise_excluded": d3["noise_excluded"],
        "oscillating": d3["oscillating"],
        "corr_ok": d3["corr_ok"],
    })

df_c3 = pd.DataFrame(rows_c3)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 15)
print(df_c3[["Condition", "zero_crossing_rate", "dominant_period", "acf_osc_score", "op_pv_corr_filt", "stiction_flag", "noise_excluded", "oscillating", "corr_ok"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 80)
print("CHECK 4: HARRIS DELAY ON HEALTHY REACTOR TEMP (d = 1, 2, 3, 5)")
print("=" * 80)
# We test delay d = 1, 2, 3, 5 across all 100 runs of healthy Reactor Temp
delays = [1, 2, 3, 5]
d4_results = {d: [] for d in delays}

for r in range(1, 101):
    sub = ff_100[ff_100.simulationRun == r].sort_values("sample")
    pv = sub["xmeas_9"].to_numpy(dtype=float)
    op = sub["xmv_10"].to_numpy(dtype=float)
    sp = np.full_like(pv, sp_nominal)
    t = sub["sample"].to_numpy(dtype=float)
    ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
    for d in delays:
        res = compute_harris_index(ldf, delay=d, settle_startup=30, settle_sp=0)
        d4_results[d].append(res["harris_index"])

for d in delays:
    arr = np.array(d4_results[d])
    print(f"  Delay d = {d}: mean I_H = {arr.mean():.4f}, std = {arr.std():.4f}, min = {arr.min():.4f}, max = {arr.max():.4f}")

print("\n" + "=" * 80)
print("CHECK 5: FAULT 15 IMPACT ON 18 SUBSET COLS & MISSING VARIABLES")
print("=" * 80)
f15_sub = df_faulty[df_faulty.faultNumber == 15]
cols_18 = [c for c in df_faulty.columns if c.startswith("xm")]

# Compute max absolute z-score shift and variance ratio on all 18 columns
print("  Evaluating Fault 15 post-onset (samples 21..500, runs 1..100) vs Fault-Free:")
f15_diffs = []
for c in cols_18:
    vals_ff = ff_100[ff_100["sample"] >= 21][c].to_numpy(dtype=float)
    vals_f15 = f15_sub[f15_sub["sample"] >= 21][c].to_numpy(dtype=float)
    mean_shift = abs(vals_f15.mean() - vals_ff.mean()) / vals_ff.std()
    var_ratio = vals_f15.var() / vals_ff.var()
    max_pt_diff = abs(vals_f15 - vals_ff).max()
    f15_diffs.append({"var": c, "mean_shift_z": mean_shift, "var_ratio": var_ratio, "max_pt_diff": max_pt_diff})

df_f15 = pd.DataFrame(f15_diffs).sort_values("max_pt_diff", ascending=False)
print("  Top 6 affected columns in the 18 subset columns:")
print(df_f15.head(6).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n  Max z-shift across all 18 columns: {df_f15['mean_shift_z'].max():.4f}")
print(f"  Variance ratio range across all 18 columns: [{df_f15['var_ratio'].min():.4f}, {df_f15['var_ratio'].max():.4f}]")

# Check all other columns in ff_100 to identify condenser-related variables
print("\n  Investigating condenser cooling water variables in full TEP (from df_ff):")
# Downs & Vogel variable check:
# xmv_11 is condenser cooling water flow.
# What are the condenser temperatures and cooling water outlet temperatures?
for c in ["xmeas_11", "xmeas_17", "xmeas_18", "xmeas_22", "xmv_11"]:
    print(f"    {c:9s}: mean={df_ff[c].mean():.4f}, std={df_ff[c].std():.4f}")
