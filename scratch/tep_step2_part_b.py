"""Phase 3 Step 2 / Part B: Noise dominance tests & Raw feature distributions on DEVELOPMENT split (runs 1..50).
Loop: OP = xmv_10, PV = xmeas_9, SP = 120.40.
Samples: 21..500 (length 480).
Classes: Healthy, Fault 14, Fault 11, Fault 4.
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

# -----------------------------------------------------------------------------
# 1. Healthy Reactor Temp Error & Comparison to Phase 2 Simulation Data
# -----------------------------------------------------------------------------
print("=" * 80)
print("1. HEALTHY REACTOR TEMP ERROR DYNAMICS VS PHASE 2 SIMULATION DATA")
print("=" * 80)

# Load TEP fault-free training development split (runs 1..50, samples 21..500)
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
ff_dev = df_ff[(df_ff.simulationRun >= 1) & (df_ff.simulationRun <= 50) & (df_ff["sample"] >= 21)].copy()

SP_NOMINAL = 120.40

# Compute error per run
tep_errors = []
tep_zcrs = []
tep_lag1s = []
tep_stds = []

for r in range(1, 51):
    sub = ff_dev[ff_dev.simulationRun == r].sort_values("sample")
    pv = sub["xmeas_9"].to_numpy(dtype=float)
    op = sub["xmv_10"].to_numpy(dtype=float)
    sp = np.full_like(pv, SP_NOMINAL)
    t = sub["sample"].to_numpy(dtype=float)
    err = pv - sp
    tep_errors.append(err)
    tep_stds.append(err.std())
    
    # lag-1 ACF
    e_demean = err - err.mean()
    c0 = np.dot(e_demean, e_demean)
    c1 = np.dot(e_demean[1:], e_demean[:-1])
    tep_lag1s.append(c1 / c0 if c0 > 0 else 0.0)
    
    # ZCR via D#2
    ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
    d2 = compute_oscillation_detection(ldf)
    tep_zcrs.append(d2["zero_crossing_rate"])

tep_stds = np.array(tep_stds)
tep_lag1s = np.array(tep_lag1s)
tep_zcrs = np.array(tep_zcrs)

print(f"TEP Healthy Reactor Temp Error (50 runs, 480 samples each):")
print(f"  Error std        : mean = {tep_stds.mean():.4f}, range = [{tep_stds.min():.4f}, {tep_stds.max():.4f}]")
print(f"  Lag-1 ACF        : mean = {tep_lag1s.mean():.4f}, range = [{tep_lag1s.min():.4f}, {tep_lag1s.max():.4f}]")
print(f"  ZCR (/100 steps) : mean = {tep_zcrs.mean():.2f}, range = [{tep_zcrs.min():.2f}, {tep_zcrs.max():.2f}]")

# Compare with Phase 2 simulation datasets (healthy loops only)
# Clean dataset
df_clean = pd.read_csv(ROOT / "data" / "simulated" / "labeled_loops.csv")
clean_h = df_clean[df_clean.fault_type == "healthy"]
clean_zcrs = []
for lid in clean_h.loop_id.unique():
    ldf = clean_h[clean_h.loop_id == lid]
    d2 = compute_oscillation_detection(ldf)
    clean_zcrs.append(d2["zero_crossing_rate"])
clean_zcrs = np.array(clean_zcrs)

# Noisy dataset (low noise loops 1..10, moderate noise loops 31..40)
df_noisy = pd.read_csv(ROOT / "data" / "simulated" / "labeled_loops_noisy.csv")
noisy_low_h = df_noisy[(df_noisy.fault_type == "healthy") & (df_noisy.noise_level == "LOW")]
low_zcrs = []
for lid in noisy_low_h.loop_id.unique():
    ldf = noisy_low_h[noisy_low_h.loop_id == lid]
    d2 = compute_oscillation_detection(ldf)
    low_zcrs.append(d2["zero_crossing_rate"])
low_zcrs = np.array(low_zcrs)

noisy_mod_h = df_noisy[(df_noisy.fault_type == "healthy") & (df_noisy.noise_level == "MODERATE")]
mod_zcrs = []
for lid in noisy_mod_h.loop_id.unique():
    ldf = noisy_mod_h[noisy_mod_h.loop_id == lid]
    d2 = compute_oscillation_detection(ldf)
    mod_zcrs.append(d2["zero_crossing_rate"])
mod_zcrs = np.array(mod_zcrs)

print(f"\nPhase 2 Simulation Healthy Loop Comparison:")
print(f"  Clean healthy ZCR        (N=10) : mean = {clean_zcrs.mean():.2f}, range = [{clean_zcrs.min():.2f}, {clean_zcrs.max():.2f}]")
print(f"  Low-noise healthy ZCR    (N=5)  : mean = {low_zcrs.mean():.2f}, range = [{low_zcrs.min():.2f}, {low_zcrs.max():.2f}]")
print(f"  Mod-noise healthy ZCR    (N=5)  : mean = {mod_zcrs.mean():.2f}, range = [{mod_zcrs.min():.2f}, {mod_zcrs.max():.2f}]")
print(f"  TEP healthy error ZCR    (N=50) : mean = {tep_zcrs.mean():.2f}, range = [{tep_zcrs.min():.2f}, {tep_zcrs.max():.2f}]")

# -----------------------------------------------------------------------------
# 2. White Noise Surrogate Test for D#2
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. WHITE NOISE SURROGATE TEST FOR D#2")
print("=" * 80)

rng = np.random.RandomState(42)
N_SURROGATES = 50
N_SAMPLES = 480
TARGET_STD = float(tep_stds.mean())

surr_acfs = []
surr_zcrs = []

for i in range(N_SURROGATES):
    e_white = rng.normal(0.0, TARGET_STD, size=N_SAMPLES)
    pv_white = SP_NOMINAL + e_white
    op_dummy = np.full(N_SAMPLES, 41.10)
    t_dummy = np.arange(21, 21 + N_SAMPLES, dtype=float)
    ldf_white = pd.DataFrame({"timestep": t_dummy, "SP": SP_NOMINAL, "PV": pv_white, "OP": op_dummy})
    d2 = compute_oscillation_detection(ldf_white)
    surr_acfs.append(d2["acf_osc_score"])
    surr_zcrs.append(d2["zero_crossing_rate"])

surr_acfs = np.array(surr_acfs)
surr_zcrs = np.array(surr_zcrs)
n_surr_flag = int(np.sum(surr_acfs >= 0.34))

print(f"50 i.i.d. Gaussian White Noise Series (length 480, std={TARGET_STD:.4f}, seed=42):")
print(f"  Surrogate ZCR (/100 steps) : mean = {surr_zcrs.mean():.2f}, range = [{surr_zcrs.min():.2f}, {surr_zcrs.max():.2f}]")
print(f"  Surrogate ACF score        : mean = {surr_acfs.mean():.4f}, range = [{surr_acfs.min():.4f}, {surr_acfs.max():.4f}]")
print(f"  Flagged oscillatory (ACF >= 0.34): {n_surr_flag} / 50 ({n_surr_flag / N_SURROGATES:.1%})")

# -----------------------------------------------------------------------------
# 3. Raw Feature Distributions on DEVELOPMENT Split (runs 1..50)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. RAW FEATURE DISTRIBUTIONS ON DEVELOPMENT SPLIT (RUNS 1..50)")
print("=" * 80)

# Load faulty data dev split
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)
faulty_dev = df_faulty[(df_faulty.simulationRun >= 1) & (df_faulty.simulationRun <= 50) & (df_faulty["sample"] >= 21)].copy()

CLASSES = {
    "Healthy": ff_dev,
    "Fault 14 (Valve Sticking)": faulty_dev[faulty_dev.faultNumber == 14],
    "Fault 11 (Cooling Water Temp Osc)": faulty_dev[faulty_dev.faultNumber == 11],
    "Fault 4 (Cooling Water Temp Step)": faulty_dev[faulty_dev.faultNumber == 4],
}

DELAYS = [1, 2, 3, 5]
class_stats = {}

for cname, cdf in CLASSES.items():
    zcr_l, period_l, acf_l, corr_l = [], [], [], []
    ih_l = {d: [] for d in DELAYS}
    
    runs = sorted(cdf.simulationRun.unique().tolist())
    assert len(runs) == 50, f"Expected 50 runs for {cname}, got {len(runs)}"
    
    for r in runs:
        sub = cdf[cdf.simulationRun == r].sort_values("sample")
        pv = sub["xmeas_9"].to_numpy(dtype=float)
        op = sub["xmv_10"].to_numpy(dtype=float)
        sp = np.full_like(pv, SP_NOMINAL)
        t = sub["sample"].to_numpy(dtype=float)
        ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
        
        # D#2
        d2 = compute_oscillation_detection(ldf)
        zcr_l.append(d2["zero_crossing_rate"])
        period_l.append(d2["dominant_period"])
        acf_l.append(d2["acf_osc_score"])
        
        # D#3
        d3 = compute_stiction_detection_v2(ldf)
        corr_l.append(d3["op_pv_corr_filt"])
        
        # D#4
        for d in DELAYS:
            d4 = compute_harris_index(ldf, delay=d, settle_startup=30, settle_sp=0)
            ih_l[d].append(d4["harris_index"])
            
    def get_mmm(arr):
        a = np.array(arr)
        return float(np.min(a)), float(np.median(a)), float(np.max(a))
        
    class_stats[cname] = {
        "ZCR": get_mmm(zcr_l),
        "dominant_period": get_mmm(period_l),
        "acf_osc_score": get_mmm(acf_l),
        "op_pv_corr_filt": get_mmm(corr_l),
    }
    for d in DELAYS:
        class_stats[cname][f"IH_d{d}"] = get_mmm(ih_l[d])

# Print table
rows = []
for cname, st in class_stats.items():
    r = {"Class": cname}
    r["ZCR (min, med, max)"] = f"{st['ZCR'][0]:.1f}, {st['ZCR'][1]:.1f}, {st['ZCR'][2]:.1f}"
    r["Period (min, med, max)"] = f"{st['dominant_period'][0]:.1f}, {st['dominant_period'][1]:.1f}, {st['dominant_period'][2]:.1f}"
    r["ACF score (min, med, max)"] = f"{st['acf_osc_score'][0]:.3f}, {st['acf_osc_score'][1]:.3f}, {st['acf_osc_score'][2]:.3f}"
    r["op_pv_corr_filt (min, med, max)"] = f"{st['op_pv_corr_filt'][0]:.3f}, {st['op_pv_corr_filt'][1]:.3f}, {st['op_pv_corr_filt'][2]:.3f}"
    r["IH d=1 (min, med, max)"] = f"{st['IH_d1'][0]:.3f}, {st['IH_d1'][1]:.3f}, {st['IH_d1'][2]:.3f}"
    r["IH d=2 (min, med, max)"] = f"{st['IH_d2'][0]:.3f}, {st['IH_d2'][1]:.3f}, {st['IH_d2'][2]:.3f}"
    r["IH d=3 (min, med, max)"] = f"{st['IH_d3'][0]:.3f}, {st['IH_d3'][1]:.3f}, {st['IH_d3'][2]:.3f}"
    r["IH d=5 (min, med, max)"] = f"{st['IH_d5'][0]:.3f}, {st['IH_d5'][1]:.3f}, {st['IH_d5'][2]:.3f}"
    rows.append(r)

df_stats = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 15)
print(df_stats.to_string(index=False))
