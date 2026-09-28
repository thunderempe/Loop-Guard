"""
Validation of D#1 to D#4 on Disturbance-Dominated Dataset
=========================================================
Runs all four diagnostics without retuning any thresholds on
data/simulated/labeled_loops_disturb.csv (seed base +4000, 30 loops).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics import (
    compute_setpoint_tracking_batch,
    compute_oscillation_detection_batch,
    compute_stiction_detection_batch,
    compute_harris_index_batch,
)

FAULTS = [
    'healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
    'oscillatory_tuning', 'noise'
]
STICTION_CLASSES = {'valve_stiction_v1', 'valve_stiction_v2'}

data_path = PROJECT_ROOT / "data" / "simulated" / "labeled_loops_disturb.csv"
df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]:,} rows, {df['loop_id'].nunique()} loops from {data_path.name}")

# Compute diagnostics
d1_res = compute_setpoint_tracking_batch(df)
d2_res = compute_oscillation_detection_batch(df)
d3_res = compute_stiction_detection_batch(df)
d4_res = compute_harris_index_batch(df, delay=3)

# --- 1. D#1 Setpoint Tracking ---
print("\n" + "=" * 80)
print("1. D#1: SETPOINT TRACKING (Sensor Drift Separation)")
print("=" * 80)
print(f"{'fault_type':<22} {'mean_error':>12} {'error_slope':>14} {'r2_linear':>14}")
print("-" * 65)
for ft in FAULTS:
    s = d1_res[d1_res['fault_type'] == ft]
    print(f"{ft:<22} {s['mean_error'].mean():>12.4f} {s['error_slope'].mean():>14.6f} {s['r2_linear'].mean():>14.4f}")

drift_d1 = d1_res[d1_res['fault_type'] == 'sensor_drift']
nondrift_d1 = d1_res[d1_res['fault_type'] != 'sensor_drift']

drift_r2_min = drift_d1['r2_linear'].min()
drift_r2_max = drift_d1['r2_linear'].max()
nondrift_r2_max = nondrift_d1['r2_linear'].max()
gap = drift_r2_min - nondrift_r2_max
next_worst_r2 = nondrift_d1.groupby('fault_type')['r2_linear'].mean().max()
r2_ratio = drift_d1['r2_linear'].mean() / next_worst_r2 if next_worst_r2 > 1e-12 else float('inf')
fp_drift = len(nondrift_d1[nondrift_d1['r2_linear'] > 0.50])

print("\nD#1 Separation Summary:")
print(f"  sensor_drift r2_linear range : [{drift_r2_min:.4f}, {drift_r2_max:.4f}]")
print(f"  non-drift r2_linear max      : {nondrift_r2_max:.4f}")
print(f"  r2_linear gap (min - max)    : {gap:+.4f} ({'SEPARATES' if gap > 0 else 'OVERLAPS'})")
print(f"  Separation ratio (drift/next): {r2_ratio:.2f}x")
print(f"  False positive drift (r2>0.5): {fp_drift}")

# --- 2. D#2 Oscillation Detection ---
print("\n" + "=" * 80)
print("2. D#2: OSCILLATION DETECTION (Signal Ranges)")
print("=" * 80)
print(f"{'fault_type':<22} {'ZCR range':>18} {'acf_osc range':>20} {'period range':>18}")
print("-" * 80)
for ft in FAULTS:
    s = d2_res[d2_res['fault_type'] == ft]
    zcr_str = f"[{s['zero_crossing_rate'].min():.2f}, {s['zero_crossing_rate'].max():.2f}]"
    acf_str = f"[{s['acf_osc_score'].min():.4f}, {s['acf_osc_score'].max():.4f}]"
    per_str = f"[{s['dominant_period'].min():.1f}, {s['dominant_period'].max():.1f}]"
    print(f"{ft:<22} {zcr_str:>18} {acf_str:>20} {per_str:>18}")

noise_zcr_min = d2_res[d2_res['fault_type'] == 'noise']['zero_crossing_rate'].min()
others_zcr_max = d2_res[d2_res['fault_type'] != 'noise']['zero_crossing_rate'].max()
h_zcr = d2_res[d2_res['fault_type'] == 'healthy']['zero_crossing_rate']

print("\nD#2 Key Checks:")
print(f"  Noise ZCR min ({noise_zcr_min:.2f}) vs Non-noise max ({others_zcr_max:.2f}): {'SEPARATES' if noise_zcr_min > others_zcr_max else 'OVERLAPS'}")
print(f"  Healthy ZCR range: [{h_zcr.min():.2f}, {h_zcr.max():.2f}] (Clean was [0.60, 1.00], Low noise was [10.4, 24.3])")

# --- 3. D#3 Valve Stiction Detection ---
print("\n" + "=" * 80)
print("3. D#3: VALVE STICTION DETECTION (corr < 0.25 + gate)")
print("=" * 80)
print(f"{'fault_type':<22} {'flagged':>8} {'expected':>8} {'op_pv_corr range':>24} {'result':>12}")
print("-" * 78)
total_fp = 0
total_miss = 0
for ft in FAULTS:
    s = d3_res[d3_res['fault_type'] == ft]
    flagged = int(s['stiction_flag'].sum())
    exp = 5 if ft in STICTION_CLASSES else 0
    corr_str = f"[{s['op_pv_corr'].min():.4f}, {s['op_pv_corr'].max():.4f}]"
    if ft in STICTION_CLASSES:
        miss = exp - flagged
        total_miss += miss
        res_str = "OK" if miss == 0 else f"MISS x{miss}"
    else:
        fp = flagged
        total_fp += fp
        res_str = "OK" if fp == 0 else f"FP x{fp}"
    print(f"{ft:<22} {flagged:>8d} {exp:>8d} {corr_str:>24} {res_str:>12}")

stict_corr_max = d3_res[d3_res['fault_type'].isin(STICTION_CLASSES)]['op_pv_corr'].max()
osc_corr_min = d3_res[d3_res['fault_type'] == 'oscillatory_tuning']['op_pv_corr'].min()
corr_gap = osc_corr_min - stict_corr_max

print(f"\nD#3 Summary:")
print(f"  Total False Positives: {total_fp}")
print(f"  Total Misses         : {total_miss}")
print(f"  op_pv_corr gap: osc_min ({osc_corr_min:.4f}) - stict_max ({stict_corr_max:.4f}) = {corr_gap:+.4f} ({'GAP HOLDS' if corr_gap > 0 else 'GAP COLLAPSED'})")

# Print loop-by-loop stiction details to see exactly what happened
print("\nD#3 Stiction loop details:")
for ft in ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2', 'oscillatory_tuning']:
    sub = d3_res[d3_res['fault_type'] == ft]
    for _, row in sub.iterrows():
        print(f"  Loop {int(row['loop_id']):2d} ({ft:<18}): flag={int(row['stiction_flag'])}, osc_gate={row['osc_flag_or_gate'] if 'osc_flag_or_gate' in row else 'N/A'}, corr={row['op_pv_corr']:.4f}")

# --- 4. D#4 Harris Index ---
print("\n" + "=" * 80)
print("4. D#4: HARRIS PERFORMANCE INDEX (delay = 3)")
print("=" * 80)
print(f"{'fault_type':<22} {'mean':>10} {'min':>10} {'max':>10} {'std':>10}")
print("-" * 65)
for ft in FAULTS:
    s = d4_res[d4_res['fault_type'] == ft]
    print(f"{ft:<22} {s['harris_index'].mean():>10.4f} {s['harris_index'].min():>10.4f} {s['harris_index'].max():>10.4f} {s['harris_index'].std():>10.4f}")

h_mean = d4_res[d4_res['fault_type'] == 'healthy']['harris_index'].mean()
drift_max = d4_res[d4_res['fault_type'] == 'sensor_drift']['harris_index'].max()
stict_mean = d4_res[d4_res['fault_type'].isin(STICTION_CLASSES)]['harris_index'].mean()
osc_mean = d4_res[d4_res['fault_type'] == 'oscillatory_tuning']['harris_index'].mean()

print("\nD#4 Benchmark Status:")
print(f"  Healthy mean ({h_mean:.4f}) vs sensor_drift max ({drift_max:.4f}): {'ABOVE' if h_mean > drift_max else 'OVERLAP'}")
print(f"  Healthy mean ({h_mean:.4f}) vs stiction mean ({stict_mean:.4f}): {'ABOVE' if h_mean > stict_mean else 'OVERLAP'}")
print(f"  Healthy mean ({h_mean:.4f}) vs osc_tuning mean ({osc_mean:.4f}): {'ABOVE' if h_mean > osc_mean else 'OVERLAP'}")
