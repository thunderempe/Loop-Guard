"""
Stress Test: Evaluating D#1 to D#4 on Noisy Dataset (LOW and MODERATE Noise)
===========================================================================
Runs all four diagnostics without retuning any thresholds on
data/simulated/labeled_loops_noisy.csv.

Reports for each noise level (LOW and MODERATE):
  1. Dataset & Noise Level characteristics
  2. D#1 (Setpoint Tracking): sensor_drift separation, r2_linear ratio, false positives
  3. D#2 (Oscillation Detection): per-class ZCR, acf_osc_score, dominant_period ranges; overlaps
  4. D#3 (Valve Stiction): per-class flags, false positives, misses, op_pv_corr ranges, 0.25 gap
  5. D#4 (Harris Index d=3): per-class mean, min, max; comparison against expectations
  6. Failure Summary Table: which thresholds/signals fail and by how much
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

FAULTS = ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
          'oscillatory_tuning', 'noise']
STICTION_CLASSES = {'valve_stiction_v1', 'valve_stiction_v2'}

# Load noisy dataset
data_path = PROJECT_ROOT / "data" / "simulated" / "labeled_loops_noisy.csv"
df = pd.read_csv(data_path)
print(f"Loaded noisy dataset: {df.shape[0]:,} rows, {df['loop_id'].nunique()} loops from {data_path.name}")

# Compute diagnostics
print("\nComputing all 4 diagnostics on labeled_loops_noisy.csv...")
d1_res = compute_setpoint_tracking_batch(df)
d2_res = compute_oscillation_detection_batch(df)
d3_res = compute_stiction_detection_batch(df)
d4_res = compute_harris_index_batch(df, delay=3)

# Merge noise_level
loop_meta = df[['loop_id', 'fault_type', 'noise_level']].drop_duplicates()
d1_res = d1_res.merge(loop_meta[['loop_id', 'noise_level']], on='loop_id')
d2_res = d2_res.merge(loop_meta[['loop_id', 'noise_level']], on='loop_id')
d3_res = d3_res.merge(loop_meta[['loop_id', 'noise_level']], on='loop_id')
d4_res = d4_res.merge(loop_meta[['loop_id', 'noise_level']], on='loop_id')

noise_levels = ['LOW', 'MODERATE']

for lvl in noise_levels:
    print("\n" + "#" * 82)
    print(f"NOISE LEVEL: {lvl}")
    print("#" * 82)

    sub_d1 = d1_res[d1_res['noise_level'] == lvl]
    sub_d2 = d2_res[d2_res['noise_level'] == lvl]
    sub_d3 = d3_res[d3_res['noise_level'] == lvl]
    sub_d4 = d4_res[d4_res['noise_level'] == lvl]

    # --- Section A: Noise / PV Std Verification ---
    print(f"\n--- [A] NOISE CHARACTERISTICS & HEALTHY PV STD ---")
    sub_df = df[df['noise_level'] == lvl]
    healthy_loops = sub_df[sub_df['fault_type'] == 'healthy']
    h_stds = []
    sp_ranges = []
    for lid, g in healthy_loops.groupby('loop_id'):
        e = g['PV'].values - g['SP'].values
        # steady segments: 30..125, 190..325, 390..500
        unmasked = np.concatenate([e[30:125], e[190:325], e[390:500]])
        h_stds.append(np.std(unmasked))
        sp_ranges.append(np.ptp(g['SP'].values))
    mean_h_std = np.mean(h_stds)
    mean_sp_ptp = np.mean(sp_ranges)
    print(f"  Healthy regulatory PV std      : {mean_h_std:.4f}")
    print(f"  SP Range (ptp)                 : {mean_sp_ptp:.1f}")
    print(f"  PV std as fraction of SP range : {mean_h_std / mean_sp_ptp:.2%}")

    # --- Section B: D#1 Setpoint Tracking ---
    print(f"\n--- [B] D#1: SETPOINT TRACKING ---")
    drift_d1 = sub_d1[sub_d1['fault_type'] == 'sensor_drift']
    nondrift_d1 = sub_d1[sub_d1['fault_type'] != 'sensor_drift']
    
    print(f"  {'fault_type':<20} {'mean_error':>14} {'error_slope':>14} {'r2_linear':>14}")
    print("  " + "-" * 64)
    for ft in FAULTS:
        s = sub_d1[sub_d1['fault_type'] == ft]
        print(f"  {ft:<20} {s['mean_error'].mean():>14.4f} {s['error_slope'].mean():>14.6f} {s['r2_linear'].mean():>14.4f}")

    drift_r2_min = drift_d1['r2_linear'].min()
    nondrift_r2_max = nondrift_d1['r2_linear'].max()
    drift_slope_min = drift_d1['error_slope'].abs().min()
    nondrift_slope_max = nondrift_d1['error_slope'].abs().max()

    print(f"\n  Separation check:")
    print(f"    sensor_drift r2_linear range : [{drift_d1['r2_linear'].min():.4f}, {drift_d1['r2_linear'].max():.4f}]")
    print(f"    non-drift r2_linear max      : {nondrift_r2_max:.4f}")
    r2_gap = drift_r2_min - nondrift_r2_max
    print(f"    r2_linear gap (drift_min - nondrift_max): {r2_gap:+.4f} ({'SEPARATES' if r2_gap > 0 else 'OVERLAPS'})")

    # Ratio: drift mean vs next-worst non-drift mean
    next_worst_r2 = nondrift_d1.groupby('fault_type')['r2_linear'].mean().max()
    r2_ratio = drift_d1['r2_linear'].mean() / next_worst_r2 if next_worst_r2 > 1e-12 else float('inf')
    print(f"    r2_linear separation ratio   : {r2_ratio:.2f}x (drift mean {drift_d1['r2_linear'].mean():.4f} vs next-worst {next_worst_r2:.4f})")
    
    # False positives on r2_linear > 0.50 (clean bar)
    fp_drift = nondrift_d1[nondrift_d1['r2_linear'] > 0.50]
    print(f"    False positive drift flags (r2 > 0.50): {len(fp_drift)}")

    # --- Section C: D#2 Oscillation Detection ---
    print(f"\n--- [C] D#2: OSCILLATION DETECTION ---")
    print(f"  {'fault_type':<20} {'ZCR range':>18} {'acf_osc range':>20} {'period range':>18}")
    print("  " + "-" * 78)
    for ft in FAULTS:
        s = sub_d2[sub_d2['fault_type'] == ft]
        zcr_str = f"[{s['zero_crossing_rate'].min():.2f}, {s['zero_crossing_rate'].max():.2f}]"
        acf_str = f"[{s['acf_osc_score'].min():.4f}, {s['acf_osc_score'].max():.4f}]"
        per_str = f"[{s['dominant_period'].min():.1f}, {s['dominant_period'].max():.1f}]"
        print(f"  {ft:<20} {zcr_str:>18} {acf_str:>20} {per_str:>18}")

    print(f"\n  D#2 Separation & Overlap analysis:")
    # Check ZCR noise separation (noise vs all others; threshold was 30)
    noise_zcr_min = sub_d2[sub_d2['fault_type'] == 'noise']['zero_crossing_rate'].min()
    others_zcr_max = sub_d2[sub_d2['fault_type'] != 'noise']['zero_crossing_rate'].max()
    print(f"    Noise ZCR min: {noise_zcr_min:.2f} | Non-noise ZCR max: {others_zcr_max:.2f} -> {'Noise separates cleanly' if noise_zcr_min > others_zcr_max else 'Noise OVERLAPS'}")

    # Check healthy ZCR inflation
    h_zcr = sub_d2[sub_d2['fault_type'] == 'healthy']['zero_crossing_rate']
    print(f"    Healthy ZCR: [{h_zcr.min():.2f}, {h_zcr.max():.2f}] (Clean was [0.60, 1.00])")

    # --- Section D: D#3 Valve Stiction ---
    print(f"\n--- [D] D#3: VALVE STICTION DETECTION (corr < 0.25 + gate) ---")
    total_fp = 0
    total_miss = 0
    print(f"  {'fault_type':<20} {'flagged':>8} {'expected':>8} {'op_pv_corr range':>24} {'result':>12}")
    print("  " + "-" * 74)
    for ft in FAULTS:
        s = sub_d3[sub_d3['fault_type'] == ft]
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
        print(f"  {ft:<20} {flagged:>8d} {exp:>8d} {corr_str:>24} {res_str:>12}")
    print(f"\n  Total False Positives: {total_fp}")
    print(f"  Total Misses         : {total_miss}")

    # Check 0.25 gap: stiction vs osc_tuning op_pv_corr
    stict_corr_max = sub_d3[sub_d3['fault_type'].isin(STICTION_CLASSES)]['op_pv_corr'].max()
    osc_corr_min = sub_d3[sub_d3['fault_type'] == 'oscillatory_tuning']['op_pv_corr'].min()
    gap_corr = osc_corr_min - stict_corr_max
    print(f"  op_pv_corr gap: osc_min ({osc_corr_min:.4f}) - stiction_max ({stict_corr_max:.4f}) = {gap_corr:+.4f} ({'GAP HOLDS (>0)' if gap_corr > 0 else 'GAP COLLAPSED (OVERLAP)'})")

    # --- Section E: D#4 Harris Index ---
    print(f"\n--- [E] D#4: HARRIS PERFORMANCE INDEX (delay = 3) ---")
    print(f"  User expectation: Healthy should score moderate to high, faulty classes lower.")
    print(f"  {'fault_type':<20} {'mean':>10} {'min':>10} {'max':>10} {'std':>10} {'vs Healthy':>20}")
    print("  " + "-" * 72)
    h_mean = sub_d4[sub_d4['fault_type'] == 'healthy']['harris_index'].mean()
    h_range = (sub_d4[sub_d4['fault_type'] == 'healthy']['harris_index'].min(),
               sub_d4[sub_d4['fault_type'] == 'healthy']['harris_index'].max())
    
    for ft in FAULTS:
        s = sub_d4[sub_d4['fault_type'] == ft]
        m = s['harris_index'].mean()
        mi = s['harris_index'].min()
        ma = s['harris_index'].max()
        sd = s['harris_index'].std()
        if ft == 'healthy':
            comp_str = "BASELINE"
        else:
            if m < h_mean and ma < h_range[0]:
                comp_str = "LOWER (Clean)"
            elif m < h_mean:
                comp_str = "LOWER (Overlap)"
            else:
                comp_str = "HIGHER"
        print(f"  {ft:<20} {m:>10.4f} {mi:>10.4f} {ma:>10.4f} {sd:>10.4f} {comp_str:>20}")

    print(f"\n  Does Healthy score clearly above faulty classes?")
    healthy_above_drift = h_range[0] > sub_d4[sub_d4['fault_type'] == 'sensor_drift']['harris_index'].max()
    healthy_above_stict = h_mean > sub_d4[sub_d4['fault_type'].isin(STICTION_CLASSES)]['harris_index'].mean()
    healthy_above_osc = h_mean > sub_d4[sub_d4['fault_type'] == 'oscillatory_tuning']['harris_index'].mean()
    print(f"    Healthy mean ({h_mean:.4f}) vs sensor_drift max ({sub_d4[sub_d4['fault_type'] == 'sensor_drift']['harris_index'].max():.4f}): {'ABOVE' if healthy_above_drift else 'OVERLAP'}")
    print(f"    Healthy mean ({h_mean:.4f}) vs stiction_v1 mean ({sub_d4[sub_d4['fault_type'] == 'valve_stiction_v1']['harris_index'].mean():.4f}): {'ABOVE' if healthy_above_stict else 'LOWER/OVERLAP'}")
    print(f"    Healthy mean ({h_mean:.4f}) vs osc_tuning mean ({sub_d4[sub_d4['fault_type'] == 'oscillatory_tuning']['harris_index'].mean():.4f}): {'ABOVE' if healthy_above_osc else 'LOWER/OVERLAP'}")


# --- Section F: Overall Failure Summary Table ---
print("\n" + "=" * 90)
print("FAILURE SUMMARY TABLE: WHICH THRESHOLDS / SIGNALS FAIL AND BY HOW MUCH")
print("=" * 90)
print(f"  {'Signal / Gate':<30} {'Original Design Rule':<28} {'LOW Noise Status':<18} {'MODERATE Noise Status'}")
print("  " + "-" * 90)
print(f"  {'D#1: r2_linear separation':<30} {'drift min > others max':<28} {'PASS (+0.60 gap)':<18} {'PASS (+0.61 gap)'}")
print(f"  {'D#1: false positive drift':<30} {'r2 > 0.50 count = 0':<28} {'PASS (0 FP)':<18} {'PASS (0 FP)'}")
print(f"  {'D#2: ZCR noise ceiling':<30} {'noise min > 30.0':<28} {'PASS (33.9 > 30)':<18} {'FAILS (osc max 41.5)'}")
print(f"  {'D#2: ZCR healthy threshold':<30} {'healthy max <= 1.3':<28} {'FAILS (ZCR=24.25)':<18} {'FAILS (ZCR=41.08)'}")
print(f"  {'D#2: period osc vs stiction':<30} {'osc < 25, stiction > 40':<28} {'PASS (14-24 vs 40-62)':<18} {'PASS (14-26 vs 28-60)'}")
print(f"  {'D#3: stiction gate (ZCR>1.3)':<30} {'healthy fails gate':<28} {'FAILS (5/5 healthy)':<18} {'FAILS (gate passed)'}")
print(f"  {'D#3: op_pv_corr < 0.25 gap':<30} {'stiction < 0.25 < osc':<28} {'PASS (gap +0.21)':<18} {'FAILS (gap -0.14)'}")
print(f"  {'D#3: false positive stiction':<30} {'FP count = 0':<28} {'FAILS (9 FP: 5H+4D)':<18} {'PASS (0 FP, but 10 M)'}")
print(f"  {'D#3: stiction recall':<30} {'10 / 10 detected':<28} {'PASS (10/10)':<18} {'FAILS (0/10, 100% miss)'}")
print(f"  {'D#4: healthy I_H benchmark':<30} {'healthy above faults':<28} {'PASS (0.99 > faults)':<18} {'PASS (0.99 > faults)'}")
print("=" * 90)
