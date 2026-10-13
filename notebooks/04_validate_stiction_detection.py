"""
Validation: Diagnostic #3 -- Valve Stiction Detection
======================================================
Runs compute_stiction_detection_batch() over all 60 loops and reports:
  1. Per-class flag summary (all 6 fault types)
  2. Explicit false positive and miss counts
  3. op_pv_corr separation between stiction and oscillatory_tuning
  4. Hard loop spotlight (21, 26, 28, 30, 33, 38)
  5. Full per-loop table sorted by stiction_flag then fault_type
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.diagnostics import compute_stiction_detection_batch

# -- Load -------------------------------------------------------------------
df = pd.read_csv(ROOT / 'data' / 'simulated' / 'labeled_loops.csv')
print(f"Loaded {df.shape[0]:,} rows, {df['loop_id'].nunique()} loops")

results = compute_stiction_detection_batch(df, settle_steps=0)

FAULTS = ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
          'oscillatory_tuning', 'noise']
HARD   = [21, 26, 28, 30, 33, 38]
STICTION_CLASSES = {'valve_stiction_v1', 'valve_stiction_v2'}

# -- 1. Gate threshold reminder ---------------------------------------------
print()
print("=" * 76)
print("THRESHOLDS USED")
print("=" * 76)
print("  Oscillation gate:  (ZCR > 1.3) OR (ZCR > 0.9 AND acf_osc_score > 0.34)")
print("                     AND ZCR < 30")
print("    ZCR > 1.3       : healthy max ZCR = 1.002, sensor_drift max = 1.202")
print("    ZCR > 0.9       : blocks sensor_drift (ZCR <= 0.802) from acf branch")
print("    acf_osc_score   : 0.34 -- above sensor_drift loop 17 max (0.332)")
print("    ZCR < 30        : noise min ZCR = 36.9")
print("  Stiction flag:     gate_pass AND op_pv_corr < 0.25")
print("    op_pv_corr gap  : stiction [-.03, 0.16] vs osc_tuning [0.41, 0.84]")

# -- 2. Per-class summary ---------------------------------------------------
print()
print("=" * 76)
print("PER-CLASS FLAG SUMMARY (10 loops each)")
print("=" * 76)
total_fp   = 0
total_miss = 0
print(f"  {'fault_type':<22}  {'flagged':>8}  {'expected':>8}  {'result':>20}")
print("  " + "-" * 64)
for ft in FAULTS:
    sub     = results[results['fault_type'] == ft]
    flagged = int(sub['stiction_flag'].sum())
    exp     = 10 if ft in STICTION_CLASSES else 0
    if ft in STICTION_CLASSES:
        missed = exp - flagged
        total_miss += missed
        note = "OK" if missed == 0 else f"MISS x{missed}"
    else:
        fp = flagged
        total_fp += fp
        note = "OK" if fp == 0 else f"FALSE POS x{fp}"
    print(f"  {ft:<22}  {flagged:>8}  {exp:>8}  {note:>20}")

print()
print(f"  Total false positives : {total_fp}")
miss_note = "none" if total_miss == 0 else "stiction_v2 mild cases -- see section 4"
print(f"  Total misses          : {total_miss}  ({miss_note})")

# -- 3. op_pv_corr separation -----------------------------------------------
print()
print("=" * 76)
print("op_pv_corr SEPARATION: stiction vs. oscillatory_tuning")
print("=" * 76)
stiction = results[results['fault_type'].isin(STICTION_CLASSES)]
osc      = results[results['fault_type'] == 'oscillatory_tuning']
print(f"  stiction (all 20 loops)  : [{stiction.op_pv_corr.min():.4f}, {stiction.op_pv_corr.max():.4f}]"
      f"  mean={stiction.op_pv_corr.mean():.4f}")
print(f"  oscillatory_tuning       : [{osc.op_pv_corr.min():.4f}, {osc.op_pv_corr.max():.4f}]"
      f"  mean={osc.op_pv_corr.mean():.4f}")
gap_lo = osc.op_pv_corr.min() - stiction.op_pv_corr.max()
print(f"  Gap (osc_min - stict_max): {gap_lo:.4f}  "
      + ("CLEAN (no overlap)" if gap_lo > 0 else "OVERLAP"))

print()
print("  op_pv_corr by class:")
for ft in FAULTS:
    sub = results[results['fault_type'] == ft]
    print(f"    {ft:<22}: [{sub.op_pv_corr.min():.4f}, {sub.op_pv_corr.max():.4f}]"
          f"  mean={sub.op_pv_corr.mean():.4f}")

# -- 4. Hard loop spotlight -------------------------------------------------
print()
print("=" * 76)
print("HARD LOOP SPOTLIGHT (6 stiction loops most similar to oscillatory_tuning)")
print("  These scored highest acf_osc_score in D#2, overlapping osc_tuning range")
print("=" * 76)
hard = results[results['loop_id'].isin(HARD)].sort_values('loop_id')
print(f"  {'loop_id':>8}  {'fault_type':<20}  {'op_pv_corr':>12}  "
      f"{'op_curvature':>14}  {'op_delta_cv':>12}  {'flag':>20}")
print("  " + "-" * 94)
for _, row in hard.iterrows():
    flag_str = "TRUE  <-- detected" if row['stiction_flag'] else "FALSE <-- MISS"
    print(f"  {int(row.loop_id):>8}  {row.fault_type:<20}  {row.op_pv_corr:>12.4f}  "
          f"{row.op_curvature:>14.4f}  {row.op_delta_cv:>12.4f}  {flag_str:>20}")
print()
print(f"  osc_tuning op_pv_corr range: [{osc.op_pv_corr.min():.4f}, {osc.op_pv_corr.max():.4f}]")

# -- 5. Miss detail ---------------------------------------------------------
missed_loops = results[
    results['fault_type'].isin(STICTION_CLASSES) & ~results['stiction_flag']
]
if not missed_loops.empty:
    print()
    print("=" * 76)
    print("MISSED STICTION LOOPS (stiction_flag=False)")
    print("  Root cause: ZCR and acf_osc_score indistinguishable from healthy/drift.")
    print("  op_pv_corr is still low (correct signal); only the gate blocks the flag.")
    print("=" * 76)
    for _, row in missed_loops.sort_values('loop_id').iterrows():
        print(f"  loop {int(row.loop_id):>2}  {row.fault_type:<20}: "
              f"op_pv_corr={row.op_pv_corr:.4f}  (corr OK; gate blocked)")

# -- 6. Full per-loop table -------------------------------------------------
print()
print("=" * 76)
print("FULL PER-LOOP TABLE (by fault_type then loop_id)")
print("=" * 76)
display = results.sort_values(['fault_type', 'loop_id']).copy()
for col in ['op_pv_corr', 'op_curvature', 'op_delta_cv']:
    display[col] = display[col].map(lambda x: f"{x:.4f}")
print(display.to_string(index=False))

print()
print("=" * 76)
print("DONE -- Diagnostic #3 validation complete.")
print("=" * 76)
