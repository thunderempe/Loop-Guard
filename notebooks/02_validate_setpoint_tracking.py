"""
Validation script for compute_setpoint_tracking().

Runs the diagnostic over all 60 loops in labeled_loops.csv and prints:
  1. Per-fault-type summary statistics (mean ± std) for all four signals
  2. Full per-loop table sorted by r2_linear descending
  3. Flag table: any non-drift loop with high mean_error, error_slope, or r2_linear
  4. Separation ratio: sensor_drift mean vs. next-worst fault type
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.diagnostics import compute_setpoint_tracking_batch

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / 'data' / 'simulated' / 'labeled_loops.csv')
print(f"Loaded: {df.shape[0]:,} rows, {df['loop_id'].nunique()} loops\n")

# ── Compute metrics over all 60 loops ──────────────────────────────────────
results = compute_setpoint_tracking_batch(df, settle_steps=0)

FAULT_ORDER = [
    'healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
    'oscillatory_tuning', 'noise',
]

# ── 1. Per-fault-type summary (mean ± std) ──────────────────────────────────
print("=" * 78)
print("SUMMARY: mean ± std per fault type (10 loops each)")
print("=" * 78)
metrics = ['mean_error', 'error_slope', 'rmse_error', 'r2_linear']

summary_rows = []
for ft in FAULT_ORDER:
    sub = results[results['fault_type'] == ft]
    row = {'fault_type': ft}
    for m in metrics:
        row[f'{m}_mean'] = sub[m].mean()
        row[f'{m}_std']  = sub[m].std()
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)

# Print in a readable grid
header = f"{'fault_type':<22} {'mean_error':>18} {'error_slope':>18} {'rmse_error':>14} {'r2_linear':>14}"
print(header)
print("-" * 90)
for _, r in summary.iterrows():
    me  = f"{r['mean_error_mean']:+.3f} ± {r['mean_error_std']:.3f}"
    es  = f"{r['error_slope_mean']:+.5f} ± {r['error_slope_std']:.5f}"
    rms = f"{r['rmse_error_mean']:.3f} ± {r['rmse_error_std']:.3f}"
    r2  = f"{r['r2_linear_mean']:.4f} ± {r['r2_linear_std']:.4f}"
    print(f"  {r['fault_type']:<20} {me:>18} {es:>18} {rms:>14} {r2:>14}")

# ── 2. Full per-loop table sorted by r2_linear ─────────────────────────────
print("\n" + "=" * 78)
print("FULL PER-LOOP TABLE (sorted by r2_linear DESC)")
print("=" * 78)
display = results.sort_values('r2_linear', ascending=False).copy()
display['mean_error']  = display['mean_error'].map(lambda x: f"{x:+.3f}")
display['error_slope'] = display['error_slope'].map(lambda x: f"{x:+.6f}")
display['rmse_error']  = display['rmse_error'].map(lambda x: f"{x:.3f}")
display['r2_linear']   = display['r2_linear'].map(lambda x: f"{x:.4f}")
print(display.to_string(index=False))

# ── 3. Flag table: suspicious non-drift loops ───────────────────────────────
# Thresholds: we use the MINIMUM sensor_drift value as the "clean" bar.
drift_loops = results[results['fault_type'] == 'sensor_drift']
non_drift   = results[results['fault_type'] != 'sensor_drift']

# Thresholds — flag if a non-drift loop crosses 50% of the sensor_drift minimum
thresh_mean_error  = drift_loops['mean_error'].abs().min() * 0.5
thresh_error_slope = drift_loops['error_slope'].abs().min() * 0.5
thresh_r2          = drift_loops['r2_linear'].min() * 0.5

flagged = non_drift[
    (non_drift['mean_error'].abs()  > thresh_mean_error)  |
    (non_drift['error_slope'].abs() > thresh_error_slope) |
    (non_drift['r2_linear']         > thresh_r2)
].copy()

print("\n" + "=" * 78)
print("FLAG TABLE: Non-drift loops that exceed 50% of min sensor_drift signal")
print(f"  Thresholds used -> |mean_error| > {thresh_mean_error:.3f} "
      f"| |error_slope| > {thresh_error_slope:.6f} "
      f"| r2_linear > {thresh_r2:.4f}")
print("=" * 78)
if flagged.empty:
    print("  [OK] No false-positive flags -- all non-drift loops are cleanly below thresholds.")
else:
    print(f"  [!!] {len(flagged)} loop(s) flagged:\n")
    flag_cols = ['loop_id', 'fault_type', 'mean_error', 'error_slope', 'rmse_error', 'r2_linear']
    print(flagged[flag_cols].to_string(index=False))

# ── 4. Separation ratio ─────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("SEPARATION ANALYSIS: sensor_drift vs. all other fault types")
print("=" * 78)
for metric in ['mean_error', 'error_slope', 'r2_linear']:
    drift_mean = drift_loops[metric].abs().mean()
    others_max  = non_drift.groupby('fault_type')[metric].apply(lambda x: x.abs().mean()).max()
    ratio = drift_mean / others_max if others_max > 1e-12 else float('inf')
    print(f"  {metric:<14}: sensor_drift abs_mean = {drift_mean:.4f}  |  "
          f"next-worst abs_mean = {others_max:.4f}  |  separation ratio = {ratio:.1f}×")

print("\n" + "=" * 78)
print("DONE — Diagnostic #1 validation complete.")
print("=" * 78)
