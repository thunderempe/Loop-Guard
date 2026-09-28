"""
Validation: windowed-max ACF for acf_osc_score.
Checks:
  1. Summary table by fault_type (new vs old scores for oscillatory_tuning)
  2. Loops 41, 45, 46 specifically: did they recover?
  3. Separation ratios
  4. Sanity check: healthy/noise/sensor_drift spurious windows
  5. Stiction ambiguous zone [0.5, 0.7] carry-forward list
"""
import sys
sys.path.insert(0, r'D:\Loop Gaurd')

import numpy as np
import pandas as pd
from src.diagnostics import compute_oscillation_detection_batch, compute_oscillation_detection

df = pd.read_csv(r'D:\Loop Gaurd\data\simulated\labeled_loops.csv')
print(f"Loaded {df.shape[0]:,} rows, {df['loop_id'].nunique()} loops\n")

results = compute_oscillation_detection_batch(df, settle_steps=0)

FAULT_ORDER = ['healthy','sensor_drift','valve_stiction_v1','valve_stiction_v2',
               'oscillatory_tuning','noise']
METRICS = ['osc_amplitude','acf_osc_score','zero_crossing_rate','dominant_period']

# ── 1. Summary table ────────────────────────────────────────────────────────
print("=" * 95)
print("SUMMARY (windowed-max ACF): mean +/- std per fault type  (10 loops each)")
print("=" * 95)
hdr = (f"{'fault_type':<22}  {'osc_amplitude':>16}  "
       f"{'acf_osc_score':>18}  {'ZCR/100steps':>14}  {'dom_period':>12}")
print(hdr)
print("-" * 95)
for ft in FAULT_ORDER:
    sub = results[results['fault_type'] == ft]
    amp = f"{sub.osc_amplitude.mean():.3f} +/- {sub.osc_amplitude.std():.3f}"
    acf = f"{sub.acf_osc_score.mean():.4f} +/- {sub.acf_osc_score.std():.4f}"
    zcr = f"{sub.zero_crossing_rate.mean():.2f} +/- {sub.zero_crossing_rate.std():.2f}"
    dp  = f"{sub.dominant_period.mean():.1f} +/- {sub.dominant_period.std():.1f}"
    print(f"  {ft:<20}  {amp:>16}  {acf:>18}  {zcr:>14}  {dp:>12}")

# ── 2. Spotlight: loops 41, 45, 46 (the previously low-scoring episodic ones)
print("\n" + "=" * 95)
print("SPOTLIGHT: Previously low-scoring episodic oscillatory loops")
print("  (Old full-series scores from investigation: 41=0.1049, 45=0.1966, 46=0.1725)")
print("=" * 95)
spotlight_ids = [41, 44, 45, 46, 49, 50]   # episodic + high-scoring for context
spot = results[results['loop_id'].isin(spotlight_ids)].sort_values('loop_id')
# Also show kp_factor context
kp_factors = {41:2.8, 42:3.2, 43:3.6, 44:4.0, 45:4.5, 46:3.0, 47:3.5, 48:3.8, 49:4.2, 50:4.8}
ki_factors = {41:2.0, 42:2.5, 43:2.8, 44:3.2, 45:3.5, 46:2.2, 47:2.6, 48:3.0, 49:3.3, 50:3.8}
for _, row in spot.iterrows():
    lid = int(row.loop_id)
    print(f"  loop {lid:>2}  kp_x={kp_factors.get(lid,'?')}  ki_x={ki_factors.get(lid,'?')}"
          f"  acf_osc_score={row.acf_osc_score:.4f}"
          f"  osc_amp={row.osc_amplitude:.3f}"
          f"  period={row.dominant_period:.0f}"
          f"  zcr={row.zero_crossing_rate:.2f}")

# ── 3. Full per-loop table sorted by acf_osc_score ─────────────────────────
print("\n" + "=" * 95)
print("FULL PER-LOOP TABLE (sorted by acf_osc_score DESC)")
print("=" * 95)
display = results.sort_values('acf_osc_score', ascending=False).copy()
for col in ['osc_amplitude','acf_osc_score','zero_crossing_rate','dominant_period']:
    display[col] = display[col].map(lambda x, c=col:
        f"{x:.4f}" if c == 'acf_osc_score' else
        f"{x:.3f}" if c in ('osc_amplitude','zero_crossing_rate') else f"{x:.0f}")
print(display.to_string(index=False))

# ── 4. Separation ratios ────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("SEPARATION: oscillatory_tuning vs. all other fault types")
print("=" * 95)
osc_loops = results[results['fault_type'] == 'oscillatory_tuning']
non_osc   = results[results['fault_type'] != 'oscillatory_tuning']
for metric in ['osc_amplitude', 'acf_osc_score']:
    osc_mean   = osc_loops[metric].mean()
    by_ft      = non_osc.groupby('fault_type')[metric].mean()
    worst_ft   = by_ft.idxmax()
    worst_val  = by_ft.max()
    ratio      = osc_mean / worst_val if worst_val > 1e-12 else float('inf')
    print(f"  {metric:<22}: osc_tuning={osc_mean:.4f}  next-worst={worst_val:.4f}"
          f"  ({worst_ft})  ratio={ratio:.1f}x")

# ── 5. Sanity check: spurious windowed scores in non-oscillatory fault types
print("\n" + "=" * 95)
print("SANITY CHECK: Do healthy/noise/sensor_drift loops get spuriously high")
print("acf_osc_score from a lucky 125-step sub-window?")
print("(If windowing introduces false positives, healthy/noise scores should jump here)")
print("=" * 95)

# Compute window-level scores for all individual 125-step windows
# across healthy, noise, and sensor_drift to see the distribution
win_scores = []
for ft in ['healthy', 'noise', 'sensor_drift']:
    ft_loops = df[df['fault_type'] == ft]
    for loop_id, grp in ft_loops.groupby('loop_id'):
        from src.diagnostics import _extract_arrays, _detrend_error, _autocorrelation
        sp, pv, op, t = _extract_arrays(grp)
        error    = pv - sp
        error_dt = _detrend_error(error, t)
        win_size, win_stride = 125, 50
        max_lag_w = win_size // 4
        for ws in range(0, max(1, len(error_dt) - win_size + 1), win_stride):
            chunk = error_dt[ws:ws+win_size]
            if len(chunk) < 4:
                continue
            acf_w = _autocorrelation(chunk, max_lag=min(max_lag_w, len(chunk)-1))
            acf_lags = acf_w[1:]
            if len(acf_lags) == 0:
                continue
            min_val = float(acf_lags[np.argmin(acf_lags)])
            score   = float(np.clip(-min_val, 0.0, 1.0))
            win_scores.append({'fault_type': ft, 'loop_id': loop_id,
                                'win_start': ws, 'window_acf_score': score})

win_df = pd.DataFrame(win_scores)
print(f"  Total windows evaluated across healthy/noise/sensor_drift: {len(win_df)}")
print()
for ft in ['healthy', 'noise', 'sensor_drift']:
    sub = win_df[win_df['fault_type'] == ft]['window_acf_score']
    print(f"  {ft:<22}: window scores: "
          f"mean={sub.mean():.4f}  max={sub.max():.4f}  "
          f"p95={sub.quantile(0.95):.4f}  p99={sub.quantile(0.99):.4f}")

# Flag any window where a non-oscillatory loop scores > 0.5
high_thresh = 0.50
suspicious = win_df[win_df['window_acf_score'] > high_thresh]
if suspicious.empty:
    print(f"\n  [OK] No healthy/noise/sensor_drift window scores above {high_thresh}.")
else:
    print(f"\n  [!!] {len(suspicious)} window(s) exceed {high_thresh}:")
    print(suspicious.sort_values('window_acf_score', ascending=False).head(10).to_string(index=False))

# ── 6. Stiction ambiguous zone ──────────────────────────────────────────────
print("\n" + "=" * 95)
print("STICTION AMBIGUOUS ZONE: acf_osc_score in [0.5, 0.7] (carry-forward for D#3)")
print("=" * 95)
stiction = results[results['fault_type'].isin(['valve_stiction_v1','valve_stiction_v2'])]
ambig = stiction[(stiction.acf_osc_score >= 0.5) & (stiction.acf_osc_score <= 0.7)]
if ambig.empty:
    print("  [OK] No stiction loops in ambiguous zone.")
else:
    print(f"  {len(ambig)} loop(s):")
    print(ambig[['loop_id','fault_type','osc_amplitude','acf_osc_score',
                  'zero_crossing_rate','dominant_period']].to_string(index=False))
print("\n  Full stiction acf_osc_score range:")
for ft in ['valve_stiction_v1','valve_stiction_v2']:
    sub = stiction[stiction['fault_type']==ft]
    vals = sorted(sub.acf_osc_score.values)
    print(f"  {ft:<20}: [{min(vals):.4f}, {max(vals):.4f}]  "
          f"values: {[f'{v:.4f}' for v in vals]}")

print("\n" + "=" * 95)
print("DONE -- Windowed ACF validation complete.")
print("=" * 95)
