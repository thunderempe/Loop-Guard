"""
Test Joint Gate: Local Prominence AND Normalized Amplitude
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r'D:\Loop Gaurd')
clean_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops.csv')
noisy_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_noisy.csv')
disturb_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_disturb.csv')

regimes = {
    'CLEAN': clean_df,
    'LOW': noisy_df[noisy_df['noise_level'] == 'LOW'],
    'MODERATE': noisy_df[noisy_df['noise_level'] == 'MODERATE'],
    'DISTURB': disturb_df,
}

# Load the local prominence results
prom_df = pd.read_csv(PROJECT_ROOT / 'scratch/prominence_results.csv')

# Load the amp_norm results from explore_d3_candidates (unmasked)
feat_df = pd.read_csv(PROJECT_ROOT / 'scratch/d3_features_explored.csv')
amp_df = feat_df[feat_df['mode'] == 'unmasked'][['regime', 'loop_id', 'amp_norm', 'corr_filt']].copy()
amp_df['regime'] = amp_df['regime'].replace({'LOW_NOISE': 'LOW', 'MOD_NOISE': 'MODERATE'})

# Merge them
merged = prom_df.merge(amp_df, on=['regime', 'loop_id'])

# Check Healthy and Drift max values across all regimes
healthy_drift = merged[merged['fault_type'].isin(['healthy', 'sensor_drift'])]
stiction = merged[merged['fault_type'].isin(['valve_stiction_v1', 'valve_stiction_v2'])]

print("=== HEALTHY & SENSOR DRIFT MAXIMA PER REGIME ===")
for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
    hd = healthy_drift[healthy_drift['regime'] == reg]
    print(f"[{reg}]")
    for ft in ['healthy', 'sensor_drift']:
        sub = hd[hd['fault_type'] == ft]
        print(f"  {ft:<15}: max prom = {sub['local_prominence'].max():.2f}, max amp_norm = {sub['amp_norm'].max():.2f}")

print("\n=== STICTION MINIMA PER REGIME ===")
for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
    st = stiction[stiction['regime'] == reg]
    print(f"[{reg}]")
    for ft in ['valve_stiction_v1', 'valve_stiction_v2']:
        sub = st[st['fault_type'] == ft]
        print(f"  {ft:<18}: min prom = {sub['local_prominence'].min():.2f}, min amp_norm = {sub['amp_norm'].min():.2f}")

# Check specific clean loops: 23, 35, 37, 39
print("\n=== SPECIFIC CLEAN LOOPS: 23, 35, 37, 39 ===")
clean_sub = merged[merged['regime'] == 'CLEAN']
for lid in [23, 35, 37, 39]:
    r = clean_sub[clean_sub['loop_id'] == lid].iloc[0]
    print(f"Clean Loop {lid:2d} ({r['fault_type']:<18}): prom = {r['local_prominence']:6.2f}, amp_norm = {r['amp_norm']:8.2f}, corr_filt = {r['corr_filt']:.4f}")

# Search for any fixed threshold pair (prom_thresh, amp_thresh)
print("\n=== SEARCHING FOR FIXED THRESHOLDS WITH ZERO HEALTHY/DRIFT FP ===")
# To have 0 FP on healthy/drift:
# prom_thresh must be > max(healthy, drift) OR amp_thresh must be > max(healthy, drift)
# For a joint AND gate: (prom >= prom_th) AND (amp_norm >= amp_th)
# To avoid FP on a given healthy/drift loop, that loop must fail AT LEAST ONE condition:
# (prom < prom_th) OR (amp_norm < amp_th).

# Let's test a grid of candidate thresholds:
prom_candidates = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0]
amp_candidates = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 50.0, 100.0, 300.0]

valid_pairs = []
for p_th in prom_candidates:
    for a_th in amp_candidates:
        # Check FP on healthy and drift across ALL 4 regimes
        hd_flagged = healthy_drift[(healthy_drift['local_prominence'] >= p_th) & (healthy_drift['amp_norm'] >= a_th)]
        if len(hd_flagged) == 0:
            # Zero FP! Compute recall per regime
            recalls = {}
            misses = {}
            for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
                st_reg = stiction[stiction['regime'] == reg]
                n_tot = len(st_reg)
                passed = st_reg[(st_reg['local_prominence'] >= p_th) & (st_reg['amp_norm'] >= a_th)]
                n_pass = len(passed)
                missed_ids = st_reg[~st_reg['loop_id'].isin(passed['loop_id'])]['loop_id'].tolist()
                recalls[reg] = f"{n_pass}/{n_tot}"
                misses[reg] = missed_ids
            valid_pairs.append((p_th, a_th, recalls, misses))

print(f"Total threshold pairs with ZERO False Positives across ALL 4 regimes: {len(valid_pairs)}")
for p_th, a_th, rec, miss in valid_pairs[:10]:
    print(f"  Thresholds: prom >= {p_th:4.1f}, amp_norm >= {a_th:5.1f} -> Recalls: {rec}")
    print(f"    Misses CLEAN: {miss['CLEAN']}")
    print(f"    Misses MOD  : {miss['MODERATE']}")
