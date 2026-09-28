import pandas as pd
import numpy as np

prom_df = pd.read_csv('scratch/prominence_results.csv')
feat_df = pd.read_csv('scratch/d3_features_explored.csv')
amp_df = feat_df[feat_df['mode'] == 'unmasked'][['regime', 'loop_id', 'amp_norm', 'corr_filt']].copy()
amp_df['regime'] = amp_df['regime'].replace({'LOW_NOISE': 'LOW', 'MOD_NOISE': 'MODERATE'})
merged = prom_df.merge(amp_df, on=['regime', 'loop_id'])

# Only test rejecting HEALTHY (assuming drift is rejected by D#1 R2 < 0.40)
healthy = merged[merged['fault_type'] == 'healthy']
stiction = merged[merged['fault_type'].isin(['valve_stiction_v1', 'valve_stiction_v2'])]

print("=== HEALTHY MAXIMA PER REGIME ===")
for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
    h = healthy[healthy['regime'] == reg]
    print(f"[{reg}] max prom = {h['local_prominence'].max():.2f}, max amp_norm = {h['amp_norm'].max():.2f}")

print("\n=== STICTION MINIMA PER REGIME ===")
for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
    st = stiction[stiction['regime'] == reg]
    print(f"[{reg}] min prom = {st['local_prominence'].min():.2f}, min amp_norm = {st['amp_norm'].min():.2f}")

# Search for any fixed pair (prom_th, amp_th) that yields 0 FP on healthy across ALL 4 regimes
prom_candidates = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
amp_candidates = [1.05, 1.15, 1.25, 1.35, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0]

valid_pairs = []
for p_th in prom_candidates:
    for a_th in amp_candidates:
        h_flagged = healthy[(healthy['local_prominence'] >= p_th) & (healthy['amp_norm'] >= a_th)]
        if len(h_flagged) == 0:
            recalls = {}
            misses = {}
            for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
                st_reg = stiction[stiction['regime'] == reg]
                n_tot = len(st_reg)
                passed = st_reg[(st_reg['local_prominence'] >= p_th) & (st_reg['amp_norm'] >= a_th)]
                recalls[reg] = f"{len(passed)}/{n_tot}"
                misses[reg] = st_reg[~st_reg['loop_id'].isin(passed['loop_id'])]['loop_id'].tolist()
            valid_pairs.append((p_th, a_th, recalls, misses))

print(f"\nTotal pairs with 0 Healthy FP: {len(valid_pairs)}")
for p_th, a_th, rec, miss in valid_pairs:
    print(f"prom >= {p_th:.1f}, amp >= {a_th:.2f} -> Recalls: {rec}")
