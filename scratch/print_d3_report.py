import pandas as pd
import numpy as np

df = pd.read_csv('scratch/d3_features_explored.csv')

features = ['amp_norm', 'peak_to_median']
regimes = ['CLEAN', 'LOW_NOISE', 'MOD_NOISE', 'DISTURB']
faults = ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2', 'oscillatory_tuning', 'noise']

for feat in features:
    print("\n" + "=" * 95)
    print(f"FEATURE: {feat.upper()}")
    print("=" * 95)
    for reg in regimes:
        print(f"\n--- REGIME: {reg} ---")
        print(f"{'fault_type':<20} | {'UNMASKED (min, mean, max)':<35} | {'FULL SERIES (min, mean, max)':<35}")
        print("-" * 95)
        sub_unm = df[(df['regime'] == reg) & (df['mode'] == 'unmasked')]
        sub_full = df[(df['regime'] == reg) & (df['mode'] == 'full')]
        for ft in faults:
            su = sub_unm[sub_unm['fault_type'] == ft][feat]
            sf = sub_full[sub_full['fault_type'] == ft][feat]
            if feat in ['amp_norm', 'peak_to_median']:
                u_str = f"[{su.min():8.2f}, {su.mean():8.2f}, {su.max():8.2f}]"
                f_str = f"[{sf.min():8.2f}, {sf.mean():8.2f}, {sf.max():8.2f}]"
            else:
                u_str = f"[{su.min():8.4f}, {su.mean():8.4f}, {su.max():8.4f}]"
                f_str = f"[{sf.min():8.4f}, {sf.mean():8.4f}, {sf.max():8.4f}]"
            print(f"{ft:<20} | {u_str:<35} | {f_str:<35}")
