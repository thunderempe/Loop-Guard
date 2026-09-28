"""
Evaluate Stiction Detection v2 on Fresh Test Sets (Seed Offset +2000)
=====================================================================
Runs compute_stiction_detection_v2_batch ONCE on:
  1. CLEAN FRESH (data/simulated/labeled_loops_clean_fresh2000.csv)
  2. LOW NOISE FRESH (data/simulated/labeled_loops_noisy_fresh2000.csv, LOW)
  3. MODERATE NOISE FRESH (data/simulated/labeled_loops_noisy_fresh2000.csv, MODERATE)
  4. DISTURBANCE FRESH (data/simulated/labeled_loops_disturb_fresh2000.csv)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics import compute_stiction_detection_v2_batch

out_dir = PROJECT_ROOT / "data" / "simulated"
clean_fresh = pd.read_csv(out_dir / "labeled_loops_clean_fresh2000.csv")
noisy_fresh = pd.read_csv(out_dir / "labeled_loops_noisy_fresh2000.csv")
disturb_fresh = pd.read_csv(out_dir / "labeled_loops_disturb_fresh2000.csv")

regimes = {
    'CLEAN FRESH': clean_fresh,
    'LOW NOISE FRESH': noisy_fresh[noisy_fresh['noise_level'] == 'LOW'],
    'MODERATE NOISE FRESH': noisy_fresh[noisy_fresh['noise_level'] == 'MODERATE'],
    'DISTURBANCE FRESH': disturb_fresh,
}

FAULTS = ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
          'oscillatory_tuning', 'noise']
STICTION_CLASSES = {'valve_stiction_v1', 'valve_stiction_v2'}

print("=" * 95)
print("ONE-TIME EVALUATION: STICTION DETECTION v2 ON FRESH-SEED (+2000) TEST SETS")
print("=" * 95)

for reg_name, r_df in regimes.items():
    print("\n" + "#" * 90)
    print(f"REGIME: {reg_name} ({r_df['loop_id'].nunique()} loops)")
    print("#" * 90)

    res = compute_stiction_detection_v2_batch(r_df)

    print(f"  {'fault_type':<20} {'flagged':>8} {'expected':>8} {'FP/Miss':>14} {'loop IDs'}")
    print("  " + "-" * 75)
    total_fp = 0
    total_miss = 0
    fp_ids = []
    miss_ids = []

    for ft in FAULTS:
        sub = res[res['fault_type'] == ft]
        flagged = int(sub['stiction_flag'].sum())
        total = len(sub)
        is_stict = (ft in STICTION_CLASSES)
        exp = total if is_stict else 0
        flagged_lids = sub[sub['stiction_flag']]['loop_id'].tolist()
        unflagged_lids = sub[~sub['stiction_flag']]['loop_id'].tolist()

        if is_stict:
            miss = exp - flagged
            total_miss += miss
            status = f"MISS x{miss}" if miss > 0 else "OK (100%)"
            lids_str = f"Missed: {unflagged_lids}" if miss > 0 else f"Flagged: {flagged_lids}"
            if miss > 0:
                miss_ids.extend(unflagged_lids)
        else:
            fp = flagged
            total_fp += fp
            status = f"FP x{fp}" if fp > 0 else "OK (0 FP)"
            lids_str = f"Flagged FP: {flagged_lids}" if fp > 0 else "-"
            if fp > 0:
                fp_ids.extend(flagged_lids)
        print(f"  {ft:<20} {flagged:>8d} {exp:>8d} {status:>14}  {lids_str}")

    n_stict_total = len(res[res['fault_type'].isin(STICTION_CLASSES)])
    n_stict_flagged = int(res[res['fault_type'].isin(STICTION_CLASSES)]['stiction_flag'].sum())
    print("  " + "-" * 75)
    print(f"  SUMMARY: Total FP = {total_fp:<3d} (IDs: {fp_ids}) | Stiction Recall = {n_stict_flagged}/{n_stict_total} (Miss IDs: {miss_ids})")
