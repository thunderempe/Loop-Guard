"""Check if simulationRun 1 has identical seed or different seed, and find where macroscopic fault onset occurs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")

# Compare run 1 of Fault 1 vs run 1 of FaultFree
f1_r1 = df_faulty[(df_faulty.faultNumber == 1) & (df_faulty.simulationRun == 1)].sort_values("sample")
ff_r1 = df_ff[df_ff.simulationRun == 1].sort_values("sample")

# Print first 10 samples of xmv_1, xmeas_9 for both
print("Sample 1..10 xmv_1 comparison (F1 vs FF):")
for s in range(1, 15):
    v_f = f1_r1[f1_r1["sample"] == s]["xmv_1"].values[0]
    v_ff = ff_r1[ff_r1["sample"] == s]["xmv_1"].values[0]
    diff = abs(v_f - v_ff)
    print(f"  sample {s:2d}: F1={v_f:.6f}  FF={v_ff:.6f}  diff={diff:.6e}")

# Check across samples 1..500: average absolute difference per sample across all 18 variables
cols = [c for c in f1_r1.columns if c.startswith("xm")]
diffs = np.abs(f1_r1[cols].values - ff_r1[cols].values) # (500, n_cols)
# normalize by std of FF to get z-score diff
stds = ff_r1[cols].std().values
stds[stds == 0] = 1.0
z_diffs = diffs / stds
mean_z_diff = z_diffs.mean(axis=1)

print("\nMean normalized difference (z-diff) across samples:")
for s in [1, 2, 5, 10, 20, 50, 100, 150, 160, 161, 162, 175, 200, 300, 400, 500]:
    if s <= len(mean_z_diff):
        print(f"  sample {s:3d}: mean z-diff = {mean_z_diff[s-1]:.6f}")

# Look for the exact jump point in mean_z_diff or individual variables
large_jump = np.where(mean_z_diff > 0.05)[0]
if len(large_jump) > 0:
    print(f"\nFirst sample where z-diff > 0.05: sample {large_jump[0]+1}")
    for i in range(max(0, large_jump[0]-5), min(500, large_jump[0]+10)):
        print(f"  sample {i+1:3d}: z-diff = {mean_z_diff[i]:.6f}, max var diff = {z_diffs[i].max():.4f}")
