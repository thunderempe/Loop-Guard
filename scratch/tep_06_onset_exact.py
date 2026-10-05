"""Check where the difference jumps above float32 roundoff (1e-4)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")

cols = [c for c in df_faulty.columns if c.startswith("xm")]

# Check for all 20 faults: at which sample does max|Faulty - FaultFree| across 100 runs exceed 1e-4?
print("Exact sample where max |Faulty - FaultFree| > 1e-4 across all 18 loop variables:")

for fn in range(1, 21):
    sub_f = df_faulty[df_faulty.faultNumber == fn].sort_values(["simulationRun", "sample"])
    sub_ff = df_ff[(df_ff.simulationRun >= 1) & (df_ff.simulationRun <= 100)].sort_values(["simulationRun", "sample"])
    
    arr_f = sub_f[cols].to_numpy(dtype=np.float32).reshape(100, 500, len(cols))
    arr_ff = sub_ff[cols].to_numpy(dtype=np.float32).reshape(100, 500, len(cols))
    
    diff = np.abs(arr_f - arr_ff) # (100, 500, n_cols)
    max_diff_t = diff.max(axis=(0, 2)) # per sample t in 0..499
    
    # Check max difference before sample 20, 50, etc.
    diverged = np.where(max_diff_t > 1e-4)[0]
    if len(diverged) > 0:
        onset_step = diverged[0] + 1
        pre_max = max_diff_t[:diverged[0]].max() if diverged[0] > 0 else 0.0
        onset_diff = max_diff_t[diverged[0]]
        print(f"  Fault {fn:2d}: onset sample = {onset_step:3d}  (max pre-onset diff = {pre_max:.2e}, diff at onset = {onset_diff:.6f})")
    else:
        print(f"  Fault {fn:2d}: NO divergence > 1e-4 across all 500 samples on these 18 columns (max diff={max_diff_t.max():.2e})")
