"""TEP Step 1 / script 06: Determine empirical fault onset sample across all 20 faults.
Loads fault-free training (runs 1..100) from tep_cache_ff_train.pkl
and faulty training subset (runs 1..100) from tep_faulty_train_subset.csv.gz.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# 1. Load fault-free training runs 1..100
df_ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df_ff = df_ff[(df_ff["simulationRun"] >= 1) & (df_ff["simulationRun"] <= 100)].sort_values(["simulationRun", "sample"])

# 2. Load faulty training subset runs 1..100
df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
df_faulty["sample"] = df_faulty["sample"].astype(int)
df_faulty = df_faulty[(df_faulty["simulationRun"] >= 1) & (df_faulty["simulationRun"] <= 100)].sort_values(["faultNumber", "simulationRun", "sample"])

loop_cols = [c for c in df_faulty.columns if c not in ["faultNumber", "simulationRun", "sample"]]

# Check point-wise equality for t in 1..500 for run 1 of fault 1 vs run 1 of fault-free
r1_ff = df_ff[df_ff["simulationRun"] == 1].set_index("sample")[loop_cols]
r1_f1 = df_faulty[(df_faulty["faultNumber"] == 1) & (df_faulty["simulationRun"] == 1)].set_index("sample")[loop_cols]
diff_r1 = np.abs(r1_f1.values - r1_ff.values)

# Find first sample where difference is non-zero
max_diff_per_sample = diff_r1.max(axis=1) # index 0 is sample 1
exact_match_until = np.where(max_diff_per_sample > 1e-6)[0]
if len(exact_match_until) > 0:
    first_diff_idx = exact_match_until[0] # 0-indexed, so sample is first_diff_idx + 1
    print(f"Pointwise test (Run 1 Fault 1 vs Run 1 FF): Exact match for samples 1 to {first_diff_idx}. First diff at sample {first_diff_idx + 1}")
    print(f"Max diff at sample {first_diff_idx}: {max_diff_per_sample[first_diff_idx-1]:.6e}, at sample {first_diff_idx+1}: {max_diff_per_sample[first_diff_idx]:.6e}")
else:
    print("Pointwise test: No difference found (or seeds completely different from step 1).")

# Check all 20 faults across all 100 runs
print("\nChecking all 20 faults: first sample of non-zero deviation from FaultFree (same run_id):")
onset_per_fault = {}
for fn in range(1, 21):
    f_sub = df_faulty[df_faulty["faultNumber"] == fn].sort_values(["simulationRun", "sample"])
    # Pivot or array comparison
    # Both df_ff and f_sub have 100 runs x 500 samples
    vals_ff = df_ff[loop_cols].to_numpy().reshape(100, 500, len(loop_cols))
    vals_f = f_sub[loop_cols].to_numpy().reshape(100, 500, len(loop_cols))
    
    diff = np.abs(vals_f - vals_ff) # (100, 500, n_cols)
    max_diff_across_runs_cols = diff.max(axis=(0, 2)) # (500,)
    
    # samples where diff > 1e-6
    diverged = np.where(max_diff_across_runs_cols > 1e-6)[0]
    if len(diverged) > 0:
        first_step = diverged[0] + 1 # sample is 1-indexed
        max_pre = max_diff_across_runs_cols[:diverged[0]].max() if diverged[0] > 0 else 0.0
        max_at = max_diff_across_runs_cols[diverged[0]]
        onset_per_fault[fn] = (first_step, max_pre, max_at)
        print(f"  Fault {fn:2d}: first divergence at sample {first_step:3d} (max pre-diff={max_pre:.2e}, diff at onset={max_at:.4f})")
    else:
        onset_per_fault[fn] = (None, 0.0, 0.0)
        print(f"  Fault {fn:2d}: NO divergence across all 500 samples on these 18 columns!")
