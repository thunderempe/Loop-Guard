"""TEP Step 1 / script 01: structure of TEP_FaultFree_Training.RData.
Read-only. Caches a float32 copy to scratch/tep_cache_ff_train.pkl for later scripts."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import pyreadr
from tep_mem import report

ROOT = Path(__file__).resolve().parent.parent
f = ROOT / "data" / "tep" / "TEP_FaultFree_Training.RData"
report("start")
t0 = time.time()
res = pyreadr.read_r(str(f))
print(f"load time {time.time()-t0:.1f}s; R objects: {list(res.keys())}")
key = list(res.keys())[0]
df = res[key]
del res
report("after load")

print(f"shape: {df.shape}")
print("dtypes:")
print(df.dtypes.value_counts().to_string())
print("first 3 cols dtypes:", df.dtypes.iloc[:3].to_dict())
cols = list(df.columns)
print("columns:", cols)
xmeas = [c for c in cols if c.lower().startswith("xmeas")]
xmv = [c for c in cols if c.lower().startswith("xmv")]
other = [c for c in cols if c not in xmeas + xmv]
print(f"n_xmeas={len(xmeas)} n_xmv={len(xmv)} other={other}")
print(f"total NaNs: {int(df.isna().sum().sum())}")

for c in other:
    u = np.unique(df[c])
    print(f"{c}: n_unique={len(u)} min={u.min()} max={u.max()}")
runs = df.groupby("simulationRun").size()
print(f"samples per run: min={runs.min()} max={runs.max()} n_runs={len(runs)}")
s = df[df.simulationRun == df.simulationRun.iloc[0]]["sample"].to_numpy()
print(f"sample index first/last in run 1: {s[:3]} ... {s[-3:]}  diffs unique: {np.unique(np.diff(s))}")

# per-variable summary (pooled)
desc = df[xmeas + xmv].describe().T[["mean", "std", "min", "max"]]
pd.set_option("display.width", 200)
print(desc.to_string(float_format=lambda v: f"{v:.4g}"))

# which variables are quantised / held (sampled analyzers): count distinct consecutive changes in run 1
r1 = df[df.simulationRun == 1]
chg = {c: float((np.diff(r1[c].to_numpy()) != 0).mean()) for c in xmeas + xmv}
held = {c: v for c, v in chg.items() if v < 0.9}
print("fraction of steps where value changes (run 1), only vars < 0.9:")
for c, v in held.items():
    print(f"  {c}: {v:.3f}")

df32 = df.copy()
for c in xmeas + xmv:
    df32[c] = df32[c].astype(np.float32)
df32.to_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
del df, df32
report("end")
