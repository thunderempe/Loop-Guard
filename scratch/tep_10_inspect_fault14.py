"""Inspect Fault 14 on Reactor Temp loop (xmv_10, xmeas_9)."""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics import compute_stiction_detection_v2, compute_oscillation_detection

df_faulty = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
f14 = df_faulty[(df_faulty.faultNumber == 14) & (df_faulty.simulationRun == 1)].sort_values("sample")

# check post onset samples 21..500
f14_post = f14[f14["sample"] >= 21]
pv = f14_post["xmeas_9"].to_numpy()
op = f14_post["xmv_10"].to_numpy()
sp = np.full_like(pv, 120.4)
t = f14_post["sample"].to_numpy()

ldf = pd.DataFrame({"timestep": t, "SP": sp, "PV": pv, "OP": op})
d2 = compute_oscillation_detection(ldf)
d3 = compute_stiction_detection_v2(ldf)

print("Fault 14 Run 1 D#2:", d2)
print("Fault 14 Run 1 D#3 v2:", d3)
print(f"PV range: [{pv.min():.2f}, {pv.max():.2f}], std={pv.std():.3f} (baseline was 0.019)")
print(f"OP range: [{op.min():.2f}, {op.max():.2f}], std={op.std():.3f} (baseline was 0.54)")
