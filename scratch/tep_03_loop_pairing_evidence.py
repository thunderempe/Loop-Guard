"""TEP Step 1 / script 03: empirical loop-pairing evidence from fault-free training data (cached float32).
Builds a (run, sample, var) cube and reports:
 A) for each XMV: top XMEAS partners by |corr| of per-run-demeaned levels and of first differences, with lag
 B) static gain test XMEAS/XMV for the top partner (is flow simply proportional to valve?)
 C) for each XMEAS: grand mean, spread of per-run means vs within-run std (constant-SP evidence), start-vs-end drift
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
df = df.sort_values(["simulationRun", "sample"])
xmeas = [f"xmeas_{i}" for i in range(1, 42)]
xmv = [f"xmv_{i}" for i in range(1, 12)]
V = xmeas + xmv
R, S = df.simulationRun.nunique(), df["sample"].nunique()
cube = df[V].to_numpy(np.float64).reshape(R, S, len(V))
assert (df["sample"].to_numpy().reshape(R, S) == np.arange(1, S + 1)).all()
idx = {v: i for i, v in enumerate(V)}
dm = cube - cube.mean(axis=1, keepdims=True)          # per-run demeaned
dd = np.diff(cube, axis=1)                             # first differences


def xcorr(a, b, lag):
    """corr(a[t], b[t+lag]) pooled over runs; a,b: (R, T). lag>0 means a leads b."""
    if lag > 0:
        a, b = a[:, :-lag], b[:, lag:]
    elif lag < 0:
        a, b = a[:, -lag:], b[:, :lag]
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


LAGS = range(-5, 11)
print("=" * 100)
print("A) XMV -> XMEAS partners (lag>0 = XMV leads).  level = per-run demeaned, diff = first differences")
print("=" * 100)
for mv in xmv:
    rows = []
    for m in xmeas:
        lv = [(xcorr(dm[:, :, idx[mv]], dm[:, :, idx[m]], L), L) for L in LAGS]
        df_ = [(xcorr(dd[:, :, idx[mv]], dd[:, :, idx[m]], L), L) for L in LAGS]
        bl = max(lv, key=lambda t: abs(t[0])); bd = max(df_, key=lambda t: abs(t[0]))
        rows.append((m, bl[0], bl[1], bd[0], bd[1]))
    rows.sort(key=lambda r: -abs(r[3]))
    print(f"\n{mv}: top-4 by |diff corr|")
    for m, cl, ll, cd, ld in rows[:4]:
        print(f"   {m:9s} diff r={cd:+.3f} @lag {ld:+d} | level r={cl:+.3f} @lag {ll:+d}")

print("\n" + "=" * 100)
print("B) Static gain test for hypothesised direct-acting pairs: ratio XMEAS/XMV (pooled), and R^2 of XMEAS ~ XMV (lag 0)")
print("=" * 100)
pairs = [("xmv_3", "xmeas_1"), ("xmv_1", "xmeas_2"), ("xmv_2", "xmeas_3"), ("xmv_4", "xmeas_4"),
         ("xmv_5", "xmeas_5"), ("xmv_6", "xmeas_10"), ("xmv_7", "xmeas_14"), ("xmv_8", "xmeas_17"),
         ("xmv_9", "xmeas_19"), ("xmv_10", "xmeas_21"), ("xmv_11", "xmeas_22")]
for mv, m in pairs:
    x = cube[:, :, idx[mv]].ravel(); y = cube[:, :, idx[m]].ravel()
    ratio = y / x
    r = np.corrcoef(x, y)[0, 1]
    print(f"  {mv:6s} vs {m:9s}: ratio mean={ratio.mean():.5g} CV={ratio.std()/abs(ratio.mean()):.3%}  corr(lag0)={r:+.4f}  R2={r*r:.4f}")

print("\n" + "=" * 100)
print("C) Constant-setpoint evidence per XMEAS: grand mean | std of run means | mean within-run std | ratio | drift(last100-first100)")
print("   small ratio (run-mean spread << within-run std) and ~0 drift => PV held at a fixed SP by integral action")
print("=" * 100)
out = []
for v in V:
    x = cube[:, :, idx[v]]
    rm = x.mean(axis=1); ws = x.std(axis=1).mean()
    drift = (x[:, -100:].mean(axis=1) - x[:, :100].mean(axis=1))
    lag1 = xcorr(dm[:, :, idx[v]], dm[:, :, idx[v]], 1)
    out.append((v, x.mean(), rm.std(), ws, rm.std() / ws, drift.mean(), drift.std(), lag1))
o = pd.DataFrame(out, columns=["var", "grand_mean", "sd_runmeans", "within_sd", "ratio", "drift_mean", "drift_sd", "acf_lag1"])
pd.set_option("display.width", 200)
print(o.sort_values("ratio").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
