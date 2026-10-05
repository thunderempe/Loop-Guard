"""Phase 3 Step 2 / consolidated re-run of Parts A and B on BOTH splits (runs 1-50 = DEV, 51-100 = REPORT).
Loop: OP = xmv_10, PV = xmeas_9, SP = 120.40 (fault-free grand mean). Samples 21..500. src/ is used read-only.

Flag rules (only D#3 has a flag inside src/; the others are applied HERE):
  D#1 drift : r2_linear >= 0.65 (README D#1 rule); also reported at >= 0.50
  D#2 noise : zero_crossing_rate > 30 (README D#2 / D#3-v2 noise ceiling)
  D#2 osc   : acf_osc_score >= 0.34 (borrowed from D#3 v1 acf_gate; README states no D#2 ACF threshold)
  D#3       : compute_stiction_detection_v2(...)["stiction_flag"] with default arguments
  D#4       : no Phase 2 threshold exists -> values only
Extra check: AR(1) surrogates with phi = healthy DEV mean lag-1 ACF, to test whether the healthy D#2 ACF flag rate
is explained by the negative lag-1 correlation alone (the white-noise surrogate already gave 0/50).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.diagnostics import (compute_setpoint_tracking, compute_oscillation_detection,
                             compute_stiction_detection_v2, compute_harris_index)

SP = 120.40
DELAYS = [1, 2, 3, 5]
ff = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl")
fa = pd.read_csv(ROOT / "scratch" / "tep_faulty_train_subset.csv.gz")
fa["sample"] = fa["sample"].astype(int)
ff = ff[ff["sample"] >= 21]
fa = fa[fa["sample"] >= 21]
SRC = {"Healthy": ff, "Fault 14": fa[fa.faultNumber == 14], "Fault 11": fa[fa.faultNumber == 11],
       "Fault 4": fa[fa.faultNumber == 4]}


def run_one(sub):
    sub = sub.sort_values("sample")
    pv = sub["xmeas_9"].to_numpy(float); op = sub["xmv_10"].to_numpy(float)
    ldf = pd.DataFrame({"timestep": sub["sample"].to_numpy(float), "SP": np.full_like(pv, SP), "PV": pv, "OP": op})
    d1 = compute_setpoint_tracking(ldf); d2 = compute_oscillation_detection(ldf); d3 = compute_stiction_detection_v2(ldf)
    e = pv - SP; ec = e - e.mean()
    row = dict(n=len(pv), r2=d1["r2_linear"], zcr=d2["zero_crossing_rate"], acf=d2["acf_osc_score"],
               period=d2["dominant_period"], corr_filt=d3["op_pv_corr_filt"], d3=d3["stiction_flag"],
               g_drift=d3["drift_excluded"], g_noise=d3["noise_excluded"], g_osc=d3["oscillating"], g_corr=d3["corr_ok"],
               lag1=float(np.dot(ec[1:], ec[:-1]) / np.dot(ec, ec)), e_std=float(e.std()), op_std=float(op.std()))
    for d in DELAYS:
        row[f"ih{d}"] = compute_harris_index(ldf, delay=d, settle_startup=30, settle_sp=0)["harris_index"]
    return row


rows = []
for split, (lo, hi) in {"DEV 1-50": (1, 50), "REPORT 51-100": (51, 100)}.items():
    for cls, df in SRC.items():
        d = df[(df.simulationRun >= lo) & (df.simulationRun <= hi)]
        runs = sorted(d.simulationRun.unique())
        assert len(runs) == 50, (split, cls, len(runs))
        for r in runs:
            rows.append(dict(split=split, cls=cls, run=int(r), **run_one(d[d.simulationRun == r])))
R = pd.DataFrame(rows)
assert (R.n == 480).all(), R.n.unique()
nan_cols = R.columns[R.isna().any()].tolist()
print(f"runs evaluated: {len(R)}  samples/run: {R.n.unique().tolist()}  columns with NaN: {nan_cols}")

pct = lambda s: f"{int(s.sum())}/{len(s)}"
print("\n=== DETECTION RATES (unchanged Phase 2 rules) ===")
det = R.groupby(["split", "cls"], sort=False).apply(lambda g: pd.Series({
    "D1 r2>=0.65": pct(g.r2 >= 0.65), "D1 r2>=0.50": pct(g.r2 >= 0.50),
    "D2 ZCR>30": pct(g.zcr > 30), "D2 ACF>=0.34": pct(g.acf >= 0.34), "D3v2 flag": pct(g.d3),
    "gate drift_ok": pct(g.g_drift), "gate noise_ok": pct(g.g_noise), "gate osc_ok": pct(g.g_osc), "gate corr_ok": pct(g.g_corr)}),
    include_groups=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(det.to_string())

print("\n=== RAW FEATURES: min / median / max ===")
mmm = lambda s: f"{s.min():.3g} / {s.median():.3g} / {s.max():.3g}"
feat = R.groupby(["split", "cls"], sort=False).apply(lambda g: pd.Series({
    "ZCR": mmm(g.zcr), "ACF": mmm(g.acf), "period": mmm(g.period), "corr_filt": mmm(g.corr_filt),
    "r2_linear": mmm(g.r2)}), include_groups=False)
print(feat.to_string())
print()
ih = R.groupby(["split", "cls"], sort=False).apply(lambda g: pd.Series({f"I_H d={d}": mmm(g[f"ih{d}"]) for d in DELAYS}),
                                                  include_groups=False)
print(ih.to_string())
print()
oth = R.groupby(["split", "cls"], sort=False).apply(lambda g: pd.Series({
    "err_std": mmm(g.e_std), "err_lag1": mmm(g.lag1), "OP_std": mmm(g.op_std),
    "I_H==1.0 count (d=3)": int((g.ih3 >= 0.99999).sum()), "period==2 count": int((g.period == 2).sum())}),
    include_groups=False)
print(oth.to_string())

print("\n=== AR(1) SURROGATE CHECK for D#2 ACF flag on healthy ===")
h = R[(R.split == "DEV 1-50") & (R.cls == "Healthy")]
phi, sd = float(h.lag1.mean()), float(h.e_std.mean())
rng = np.random.RandomState(7)
acfs, zcrs = [], []
for i in range(200):
    a = rng.normal(0, 1, 480 + 200); x = np.zeros_like(a)
    for t in range(1, len(a)):
        x[t] = phi * x[t - 1] + a[t]
    x = x[200:]; x = x / x.std() * sd
    ldf = pd.DataFrame({"timestep": np.arange(480.0), "SP": SP, "PV": SP + x, "OP": 41.1})
    d2 = compute_oscillation_detection(ldf); acfs.append(d2["acf_osc_score"]); zcrs.append(d2["zero_crossing_rate"])
acfs = np.array(acfs)
print(f"phi={phi:.4f} sd={sd:.4f}  200 surrogates (seed 7): ACF min/med/max = {acfs.min():.3f}/{np.median(acfs):.3f}/{acfs.max():.3f}"
      f"  ACF>=0.34: {(acfs >= 0.34).sum()}/200  ZCR median={np.median(zcrs):.1f}")
print(f"healthy DEV actual: ACF>=0.34 {(h.acf >= 0.34).sum()}/50, ACF median {h.acf.median():.3f}")
