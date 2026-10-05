"""TEP Step 1 / script 04: identify controller law + implied setpoint per candidate loop (fault-free training).

For each (OP=xmv, PV=xmeas, lag L) candidate:
  PI test (velocity form):  dU[t] = Kc*dY[t-L] + Ki*Y[t-L] + c0     -> SP_hat = -c0/Ki   (per run)
  P  test (position form):   U[t]  = b*Y[t-L] + a                    (exact affine => P-only, SP not separable from bias)
Picks best lag in 0..5 by PI R^2. Also reports signal-character metrics used by D#1-D#4:
  OP within-run std, PV within-run std, PV lag-1 ACF of (PV - SP), ZCR per 100 steps of (PV - SP), fraction of PV steps unchanged (analyzer hold).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_pickle(ROOT / "scratch" / "tep_cache_ff_train.pkl").sort_values(["simulationRun", "sample"])
R, S = df.simulationRun.nunique(), df["sample"].nunique()
get = lambda c: df[c].to_numpy(np.float64).reshape(R, S)

CANDS = [("xmv_10", "xmeas_9"), ("xmv_4", "xmeas_4"), ("xmv_3", "xmeas_23"), ("xmv_1", "xmeas_26"),
         ("xmv_2", "xmeas_27"), ("xmv_6", "xmeas_30"), ("xmv_5", "xmeas_7"), ("xmv_5", "xmeas_13"),
         ("xmv_9", "xmeas_38"), ("xmv_9", "xmeas_19"), ("xmv_7", "xmeas_12"), ("xmv_8", "xmeas_15"),
         ("xmv_11", "xmeas_17"), ("xmv_3", "xmeas_1"), ("xmv_6", "xmeas_10")]


def pi_fit(U, Y, L):
    dU = np.diff(U, axis=1)[:, L:]                 # dU[t], t=1..S-1 (shifted so Y lag aligns)
    dY = np.diff(Y, axis=1)[:, :dU.shape[1]]
    Yl = Y[:, 1:][:, :dU.shape[1]]
    X = np.column_stack([dY.ravel(), Yl.ravel(), np.ones(dU.size)])
    coef, *_ = np.linalg.lstsq(X, dU.ravel(), rcond=None)
    res = dU.ravel() - X @ coef
    r2 = 1 - res.var() / dU.ravel().var()
    # per-run SP estimate
    sps = []
    for r in range(R):
        Xr = np.column_stack([dY[r], Yl[r], np.ones(dU.shape[1])])
        c, *_ = np.linalg.lstsq(Xr, dU[r], rcond=None)
        if abs(c[1]) > 1e-12:
            sps.append(-c[2] / c[1])
    return coef, r2, np.array(sps)


def p_fit(U, Y, L):
    u = U[:, L:].ravel(); y = Y[:, :S - L].ravel()
    b, a = np.polyfit(y, u, 1)
    res = u - (a + b * y)
    return b, a, 1 - res.var() / u.var(), res.std()


rows = []
for mv, pv in CANDS:
    U, Y = get(mv), get(pv)
    best = max(range(6), key=lambda L: pi_fit(U, Y, L)[1])
    coef, r2pi, sps = pi_fit(U, Y, best)
    b, a, r2p, rsd = p_fit(U, Y, best)
    sp_med = np.median(sps)
    e = Y - Y.mean()                                   # deviation from fault-free grand mean
    ec = e - e.mean(axis=1, keepdims=True)
    acf1 = (ec[:, 1:] * ec[:, :-1]).sum() / (ec * ec).sum()
    zcr = (np.sign(e[:, 1:]) != np.sign(e[:, :-1])).sum(axis=1).mean() / (S - 1) * 100
    hold = (np.diff(Y, axis=1) == 0).mean()
    rows.append(dict(OP=mv, PV=pv, lag=best, PI_R2=r2pi, Kc=coef[0], Ki=coef[1], SP_hat_median=sp_med,
                     SP_hat_IQR=np.subtract(*np.percentile(sps, [75, 25])), PV_grand_mean=Y.mean(),
                     P_R2=r2p, P_resid_sd=rsd, OP_sd=U.std(axis=1).mean(), PV_sd=Y.std(axis=1).mean(),
                     PVdev_acf1=acf1, ZCR_per100=zcr, PV_hold_frac=hold))

o = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("Controller-law identification (fault-free training, 500 runs x 500 samples)")
print(o[["OP", "PV", "lag", "PI_R2", "Kc", "Ki", "SP_hat_median", "SP_hat_IQR", "PV_grand_mean", "P_R2", "P_resid_sd"]]
      .to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print("\nSignal character (deviation from fault-free grand mean)")
print(o[["OP", "PV", "OP_sd", "PV_sd", "PVdev_acf1", "ZCR_per100", "PV_hold_frac"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
