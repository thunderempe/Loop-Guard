"""
Loop Guard - Diagnostics Module
================================
Phase 2 diagnostic functions for PID control loop fault detection.

Diagnostic #1 — Setpoint Tracking:
    compute_setpoint_tracking()
        Computes four signals that characterise how well (and in what way)
        a loop's PV tracks its SP over the run:
            - mean_error   : mean signed bias of (PV - SP)
            - error_slope  : OLS linear drift rate of error vs. time
            - rmse_error   : root-mean-square tracking error (general severity)
            - r2_linear    : R2 of the linear fit to error(t); ~1.0 means a
                             clean monotonic drift, ~0 means noise/oscillation

Diagnostic #2 — Oscillation Detection:
    compute_oscillation_detection()
        Four ACF-based signals that characterise oscillatory behaviour:
            - osc_amplitude      : std of the detrended error signal
            - acf_osc_score      : |min ACF value at lags 1..N//4|, in [0,1]
            - zero_crossing_rate : sign-change rate of detrended error per 100 steps
            - dominant_period    : estimated oscillation period (2 x lag of ACF minimum)
Diagnostic #3 — Stiction Detection:
    compute_stiction_detection()
        Two-stage stiction detector using the joint OP-PV relationship.
        Stage 1 — Oscillation gate: checks zero_crossing_rate and acf_osc_score
            (D#2 signals) to confirm the loop is oscillating and is not noise.
        Stage 2 — Stiction test: when oscillation is present, uses op_pv_corr
            (Pearson correlation of |delta-OP| and |delta-PV|) to distinguish
            stiction (near-zero corr) from linear oscillation (high corr).
        Also returns op_curvature and op_delta_cv as informational signals.
Diagnostic #4 — Harris Performance Index:
    compute_harris_index()
        Estimates the minimum variance control benchmark (Harris 1989)
        from routine regulatory closed-loop data. Demeans contiguous
        steady-state segments separately without concatenation, selects AR order
        by AIC (up to max_order), expands impulse response up to the process delay,
        and computes the ratio of minimum achievable variance to actual error variance.
"""

from typing import Union
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_arrays(loop_data):
    """
    Unpack SP, PV, OP, and time arrays from any supported input type.

    Accepted inputs
    ---------------
    SimulationResult  : accesses .sp, .pv, .op, .time directly.
    pandas DataFrame  : expects columns SP, PV, OP, and optionally timestep.
    dict              : expects keys 'SP'/'sp', 'PV'/'pv', 'OP'/'op',
                        and optionally 'time'/'t'.

    Returns
    -------
    (sp, pv, op, time) as float64 numpy arrays, all length N.
    """
    # Import here to avoid circular dependency if diagnostics is imported
    # before simulator (also lets the module work standalone with plain
    # dicts/DataFrames without requiring the simulator package).
    try:
        from src.simulator import SimulationResult
        if isinstance(loop_data, SimulationResult):
            return (
                np.asarray(loop_data.sp,   dtype=float),
                np.asarray(loop_data.pv,   dtype=float),
                np.asarray(loop_data.op,   dtype=float),
                np.asarray(loop_data.time, dtype=float),
            )
    except ImportError:
        pass

    if isinstance(loop_data, pd.DataFrame):
        sp = loop_data["SP"].to_numpy(dtype=float)
        pv = loop_data["PV"].to_numpy(dtype=float)
        op = loop_data["OP"].to_numpy(dtype=float)
        t  = (
            loop_data["timestep"].to_numpy(dtype=float)
            if "timestep" in loop_data.columns
            else np.arange(len(sp), dtype=float)
        )
        return sp, pv, op, t

    if isinstance(loop_data, dict):
        sp = np.asarray(loop_data.get("SP", loop_data.get("sp")), dtype=float)
        pv = np.asarray(loop_data.get("PV", loop_data.get("pv")), dtype=float)
        op = np.asarray(loop_data.get("OP", loop_data.get("op")), dtype=float)
        t_raw = loop_data.get("time", loop_data.get("t", loop_data.get("timestep")))
        t = np.asarray(t_raw, dtype=float) if t_raw is not None else np.arange(len(sp), dtype=float)
        return sp, pv, op, t

    raise TypeError(
        f"Unsupported loop_data type: {type(loop_data)}. "
        "Expected SimulationResult, dict, or pandas DataFrame."
    )


def _ols_slope_and_r2(x, y):
    """
    Fit a simple OLS line  y = a + b*x  and return (slope b, R2).

    Uses the closed-form solution — avoids scipy/statsmodels dependency
    for this hot path.

    Parameters
    ----------
    x : 1-D float array of length N  — independent variable (time indices)
    y : 1-D float array of length N  — dependent variable (error signal)

    Returns
    -------
    slope : float  — OLS slope coefficient b
    r2    : float  — coefficient of determination in [0, 1].
        Near 1.0 → linear model explains error well (monotonic drift).
        Near 0.0 → error has no linear trend (oscillation / noise / good tracking).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0

    # De-mean for numerical stability
    x_mean = x.mean()
    y_mean = y.mean()
    x_c = x - x_mean
    y_c = y - y_mean

    ss_xx = float(np.dot(x_c, x_c))
    if ss_xx == 0.0:
        # All x values identical — slope undefined, R2 = 0
        return 0.0, 0.0

    slope = float(np.dot(x_c, y_c)) / ss_xx

    # R2 = 1 - SS_res / SS_tot
    y_pred_c = slope * x_c           # Predicted deviations from mean
    ss_res = float(np.dot(y_c - y_pred_c, y_c - y_pred_c))
    ss_tot = float(np.dot(y_c, y_c))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    # Clamp to [0, 1]: negative R2 means the linear model is worse than
    # the horizontal mean — we treat that as "no trend" (= 0).
    r2 = float(np.clip(r2, 0.0, 1.0))

    return slope, r2


# ---------------------------------------------------------------------------
# Diagnostic #1: Setpoint Tracking
# ---------------------------------------------------------------------------

def compute_setpoint_tracking(loop_data, settle_steps=0):
    """
    Compute four setpoint-tracking metrics for a single PID control loop run.

    The four signals are designed to be *orthogonal* diagnostic indicators
    so that different fault types can be fingerprinted rather than just
    ranked by a single severity score:

    +--------------+----------------------------------------------------------+
    | Signal       | What it measures                                         |
    +--------------+----------------------------------------------------------+
    | mean_error   | Mean signed bias of (PV - SP).                           |
    |              | Healthy ~= 0; sensor_drift -> large non-zero value.      |
    |              | Oscillatory faults ~= 0 (symmetric around SP).          |
    +--------------+----------------------------------------------------------+
    | error_slope  | OLS slope of error(t) vs. time (units: PV-units/step).  |
    |              | Healthy ~= 0; sensor_drift -> slope ~= drift_rate.       |
    |              | Oscillation/stiction ~= 0 (no monotonic trend).         |
    +--------------+----------------------------------------------------------+
    | rmse_error   | sqrt(mean((PV - SP)^2)).                                 |
    |              | General tracking severity — elevated for ALL fault types |
    |              | to varying degrees. Not a solo discriminator but useful  |
    |              | for ranking overall loop health.                         |
    +--------------+----------------------------------------------------------+
    | r2_linear    | R2 of the linear fit to error(t), in [0, 1].            |
    |              | ~1.0 means error grows linearly (clean monotonic drift). |
    |              | ~0.0 means error is noisy/oscillatory (no trend).       |
    |              | Best single "driftiness" score — normalised and bounded. |
    +--------------+----------------------------------------------------------+

    Why four signals instead of one?
    ---------------------------------
    A single RMSE score cannot separate sensor_drift from oscillatory_tuning
    — both produce high RMSE but for geometrically different reasons:

    * sensor_drift    : monotonically trending error
                        -> high |mean_error|, high |error_slope|, high r2_linear
    * oscillatory     : zero-mean cyclic error
                        -> mean_error ~= 0, error_slope ~= 0, r2_linear ~= 0,
                           high rmse_error
    * valve_stiction  : small asymmetric limit cycle
                        -> moderate rmse, mean_error ~= 0, r2_linear ~= 0
    * noise           : random zero-mean jitter
                        -> everything ~= 0 except slightly elevated rmse_error
    * healthy         : all four signals near zero

    Parameters
    ----------
    loop_data : SimulationResult | dict | DataFrame
        Single loop run exposing SP and PV signals.
        See _extract_arrays() for accepted input formats.
    settle_steps : int, default=0
        Number of initial timesteps to skip. Useful if the process has a
        transient startup period that would inflate early-sample statistics.
        Suggested: 0 (default) for our labeled dataset since initial_pv == SP.

    Returns
    -------
    dict with keys:
        "mean_error"  (float) -- mean(PV - SP), signed, same units as PV
        "error_slope" (float) -- OLS slope of (PV - SP) vs. timestep
        "rmse_error"  (float) -- RMSE of tracking error, always >= 0
        "r2_linear"   (float) -- R2 of linear fit to error(t), in [0, 1]
    """
    sp, pv, op, t = _extract_arrays(loop_data)

    n = len(sp)
    if n == 0:
        raise ValueError("loop_data contains zero samples.")

    # Apply settle-time trimming — skip early transient if requested
    start = min(int(settle_steps), n - 1)
    sp_w  = sp[start:]
    pv_w  = pv[start:]
    t_w   = t[start:]

    # Core error signal: positive when PV > SP (above setpoint)
    error = pv_w - sp_w

    # --- Signal 1: Mean signed error (bias / offset) -----------------------
    # A healthy or oscillatory loop will have mean ~= 0.
    # A drifting sensor biases the error over the window.
    mean_error = float(np.mean(error))

    # --- Signals 2 & 4: OLS slope + R2 on the full error window -----------
    # t_w is the x-axis so slope has units of (PV units) per timestep —
    # directly interpretable as drift rate.
    # R2 tells us how well the linear model explains the error variance:
    # a clean drift -> R2 near 1; noisy/oscillatory error -> R2 near 0.
    error_slope, r2_linear = _ols_slope_and_r2(t_w, error)

    # --- Signal 3: RMSE tracking error -------------------------------------
    # Captures overall magnitude of tracking error regardless of sign/shape.
    rmse_error = float(np.sqrt(np.mean(error ** 2)))

    return {
        "mean_error":  mean_error,
        "error_slope": error_slope,
        "rmse_error":  rmse_error,
        "r2_linear":   r2_linear,
    }


def compute_setpoint_tracking_batch(df, loop_id_col="loop_id", settle_steps=0):
    """
    Run compute_setpoint_tracking() over every loop in a labeled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (e.g. labeled_loops.csv).
        Must have columns: loop_id, SP, PV, OP, timestep, fault_type.
    loop_id_col : str, default='loop_id'
        Column name for the loop identifier.
    settle_steps : int, default=0
        Passed through to compute_setpoint_tracking().

    Returns
    -------
    pd.DataFrame
        One row per loop with columns:
        loop_id, fault_type, mean_error, error_slope, rmse_error, r2_linear
    """
    records = []
    for loop_id, group in df.groupby(loop_id_col, sort=True):
        fault_type = group["fault_type"].iloc[0]
        metrics    = compute_setpoint_tracking(group, settle_steps=settle_steps)
        records.append({
            loop_id_col:  loop_id,
            "fault_type": fault_type,
            **metrics,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Diagnostic #2: Oscillation Detection
# ---------------------------------------------------------------------------

def _detrend_error(error, t):
    """
    Remove the OLS linear trend from the error signal.

    Subtracting the best-fit line eliminates the monotonic slope introduced
    by sensor_drift, so that the oscillation metrics are not contaminated by
    a slowly drifting baseline.

    Parameters
    ----------
    error : 1-D float array  — raw (PV - SP) signal
    t     : 1-D float array  — time axis (same length as error)

    Returns
    -------
    error_dt : 1-D float array  — residuals after OLS line removal
    """
    slope, _ = _ols_slope_and_r2(t, error)
    intercept = error.mean() - slope * t.mean()
    return error - (intercept + slope * t)


def _autocorrelation(x, max_lag=None):
    """
    Compute the normalised autocorrelation function (ACF) of signal x.

    Uses the unbiased estimator with the signal de-meaned before computation.
    ACF[0] = 1 by definition; values at positive lags are in [-1, 1].

    Parameters
    ----------
    x       : 1-D float array
    max_lag : int, optional
        If provided, only lags 0..max_lag are returned.
        Defaults to len(x) - 1.

    Returns
    -------
    acf : 1-D float array of length min(max_lag+1, len(x))
        acf[k] = normalised correlation of x with x shifted by k steps.
    """
    n = len(x)
    if n < 2:
        return np.array([1.0])

    x_dm  = x - x.mean()           # de-mean
    var   = float(np.dot(x_dm, x_dm))
    if var == 0.0:
        # Constant signal — ACF is 1 at lag 0, undefined elsewhere; return zeros
        ml = min(max_lag + 1, n) if max_lag is not None else n
        out = np.zeros(ml)
        out[0] = 1.0
        return out

    if max_lag is None:
        max_lag = n - 1

    # Full linear correlation via np.correlate is O(N^2) but clean for N=500
    full = np.correlate(x_dm, x_dm, mode='full')   # length 2N-1
    # full[N-1+k] is the correlation at lag k
    mid = n - 1
    acf = full[mid : mid + max_lag + 1] / var       # normalise so acf[0]=1
    return acf


def compute_oscillation_detection(loop_data, settle_steps=0,
                                   window_size=125, window_stride=50):
    """
    Compute four ACF-based oscillation metrics for a single PID loop run.

    The four signals jointly separate oscillatory_tuning, valve_stiction,
    noise, healthy, and sensor_drift loops:

    +---------------------+----------------------------------------------------+
    | Signal              | What it measures                                   |
    +---------------------+----------------------------------------------------+
    | osc_amplitude       | std of the detrended (PV - SP) error.              |
    |                     | Full-series statistic. Captures overall oscillation|
    |                     | magnitude in PV units.                             |
    +---------------------+----------------------------------------------------+
    | acf_osc_score       | MAX over sliding windows of |min ACF value| at     |
    |                     | lags 1..window_size//4,  in [0, 1].                |
    |                     | Computed per window, then the maximum is taken.    |
    |                     | This correctly scores *episodic* oscillators (loops|
    |                     | where OP saturation creates flat dead segments     |
    |                     | between oscillation bursts) by isolating the most  |
    |                     | active window rather than averaging in the silence.|
    +---------------------+----------------------------------------------------+
    | zero_crossing_rate  | Sign-change count per 100 steps of detrended error.|
    |                     | Full-series statistic. Noise: VERY HIGH (~40-48).  |
    |                     | Oscillatory/stiction: MODERATE (~4-15).            |
    |                     | Healthy/drift: LOW (<3).                           |
    +---------------------+----------------------------------------------------+
    | dominant_period     | Estimated oscillation period (steps) from the      |
    |                     | winning window (the one with max acf_osc_score).   |
    |                     | = 2 x lag of ACF minimum within that window.       |
    |                     | 0.0 if no negative ACF dip found anywhere.         |
    +---------------------+----------------------------------------------------+

    Why windowed ACF for acf_osc_score?
    -------------------------------------
    Aggressive tuning with output saturation (OP hitting 0% or 100%) creates
    episodic oscillation: the loop bursts into oscillation after each SP step,
    then the integrator winds up against the saturation wall and the loop goes
    flat until the next step. A full-series ACF averages the oscillatory burst
    with the flat saturated segments, severely diluting the negative ACF dip.
    A 125-step window is long enough to capture ~5-9 full oscillation cycles
    at the fastest expected period (~14 steps) and short enough to isolate one
    burst. Taking the maximum across windows recovers the true oscillation
    score of the active segments.

    osc_amplitude and zero_crossing_rate remain full-series statistics because
    they are not distorted by the averaging effect in the same way.

    Window size note
    ----------------
    max_lag within each window = window_size // 4 = 31 steps for the default
    window of 125. This covers oscillation periods up to 62 steps, which spans
    the full range of oscillatory_tuning periods (14-24 steps) while naturally
    being insensitive to stiction limit cycles (period > 130 steps) — those
    cannot complete even half a cycle within a 125-step window, so their
    windowed ACF stays near zero, maintaining the separation from oscillatory.

    Parameters
    ----------
    loop_data : SimulationResult | dict | DataFrame
        Single loop run. Must expose SP and PV signals.
        See _extract_arrays() for accepted formats.
    settle_steps : int, default=0
        Number of initial timesteps to skip before computing metrics.
    window_size : int, default=125
        Length of each ACF sub-window in steps.
        Must be at least 10. Default captures ~5-9 oscillation cycles at
        the fastest expected period.
    window_stride : int, default=50
        Step size between consecutive window start positions.
        Smaller stride = more overlap = more windows evaluated = more
        compute, but finer coverage of burst-onset timing.

    Returns
    -------
    dict with keys:
        "osc_amplitude"      (float) -- std of detrended error, in PV units
        "acf_osc_score"      (float) -- max windowed |min ACF|, in [0, 1]
        "zero_crossing_rate" (float) -- sign changes per 100 steps (full series)
        "dominant_period"    (float) -- estimated oscillation period (steps);
                                        from the highest-scoring window;
                                        0.0 if no negative ACF dip detected
    """
    sp, pv, op, t = _extract_arrays(loop_data)

    n = len(sp)
    if n == 0:
        raise ValueError("loop_data contains zero samples.")

    # Apply settle-time trimming
    start = min(int(settle_steps), n - 1)
    sp_w  = sp[start:]
    pv_w  = pv[start:]
    t_w   = t[start:]
    n_w   = len(sp_w)

    # Raw error (positive when PV is above SP)
    error = pv_w - sp_w

    # --- Detrend: remove OLS linear slope from error -----------------------
    # This is critical: sensor_drift loops have a large linear trend that
    # would inflate osc_amplitude and corrupt the ACF into looking like a
    # very slowly oscillating signal. Subtracting the best-fit line makes
    # all oscillation metrics comparable across fault types.
    error_dt = _detrend_error(error, t_w)

    # --- Signal 1: Oscillation amplitude (std of detrended error) ----------
    # std is preferred over peak-to-peak because it is robust to single
    # outlier samples and directly related to energy in the signal.
    osc_amplitude = float(np.std(error_dt, ddof=1)) if n_w > 1 else 0.0

    # --- Signals 2 & 4: Windowed ACF — max acf_osc_score + dominant period --
    # Problem being solved: oscillatory loops with OP saturation produce
    # *episodic* oscillation (burst → flat → burst). A full-series ACF averages
    # the oscillatory bursts with the flat dead segments, heavily diluting the
    # negative ACF dip and making the loop look barely different from healthy.
    #
    # Fix: slide a window across the detrended error, compute the ACF-based
    # score within each window, and take the MAXIMUM across all windows.
    # This isolates the most active burst and reports its true oscillation
    # strength, regardless of how much flat/silent time surrounds it.
    #
    # Window parameters:
    #   window_size  = 125 steps → max_lag = 31; covers periods up to 62 steps
    #   window_stride = 50 steps  → ~8 windows over a 500-step series
    # osc_amplitude and ZCR remain full-series stats (not affected by episodic
    # averaging in the same way — std is already inflated by bursts, ZCR is
    # dominated by noise which is always high-frequency).

    win_size   = max(10, int(window_size))
    win_stride = max(1, int(window_stride))
    # max_lag per window: window_size // 4 covers oscillation periods up to
    # 2 * (window_size // 4) steps. Default: lags 1..31 → periods ≤ 62 steps.
    # This is deliberately insensitive to stiction (period > 130 steps).
    max_lag_w  = max(1, win_size // 4)

    best_score  = 0.0          # best (max) acf_osc_score found so far
    best_period = 0.0          # dominant_period from the winning window

    # Slide window across the full detrended error series
    win_starts = range(0, max(1, n_w - win_size + 1), win_stride)
    for ws in win_starts:
        chunk = error_dt[ws : ws + win_size]
        if len(chunk) < 4:          # too short to compute a meaningful ACF
            continue

        acf_w    = _autocorrelation(chunk, max_lag=min(max_lag_w, len(chunk) - 1))
        acf_lags = acf_w[1:]        # exclude lag-0 (always 1.0)

        if len(acf_lags) == 0:
            continue

        min_idx = int(np.argmin(acf_lags))
        min_val = float(acf_lags[min_idx])
        score   = float(np.clip(-min_val, 0.0, 1.0))

        if score > best_score:
            best_score  = score
            acf_lag     = min_idx + 1      # 1-based lag of the ACF minimum
            # Only record a period if the dip is meaningfully negative (> 0.05);
            # shallow dips near 0 are just finite-sample noise.
            best_period = float(2 * acf_lag) if min_val < -0.05 else 0.0

    acf_osc_score   = best_score
    dominant_period = best_period

    # --- Signal 3: Zero-Crossing Rate of detrended error -------------------
    # Count how many times the sign of error_dt flips between consecutive
    # samples, then scale to "per 100 steps" for readability.
    # np.sign returns 0 for exactly-zero samples; we treat 0 as non-crossing.
    signs = np.sign(error_dt)
    # Replace zeros with the last non-zero sign (forward-fill) so a sample
    # that lands exactly on zero doesn't contribute spurious crossings.
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    # A crossing occurs where consecutive signs differ
    crossings = int(np.sum(np.diff(signs) != 0))
    zero_crossing_rate = float(crossings / max(n_w - 1, 1) * 100)

    return {
        "osc_amplitude":      osc_amplitude,
        "acf_osc_score":      acf_osc_score,
        "zero_crossing_rate": zero_crossing_rate,
        "dominant_period":    dominant_period,
    }


def compute_oscillation_detection_batch(df, loop_id_col="loop_id", settle_steps=0):
    """
    Run compute_oscillation_detection() over every loop in a labeled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (e.g. labeled_loops.csv).
        Must have columns: loop_id, SP, PV, OP, timestep, fault_type.
    loop_id_col : str, default='loop_id'
        Column name for the loop identifier.
    settle_steps : int, default=0
        Passed through to compute_oscillation_detection().

    Returns
    -------
    pd.DataFrame
        One row per loop with columns:
        loop_id, fault_type, osc_amplitude, acf_osc_score,
        zero_crossing_rate, dominant_period
    """
    records = []
    for loop_id, group in df.groupby(loop_id_col, sort=True):
        fault_type = group["fault_type"].iloc[0]
        metrics    = compute_oscillation_detection(group, settle_steps=settle_steps)
        records.append({
            loop_id_col:  loop_id,
            "fault_type": fault_type,
            **metrics,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Diagnostic #3 — Valve Stiction Detection
# ---------------------------------------------------------------------------

def _op_pv_corr(op, pv):
    """Pearson correlation of |delta-OP| and |delta-PV|.

    Measures how tightly OP and PV movements are coupled step-by-step.
    In a linear loop (oscillatory_tuning) the process converts OP changes to
    proportional PV changes (with a short lag), so the two magnitude signals
    are strongly correlated.  In a stiction loop the valve is stuck during part
    of each limit-cycle: OP ramps while PV is frozen, then PV jumps while OP
    reverses -- these two phases are anti-correlated, driving the overall
    Pearson correlation toward zero.

    Returns float in [-1, 1], or np.nan when either signal has no variance.
    """
    dop = np.abs(np.diff(op))
    dpv = np.abs(np.diff(pv))
    if dop.std() < 1e-9 or dpv.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(dop, dpv)[0, 1])


def _op_curvature(op):
    """Mean second-difference magnitude of OP, normalised by std of first difference.

    A triangle-wave OP (stiction: constant-slope ramps) has near-zero curvature
    everywhere except at the reversal points, giving a low value.  A sinusoidal
    OP (oscillatory_tuning) has continuously varying curvature and therefore a
    higher value.  Pure noise gives very high values.

    Returns float >= 0, or np.nan when OP has no variance.
    """
    dop     = np.diff(op)
    d2op    = np.abs(np.diff(dop))     # equivalent to |diff(op, n=2)|
    dop_std = dop.std()
    if dop_std < 1e-9:
        return np.nan
    return float(d2op.mean() / dop_std)


def _op_delta_cv(op):
    """Coefficient of variation (std / mean) of |delta-OP|.

    A pure triangle wave has constant-magnitude slope -> very low CV.
    A sinusoid has slope that varies from 0 (at peaks) to maximum (at
    zero-crossings) -> higher CV.  Noise produces very low CV (all steps
    similar magnitude).

    Returns float >= 0, or np.nan when mean |delta-OP| ~ 0.
    """
    dop = np.abs(np.diff(op))
    mu  = dop.mean()
    if mu < 1e-9:
        return np.nan
    return float(dop.std() / mu)


def compute_stiction_detection(loop_data, settle_steps=0,
                               zcr_primary=1.3, zcr_acf_floor=0.9,
                               acf_gate=0.34, zcr_noise_ceil=30.0,
                               corr_threshold=0.25):
    """Detect valve stiction in a PID loop using OP-PV movement coupling.

    Two-stage logic
    ---------------
    Stage 1 -- Oscillation gate (is the loop in an oscillatory state at all?)
        Reuses zero_crossing_rate and acf_osc_score from Diagnostic #2
        (computed internally by calling compute_oscillation_detection).

        gate_pass = (
            zero_crossing_rate > zcr_primary
            OR
            (zero_crossing_rate > zcr_acf_floor AND acf_osc_score > acf_gate)
        ) AND zero_crossing_rate < zcr_noise_ceil

        Threshold choices calibrated on the 60-loop labelled dataset:
            zcr_primary   = 1.3   healthy max ZCR = 1.002, sensor_drift max = 1.202.
                                   Strict > 1.3 excludes both; stiction_v1 next
                                   lowest ZCR = 1.403 passes.
            zcr_acf_floor = 0.9   Blocks sensor_drift loops (ZCR <= 0.802) from
                                   reaching the acf branch.
            acf_gate      = 0.34  sensor_drift max acf_osc_score (at ZCR > 0.9) is
                                   0.332 (loop 17); threshold of 0.34 excludes it.
                                   Catches stiction_v2 loops at ZCR ~ 1.2 (acf >= 0.26).
            zcr_noise_ceil= 30.0  noise min ZCR = 36.9; excluded cleanly.

        Known limitations on this dataset (inherent ambiguity):
            loop 29  (stiction_v1): ZCR=1.202, acf=0.235 -- indistinct from sensor_drift.
            loops 34,35,36,37,39 (stiction_v2): ZCR <= 1.202, acf <= 0.305 -- very
                mild stiction, nearly identical to healthy on all D#2 signals.
            These 6 loops return stiction_flag=False (reported as misses in validation).

    Stage 2 -- Stiction vs. linear oscillation (only when gate_pass is True)
        op_pv_corr (Pearson correlation of |delta-OP| and |delta-PV|) measures
        whether OP and PV changes occur together (linear loop) or independently
        (stuck valve).
            stiction        all gate-passing loops: op_pv_corr in [-0.03, 0.16]
            oscillatory_tuning all loops:           op_pv_corr in [0.41, 0.84]
        Clean gap of 0.25 units with zero overlap on the 60-loop dataset.

        stiction_flag = gate_pass AND op_pv_corr < corr_threshold
        corr_threshold = 0.25  (midpoint of the 0.16 .. 0.41 gap).

    Parameters
    ----------
    loop_data : SimulationResult | dict | DataFrame
        Single loop run.  Must expose SP, PV, and OP signals.
        See _extract_arrays() for accepted formats.
    settle_steps : int, default=0
        Number of initial timesteps to skip before computing metrics.
    zcr_primary : float, default=1.3
        Primary ZCR threshold for the oscillation gate.
    zcr_acf_floor : float, default=0.9
        Minimum ZCR required before the acf fallback branch activates.
    acf_gate : float, default=0.34
        Minimum acf_osc_score for the acf fallback branch.
    zcr_noise_ceil : float, default=30.0
        ZCR above this value is classified as noise; gate fails.
    corr_threshold : float, default=0.25
        Loops that pass the gate and have op_pv_corr below this threshold
        are flagged as stiction.

    Returns
    -------
    dict with keys:
        "op_pv_corr"    (float) -- corr(|dOP|, |dPV|); primary stiction signal
        "op_curvature"  (float) -- mean|d2OP|/std(dOP); informational only
        "op_delta_cv"   (float) -- std(|dOP|)/mean(|dOP|); informational only
        "stiction_flag" (bool)  -- True when stiction is detected
    """
    sp, pv, op, t = _extract_arrays(loop_data)

    if settle_steps > 0:
        sp = sp[settle_steps:]
        pv = pv[settle_steps:]
        op = op[settle_steps:]
        t  = t[settle_steps:]

    # -- Stage 1: oscillation gate (reusing D#2 signals) ---------------------
    osc = compute_oscillation_detection(
        {"SP": sp, "PV": pv, "OP": op, "timestep": t},
        settle_steps=0,   # already trimmed above
    )
    zcr = osc["zero_crossing_rate"]
    acf = osc["acf_osc_score"]

    gate_pass = (
        (zcr > zcr_primary)
        or (zcr > zcr_acf_floor and acf > acf_gate)
    ) and (zcr < zcr_noise_ceil)

    # -- Stage 2: stiction vs. linear oscillation ----------------------------
    corr = _op_pv_corr(op, pv)
    curv = _op_curvature(op)
    cv   = _op_delta_cv(op)

    stiction_flag = bool(
        gate_pass
        and (not np.isnan(corr))
        and corr < corr_threshold
    )

    return {
        "op_pv_corr":    float(corr) if not np.isnan(corr) else np.nan,
        "op_curvature":  float(curv) if not np.isnan(curv) else np.nan,
        "op_delta_cv":   float(cv)   if not np.isnan(cv)   else np.nan,
        "stiction_flag": stiction_flag,
    }


def compute_stiction_detection_batch(df, loop_id_col="loop_id", settle_steps=0):
    """Run compute_stiction_detection() over every loop in a labelled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: loop_id_col, 'SP', 'PV', 'OP', 'timestep'.
        Optionally 'fault_type' (passed through unchanged if present).
    loop_id_col : str, default='loop_id'
    settle_steps : int, default=0

    Returns
    -------
    pd.DataFrame with one row per loop and columns:
        loop_id, fault_type (if present), op_pv_corr, op_curvature,
        op_delta_cv, stiction_flag.
    """
    records       = []
    has_fault_col = "fault_type" in df.columns
    for loop_id, group in df.groupby(loop_id_col, sort=True):
        metrics = compute_stiction_detection(group, settle_steps=settle_steps)
        row     = {loop_id_col: loop_id, **metrics}
        if has_fault_col:
            row["fault_type"] = group["fault_type"].iloc[0]
        records.append(row)

    out  = pd.DataFrame(records)
    cols = [loop_id_col]
    if has_fault_col:
        cols.append("fault_type")
    cols += ["op_pv_corr", "op_curvature", "op_delta_cv", "stiction_flag"]
    return out[cols]


# ---------------------------------------------------------------------------
# Diagnostic #3 (v2) — Valve Stiction Detection v2 (Decoupled Robust Pipeline)
# ---------------------------------------------------------------------------

def compute_stiction_detection_v2(
    loop_data,
    settle_steps=0,
    r2_max=0.40,
    zcr_noise_ceil=30.0,
    acf_min=0.22,
    period_min=35.0,
    corr_filt_max=0.50,
    ma_window=5,
):
    """Detect valve stiction using decoupled multi-stage diagnostics and filtered OP-PV correlation.

    Stage 1 -- Oscillation & Fault Pre-gates:
        1. drift_excluded: r2_linear < r2_max (D#1 check; rejects sensor drift).
        2. noise_excluded: zero_crossing_rate < zcr_noise_ceil (D#2 check; rejects measurement noise).
        3. oscillating: acf_osc_score >= acf_min AND dominant_period >= period_min
           (D#2 check; rejects non-oscillatory loops and short-period oscillatory tuning).

    Stage 2 -- Stiction vs. Linear Oscillation:
        corr_ok: op_pv_corr_filt < corr_filt_max
        Computes Pearson correlation of |delta-OP| and |delta-PV| smoothed by a
        moving average filter (ma_window). Filtering removes high-frequency jitter
        and restores the stiction vs. oscillatory tuning separation gap in noisy regimes.

    Final Flag:
        stiction_flag = drift_excluded AND noise_excluded AND oscillating AND corr_ok

    Parameters
    ----------
    loop_data : SimulationResult | dict | DataFrame
        Single loop run containing SP, PV, and OP.
    settle_steps : int, default=0
        Number of initial timesteps to trim.
    r2_max : float, default=0.40
        Ceiling on linear fit R2 from D#1 to exclude sensor drift.
    zcr_noise_ceil : float, default=30.0
        Ceiling on zero-crossing rate from D#2 to exclude high-frequency noise.
    acf_min : float, default=0.22
        Minimum ACF oscillation score from D#2 to require meaningful periodic energy.
    period_min : float, default=35.0
        Minimum dominant period from D#2 to exclude fast oscillatory tuning (periods 14-26).
    corr_filt_max : float, default=0.50
        Maximum filtered OP-PV correlation to declare stiction.
    ma_window : int, default=5
        Moving average window size for low-pass filtering OP and PV increments.

    Returns
    -------
    dict with boolean gate conditions, raw metric signals, and stiction_flag.
    """
    sp, pv, op, t = _extract_arrays(loop_data)

    if settle_steps > 0:
        sp = sp[settle_steps:]
        pv = pv[settle_steps:]
        op = op[settle_steps:]
        t  = t[settle_steps:]

    loop_dict = {"SP": sp, "PV": pv, "OP": op, "timestep": t}

    # 1. D#1 Tracking (r2_linear)
    trk = compute_setpoint_tracking(loop_dict, settle_steps=0)
    r2_linear = trk["r2_linear"]

    # 2. D#2 Oscillation (zcr, acf_osc_score, dominant_period)
    osc = compute_oscillation_detection(loop_dict, settle_steps=0)
    zcr = osc["zero_crossing_rate"]
    acf = osc["acf_osc_score"]
    period = osc["dominant_period"]

    # 3. Filtered OP-PV correlation (MA-5)
    ma_w = max(1, int(ma_window))
    op_s = pd.Series(op).rolling(ma_w, min_periods=1).mean().values
    pv_s = pd.Series(pv).rolling(ma_w, min_periods=1).mean().values
    dop = np.abs(np.diff(op_s))
    dpv = np.abs(np.diff(pv_s))
    if dop.std() > 1e-9 and dpv.std() > 1e-9:
        corr_filt = float(np.corrcoef(dop, dpv)[0, 1])
    else:
        corr_filt = np.nan

    # Raw unfiltered correlation
    dop_raw = np.abs(np.diff(op))
    dpv_raw = np.abs(np.diff(pv))
    if dop_raw.std() > 1e-9 and dpv_raw.std() > 1e-9:
        corr_raw = float(np.corrcoef(dop_raw, dpv_raw)[0, 1])
    else:
        corr_raw = np.nan

    # Gate logic
    drift_excluded = bool(r2_linear < r2_max)
    noise_excluded = bool(zcr < zcr_noise_ceil)
    oscillating    = bool(acf >= acf_min and period >= period_min)
    corr_ok        = bool(not np.isnan(corr_filt) and corr_filt < corr_filt_max)

    stiction_flag = bool(drift_excluded and noise_excluded and oscillating and corr_ok)

    return {
        "stiction_flag":      stiction_flag,
        "drift_excluded":     drift_excluded,
        "noise_excluded":     noise_excluded,
        "oscillating":        oscillating,
        "corr_ok":            corr_ok,
        "r2_linear":          float(r2_linear),
        "zero_crossing_rate": float(zcr),
        "acf_osc_score":      float(acf),
        "dominant_period":    float(period),
        "op_pv_corr_filt":    float(corr_filt) if not np.isnan(corr_filt) else np.nan,
        "op_pv_corr_raw":     float(corr_raw) if not np.isnan(corr_raw) else np.nan,
    }


def compute_stiction_detection_v2_batch(
    df,
    loop_id_col="loop_id",
    settle_steps=0,
    r2_max=0.40,
    zcr_noise_ceil=30.0,
    acf_min=0.22,
    period_min=35.0,
    corr_filt_max=0.50,
    ma_window=5,
):
    """Run compute_stiction_detection_v2() over every loop in a labelled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: loop_id_col, 'SP', 'PV', 'OP', 'timestep'.
        Optionally 'fault_type' (passed through unchanged if present).
    loop_id_col : str, default='loop_id'
    settle_steps : int, default=0

    Returns
    -------
    pd.DataFrame with one row per loop and gate columns + raw signals.
    """
    records       = []
    has_fault_col = "fault_type" in df.columns
    for loop_id, group in df.groupby(loop_id_col, sort=True):
        metrics = compute_stiction_detection_v2(
            group,
            settle_steps=settle_steps,
            r2_max=r2_max,
            zcr_noise_ceil=zcr_noise_ceil,
            acf_min=acf_min,
            period_min=period_min,
            corr_filt_max=corr_filt_max,
            ma_window=ma_window,
        )
        row = {loop_id_col: loop_id, **metrics}
        if has_fault_col:
            row["fault_type"] = group["fault_type"].iloc[0]
        records.append(row)

    out = pd.DataFrame(records)
    cols = [loop_id_col]
    if has_fault_col:
        cols.append("fault_type")
    cols += [
        "stiction_flag",
        "drift_excluded",
        "noise_excluded",
        "oscillating",
        "corr_ok",
        "r2_linear",
        "zero_crossing_rate",
        "acf_osc_score",
        "dominant_period",
        "op_pv_corr_filt",
        "op_pv_corr_raw",
    ]
    return out[cols]


# ---------------------------------------------------------------------------
# Diagnostic #4: Harris Performance Index
# ---------------------------------------------------------------------------

def _extract_unmasked_segments(sp, settle_startup=30, settle_sp=65, sp_change_thresh=0.01):
    """
    Detect setpoint change points and return slices of regulatory steady state.

    Masks out the initial startup transient (0..settle_startup) and a settling
    window (idx..idx+settle_sp) after each detected step change in SP.
    Returns contiguous (start_idx, end_idx) index ranges.
    """
    N = len(sp)
    mask = np.ones(N, dtype=bool)
    mask[:settle_startup] = False

    dsp = np.abs(np.diff(sp, prepend=sp[0]))
    change_idx = np.where(dsp > sp_change_thresh)[0]
    for idx in change_idx:
        mask[idx : min(N, idx + settle_sp)] = False

    segments = []
    in_seg = False
    start = 0
    for i in range(N):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif not mask[i] and in_seg:
            in_seg = False
            if i - start > 0:
                segments.append((start, i))
    if in_seg and N - start > 0:
        segments.append((start, N))

    return segments


def compute_harris_index(loop_data, delay=3, settle_startup=30, settle_sp=65,
                         sp_change_thresh=0.01, max_order=30):
    """
    Estimate the Harris Performance Index (Harris 1989) from closed-loop routine data.

    Computes the minimum achievable variance benchmark (MVC) relative to actual
    tracking error variance. Uses separate unmasked regulatory steady-state segments
    (without cross-segment concatenation), demeans each segment separately, chooses
    the AR order via AIC (up to max_order), and calculates the impulse response
    weights up to the loop dead-time horizon `delay`.

    Parameters
    ----------
    loop_data : SimulationResult | dict | DataFrame
        Single loop run containing 'SP', 'PV', 'OP'.
    delay : int, default=3
        Process dead-time horizon in integer sampling intervals.
    settle_startup : int, default=30
        Number of initial startup steps to mask out.
    settle_sp : int, default=65
        Settling window masked after each detected setpoint change.
    sp_change_thresh : float, default=0.01
        Absolute step magnitude on SP to detect setpoint transitions.
    max_order : int, default=30
        Maximum autoregressive order to evaluate during AIC selection.

    Returns
    -------
    dict with keys:
        "harris_index"      (float) : Minimum achievable variance / actual variance, in [0, 1]
        "ar_order"          (int)   : Best AR order chosen by AIC
        "var_total"         (float) : Actual pooled variance of demeaned regulatory error
        "var_mv"            (float) : Minimum achievable variance under MVC
        "var_innovations"   (float) : Estimated white-noise innovation variance sigma_a^2
        "harris_index_p10"  (float) : Harris Index computed with fixed AR order p=10
        "harris_index_p25"  (float) : Harris Index computed with fixed AR order p=25
        "n_samples_used"    (int)   : Number of unmasked samples used in the benchmark
        "trend_r2"          (float) : R2 of linear trend over unmasked time (non-stationarity test)
    """
    sp, pv, op, t = _extract_arrays(loop_data)
    segments = _extract_unmasked_segments(
        sp, settle_startup=settle_startup, settle_sp=settle_sp,
        sp_change_thresh=sp_change_thresh,
    )

    n_samples_used = sum(end - start for start, end in segments)

    # Degenerate condition 1: too few unmasked samples
    if n_samples_used < max_order + 5 or len(segments) == 0:
        return {
            "harris_index": np.nan,
            "ar_order": np.nan,
            "var_total": np.nan,
            "var_mv": np.nan,
            "var_innovations": np.nan,
            "harris_index_p10": np.nan,
            "harris_index_p25": np.nan,
            "n_samples_used": n_samples_used,
            "trend_r2": np.nan,
        }

    # Demean each segment separately
    demeaned_segs = []
    all_demeaned_samples = []
    t_unmasked = []
    for start, end in segments:
        seg_e = pv[start:end] - sp[start:end]
        seg_demeaned = seg_e - np.mean(seg_e)
        demeaned_segs.append(seg_demeaned)
        all_demeaned_samples.extend(seg_demeaned)
        t_unmasked.extend(t[start:end])

    var_total = float(np.var(all_demeaned_samples))

    # Degenerate condition 2: zero error variance
    if var_total < 1e-12:
        return {
            "harris_index": 1.0,
            "ar_order": 0,
            "var_total": var_total,
            "var_mv": 0.0,
            "var_innovations": 0.0,
            "harris_index_p10": 1.0,
            "harris_index_p25": 1.0,
            "n_samples_used": n_samples_used,
            "trend_r2": 0.0,
        }

    # Fit AR(p) by stacking regression rows only within individual segments
    def fit_ar_for_order(p):
        X_list, y_list = [], []
        for x in demeaned_segs:
            L = len(x)
            if L > p:
                X_seg = np.column_stack([x[p - i - 1 : L - i - 1] for i in range(p)])
                y_seg = x[p:]
                X_list.append(X_seg)
                y_list.append(y_seg)
        if not X_list:
            return None, np.nan, 0
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ coef
            var_a = np.var(res)
            return coef, var_a, len(y)
        except (np.linalg.LinAlgError, ValueError):
            return None, np.nan, len(y)

    def calc_ih(coef, var_a, d):
        f = np.zeros(d)
        f[0] = 1.0
        p = len(coef)
        for k in range(1, d):
            f[k] = sum(coef[j - 1] * f[k - j] for j in range(1, min(k, p) + 1))
        var_mv = var_a * float(np.sum(f**2))
        ih = np.clip(var_mv / var_total, 0.0, 1.0)
        return float(ih), float(var_mv), float(var_a)

    # Order selection by AIC
    best_aic = np.inf
    best_p = 1
    best_coef = None
    best_var_a = var_total

    max_p_eval = min(max_order, max(len(x) - 1 for x in demeaned_segs))
    for p in range(1, max_p_eval + 1):
        coef, var_a, M = fit_ar_for_order(p)
        if coef is not None and var_a > 1e-14 and M > p:
            aic = M * np.log(var_a) + 2 * p
            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_coef = coef
                best_var_a = var_a

    # Degenerate condition 3: AR fitting failed
    if best_coef is None:
        return {
            "harris_index": np.nan,
            "ar_order": np.nan,
            "var_total": var_total,
            "var_mv": np.nan,
            "var_innovations": np.nan,
            "harris_index_p10": np.nan,
            "harris_index_p25": np.nan,
            "n_samples_used": n_samples_used,
            "trend_r2": np.nan,
        }

    ih_aic, vmv_aic, va_aic = calc_ih(best_coef, best_var_a, delay)

    # Fixed p=10 and p=25 for comparison
    c10, va10, _ = fit_ar_for_order(10)
    ih_p10, _, _ = calc_ih(c10, va10, delay) if c10 is not None else (np.nan, np.nan, np.nan)

    c25, va25, _ = fit_ar_for_order(25)
    ih_p25, _, _ = calc_ih(c25, va25, delay) if c25 is not None else (np.nan, np.nan, np.nan)

    # Non-stationarity test: linear trend R2 over unmasked time
    t_arr = np.asarray(t_unmasked, dtype=float)
    y_arr = np.asarray(all_demeaned_samples, dtype=float)
    if len(t_arr) > 2 and var_total > 1e-12:
        slope, intercept = np.polyfit(t_arr, y_arr, 1)
        r2_trend = 1.0 - float(np.var(y_arr - (slope * t_arr + intercept))) / var_total
        r2_trend = float(np.clip(r2_trend, 0.0, 1.0))
    else:
        r2_trend = 0.0

    return {
        "harris_index": ih_aic,
        "ar_order": int(best_p),
        "var_total": var_total,
        "var_mv": vmv_aic,
        "var_innovations": va_aic,
        "harris_index_p10": ih_p10,
        "harris_index_p25": ih_p25,
        "n_samples_used": int(n_samples_used),
        "trend_r2": r2_trend,
    }


def compute_harris_index_batch(df, loop_id_col="loop_id", delay=3, settle_startup=30,
                               settle_sp=65, sp_change_thresh=0.01, max_order=30):
    """
    Run compute_harris_index() over every loop in a labelled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: loop_id_col, 'SP', 'PV', 'OP', 'timestep'.
        Optionally 'fault_type' (passed through unchanged if present).
    loop_id_col : str, default='loop_id'
    delay : int, default=3
    settle_startup : int, default=30
    settle_sp : int, default=65
    sp_change_thresh : float, default=0.01
    max_order : int, default=30

    Returns
    -------
    pd.DataFrame with one row per loop and columns:
        loop_id, fault_type (if present), harris_index, ar_order,
        var_total, var_mv, var_innovations, harris_index_p10,
        harris_index_p25, n_samples_used, trend_r2.
    """
    records = []
    has_fault_col = "fault_type" in df.columns
    for loop_id, group in df.groupby(loop_id_col, sort=True):
        metrics = compute_harris_index(
            group, delay=delay, settle_startup=settle_startup,
            settle_sp=settle_sp, sp_change_thresh=sp_change_thresh,
            max_order=max_order,
        )
        row = {loop_id_col: loop_id, **metrics}
        if has_fault_col:
            row["fault_type"] = group["fault_type"].iloc[0]
        records.append(row)

    out = pd.DataFrame(records)
    cols = [loop_id_col]
    if has_fault_col:
        cols.append("fault_type")
    cols += [
        "harris_index", "ar_order", "var_total", "var_mv",
        "var_innovations", "harris_index_p10", "harris_index_p25",
        "n_samples_used", "trend_r2"
    ]
    return out[cols]
