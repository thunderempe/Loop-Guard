"""
Exploration of D#3 Candidate Features across Regimes
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
clean_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops.csv')
noisy_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_noisy.csv')
disturb_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_disturb.csv')

regimes = {
    'CLEAN': clean_df,
    'LOW_NOISE': noisy_df[noisy_df['noise_level'] == 'LOW'],
    'MOD_NOISE': noisy_df[noisy_df['noise_level'] == 'MODERATE'],
    'DISTURB': disturb_df,
}

FAULTS = ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2',
          'oscillatory_tuning', 'noise']
STICTION_CLASSES = {'valve_stiction_v1', 'valve_stiction_v2'}

# Regulatory masks: 30..125, 190..325, 390..500
MASKS = [(30, 125), (190, 325), (390, 500)]

def get_unmasked(arr):
    return np.concatenate([arr[s:e] for s, e in MASKS])

def detrend_linear(x):
    t = np.arange(len(x))
    p = np.polyfit(t, x, 1)
    return x - np.polyval(p, t)

def compute_features(sp, pv, op, use_unmasked=False):
    if use_unmasked:
        # demean each segment separately to remove DC step jumps
        segments_e = []
        segments_pv = []
        segments_op = []
        for s, e in MASKS:
            seg_e = (pv[s:e] - sp[s:e]) - np.mean(pv[s:e] - sp[s:e])
            segments_e.append(seg_e)
            segments_pv.append(pv[s:e] - np.mean(pv[s:e]))
            segments_op.append(op[s:e] - np.mean(op[s:e]))
        e_sig = np.concatenate(segments_e)
        pv_sig = np.concatenate(segments_pv)
        op_sig = np.concatenate(segments_op)
    else:
        # full series with linear detrending on error
        e_sig = detrend_linear(pv - sp)
        pv_sig = pv
        op_sig = op

    # --- Feature A: Oscillation amplitude normalized by noise floor ---
    osc_amp = np.std(e_sig)
    # Robust noise estimator on first differences of PV:
    # Under Gaussian noise, diff(noise) ~ N(0, 2*sigma^2)
    # MAD / 0.6745 / sqrt(2) is consistent estimator of white noise std
    if use_unmasked:
        # compute diff within segments to avoid step jump artifacts
        diffs = np.concatenate([np.diff(pv[s:e]) for s, e in MASKS])
    else:
        diffs = np.diff(pv)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    noise_est = mad / (0.67448975 * np.sqrt(2.0))
    if noise_est < 1e-4:
        noise_est = 1e-4  # prevent division by zero in noiseless clean data
    amp_norm = osc_amp / noise_est

    # --- Feature B: Periodogram peak significance in [20, 100] period band ---
    # Frequencies: N = len(e_sig)
    N = len(e_sig)
    fft_vals = np.fft.rfft(e_sig - np.mean(e_sig))
    psd = np.abs(fft_vals)**2 / N
    freqs = np.fft.rfftfreq(N, d=1.0)
    
    # Exclude DC (f=0)
    psd_pos = psd[1:]
    freqs_pos = freqs[1:]
    
    # Band [20, 100] steps -> freq in [0.01, 0.05]
    in_band = (freqs_pos >= 0.01) & (freqs_pos <= 0.05)
    if np.any(in_band):
        band_peak = np.max(psd_pos[in_band])
        median_psd = np.median(psd_pos)
        peak_to_median = band_peak / median_psd if median_psd > 1e-9 else band_peak / 1e-9
        fisher_g = band_peak / np.sum(psd_pos) if np.sum(psd_pos) > 1e-9 else 0.0
    else:
        peak_to_median = 0.0
        fisher_g = 0.0

    # --- Feature C: Persistence across 125-step windows ---
    # Slide 125-step window with stride 25 across full error or unmasked
    # On full series: 500 steps, window=125, stride=25 -> 16 windows
    # For each window, compute ACF up to lag 50.
    # Check if there is a minimum < -0.15 (indicating period 2 * lag)
    sig_for_win = detrend_linear(pv - sp)
    win_len = 125
    stride = 25
    win_periods = []
    for w_start in range(0, len(sig_for_win) - win_len + 1, stride):
        w_chunk = sig_for_win[w_start : w_start + win_len]
        w_chunk = w_chunk - np.mean(w_chunk)
        if np.std(w_chunk) < 1e-6:
            continue
        # ACF
        acf = np.correlate(w_chunk, w_chunk, mode='full')
        acf = acf[len(w_chunk)-1:]
        acf = acf / acf[0]
        # Check lags 5 to 50 (periods 10 to 100)
        lags = acf[1:51]
        min_idx = np.argmin(lags)
        min_val = lags[min_idx]
        if min_val < -0.15:
            est_p = 2 * (min_idx + 1)
            win_periods.append(est_p)
        else:
            win_periods.append(0.0)

    # Persistence: fraction of windows with period in [20, 100]
    total_wins = len(win_periods)
    stiction_band_wins = [p for p in win_periods if 20 <= p <= 100]
    persistence = len(stiction_band_wins) / max(total_wins, 1)

    # Also check period consistency: fraction within 25% of median period
    if len(stiction_band_wins) >= 2:
        med_p = np.median(stiction_band_wins)
        consistent = [p for p in stiction_band_wins if 0.75 * med_p <= p <= 1.25 * med_p]
        persistence_consistent = len(consistent) / max(total_wins, 1)
    else:
        persistence_consistent = persistence

    # --- Feature D: Filtered op_pv_corr ---
    # Moving average window W=5 on OP and PV
    w_filt = 5
    if use_unmasked:
        # filter per segment
        dop_list = []
        dpv_list = []
        for s, e in MASKS:
            op_s = pd.Series(op[s:e]).rolling(w_filt, min_periods=1).mean().values
            pv_s = pd.Series(pv[s:e]).rolling(w_filt, min_periods=1).mean().values
            dop_list.append(np.abs(np.diff(op_s)))
            dpv_list.append(np.abs(np.diff(pv_s)))
        dop_f = np.concatenate(dop_list)
        dpv_f = np.concatenate(dpv_list)
    else:
        op_s = pd.Series(op).rolling(w_filt, min_periods=1).mean().values
        pv_s = pd.Series(pv).rolling(w_filt, min_periods=1).mean().values
        dop_f = np.abs(np.diff(op_s))
        dpv_f = np.abs(np.diff(pv_s))

    if np.std(dop_f) > 1e-9 and np.std(dpv_f) > 1e-9:
        corr_filt = float(np.corrcoef(dop_f, dpv_f)[0, 1])
    else:
        corr_filt = np.nan

    # Raw unfiltered corr for comparison
    if use_unmasked:
        dop_r = np.concatenate([np.abs(np.diff(op[s:e])) for s, e in MASKS])
        dpv_r = np.concatenate([np.abs(np.diff(pv[s:e])) for s, e in MASKS])
    else:
        dop_r = np.abs(np.diff(op))
        dpv_r = np.abs(np.diff(pv))
    if np.std(dop_r) > 1e-9 and np.std(dpv_r) > 1e-9:
        corr_raw = float(np.corrcoef(dop_r, dpv_r)[0, 1])
    else:
        corr_raw = np.nan

    return {
        'amp_norm': amp_norm,
        'peak_to_median': peak_to_median,
        'fisher_g': fisher_g,
        'persistence': persistence_consistent,
        'corr_filt': corr_filt,
        'corr_raw': corr_raw,
    }

print("Running feature extraction on all datasets...")
results = []
for reg_name, r_df in regimes.items():
    for mode in ['unmasked', 'full']:
        use_unm = (mode == 'unmasked')
        for lid, group in r_df.groupby('loop_id', sort=True):
            ft = group['fault_type'].iloc[0]
            sp = group['SP'].values
            pv = group['PV'].values
            op = group['OP'].values
            feats = compute_features(sp, pv, op, use_unmasked=use_unm)
            results.append({
                'regime': reg_name,
                'mode': mode,
                'loop_id': lid,
                'fault_type': ft,
                **feats,
            })

res_df = pd.DataFrame(results)
res_df.to_csv(PROJECT_ROOT / 'scratch/d3_features_explored.csv', index=False)
print("Done! Saved to scratch/d3_features_explored.csv")
