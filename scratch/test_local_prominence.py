"""
Test Local Peak Prominence on Segment-Averaged Periodograms
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r'D:\Loop Gaurd')
clean_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops.csv')
noisy_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_noisy.csv')
disturb_df = pd.read_csv(PROJECT_ROOT / 'data/simulated/labeled_loops_disturb.csv')

regimes = {
    'CLEAN': clean_df,
    'LOW': noisy_df[noisy_df['noise_level'] == 'LOW'],
    'MODERATE': noisy_df[noisy_df['noise_level'] == 'MODERATE'],
    'DISTURB': disturb_df,
}

MASKS = [(30, 125), (190, 325), (390, 500)]
N_FFT = 256
DT = 1.0

def compute_segment_periodogram_metrics(pv, sp):
    # Compute periodogram for each segment, padded to N_FFT
    psd_list = []
    for s, e in MASKS:
        seg_e = pv[s:e] - sp[s:e]
        # Detrend linearly
        t = np.arange(len(seg_e))
        p = np.polyfit(t, seg_e, 1)
        seg_dt = seg_e - np.polyval(p, t)
        
        # Padded FFT
        fft_v = np.fft.rfft(seg_dt, n=N_FFT)
        # Power spectral density normalized by actual sample count
        psd_k = (np.abs(fft_v) ** 2) / len(seg_dt)
        psd_list.append(psd_k)
        
    avg_psd = np.mean(psd_list, axis=0)
    freqs = np.fft.rfftfreq(N_FFT, d=DT)
    
    # Exclude DC (f=0)
    psd_pos = avg_psd[1:]
    freqs_pos = freqs[1:]
    
    # Target band: period 20 to 100 steps -> freq [0.01, 0.05]
    in_band = (freqs_pos >= 0.01) & (freqs_pos <= 0.05)
    
    if not np.any(in_band) or np.max(psd_pos[in_band]) < 1e-12:
        return {'local_prominence': 0.0, 'peak_to_median': 0.0, 'peak_period': 0.0}
    
    band_indices = np.where(in_band)[0]
    peak_idx_in_band = np.argmax(psd_pos[in_band])
    peak_idx = band_indices[peak_idx_in_band]
    
    peak_f = freqs_pos[peak_idx]
    peak_power = psd_pos[peak_idx]
    peak_period = 1.0 / peak_f if peak_f > 0 else 0.0
    
    # Peak-to-median
    median_power = np.median(psd_pos)
    p2m = peak_power / median_power if median_power > 1e-9 else peak_power / 1e-9
    
    # Local prominence: flanking bands [0.5*f, 0.8*f] and [1.25*f, 2.0*f]
    lower_flank = (freqs_pos >= 0.5 * peak_f) & (freqs_pos <= 0.8 * peak_f)
    upper_flank = (freqs_pos >= 1.25 * peak_f) & (freqs_pos <= 2.0 * peak_f)
    
    flank_mask = lower_flank | upper_flank
    # Also exclude the peak bin and immediate neighbors (+/- 1 bin)
    flank_mask[max(0, peak_idx - 1) : min(len(flank_mask), peak_idx + 2)] = False
    
    if np.sum(flank_mask) > 0:
        flank_power = np.mean(psd_pos[flank_mask])
    else:
        # Fallback to entire spectrum outside the peak bin
        rest_mask = np.ones(len(psd_pos), dtype=bool)
        rest_mask[max(0, peak_idx - 2) : min(len(rest_mask), peak_idx + 3)] = False
        flank_power = np.mean(psd_pos[rest_mask]) if np.sum(rest_mask) > 0 else 1e-9
        
    prominence = peak_power / flank_power if flank_power > 1e-9 else peak_power / 1e-9
    
    return {
        'local_prominence': float(prominence),
        'peak_to_median': float(p2m),
        'peak_period': float(peak_period),
    }

records = []
for reg_name, r_df in regimes.items():
    for lid, g in r_df.groupby('loop_id', sort=True):
        ft = g['fault_type'].iloc[0]
        pv = g['PV'].values
        sp = g['SP'].values
        res = compute_segment_periodogram_metrics(pv, sp)
        records.append({
            'regime': reg_name,
            'loop_id': lid,
            'fault_type': ft,
            **res,
        })

df_res = pd.DataFrame(records)

print("=" * 85)
print("LOCAL PROMINENCE & PEAK-TO-MEDIAN (SEGMENT-AVERAGED)")
print("=" * 85)
for reg in ['CLEAN', 'LOW', 'MODERATE', 'DISTURB']:
    print(f"\n--- REGIME: {reg} ---")
    print(f"{'fault_type':<20} | {'LOCAL PROMINENCE (min, mean, max)':<35} | {'PEAK-TO-MEDIAN (min, mean, max)':<30}")
    print("-" * 90)
    sub = df_res[df_res['regime'] == reg]
    for ft in ['healthy', 'sensor_drift', 'valve_stiction_v1', 'valve_stiction_v2', 'oscillatory_tuning', 'noise']:
        s = sub[sub['fault_type'] == ft]
        pr = s['local_prominence']
        pm = s['peak_to_median']
        pr_str = f"[{pr.min():6.2f}, {pr.mean():6.2f}, {pr.max():6.2f}]"
        pm_str = f"[{pm.min():6.2f}, {pm.mean():6.2f}, {pm.max():6.2f}]"
        print(f"{ft:<20} | {pr_str:<35} | {pm_str:<30}")

df_res.to_csv(PROJECT_ROOT / 'scratch/prominence_results.csv', index=False)
