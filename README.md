# Loop Guard

Loop Guard is an automated diagnostic and health monitoring system for industrial PID control loops. It detects, isolates, and assesses common control loop degradation modes from routine closed-loop operating data.

---

## Diagnostics

| Diagnostic | Fault / Condition Detected | Key Signals & Methods | Implementation Functions | Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **D#1: Setpoint Tracking** | Sensor Drift (slow sensor decalibration / bias ramp) | • `error_slope`: OLS linear slope of $(PV - SP)$ vs time<br>• `r2_linear`: $R^2$ of linear fit to tracking error<br>• `mean_error`, `rmse_error` | `compute_setpoint_tracking`<br>`compute_setpoint_tracking_batch` | **Validated**: $R^2 \ge 0.65$ separates drift from non-drift with $21\times$–$33\times$ margin across clean, noisy, and disturbance regimes (0 false positives). |
| **D#2: Oscillation Detection** | Loop Oscillations & High-Frequency Noise Fault | • `acf_osc_score`: windowed-max negative autocorrelation peak ($W=125$, stride=50)<br>• `zero_crossing_rate`: sign crossings of detrended error per 100 steps<br>• `dominant_period`: $2 \times$ lag of ACF minimum | `compute_oscillation_detection`<br>`compute_oscillation_detection_batch` | **Validated**: Windowed ACF isolates episodic oscillations; ZCR $> 30$ isolates the noise class in clean and LOW regimes and in the disturbance set, but breaks in MODERATE noise (healthy ZCR 22 to 41), where `dominant_period` is the more reliable signal. |
| **D#3: Stiction Detection (v2)** | Suspected Valve Stiction (deadband & stick-slip) | • Pre-gate: $R^2_{\text{linear}} < 0.40$ (rejects drift), $\text{ZCR} < 30.0$ (rejects noise), $\text{ACF} \ge 0.22$, $\text{Period} \ge 35$<br>• Classifier: `op_pv_corr_filt` (MA-5 filtered Pearson correlation of $|\Delta OP|$ and $|\Delta PV|$) $< 0.50$ | `compute_stiction_detection_v2`<br>`compute_stiction_detection_v2_batch` | **Validated (v2)**: Moving-average filter restores the stiction vs. linear oscillation gap ($>0.28$) in noisy data; evaluated across 4 operating regimes. |
| **D#4: Harris Performance Index** | Control Loop Performance Degradation (distance to MV benchmark) | • $I_H = 1 - \sigma_{\text{min}}^2 / \sigma_y^2$<br>• Segmented AR model (AIC order selection up to $p=30$) fitted across unmasked steady segments ($d=3$) | `compute_harris_index`<br>`compute_harris_index_batch` | **Validated**: 8/8 known-answer unit tests passed; invariant class ordering across delays $d=2, 3, 4$; reflects colored disturbance correlation ($I_H \approx 0.38$). |

---

## Validation and Known Limits

### Diagnostic #3 (v2) Empirical Results Across Regimes

Evaluated on the four design sets and tested out-of-sample on fresh-seed (+2000) sets:

| Regime | Condition | Stiction Flagged / Total | False Positives | Missed Stiction Loop IDs |
| :--- | :--- | :---: | :---: | :--- |
| **CLEAN (Design)** | Ideal simulation, $\sigma_m=0, \sigma_d=0$ (60 loops, 20 stiction) | **19 / 20 (95%)** | 1 (Healthy Loop 2) | Loop 23 |
| **CLEAN (Fresh +2000)** | Fresh seeds, same configs/params (30 loops, 10 stiction) | **9 / 10 (90%)** | 1 (Healthy Loop 2) | Loop 13 |
| **LOW NOISE (Design)** | $\sigma_m=0.15, \sigma_d=0.20$ (30 loops, 10 stiction) | **10 / 10 (100%)** | 0 | None |
| **LOW NOISE (Fresh +2000)** | Fresh seeds, same configs/params (30 loops, 10 stiction) | **9 / 10 (90%)** | 1 (Healthy Loop 2) | Loop 11 |
| **MODERATE NOISE (Design)** | $\sigma_m=0.50, \sigma_d=0.80$ (30 loops, 10 stiction) | **7 / 10 (70%)** | 0 | Loops 41, 43, 48 |
| **MODERATE NOISE (Fresh +2000)** | Fresh seeds, same configs/params (30 loops, 10 stiction) | **5 / 10 (50%)** | 0 | Loops 41, 43, 46, 47, 48 |
| **DISTURBANCE (Design)** | $\sigma_m=0.05$, AR(1) $\phi=0.9$, $\sigma_w=0.15$ (30 loops, 10 stiction) | **10 / 10 (100%)** | 5 (Healthy Loops 1–5) | None |
| **DISTURBANCE (Fresh +2000)** | Fresh seeds, same configs/params (30 loops, 10 stiction) | **10 / 10 (100%)** | 4 (Healthy Loops 1, 2, 3, 5) | None |

*In all regimes without exception, sensor drift ($R^2 < 0.40$), linear oscillatory tuning (`corr_filt` $\ge 0.64$), and high-frequency noise faults are rejected with 0 false positives.*

---

### Known Limitations

1. **Suspected Stiction False Positives Under Dominant Load Disturbance:**  
   In regimes where low-frequency colored load disturbance accounts for $>80\%$ of regulatory PV variance without high-frequency measurement noise, healthy closed-loop wandering produces low step-by-step PV-OP correlation (`corr_filt` $< 0.50$) and slow apparent periods ($48$–$62$ steps). D#3 flags these healthy loops as suspected stiction. Confirming stiction in such loops requires an active valve step test or process model.
2. **Reduced Stiction Recall Under Heavy Measurement Noise:**  
   When sensor noise is severe ($\sigma_m = 0.50$, where noise variance exceeds stiction limit cycle amplitude), high-frequency zero crossings push loop ZCR above the noise ceiling ($\text{ZCR} > 30$), causing D#3 recall to drop to $50\%$–$70\%$.
3. **Harris Index (D#4) is a Coarse Severity Band, Not an Exact Point Metric:**  
   Monte Carlo unit testing at finite sample lengths ($N=500$, 200 seeds) demonstrates that while the asymptotic theoretical value is exact ($I_H = 0.4686$ for AR(1) $\phi=0.9, d=3$), finite regulatory records exhibit a sample standard deviation of $\approx 0.09$ and an upward estimation bias of $\approx +0.05$ (mean measured $0.5208$ on segmented data). It functions as a relative performance health band rather than an exact regulator index.
4. **Harris Index Invariance to Pure Sensor Noise:**  
   White measurement noise cannot be regulated away by feedback control. Because an AR model fits white noise with zero predictive lags beyond delay $d$, the minimum variance bound equals the total error variance, driving $I_H \to 1.0$. D#4 does not penalize sensor noise; sensor noise must be diagnosed via D#2 ZCR.
5. **Fresh Test Set Parameter Continuity:**  
   The fresh-seed evaluation sets (`labeled_loops_*_fresh2000.csv`) evaluate out-of-sample stochastic robustness (different pseudo-random trajectories for setpoint steps, measurement noise, and load disturbances), but reuse the identical 5 FOPDT process configurations and fault parameter grids ($S, J$, deadband, drift rate, tuning factors) as the design sets.
6. **Self-Simulated Scope:**  
   All Phase 2 diagnostic evaluations have been conducted on self-simulated first-order plus dead time (FOPDT) systems with simulated non-linearities. Industrial multi-unit validation on the Tennessee Eastman Process (TEP) benchmark dataset is the scope of Phase 3.
