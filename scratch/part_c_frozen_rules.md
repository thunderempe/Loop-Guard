# Part C: Frozen Rules Note (Calibrated on DEV Split Runs 1..50 Only)

These rules are derived exclusively from the empirical feature distributions on the DEVELOPMENT split (runs 1..50) of the TEP Reactor Temp loop (OP: `xmv_10`, PV: `xmeas_9`, SP: 120.40, samples 21..500). They are frozen prior to loading or evaluating any run from the REPORT split (runs 51..100).

---

## 1. Diagnostic #1 (Setpoint Tracking / Sensor Drift)
- **Rule:** `d1_drift_flag = (r2_linear >= 0.50)`
- **One-line rationale:** Preserved from Phase 2 benchmark because neither healthy nor faulty Reactor Temp runs exhibit linear ramp drift (DEV max $R^2 = 0.00725$), preventing false drift detection.

---

## 2. Diagnostic #2 (Adapted Thermal Oscillation Detector)
- **Rule:** `d2_osc_adapted_flag = (acf_osc_score >= 0.55) or (zero_crossing_rate <= 30.0 and acf_osc_score >= 0.35)`
- **One-line rationale:** Healthy DEV regulation is noise-dominated ($\text{ZCR} \in [48.0, 59.7]$, max ACF $= 0.497$); true thermal oscillations either generate slow zero-crossings (Fault 11 $\text{ZCR} \le 25.9, \text{ACF} \ge 0.364$) or dominant correlation peaks (Fault 14 $\text{ACF} \ge 0.941$).

---

## 3. Diagnostic #3: Two Distinct Flags Reported Side-by-Side

### 3a. Unmodified Phase 2 Stiction Detector (`stiction_v2_flag`)
- **Rule:** `stiction_v2_flag = compute_stiction_detection_v2(df)["stiction_flag"]` (default arguments: $R^2 < 0.40, \text{ZCR} < 30.0, \text{ACF} \ge 0.22, T \ge 35.0, \text{corr\_filt} < 0.50$)
- **One-line rationale:** Evaluated without modification to benchmark Phase 2 stiction recall and false alarm behavior on industrial plant data.

### 3b. Period-2 Actuator Chatter Detector (`chatter_period2_flag`)
- **Rule:** `chatter_period2_flag = (zero_crossing_rate >= 80.0 and dominant_period == 2.0 and acf_osc_score >= 0.80)`
- **One-line rationale:** Specifically fingerprints the Nyquist-frequency alternating limit-cycle chatter produced by TEP's discrete-time valve sticking model (Fault 14 DEV: $\text{ZCR} \ge 89.6, T = 2.0, \text{ACF} \ge 0.941$), making clear this is a period-2 chatter detector rather than validation of Phase 2 stiction v2.

---

## 4. Diagnostic #4 (Adapted Harris Index Performance Degradation)
- **Rule 4a (Moderate Regulatory Degradation):** `d4_degraded_mod_flag = (harris_index_d1 < 0.50)`
- **One-line rationale:** Flags degradation when minimum-variance benchmark at dead time $d=1$ drops below $0.50$, cleanly separated from Healthy DEV where $I_H(d=1) \ge 0.784$ (Fault 11 DEV max $= 0.347$, Fault 14 DEV max $= 0.0232$).
- **Rule 4b (Severe Regulatory Degradation):** `d4_degraded_sev_flag = (harris_index_d3 < 0.20)`
- **One-line rationale:** Flags catastrophic regulatory collapse at dead time $d=3$ below $0.20$, cleanly separating Fault 14 (DEV max $= 0.0859$) from Healthy DEV (min $= 0.881$).

---

## 5. Composite System Flag ("Any Feature Flagged")
- **Rule:** `any_flagged = (d1_drift_flag or d2_osc_adapted_flag or chatter_period2_flag or d4_degraded_mod_flag)`
- **One-line rationale:** Combines orthogonal detectors to assess overall anomaly detection coverage per class.
