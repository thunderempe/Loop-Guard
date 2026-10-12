# Loop Guard

> An automated health monitor for industrial PID control loops that spots hidden controller faults from routine operational data.

---

## What Problem Does This Solve?

Imagine you are driving a car on cruise control set to 60 mph.
* If your speedometer slowly loses calibration, the car might gradually speed up to 75 mph while still insisting you are going 60.
* If your gas pedal gets sticky, the engine will lag behind your foot, then suddenly jerk forward.
* If the computer reacts too aggressively, the car will constantly lurch faster and slower in an endless wave.
* If you are driving on a bumpy gravel road, you would want a score telling you how much fuel and stability you are wasting compared to ideal driving.

In industrial factories, such as chemical plants, oil refineries, and pharmaceutical manufacturing sites, thousands of automated gadgets do the exact same job as cruise control. They are called **PID control loops**. They open and close valves, adjust heaters, and throttle pumps to keep temperatures, pressures, liquid levels, and flows steady at a target value.

In a typical plant running 500 to 3,000 of these loops, many operate poorly at any given time (valves stick, sensors drift, tuning becomes unstable, or noise rattles the equipment). Human plant engineers cannot watch thousands of graphs every day, so problems often go unnoticed until a safety system trips, equipment breaks, or an entire batch of product is ruined.

**Loop Guard** acts as an automated health monitor. It continuously reads the routine data a factory already records (target setpoint, actual sensor reading, valve command) and flags what is broken before operations fail.

---

## What is a PID Loop?

A **PID (Proportional-Integral-Derivative) loop** is a classic automated control loop. It looks at the gap between where a process currently is and where you want it to be, and continuously nudges an actuator (like a valve) to close that gap.

### Glossary of Key Terms
* **Setpoint (SP):** The target value you want to maintain (e.g., target temperature: 120 °C).
* **Process Variable (PV):** The actual measured sensor reading coming from the plant (e.g., current thermometer reading).
* **Controller Output (OP):** The instruction sent by the controller to the actuator (e.g., valve position command: 45% open).
* **Sensor Drift:** A sensor that slowly decalibrates over time, creating a steady ramp error that the controller cannot see on its own.
* **Oscillation:** A repeating back-and-forth swing where the process variable repeatedly overshoots and undershoots the target.
* **Valve Stiction:** Static friction in a mechanical valve stem. The valve sticks in place while the controller pushes harder, then suddenly breaks free and jumps, causing a choppy limit cycle.
* **Noise:** Random, high-frequency measurement electrical chatter or fluid turbulence on a sensor signal.
* **Harris Performance Index:** A standard engineering benchmark (Harris, 1989) that asks: *given unavoidable physical pipeline delays, how close is this loop's error variance to the theoretical minimum variance achievable by an ideal controller?*
* **TEP Benchmark:** The Tennessee Eastman Process, a renowned simulation benchmark of an industrial chemical plant developed by Eastman Chemical Company (Downs & Vogel, 1993; extended by Rieth et al., 2017).

---

## What Does Loop Guard Check?

Loop Guard runs four automated diagnostic checks:

| Check | What It Looks For (Plain Words) | What It Is Good At | Where It Struggles |
| :--- | :--- | :--- | :--- |
| **D#1: Setpoint Tracking** | Checks if the sensor is slowly decalibrating and drifting away from the target over time. | Extremely clean separation of linear drift ramps: dashboard flags drift at $R^2 \ge 0.50$, and $0.65$ represents the midpoint of the gap seen on simulated data (weakest drift loop $0.68$, strongest non-drift loop $0.08$), with a $21\times$ to $33\times$ ratio of average $R^2$ values between drift and non-drift loops and 0 false alarms across simulated tests. | Misses sudden abrupt sensor jumps or nonlinear curves; assumes stationary setpoints. |
| **D#2: Oscillation & Noise Detection** | Checks if the loop is swinging back and forth in a rhythm, or if the sensor is jittering with rapid noise. | Sliding-window autocorrelation captures episodic bursts even when the valve temporarily flattens against safety limits. | In high-noise environments, random noise causes healthy loops to cross zero frequently, tripping fixed noise thresholds. |
| **D#3: Valve Stiction Detection (v2)** | Checks if a mechanical valve is sticking and jumping rather than moving smoothly with commands. | Moving-average filtering (MA-5) cleans sensor jitter, restoring a wide separation gap between stiction limit cycles and normal oscillations. | Low-frequency load disturbances make healthy loops wander slowly, mimicking stiction; fast valve chatter at period 2 is blocked by standard gates. |
| **D#4: Harris Performance Index** | Grades control quality from 0.0 (severely degraded) to 1.0 (ideal control or pure random white noise). | Model-free benchmark calculated from routine data without pausing plant production or running intrusive step tests. | Blind to pure white sensor noise (scores white noise as 1.0); estimates have sample variance and slight upward bias on short time records. |

---

## How It Was Built and Tested

* **Phase 0: Project Setup & Benchmark Sourcing**  
  Established the project structure, dependencies, and retrieved the extended Tennessee Eastman Process (TEP) benchmark dataset (Rieth et al., 2017) from Harvard Dataverse to serve as an industrial simulation testbed alongside custom synthetic models.
* **Phase 1: Generative Physics & Fault Injection Simulator**  
  Built a discrete-time First-Order Plus Dead-Time (FOPDT) process simulator coupled to a position-form digital PID controller with anti-windup clamping (`src/simulator.py`). Implemented five realistic closed-loop fault models (`src/faults.py`): sensor drift, 1-parameter deadband stiction (v1), 2-parameter Choudhury $(S, J)$ valve stiction (v2), aggressive oscillatory tuning, and high-frequency measurement noise.
* **Phase 2: Diagnostic Suite & Stress Testing**  
  Built four diagnostic modules (`src/diagnostics.py`). Proved D#4 with an 8-test known-answer unit suite against closed-form theoretical equations. Stress-tested the diagnostics under realistic field conditions (clean, low noise, moderate noise, and colored load disturbances). Diagnosed where initial rules broke under noise, created Stiction Detection v2 with a low-pass moving-average filter, and verified generalization out-of-sample on 120 loops generated with fresh random seeds (+2000 offset).
* **Phase 3: Benchmark Validation on TEP Industrial Simulation**  
  Extracted and evaluated the critical Reactor Temperature loop (`xmv_10` cooling water valve, `xmeas_9` reactor temperature) from the Tennessee Eastman Process. Calibrated decision thresholds exclusively on development runs 1 to 50 (frozen in `scratch/part_c_frozen_rules.md`) and executed a strict single evaluation on unseen report runs 51 to 100 across Faults 14, 11, 4, and healthy baselines.
* **Phase 4: Operator Triage Dashboard (Built - v1)**  
  Built an interactive Streamlit web application (`app/streamlit_app.py`) allowing control engineers to triage loops. Includes a multi-loop overview with red/yellow/green status badges, single-loop drill-down with time-series and phase-plane plots, diagnostic explanations, CSV loop upload validation, and demonstration suites for both synthetic and TEP benchmark loops.

---

## Results

### 1. Simulated Data Results (Phase 2)

#### Diagnostic #3 (Stiction v2) Across All 4 Operating Regimes
Stiction detection was tested across the initial design sets and re-tested out-of-sample on completely unseen random seeds (fresh +2000 sets):

| Operating Regime | Environment Description | Stiction Detected (Recall) | False Alarms (Healthy/Drift/Tuning/Noise) |
| :--- | :--- | :---: | :---: |
| **Clean (Design)** | Ideal simulation, $\sigma_m=0, \sigma_d=0$ (60 loops, 20 stiction) | 19 / 20 (95%) | 1 (Healthy Loop 2) |
| **Clean (Fresh +2000)** | Fresh random seeds, same process grid (30 loops, 10 stiction) | 9 / 10 (90%) | 1 (Healthy Loop 2) |
| **Low Noise (Design)** | Realistic noise: $\sigma_m=0.15, \sigma_d=0.20$ (30 loops, 10 stiction) | 10 / 10 (100%) | 0 |
| **Low Noise (Fresh +2000)** | Fresh random seeds (30 loops, 10 stiction) | 9 / 10 (90%) | 1 (Healthy Loop 2) |
| **Moderate Noise (Design)** | Heavy noise: $\sigma_m=0.50, \sigma_d=0.80$ (30 loops, 10 stiction) | 7 / 10 (70%) | 0 |
| **Moderate Noise (Fresh +2000)** | Fresh random seeds (30 loops, 10 stiction) | 5 / 10 (50%) | 0 |
| **Disturbance (Design)** | Colored load disturbance: AR(1) $\phi=0.9, \sigma_w=0.15$ (30 loops, 10 stiction) | 10 / 10 (100%) | 5 (Healthy Loops 1 to 5) |
| **Disturbance (Fresh +2000)** | Fresh random seeds (30 loops, 10 stiction) | 10 / 10 (100%) | 4 (Healthy Loops 1, 2, 3, 5) |

*What this means for a non-technical reader:* Loop Guard correctly spots sticking valves 90% to 100% of the time under normal factory conditions. However, when an unmeasured disturbance slowly pushes the process, healthy valves slowly wander to compensate, which can trick the system into suspecting valve stiction.

#### Diagnostics D#1, D#2, and D#4 Summary on Simulated Data
* **D#1 (Sensor Drift):** Maintained $100\%$ detection with $0$ false alarms across every simulated regime (dashboard flags drift at $R^2 \ge 0.50$, while $0.65$ serves as the midpoint between the weakest drift loop at $0.68$ and strongest non-drift loop at $0.08$, exhibiting a $21\times$ to $33\times$ ratio of average $R^2$ values between drift and non-drift loops).
* **D#2 (Oscillation & Noise):** Windowed autocorrelation cleanly isolated oscillatory tuning. Zero-crossing rate ($>30.0$) separated noise faults in clean and low-noise regimes, but broke down in moderate noise where healthy loops naturally crossed zero 22 to 41 times per 100 steps.
* **D#4 (Harris Performance Index):** Passed 8 out of 8 theoretical known-answer unit tests. Maintained invariant ranking across process delays ($d=2, 3, 4$). In colored disturbance regimes, scored realistic performance degradation ($I_H \approx 0.38$).

---

### 2. TEP Benchmark Results (Phase 3: Reactor Temperature Loop)

Evaluated on the Tennessee Eastman Process (TEP) benchmark for the Reactor Temperature loop (`xmv_10` cooling water valve command, `xmeas_9` reactor temperature sensor). Decision rules were frozen on development runs 1 to 50 (`scratch/part_c_frozen_rules.md`) and evaluated in a single test run on report runs 51 to 100 (50 runs per class, 48 hours per run, samples 21 to 500):

| Diagnostic / Detector | Rule / Threshold Applied | Healthy (False Alarms) | Fault 14: Valve Sticking | Fault 11: Cooling Temp Osc | Fault 4: Cooling Temp Step |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D#4 (Harris Index)** | `I_H(d=1) < 0.50` | **0 / 50 (0%)** | **50 / 50 (100%)** | **50 / 50 (100%)** | **0 / 50 (0%)** |
| **Adapted D#2 (Oscillation)** | `ACF >= 0.55` or `(ZCR <= 30 and ACF >= 0.35)` | **1 / 50 (2%)** | **50 / 50 (100%)** | **50 / 50 (100%)** | **0 / 50 (0%)** |
| **D#3 v2 (Unmodified Stiction)** | Default Phase 2 thresholds (`R2 < 0.4, ZCR < 30, T >= 35, corr < 0.5`) | **0 / 50 (0%)** | **0 / 50 (0%)** | **2 / 50 (4%)** | **0 / 50 (0%)** |
| **Period-2 Chatter Detector** | `ZCR >= 80 and T = 2.0 and ACF >= 0.80` | **0 / 50 (0%)** | **50 / 50 (100%)** | **0 / 50 (0%)** | **0 / 50 (0%)** |
| **Any Diagnostic Flagged** | Union of orthogonal detectors | **1 / 50 (2%)** | **50 / 50 (100%)** | **50 / 50 (100%)** | **0 / 50 (0%)** |

*What this means for a non-technical reader:* On the chemical plant benchmark, our health monitoring system detected 100% of the oscillatory and sticking valve disturbances with only 1 false alarm across 50 healthy runs. However, fixed rules created in simple simulations could not directly detect the sticking valve without recalibration, because the industrial simulation naturally has more baseline noise and faster valve cycling than the synthetic equations assumed.

---

## Honest Limitations

1. **Simulated Healthy Baselines Are Cleaner Than Real Plants:**  
   Our synthetic healthy loops had zero measurement noise, making healthy baseline tracking artificially near-perfect. In the TEP benchmark, healthy reactor temperature has a natural baseline zero-crossing rate of ~52 per 100 steps and lag-1 autocorrelation of $-0.15$, proving that real plant loops are noise-dominated.
2. **Stiction Detection (v2) Misses in Heavy Noise and Alarms Under Load Disturbances:**  
   When sensor noise is severe ($\sigma_m = 0.50$), stiction limit-cycle recall drops to 50% to 70% because noise jitter pushes zero crossings above the gate ceiling. In disturbance-dominated regimes, slow load wandering produces low step-by-step correlation, generating false stiction alarms on healthy loops (5/5 on design set, 4/5 on fresh set).
3. **Fault 4 Was Undetected by All Diagnostics:**  
   Fault 4 (a step change in cooling water inlet temperature) was missed by every PV-based diagnostic (0/50). Because the closed-loop PI controller successfully absorbed the disturbance by shifting the valve position by $+6.9\sigma$, the temperature PV showed $0\sigma$ change. Diagnosing disturbances absorbed by control requires inspecting valve OP shifts, not just error tracking.
4. **Harris Index is Coarse and Biased on Short Data:**  
   Finite-sample regression bias shifts the estimated Harris Index upward: after masking (about 340 usable samples from a 500-step run), the bias is $\approx +0.05$ with a standard deviation of $\approx 0.09$; on unmasked 500-sample records, the bias is about $+0.02$ with a standard deviation of about $0.07$. It functions as a relative health indicator, not an exact point metric. Additionally, pure white noise drives $I_H \to 1.0$, falsely making noise look like ideal control.
5. **Only One TEP Loop Was Evaluated:**  
   Phase 3 validation was performed exclusively on the Reactor Temperature loop (`xmv_10` / `xmeas_9`). Multi-loop plant-wide interactions have not yet been evaluated.
6. **TEP Rules Required Hand Adaptation:**  
   Unmodified Phase 2 thresholds did not transfer out-of-the-box to TEP due to baseline shift. The reported Phase 3 rules had to be calibrated on development runs 1 to 50 before being evaluated once on runs 51 to 100.
7. **Dashboard Scope (v1.2 Built, Not Connected to Live Plant Historian Streams):**  
   The Phase 4 Streamlit dashboard is built (v1.2: Intro landing page with 3D closed-loop animation, overview cards with one-line plain summaries, drill-down with technician suggestions and fault knowledge base, dark theme visual polish, CSV upload, TEP demo) for offline file-based analysis. It does not yet connect directly to live distributed control system (DCS) protocols or process multi-loop wide CSV files in batch.

---

## How to Run It

### 1. Environment Setup

Clone the repository and install the dependencies in a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# On Linux/macOS: source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Downloading the TEP Benchmark Data

The raw Tennessee Eastman Process simulation files are large and excluded via `.gitignore`.
1. Download the extended TEP training dataset published by Rieth et al. (2017) from the Harvard Dataverse:
   * **URL:** [https://doi.org/10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1)
2. Place the downloaded `.RData` files into the `data/tep/` folder:
   * `data/tep/TEP_FaultFree_Training.RData`
   * `data/tep/TEP_Faulty_Training.RData`

### 3. Step-by-Step Reproduction Pipeline

Run the scripts in the following order to reproduce the complete project results from scratch:

```bash
# --- Step 1: Sanity Check & Dataset Generation ---
python notebooks/00_sanity_check.py
python notebooks/01_generate_dataset.py

# --- Step 2: Validate Baseline Diagnostics (Phase 2) ---
python notebooks/02_validate_setpoint_tracking.py
python notebooks/03_validate_oscillation_detection.py
python notebooks/04_validate_stiction_detection.py
python notebooks/05_validate_harris_index.py
python notebooks/06_test_harris_known_answer.py

# --- Step 3: Stress Testing (Noise & Disturbance Regimes) ---
python notebooks/07_generate_noisy_dataset.py
python notebooks/08_validate_noisy_stress_test.py
python notebooks/09_generate_disturb_dataset.py
python notebooks/10_validate_disturb_dataset.py

# --- Step 4: Stiction v2 & Fresh Seed (+2000) Out-of-Sample Evaluation ---
python notebooks/11_validate_stiction_v2.py
python notebooks/12_generate_fresh2000_datasets.py
python notebooks/13_evaluate_fresh2000.py

# --- Step 5: Phase 3 TEP Evaluation (Report Split Runs 51-100) ---
python scratch/tep_step2_eval_part_c.py
```

### 4. Running the Operator Dashboard (Phase 4)

Launch the Streamlit web dashboard to inspect and triage control loops:

```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# On Linux/macOS: source venv/bin/activate

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
Loop Guard/
├── app/                                 # Streamlit operator dashboard (Phase 4)
│   ├── helpers.py                       # Data loading, validation, and diagnostic execution
│   ├── intro_scene.py                   # 3D closed-loop animation scene and static fallback
│   └── streamlit_app.py                 # Multi-page dashboard interface
├── data/
│   ├── simulated/                       # Synthetic labeled loop CSV datasets
│   └── tep/                             # TEP RData benchmark files (downloaded, gitignored)
├── notebooks/                           # Reproducible experimental and validation scripts
│   ├── 00_sanity_check.py               # Checks environment and verifies TEP/CSV data assets
│   ├── 01_generate_dataset.py           # Generates initial 60-loop ideal labeled dataset
│   ├── 02_validate_setpoint_tracking.py # Validates D#1 (drift separation & R2 metric)
│   ├── 03_validate_oscillation_detection.py # Validates D#2 (sliding-window ACF & ZCR)
│   ├── 04_validate_stiction_detection.py    # Validates D#3 v1 stiction algorithm
│   ├── 05_validate_harris_index.py      # Validates D#4 across multiple delay horizons
│   ├── 06_test_harris_known_answer.py   # 8-test unit test suite for Harris Index theoretical bounds
│   ├── 07_generate_noisy_dataset.py     # Generates LOW and MODERATE noise stress datasets
│   ├── 08_validate_noisy_stress_test.py # Evaluates diagnostics under noise stress
│   ├── 09_generate_disturb_dataset.py   # Generates colored load disturbance stress dataset
│   ├── 10_validate_disturb_dataset.py   # Evaluates diagnostics under disturbance stress
│   ├── 11_validate_stiction_v2.py       # Benchmarks D#3 v1 vs D#3 v2 across all 4 design sets
│   ├── 12_generate_fresh2000_datasets.py# Generates out-of-sample fresh-seed (+2000) datasets
│   └── 13_evaluate_fresh2000.py         # One-time out-of-sample evaluation of D#3 v2
├── scratch/                             # Exploratory analysis and Phase 3 TEP evaluation
│   ├── part_c_frozen_rules.md           # Formal rules frozen on TEP DEV split (runs 1-50)
│   ├── tep_05_faulty_train_extract.py   # Streams and extracts TEP candidate loop subset
│   ├── tep_06_onset_exact.py            # Verifies exact sample 21 fault introduction
│   ├── tep_10_inspect_fault14.py        # Inspects Fault 14 period-2 actuator chatter
│   ├── tep_step2_eval_part_c.py         # One-time evaluation on TEP REPORT split (runs 51-100)
│   └── tep_step2_summary.py             # Feature distribution summary between DEV and REPORT
├── src/                                 # Core importable package library
│   ├── __init__.py                      # Package exports exposing simulator, faults, and diagnostics
│   ├── simulator.py                     # FOPDT physics engine, digital PID, and valve models
│   ├── faults.py                        # Closed-loop physical fault injection routines
│   └── diagnostics.py                   # The 4 diagnostic algorithms and batch calculators
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## What's Next (Dashboard v2 & Future Work)

* **Multi-Loop Batch CSV Ingestion:**  
  Support uploading wide-format plant historian exports containing dozens of loops simultaneously, auto-grouping by tag name prefixes.
* **Automated Process Delay Estimator:**  
  Automatically estimate loop dead time $d$ via cross-correlation between OP and PV, eliminating manual delay selection.
* **Per-Loop Baseline Auto-Calibration:**  
  Implement unsupervised baseline learning (e.g., using normal operating periods) to automatically adjust zero-crossing and ACF thresholds to each loop's natural noise level, eliminating manual threshold tuning.
* **Shift Handover Report Export:**  
  One-click generation of printable summary reports and executive summaries for engineering shift handovers.
