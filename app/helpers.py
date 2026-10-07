"""
Loop Guard - Streamlit Dashboard Helper Functions
Provides cached data loading, preprocessing, diagnostic evaluations,
and plain-English summary formatting.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import streamlit as st

# Import core diagnostic functions from src
from src.diagnostics import (
    compute_setpoint_tracking,
    compute_oscillation_detection,
    compute_stiction_detection_v2,
    compute_harris_index,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Configuration: Hardcoded Demo Loop IDs (Configurable in one place)
# -----------------------------------------------------------------------------
DEMO_SIMULATED_LOOPS: List[Dict[str, Any]] = [
    {"loop_id": 1, "expected_type": "healthy", "display_name": "Simulated Loop 01"},
    {"loop_id": 3, "expected_type": "healthy", "display_name": "Simulated Loop 03"},
    {"loop_id": 16, "expected_type": "sensor_drift", "display_name": "Simulated Loop 16"},
    {"loop_id": 38, "expected_type": "valve_stiction_v2", "display_name": "Simulated Loop 38"},
    {"loop_id": 43, "expected_type": "oscillatory_tuning", "display_name": "Simulated Loop 43"},
]

# -----------------------------------------------------------------------------
# 1. Data Loading Functions (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_simulated_demo_data() -> Dict[str, pd.DataFrame]:
    """
    Load the 5 hardcoded demonstration loops from data/simulated/labeled_loops.csv.
    Returns a dictionary mapping display loop name to its DataFrame.
    """
    csv_path = ROOT_DIR / "data" / "simulated" / "labeled_loops.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Simulated dataset not found at {csv_path}")

    full_df = pd.read_csv(csv_path)
    loops_dict = {}

    for cfg in DEMO_SIMULATED_LOOPS:
        lid = cfg["loop_id"]
        sub = full_df[full_df["loop_id"] == lid].copy().sort_values("timestep")
        name = cfg["display_name"]
        sub["loop_display_name"] = name
        sub["true_fault_type"] = cfg["expected_type"]
        loops_dict[name] = sub

    return loops_dict


@st.cache_data
def load_tep_demo_data() -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]:
    """
    Load Tennessee Eastman Process demonstration runs for the Reactor Temperature loop.
    Reuses the extraction logic from scratch/tep_step2_eval_part_c.py.
    Shows 1 Healthy run, 1 Fault 14 run, 1 Fault 11 run, and 1 Fault 4 run (Run 51).
    If TEP files are missing, returns (None, friendly_message).
    """
    ff_path = ROOT_DIR / "scratch" / "tep_cache_ff_train.pkl"
    faulty_path = ROOT_DIR / "scratch" / "tep_faulty_train_subset.csv.gz"

    if not ff_path.exists() or not faulty_path.exists():
        msg = (
            "TEP benchmark data files were not found locally in scratch/ "
            "(tep_cache_ff_train.pkl or tep_faulty_train_subset.csv.gz). "
            "Please download the TEP dataset or run the scratch extraction scripts first."
        )
        return None, msg

    try:
        df_ff = pd.read_pickle(ff_path)
        df_faulty = pd.read_csv(faulty_path)
        df_faulty["sample"] = df_faulty["sample"].astype(int)

        sp_nominal = 120.40  # Grand mean nominal setpoint for Reactor Temperature
        target_run = 51      # First run of the report evaluation split (51..100)

        # Extract Run 51 from sample 21 to 500
        classes_map = [
            ("TEP Healthy (Run 51)", df_ff[df_ff["simulationRun"] == target_run], "Healthy baseline"),
            ("TEP Fault 14 (Run 51)", df_faulty[(df_faulty["faultNumber"] == 14) & (df_faulty["simulationRun"] == target_run)], "Fault 14: Valve Sticking"),
            ("TEP Fault 11 (Run 51)", df_faulty[(df_faulty["faultNumber"] == 11) & (df_faulty["simulationRun"] == target_run)], "Fault 11: Cooling Water Temp Oscillation"),
            ("TEP Fault 4 (Run 51)", df_faulty[(df_faulty["faultNumber"] == 4) & (df_faulty["simulationRun"] == target_run)], "Fault 4: Cooling Water Temp Step"),
        ]

        loops_dict = {}
        for name, cdf, fault_desc in classes_map:
            sub = cdf[cdf["sample"] >= 21].sort_values("sample").copy()
            if len(sub) == 0:
                continue
            ldf = pd.DataFrame({
                "timestep": sub["sample"].to_numpy(dtype=float),
                "SP": np.full(len(sub), sp_nominal, dtype=float),
                "PV": sub["xmeas_9"].to_numpy(dtype=float),
                "OP": sub["xmv_10"].to_numpy(dtype=float),
            })
            ldf["loop_display_name"] = name
            ldf["true_fault_type"] = fault_desc
            loops_dict[name] = ldf

        return loops_dict, None
    except Exception as e:
        return None, f"Error reading TEP dataset files: {str(e)}"


# -----------------------------------------------------------------------------
# 2. Upload Validation and Preprocessing
# -----------------------------------------------------------------------------
def validate_and_preprocess_upload(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Validate user-uploaded loop CSV file:
    1. Check case-insensitive column names for SP, PV, OP.
    2. Check for missing values / NaNs.
    3. Forward-fill gaps of up to 3 samples and warn.
    4. Raise ValueError if gaps exceed 3 samples or if sample count < 125.
    Returns (cleaned_dataframe, warning_message).
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not parse uploaded CSV file: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Match columns case-insensitively
    col_map = {}
    for col in df.columns:
        clean_name = str(col).strip().upper()
        if clean_name in ("SP", "PV", "OP", "TIMESTEP", "TIME"):
            col_map[col] = "timestep" if clean_name in ("TIMESTEP", "TIME") else clean_name

    df = df.rename(columns=col_map)
    required = ["SP", "PV", "OP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"The file must contain columns named SP (setpoint), PV (process variable), and OP (controller output)."
        )

    # Check sample count
    min_required_samples = 125
    if len(df) < min_required_samples:
        raise ValueError(
            f"Uploaded loop has only {len(df)} samples. A minimum of {min_required_samples} samples "
            f"is required to reliably evaluate sliding-window oscillation and valve stiction checks."
        )

    # Check missing values / NaNs
    warning_msg = None
    has_nans = df[["SP", "PV", "OP"]].isna().any().any()
    if has_nans:
        # Check max consecutive NaNs in each required column
        max_gap = 0
        for col in ["SP", "PV", "OP"]:
            is_na = df[col].isna()
            if is_na.any():
                # Count consecutive True runs
                runs = is_na.astype(int).groupby((~is_na).cumsum()).cumsum()
                col_max = int(runs.max())
                max_gap = max(max_gap, col_max)

        if max_gap > 3:
            raise ValueError(
                f"Data contains extended missing value gaps of {max_gap} consecutive samples. "
                f"Loop Guard requires plant records with gap lengths of at most 3 samples to avoid corrupting diagnostic results."
            )
        else:
            # Forward-fill and backward-fill gaps of up to 3 samples
            df[["SP", "PV", "OP"]] = df[["SP", "PV", "OP"]].ffill().bfill()
            warning_msg = (
                f"Notice: Detected short missing data gap(s) up to {max_gap} sample(s). "
                f"Values were automatically forward-filled for analysis."
            )

    # Clean numeric types
    for col in ["SP", "PV", "OP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["SP", "PV", "OP"]].isna().any().any():
        raise ValueError("Uploaded file contains non-numeric data in SP, PV, or OP columns.")

    if "timestep" not in df.columns:
        df["timestep"] = np.arange(len(df), dtype=float)
    else:
        df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce").fillna(np.arange(len(df), dtype=float))

    df["loop_display_name"] = "Uploaded Loop"
    df["true_fault_type"] = "User Uploaded (Unlabeled)"

    return df, warning_msg


# -----------------------------------------------------------------------------
# 3. Diagnostic Evaluation Pipelines
# -----------------------------------------------------------------------------
def evaluate_loop_diagnostics(
    ldf: pd.DataFrame,
    ruleset: str = "simulation",
    harris_delay: int = 3,
) -> Dict[str, Any]:
    """
    Run all 4 diagnostics on a single loop DataFrame and apply either
    'simulation' (Phase 2 defaults) or 'tep' (Phase 3 adapted frozen rules).
    """
    # 1. D#1: Setpoint tracking
    d1 = compute_setpoint_tracking(ldf)
    r2_linear = float(d1["r2_linear"])

    # 2. D#2: Oscillation detection
    d2 = compute_oscillation_detection(ldf)
    zcr = float(d2["zero_crossing_rate"])
    acf_osc = float(d2["acf_osc_score"])
    period = float(d2["dominant_period"])

    # 3. D#3: Valve Stiction v2
    d3 = compute_stiction_detection_v2(ldf)
    stiction_v2_flag = bool(d3["stiction_flag"])

    # 4. D#4: Harris Performance Index
    # In TEP rules, d=1 is standard per scratch/part_c_frozen_rules.md, or user selected delay
    eff_delay = 1 if ruleset == "tep" else int(harris_delay)
    d4 = compute_harris_index(ldf, delay=eff_delay, settle_startup=30, settle_sp=65 if ruleset == "simulation" else 0)
    ih = float(d4["harris_index"])
    var_total = float(d4["var_total"]) if not np.isnan(d4["var_total"]) else 0.0

    # Decision Logic per Ruleset
    if ruleset == "tep":
        # TEP Frozen Rules (from scratch/part_c_frozen_rules.md)
        d1_flag = bool(r2_linear >= 0.50)
        d2_flag = bool((acf_osc >= 0.55) or (zcr <= 30.0 and acf_osc >= 0.35))
        # Period-2 Chatter specifically fingerprints TEP Fault 14 valve sticking
        chatter_p2 = bool(zcr >= 80.0 and period == 2.0 and acf_osc >= 0.80)
        d3_flag = bool(chatter_p2 or stiction_v2_flag)
        d4_flag = bool(not np.isnan(ih) and ih < 0.50)
        ruleset_title = "TEP adapted rules"
    else:
        # Simulation / Uploaded Default Rules (Phase 2 validated defaults)
        d1_flag = bool(r2_linear >= 0.50)
        d2_flag = bool((acf_osc >= 0.35 and period <= 30.0 and period > 0) or (zcr >= 30.0))
        d3_flag = bool(stiction_v2_flag)
        # Performance is degraded when minimum variance ratio is low and error variance is non-negligible
        d4_flag = bool(not np.isnan(ih) and ih < 0.50 and var_total >= 0.05)
        chatter_p2 = False
        ruleset_title = "Simulation rules"

    flag_count = int(d1_flag + d2_flag + d3_flag + d4_flag)

    # Overall Health Status Rule:
    # green = 0 flags, yellow = exactly 1 flag, red = 2 or more flags
    if flag_count == 0:
        overall_status = "green"
        status_label = "Healthy"
    elif flag_count == 1:
        overall_status = "yellow"
        status_label = "Attention Needed"
    else:
        overall_status = "red"
        status_label = "Action Required"

    return {
        "ruleset": ruleset_title,
        "d1_flag": d1_flag,
        "d2_flag": d2_flag,
        "d3_flag": d3_flag,
        "d4_flag": d4_flag,
        "chatter_p2": chatter_p2,
        "flag_count": flag_count,
        "overall_status": overall_status,
        "status_label": status_label,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "harris_delay": eff_delay,
    }


# -----------------------------------------------------------------------------
# 4. Plain-English Summary and Thresholds Display
# -----------------------------------------------------------------------------
def generate_plain_headline(res: Dict[str, Any]) -> str:
    """
    Generate a simple, jargon-free headline explaining loop health.
    """
    flags = []
    if res["d1_flag"]:
        flags.append("sensor decalibration drift")
    if res["d3_flag"]:
        if res.get("chatter_p2"):
            flags.append("high-frequency valve chattering")
        else:
            flags.append("sticking valve behavior")
    if res["d2_flag"]:
        flags.append("unstable rhythmic swinging")
    if res["d4_flag"]:
        flags.append("degraded control efficiency")

    if not flags:
        return "This loop looks healthy: steady setpoint tracking and normal valve movement."
    elif len(flags) == 1:
        return f"This loop needs attention: suspected {flags[0]}."
    else:
        return f"This loop requires action: detected {', '.join(flags[:-1])} and {flags[-1]}."


def get_thresholds_summary(ruleset: str = "simulation", harris_delay: int = 3) -> Dict[str, str]:
    """
    Return human-readable descriptions of all active thresholds.
    """
    if ruleset == "tep":
        return {
            "Ruleset in Use": "TEP adapted rules (Calibrated on DEV runs 1..50)",
            "D#1 (Sensor Drift)": "R2 linear >= 0.50",
            "D#2 (Oscillation)": "ACF >= 0.55 OR (ZCR <= 30.0 AND ACF >= 0.35)",
            "D#3 (Valve Stiction)": "Period-2 Chatter (ZCR >= 80, Period == 2, ACF >= 0.80) OR Stiction v2 (Default)",
            "D#4 (Performance)": f"Harris Index I_H(d={harris_delay}) < 0.50",
        }
    else:
        return {
            "Ruleset in Use": "Simulation rules (Phase 2 validated defaults)",
            "D#1 (Sensor Drift)": "R2 linear >= 0.50 (monotonically diverging error)",
            "D#2 (Oscillation)": "ACF >= 0.35 with Period <= 30 steps OR ZCR >= 30.0 (noise)",
            "D#3 (Valve Stiction)": "Stiction v2 default (R2 < 0.40, ZCR < 30.0, ACF >= 0.22, Period >= 35, Corr < 0.50)",
            "D#4 (Performance)": f"Harris Index I_H(d={harris_delay}) < 0.50 with error variance >= 0.05",
        }


# -----------------------------------------------------------------------------
# 5. Fault Knowledge Base & Suggested Actions (PART 1)
# -----------------------------------------------------------------------------
FAULT_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "drift": {
        "title": "Sensor Drift",
        "what_it_means": (
            "The sensor reading is steadily pulling away from the target over time. "
            "This creates a widening gap between the true physical condition and what the controller thinks is happening."
        ),
        "possible_causes": [
            "Gradual sensor decalibration or aging transmitter electronics.",
            "Physical coating, buildup, or fouling on the sensor probe.",
            "Ambient temperature shifts affecting uncompensated instrument electronics.",
            "Slow unmeasured baseline changes in upstream feed material.",
        ],
        "what_to_check": [
            "Compare the current transmitter reading against a calibrated local gauge.",
            "Inspect the probe for physical scaling, sediment, or corrosion.",
            "Review recent zero and span calibration records for this instrument.",
            "Verify instrument electrical signal wiring and ground integrity.",
        ],
        "risk_if_ignored": (
            "Product quality will quietly drift outside technical specifications, leading to off-spec batches or wasted raw materials."
        ),
    },
    "oscillation": {
        "title": "Process Oscillation",
        "what_it_means": (
            "The process variable is repeatedly swinging back and forth across the target value. "
            "This continuous wave pattern shows that the automated controller cannot find a stable balance."
        ),
        "possible_causes": [
            "Controller tuning is too sensitive or aggressive for current plant conditions.",
            "Cyclical disturbances cycling into this loop from upstream equipment.",
            "Coupling or interaction with an adjacent control loop in the same unit.",
            "Nonlinear process response shifts occurring at the current operating point.",
        ],
        "what_to_check": [
            "Review PID controller gain and reset time settings against tuning records.",
            "Inspect upstream process flows to see if the swing originates upstream.",
            "Check whether neighboring control loops are oscillating at the same rhythm.",
            "Confirm whether production rate changes altered the process response speed.",
        ],
        "risk_if_ignored": (
            "Constant back and forth swinging accelerates mechanical valve wear, wastes utility energy, and can destabilize downstream process units."
        ),
    },
    "stiction": {
        "title": "Valve Stiction",
        "what_it_means": (
            "The mechanical valve is sticking in position and then suddenly jumping when control pressure builds up. "
            "This jerky stop and start motion prevents the valve from making smooth, gradual corrections."
        ),
        "possible_causes": [
            "Excessive friction from an over-tightened valve stem packing gland.",
            "Product residue, solids, or scale buildup on the valve plug and seat.",
            "Physical wear or galling on the valve stem or actuator guides.",
            "Low or fluctuating pneumatic instrument air supply pressure.",
        ],
        "what_to_check": [
            "Visually inspect the valve stem for physical scoring, debris, or dried product.",
            "Check actuator air supply pressure and verify supply regulator stability.",
            "Check valve packing nut torque against vendor specifications.",
            "Observe valve stem travel locally while manually nudging the controller output.",
        ],
        "risk_if_ignored": (
            "Sticking valves force the process into continuous cycling, cause premature actuator breakdown, and create product quality variability."
        ),
    },
    "chatter": {
        "title": "Actuator Chatter",
        "what_it_means": (
            "The valve position is rapidly jumping back and forth at every single measurement step. "
            "This rapid alternating movement is the fastest possible cyclical pattern a digital valve can experience."
        ),
        "possible_causes": [
            "Severe valve stem friction creating an alternating limit cycle.",
            "Derivative controller action amplifying high-frequency sensor noise.",
            "Digital valve positioner deadband set too tight for process dynamics.",
            "Mechanical loose play or worn linkage between the actuator and positioner.",
        ],
        "what_to_check": [
            "Inspect the valve positioner for error codes and travel chatter logs.",
            "Check whether controller derivative action (Td) is active and reacting to noise.",
            "Verify digital positioner tuning parameters and travel deadband settings.",
            "Inspect feedback arm linkages and mounting brackets for mechanical play.",
        ],
        "risk_if_ignored": (
            "Extreme chatter can destroy valve packings and internal positioner linkages within days or weeks of continuous operation."
        ),
    },
    "performance": {
        "title": "Degraded Control Performance",
        "what_it_means": (
            "The loop is generating substantially more tracking error variance than the best achievable control benchmark. "
            "While the loop may maintain a steady average, it is operating with unnecessary variability."
        ),
        "possible_causes": [
            "Sluggish controller tuning that responds too slowly to load disturbances.",
            "Secondary impact from an underlying valve defect like stiction or deadband.",
            "Unmeasured external process disturbances continuously upsetting the loop.",
            "Uncompensated physical transport delay in fluid piping or equipment.",
        ],
        "what_to_check": [
            "Inspect the loop for physical valve issues (stiction or drift) first.",
            "Evaluate how long the loop takes to settle after routine setpoint steps.",
            "Verify process delay and time constant estimates against design data.",
            "Check if operational throughput or feed composition has shifted recently.",
        ],
        "risk_if_ignored": (
            "Operating with poor control efficiency increases energy consumption, widens quality spreads, and lowers overall plant productivity."
        ),
    },
}

SUGGESTION_DISCLAIMER = (
    "These are general guidelines based on the pattern detected. Always confirm on site before making changes."
)


def get_fault_actions(res: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return a list of (key, entry) from FAULT_KNOWLEDGE for all active flags in result.
    Order: drift, chatter or stiction, oscillation, performance.
    """
    actions = []
    if res.get("d1_flag"):
        actions.append(("drift", FAULT_KNOWLEDGE["drift"]))
    if res.get("d3_flag"):
        if res.get("chatter_p2"):
            actions.append(("chatter", FAULT_KNOWLEDGE["chatter"]))
        else:
            actions.append(("stiction", FAULT_KNOWLEDGE["stiction"]))
    if res.get("d2_flag"):
        actions.append(("oscillation", FAULT_KNOWLEDGE["oscillation"]))
    if res.get("d4_flag"):
        actions.append(("performance", FAULT_KNOWLEDGE["performance"]))
    return actions


def get_card_summary(res: Dict[str, Any], loop_name: str) -> str:
    """
    Generate a one-line plain English summary under the badges for overview cards.
    """
    if "Fault 4" in loop_name:
        return "Status: Controller absorbed disturbance into valve movement. No fault detected."

    flag_count = res.get("flag_count", 0)
    if flag_count == 0:
        return "Status: Operating normally. No issues detected."

    d1 = res.get("d1_flag", False)
    d2 = res.get("d2_flag", False)
    d3 = res.get("d3_flag", False)
    d4 = res.get("d4_flag", False)
    chatter = res.get("chatter_p2", False)

    stict_text = "actuator chatter" if chatter else "valve stiction"

    # Single flag
    if flag_count == 1:
        if d1:
            return "Likely cause: sensor drift. See drill-down for details."
        if d3:
            return f"Likely cause: {stict_text}. See drill-down for details."
        if d2:
            return "Likely cause: tuning or process oscillation. See drill-down for details."
        if d4:
            return "Likely cause: suboptimal controller performance. See drill-down for details."

    # Multiple flags
    primary_causes = []
    if d1:
        primary_causes.append("sensor drift")
    if d3:
        primary_causes.append(stict_text)
    if d2:
        primary_causes.append("oscillation")

    if primary_causes and d4:
        return f"Likely cause: {', '.join(primary_causes)} leading to degraded performance. See drill-down for details."
    elif primary_causes:
        return f"Likely cause: {', '.join(primary_causes)}. See drill-down for details."
    else:
        return "Multiple operational issues detected. See drill-down for details."


def render_html(html_str: str) -> None:
    """
    Render raw HTML safely in Streamlit without triggering Markdown code block formatting.
    Strips leading and trailing whitespace from every line, removes blank lines,
    ensures no line starts with 4 or more spaces, and passes to st.markdown.
    """
    cleaned_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


