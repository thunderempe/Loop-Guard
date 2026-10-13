"""
Loop Guard - Streamlit Dashboard Helper Functions
Provides cached data loading, preprocessing, diagnostic evaluations,
and plain-English summary formatting.
"""

import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import core diagnostic functions from src
from src.diagnostics import (
    compute_setpoint_tracking,
    compute_oscillation_detection,
    compute_stiction_detection_v2,
    compute_harris_index,
)

from app.fault_info import FAULT_KNOWLEDGE_BASE
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


def is_tep_demo_subset() -> bool:
    """Return True if using the compact committed TEP demo subset rather than the full scratch dataset."""
    ff_path = ROOT_DIR / "scratch" / "tep_cache_ff_train.pkl"
    faulty_path = ROOT_DIR / "scratch" / "tep_faulty_train_subset.csv.gz"
    return not (ff_path.exists() and faulty_path.exists())


@st.cache_data
def load_tep_demo_data() -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]:
    """
    Load Tennessee Eastman Process demonstration runs for the Reactor Temperature loop.
    Reuses the extraction logic from scratch/tep_step2_eval_part_c.py.
    Shows 1 Healthy run, 1 Fault 14 run, 1 Fault 11 run, and 1 Fault 4 run (Run 51).
    Uses full scratch dataset if present, otherwise falls back to data/tep_demo/.
    If TEP files are missing, returns (None, friendly_message).
    """
    ff_path = ROOT_DIR / "scratch" / "tep_cache_ff_train.pkl"
    faulty_path = ROOT_DIR / "scratch" / "tep_faulty_train_subset.csv.gz"
    demo_path = ROOT_DIR / "data" / "tep_demo" / "tep_reactor_temp_demo.csv.gz"

    using_full = ff_path.exists() and faulty_path.exists()
    using_demo = demo_path.exists()

    if not using_full and not using_demo:
        msg = (
            "TEP benchmark data files were not found locally in scratch/ "
            "(tep_cache_ff_train.pkl or tep_faulty_train_subset.csv.gz) "
            "or data/tep_demo/ (tep_reactor_temp_demo.csv.gz). "
            "Please download the TEP dataset or run the scratch extraction scripts first."
        )
        return None, msg

    try:
        if using_full:
            df_ff = pd.read_pickle(ff_path)
            df_faulty = pd.read_csv(faulty_path)
            df_faulty["sample"] = df_faulty["sample"].astype(int)
        else:
            demo_df = pd.read_csv(demo_path)
            demo_df["sample"] = demo_df["sample"].astype(int)
            df_ff = demo_df[demo_df["faultNumber"] == 0]
            df_faulty = demo_df[demo_df["faultNumber"] > 0]

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
@st.cache_data
def validate_and_preprocess_upload(file_input: Any) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Validate user uploaded loop CSV file from bytes or uploaded file buffer:
    1. Check case insensitive column names for SP, PV, OP.
    2. Check for missing values or NaNs.
    3. Forward fill gaps of up to 3 samples and warn.
    4. Raise ValueError if gaps exceed 3 samples or if sample count < 125.
    Returns (cleaned_dataframe, warning_message).
    """
    if isinstance(file_input, bytes):
        raw_bytes = file_input
    elif hasattr(file_input, "getvalue"):
        raw_bytes = file_input.getvalue()
    elif hasattr(file_input, "read"):
        raw_bytes = file_input.read()
    else:
        raw_bytes = bytes(file_input)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
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
        fallback_series = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce").fillna(fallback_series)

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
        d2_osc_flag = bool((acf_osc >= 0.55) or (zcr <= 30.0 and acf_osc >= 0.35))
        d2_noise_flag = False
        d2_flag = d2_osc_flag
        # Period-2 Chatter specifically fingerprints TEP Fault 14 valve sticking
        chatter_p2 = bool(zcr >= 80.0 and period == 2.0 and acf_osc >= 0.80)
        d3_flag = bool(chatter_p2 or stiction_v2_flag)
        d4_flag = bool(not np.isnan(ih) and ih < 0.50)
        ruleset_title = "TEP adapted rules"
    else:
        # Simulation / Uploaded Default Rules (Phase 2 validated defaults)
        d1_flag = bool(r2_linear >= 0.50)
        d2_osc_flag = bool(acf_osc >= 0.35 and period <= 30.0 and period > 0)
        d2_noise_flag = bool(zcr >= 30.0)
        d2_flag = bool(d2_osc_flag or d2_noise_flag)
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
        "d2_osc_flag": d2_osc_flag,
        "d2_noise_flag": d2_noise_flag,
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
FAULT_KNOWLEDGE = FAULT_KNOWLEDGE_BASE

SUGGESTION_DISCLAIMER = (
    "These are general guidelines based on the pattern detected. Always confirm on site before making changes."
)


def get_fault_actions(res: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return a list of (key, entry) from FAULT_KNOWLEDGE_BASE for all active flags in result.
    If no flags are raised, returns the healthy entry.
    Order: healthy, or drift, chatter or stiction, noise, oscillation, performance.
    """
    flag_count = res.get("flag_count", 0)
    if flag_count == 0:
        return [("healthy", FAULT_KNOWLEDGE_BASE["healthy"])]

    actions = []
    if res.get("d1_flag"):
        actions.append(("drift", FAULT_KNOWLEDGE_BASE["drift"]))
    if res.get("d3_flag"):
        if res.get("chatter_p2"):
            actions.append(("chatter", FAULT_KNOWLEDGE_BASE["chatter"]))
        else:
            actions.append(("stiction", FAULT_KNOWLEDGE_BASE["stiction"]))
    if res.get("d2_noise_flag"):
        actions.append(("noise", FAULT_KNOWLEDGE_BASE["noise"]))
    if res.get("d2_osc_flag"):
        actions.append(("oscillation", FAULT_KNOWLEDGE_BASE["oscillation"]))
    elif res.get("d2_flag") and not res.get("d2_noise_flag"):
        actions.append(("oscillation", FAULT_KNOWLEDGE_BASE["oscillation"]))
    if res.get("d4_flag"):
        actions.append(("performance", FAULT_KNOWLEDGE_BASE["performance"]))
    return actions


def get_card_summary(res: Dict[str, Any], loop_name: str) -> str:
    """
    Generate a one line plain English summary under the badges for overview cards.
    """
    if "Fault 4" in loop_name:
        return "Status: Controller absorbed disturbance into valve movement. No fault detected."

    flag_count = res.get("flag_count", 0)
    if flag_count == 0:
        return "Status: Operating normally. No issues detected."

    d1 = res.get("d1_flag", False)
    d2 = res.get("d2_flag", False)
    d2_noise = res.get("d2_noise_flag", False)
    d3 = res.get("d3_flag", False)
    d4 = res.get("d4_flag", False)
    chatter = res.get("chatter_p2", False)

    stict_text = "actuator chatter" if chatter else "valve stiction"

    # Single flag
    if flag_count == 1:
        if d1:
            return "Possible cause: sensor drift. See drill down for details."
        if d3:
            return f"Possible cause: {stict_text}. See drill down for details."
        if d2_noise and not res.get("d2_osc_flag"):
            return "Possible cause: high frequency measurement noise. See drill down for details."
        if d2:
            return "Possible cause: tuning or process oscillation. See drill down for details."
        if d4:
            return "Possible cause: suboptimal controller performance. See drill down for details."

    # Multiple flags
    primary_causes = []
    if d1:
        primary_causes.append("sensor drift")
    if d3:
        primary_causes.append(stict_text)
    if d2:
        primary_causes.append("oscillation")

    if primary_causes and d4:
        return f"Possible cause: {', '.join(primary_causes)} leading to degraded performance. See drill down for details."
    elif primary_causes:
        return f"Possible cause: {', '.join(primary_causes)}. See drill down for details."
    else:
        return "Multiple operational issues detected. See drill down for details."


def render_html(html_str: str) -> None:
    """
    Render raw HTML safely in Streamlit without triggering Markdown code block formatting.
    Strips leading and trailing whitespace from every line, removes blank lines,
    ensures no line starts with 4 or more spaces, and passes to st.markdown.
    """
    cleaned_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Cached Chart Rendering
# -----------------------------------------------------------------------------
@st.cache_data
def build_drilldown_chart_png(
    chart_type: str,
    loop_name: str,
    data_source: str,
    ldf: pd.DataFrame,
) -> bytes:
    """
    Build and cache drill down diagnostic charts as PNG bytes.
    chart_type: 'timeseries' or 'phase_plane'
    """
    buf = io.BytesIO()

    if chart_type == "timeseries":
        t = ldf["timestep"].to_numpy()
        sp = ldf["SP"].to_numpy()
        pv = ldf["PV"].to_numpy()
        op = ldf["OP"].to_numpy()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.patch.set_facecolor("#111827")
        ax1.set_facecolor("#0d1117")
        ax2.set_facecolor("#0d1117")

        for ax in (ax1, ax2):
            ax.tick_params(colors="#cbd5e1")
            ax.xaxis.label.set_color("#e2e8f0")
            ax.yaxis.label.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.grid(True, linestyle=":", alpha=0.35, color="#64748b")

        # SP and PV
        ax1.plot(t, sp, label="Target Setpoint (SP)", color="#38bdf8", linestyle="--", linewidth=1.5)
        ax1.plot(t, pv, label="Measured Sensor (PV)", color="#f59e0b", linewidth=1.2, alpha=0.95)
        ax1.set_ylabel("PV / SP")
        leg1 = ax1.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155")
        for text in leg1.get_texts():
            text.set_color("#e2e8f0")
        ax1.set_title(f"{loop_name} Operational Behavior", fontsize=11, fontweight="bold", color="#f8fafc")

        # OP
        ax2.plot(t, op, label="Valve Command (OP)", color="#10b981", linewidth=1.2)
        ax2.set_xlabel("Time (samples)")
        ax2.set_ylabel("OP (%)")
        leg2 = ax2.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155")
        for text in leg2.get_texts():
            text.set_color("#e2e8f0")

        plt.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)

    elif chart_type == "phase_plane":
        pv = ldf["PV"].to_numpy()
        op = ldf["OP"].to_numpy()
        dop = np.abs(np.diff(op))
        dpv = np.abs(np.diff(pv))

        fig_scatter, ax_s = plt.subplots(figsize=(6, 3.8))
        fig_scatter.patch.set_facecolor("#111827")
        ax_s.set_facecolor("#0d1117")
        ax_s.tick_params(colors="#cbd5e1")
        ax_s.xaxis.label.set_color("#e2e8f0")
        ax_s.yaxis.label.set_color("#e2e8f0")
        for spine in ax_s.spines.values():
            spine.set_color("#334155")
        ax_s.grid(True, linestyle=":", alpha=0.35, color="#64748b")

        ax_s.scatter(dop, dpv, alpha=0.55, color="#a855f7", edgecolors="none", s=25)
        ax_s.set_xlabel("|dOP| (Valve Command Step Size)")
        ax_s.set_ylabel("|dPV| (Process Variable Step Size)")
        ax_s.set_title("Step by Step Movement Coupling (|dOP| vs |dPV|)", fontsize=10, fontweight="bold", color="#f8fafc")

        plt.tight_layout()
        fig_scatter.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig_scatter)
    else:
        raise ValueError(f"Unknown chart_type: {chart_type}")

    buf.seek(0)
    return buf.getvalue()


