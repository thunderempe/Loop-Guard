"""
Loop Guard - Streamlit Operator Dashboard (Phase 4)
Automated health monitoring for industrial PID control loops.
"""

import sys
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in Python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.helpers import (
    load_simulated_demo_data,
    load_tep_demo_data,
    is_tep_demo_subset,
    validate_and_preprocess_upload,
    evaluate_loop_diagnostics,
    generate_plain_headline,
    get_thresholds_summary,
    get_fault_actions,
    get_card_summary,
    build_drilldown_chart_png,
    render_html,
    SUGGESTION_DISCLAIMER,
)
from app.intro_scene import build_intro_scene_html

# Layout Spacing Constants
INTRO_BUTTONS_TOP_SPACE_PX = 48
INTRO_BUTTONS_BOTTOM_SPACE_PX = 24

# -----------------------------------------------------------------------------
# App Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loop Guard - Controller Health Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme, translucent cards, and high contrast badges (No em dashes or en dashes anywhere)
render_html(
    """
    <style>
    /* Dark background gradient across the app */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #172033 0%, #0d1117 55%, #090d16 100%);
        background-attachment: fixed;
        color: #e2e8f0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 17, 23, 0.92);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Soft translucent card styling */
    .loop-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(22, 29, 45, 0.65);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .loop-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }

    .metric-box {
        background: rgba(22, 29, 45, 0.6);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 8px;
        padding: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
    }

    /* Clear, high contrast status badges */
    .badge-green {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }

    .badge-yellow {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }

    .badge-red {
        background-color: rgba(239, 68, 68, 0.18);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }

    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        padding: 4px 8px;
        border-radius: 10px;
        font-size: 0.80rem;
        display: inline-block;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Consistent heading font sizes and spacing */
    h1 {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-bottom: 0.8rem !important;
    }
    h2 {
        font-size: 1.4rem !important;
        font-weight: 650 !important;
        color: #f1f5f9 !important;
        margin-top: 1.1rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-top: 1.0rem !important;
    }
    h4 {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* Expander styling for dark mode */
    [data-testid="stExpander"] {
        background: rgba(22, 29, 45, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """
)


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Loop Guard")
st.sidebar.caption("Automated PID Loop Diagnostics")

nav_options = ["Intro", "Overview", "Loop Drill-down", "About & Limits"]
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = "Intro"

def inspect_loop_callback(target_loop: str) -> None:
    st.session_state.selected_loop_name = target_loop
    st.session_state.nav_radio = "Loop Drill-down"


def intro_demo_callback() -> None:
    st.session_state.nav_radio = "Overview"
    st.session_state.data_source_select = "Demo: simulated loops"


def intro_upload_callback() -> None:
    st.session_state.nav_radio = "Overview"
    st.session_state.data_source_select = "Upload your own loop CSV"


nav_choice = st.sidebar.radio(
    "Navigation",
    nav_options,
    key="nav_radio",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")

data_source_options = [
    "Demo: simulated loops",
    "Demo: TEP Reactor Temperature",
    "Upload your own loop CSV",
]
if "data_source_select" not in st.session_state:
    st.session_state.data_source_select = "Demo: simulated loops"

data_source = st.sidebar.selectbox(
    "Select Input Source",
    data_source_options,
    key="data_source_select",
)

# Reveal true fault toggle for simulated demo
reveal_faults = False
if data_source == "Demo: simulated loops":
    reveal_faults = st.sidebar.checkbox("Reveal true fault (for demo)", value=False)

# Harris delay input in sidebar / advanced
harris_delay_input = 3
if data_source != "Demo: TEP Reactor Temperature":
    with st.sidebar.expander("Advanced Settings"):
        harris_delay_input = st.number_input(
            "Harris benchmark delay (steps)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Process dead time horizon in integer sample steps. Default is 3.",
        )


# -----------------------------------------------------------------------------
# Data Ingestion & State Management
# -----------------------------------------------------------------------------
active_loops_dict = {}
active_ruleset = "simulation"
upload_warning = None
data_error = None

if data_source == "Demo: simulated loops":
    active_ruleset = "simulation"
    try:
        active_loops_dict = load_simulated_demo_data()
    except Exception as e:
        data_error = f"Error loading simulated demo data: {str(e)}"

elif data_source == "Demo: TEP Reactor Temperature":
    active_ruleset = "tep"
    loops_dict, err_msg = load_tep_demo_data()
    if err_msg:
        st.sidebar.warning(err_msg)
        data_error = err_msg
    else:
        active_loops_dict = loops_dict
        if is_tep_demo_subset():
            st.sidebar.caption("Using committed demo subset (Run 51 of 5). Full TEP dataset contains 100 runs per fault.")

elif data_source == "Upload your own loop CSV":
    active_ruleset = "simulation"
    st.sidebar.info(
        "Thresholds were calibrated on simulated loops and one TEP loop, and may need recalibrating for other plants."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload loop CSV (SP, PV, OP)",
        type=["csv"],
        help="Upload a CSV with columns SP (target), PV (measurement), and OP (valve command).",
    )
    if uploaded_file is not None:
        try:
            cleaned_df, upload_warning = validate_and_preprocess_upload(uploaded_file.getvalue())
            active_loops_dict = {"Uploaded Loop": cleaned_df}
        except Exception as e:
            data_error = str(e)


# -----------------------------------------------------------------------------
# Session State for Active Loop Selection
# -----------------------------------------------------------------------------
loop_names = list(active_loops_dict.keys()) if active_loops_dict else []
if "selected_loop_name" not in st.session_state or st.session_state.selected_loop_name not in loop_names:
    if loop_names:
        st.session_state.selected_loop_name = loop_names[0]


# -----------------------------------------------------------------------------
# PAGE 0: INTRO (LANDING PAGE)
# -----------------------------------------------------------------------------
if nav_choice == "Intro":
    st.title("Loop Guard")
    st.caption("A health check for the control loops that keep a plant running.")

    # 3D Loop Animation Scene
    components.html(build_intro_scene_html(), height=800)

    # Vertical breathing room above action buttons
    render_html(f"<div style='height: {INTRO_BUTTONS_TOP_SPACE_PX}px;'></div>")

    # Call to action buttons
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.button(
            "Explore the demo",
            type="primary",
            key="btn_intro_demo",
            use_container_width=True,
            on_click=intro_demo_callback,
        )
    with col_b2:
        st.button(
            "Try your own loop data",
            key="btn_intro_upload",
            use_container_width=True,
            on_click=intro_upload_callback,
        )

    # Vertical breathing room below action buttons
    render_html(f"<div style='height: {INTRO_BUTTONS_BOTTOM_SPACE_PX}px;'></div>")

    st.markdown("---")

    st.markdown("### The Problem")
    st.markdown(
        textwrap.dedent(
            """
            Imagine driving a car where the cruise control slowly decalibrates, the accelerator pedal gets sticky, or the engine constantly surges back and forth.
            In an industrial plant, thousands of automated PID control loops adjust valves and pumps to keep temperatures, pressures, and flows steady.
            When these loops degrade unnoticed, they quietly waste utility energy, accelerate mechanical wear, and risk sudden shutdowns.
            """
        )
    )

    st.markdown("### What Loop Guard Does")
    c1, c2 = st.columns(2)
    with c1:
        card_d1 = """
        <div class='metric-box'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>🧭 Sensor Drift</div>
            <div style='color: #cbd5e1; font-size: 0.90rem;'>Spots sensors slowly wandering away from their true target.</div>
        </div>
        """
        render_html(card_d1)

        card_d3 = """
        <div class='metric-box'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>⚙️ Valve Stiction</div>
            <div style='color: #cbd5e1; font-size: 0.90rem;'>Detects sticky mechanical valves that jerk instead of moving smoothly.</div>
        </div>
        """
        render_html(card_d3)

    with c2:
        card_d2 = """
        <div class='metric-box'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>〰️ Process Oscillation</div>
            <div style='color: #cbd5e1; font-size: 0.90rem;'>Catches aggressive controller tuning and cyclical process swings.</div>
        </div>
        """
        render_html(card_d2)

        card_d4 = """
        <div class='metric-box'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>📊 Control Performance</div>
            <div style='color: #cbd5e1; font-size: 0.90rem;'>Measures excess tracking variability compared to ideal control benchmarks.</div>
        </div>
        """
        render_html(card_d4)

    st.markdown("### How It Was Tested")
    st.markdown(
        textwrap.dedent(
            """
            1. **Simulated faults with known answers:** Validated detection logic across 60 synthetic control loops with known physical faults in clean, noisy, and load-disturbed regimes.
            2. **A standard industrial benchmark:** Evaluated on the Tennessee Eastman Process benchmark for the Reactor Temperature loop, achieving 100 percent detection of Fault 14 and Fault 11 with 1 false alarm in 50 healthy runs.
            3. **Honest limits:** Demonstrated that closed-loop control absorbs some disturbances (such as Fault 4, which is not visible to PV based checks), proving that valve output shifts must be tracked alongside sensor errors.
            """
        )
    )

    st.info("Validated on simulated data and one benchmark loop. Not yet validated on real plant data.")


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
elif nav_choice == "Overview":
    st.title("Control Loop Health Overview")

    # Display active ruleset banner
    ruleset_name = "Simulation rules (Phase 2 defaults)" if active_ruleset == "simulation" else "TEP adapted rules (Frozen DEV benchmark)"
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        st.info(f"Ruleset in use: **{ruleset_name}**")
    with col_r2:
        st.metric("Total Monitored Loops", len(active_loops_dict))

    if data_source == "Upload your own loop CSV":
        st.info("Thresholds were calibrated on simulated loops and one TEP loop, and may need recalibrating for other plants.")

    if upload_warning:
        st.warning(upload_warning)

    if data_error:
        st.error(f"Cannot display loops: {data_error}")
    elif not active_loops_dict:
        st.info("No loop data currently loaded. Please upload a CSV file or select a demonstration source in the sidebar.")
    else:
        # Status rule explanation expander
        with st.expander("How Overall Loop Status is Determined", expanded=False):
            st.markdown(
                textwrap.dedent(
                    """
                    Each control loop is evaluated across four independent diagnostic checks:
                    1. **Sensor Drift (D#1):** Checks if the measurement is slowly wandering away from the setpoint.
                    2. **Oscillation (D#2):** Checks for rhythmic swinging or high-frequency measurement noise.
                    3. **Valve Stiction (D#3):** Checks if the mechanical valve is sticking and jumping.
                    4. **Performance (D#4):** Grades regulatory variance against theoretical minimum variance.

                    **Classification Criteria:**
                    * **Green (Healthy):** 0 diagnostics flagged.
                    * **Yellow (Attention Needed):** Exactly 1 diagnostic flagged.
                    * **Red (Action Required):** 2 or more diagnostics flagged.

                    *Overall score is simply the count of flagged diagnostics out of 4.*
                    """
                )
            )

        # Loop Cards
        for loop_name, loop_df in active_loops_dict.items():
            res = evaluate_loop_diagnostics(
                loop_df,
                ruleset=active_ruleset,
                harris_delay=harris_delay_input,
            )

            # Styling for overall status
            status_code = res["overall_status"]
            if status_code == "green":
                status_html = "<span class='badge-green'>HEALTHY (0/4 flagged)</span>"
            elif status_code == "yellow":
                status_html = "<span class='badge-yellow'>ATTENTION NEEDED (1/4 flagged)</span>"
            else:
                status_html = f"<span class='badge-red'>ACTION REQUIRED ({res['flag_count']}/4 flagged)</span>"

            # Badge helpers for individual diagnostics
            def make_badge(flag: bool, label: str) -> str:
                if flag:
                    return f"<span class='badge-red'>{label}: FLAGGED</span>"
                else:
                    return f"<span class='badge-neutral'>{label}: OK</span>"

            b_drift = make_badge(res["d1_flag"], "Drift")
            b_osc = make_badge(res["d2_flag"], "Oscillation")
            stict_label = "Chatter" if res.get("chatter_p2") else "Stiction"
            b_stict = make_badge(res["d3_flag"], stict_label)
            b_perf = make_badge(res["d4_flag"], "Performance")

            true_fault_text = ""
            if (reveal_faults or active_ruleset == "tep") and "true_fault_type" in loop_df.columns:
                tf = loop_df["true_fault_type"].iloc[0]
                true_fault_text = f"&nbsp;&nbsp;<span style='color: #6c757d; font-size: 0.85rem;'>[Label: <b>{tf}</b>]</span>"

            card_summary = get_card_summary(res, loop_name)

            # Render card container
            with st.container():
                card_html = f"""
                <div class='loop-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <div>
                            <span style='font-size: 1.15rem; font-weight: 700; color: #f8fafc;'>{loop_name}</span>{true_fault_text}
                        </div>
                        <div>{status_html}</div>
                    </div>
                    <div style='margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;'>
                        {b_drift}
                        {b_osc}
                        {b_stict}
                        {b_perf}
                    </div>
                    <div style='margin-top: 10px; font-size: 0.90rem; color: #cbd5e1; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px;'>{card_summary}</div>
                </div>
                """
                render_html(card_html)

                if "Fault 4" in loop_name:
                    st.info(
                        "Note on Fault 4: The feedback PI controller absorbed the cooling water inlet temperature step "
                        "by shifting valve command OP (+6.9 sigma), keeping reactor temperature PV unchanged. "
                        "Because PV tracking remained undisturbed, PV-based error checks show healthy (green)."
                    )

                col_btn, _ = st.columns([2, 8])
                with col_btn:
                    st.button(
                        f"Inspect {loop_name}",
                        key=f"btn_{loop_name}",
                        on_click=inspect_loop_callback,
                        args=(loop_name,),
                    )


# -----------------------------------------------------------------------------
# PAGE 2: LOOP DRILL-DOWN
# -----------------------------------------------------------------------------
elif nav_choice == "Loop Drill-down":
    st.title("Loop Diagnostic Drill-Down")

    if data_error:
        st.error(f"Cannot inspect loop: {data_error}")
    elif not active_loops_dict:
        st.info("No loops available. Select a data source in the sidebar.")
    else:
        # Loop selection box
        selected_loop = st.selectbox(
            "Select Loop to Inspect",
            loop_names,
            index=loop_names.index(st.session_state.selected_loop_name) if st.session_state.selected_loop_name in loop_names else 0,
        )
        st.session_state.selected_loop_name = selected_loop

        ldf = active_loops_dict[selected_loop]
        res = evaluate_loop_diagnostics(
            ldf,
            ruleset=active_ruleset,
            harris_delay=harris_delay_input,
        )

        # Plain-English Headline
        headline = generate_plain_headline(res)
        status_code = res["overall_status"]
        if status_code == "green":
            st.success(f"**Health Assessment:** {headline}")
        elif status_code == "yellow":
            st.warning(f"**Health Assessment:** {headline}")
        else:
            st.error(f"**Health Assessment:** {headline}")

        st.caption("Thresholds were calibrated on simulated loops and one TEP loop, and may need recalibrating for other plants.")

        # True fault label if revealed
        if (reveal_faults or active_ruleset == "tep") and "true_fault_type" in ldf.columns:
            st.caption(f"Ground truth demonstration label: **{ldf['true_fault_type'].iloc[0]}**")

        if "Fault 4" in selected_loop:
            st.info(
                "Physical Context: Fault 4 is an inlet cooling temperature step disturbance. "
                "The feedback PI controller successfully absorbed it by shifting valve position OP by +6.9 sigma, "
                "maintaining reactor temperature PV at target (0 sigma deviation). "
                "Because PV tracking remained undisturbed, PV-based error diagnostics show healthy (green). "
                "Detecting disturbances absorbed by control action requires monitoring valve OP shifts, not error metrics alone."
            )

        # Section: What was found and what to do (PART 1)
        st.markdown("### What was found and what to do")
        actions = get_fault_actions(res)

        if not actions:
            st.success("No action needed: All diagnostic checks are operating within normal boundaries.")
        else:
            # Performance flag rule: if Performance is flagged together with another flag,
            # add one sentence explaining that poor control quality is often a side effect of the other problem, so fix that one first.
            other_flags = [k for k, _ in actions if k != "performance"]
            has_perf = any(k == "performance" for k, _ in actions)
            if has_perf and other_flags:
                st.info("Note on priority: Poor control quality is often a side effect of the other problem, so fix that one first.")

            # When stiction and oscillation both fire, show oscillation as a one line secondary note, not a full card.
            has_stict = any(k in ("stiction", "chatter") for k, _ in actions)
            has_osc = any(k == "oscillation" for k, _ in actions)
            if has_stict and has_osc:
                st.info("Secondary observation: Process oscillation was also detected, but valve stiction can cause cycling. Check and fix valve stiction first.")
                actions = [(k, v) for k, v in actions if k != "oscillation"]

            for fault_key, entry in actions:
                card_title = f"Status Details: {entry['title']}" if fault_key == "healthy" else f"Suggested Action: {entry['title']}"
                with st.expander(card_title, expanded=True):
                    st.markdown(f"**What it means:**\n\n{entry['what_it_means']}")
                    if entry.get("jargon_explained"):
                        st.markdown(f"*{entry['jargon_explained']}*")
                    if entry.get("reliability_short"):
                        st.markdown(f"**Reliability:** {entry['reliability_short']}")
                    st.markdown("**Possible causes:**")
                    for cause in entry["possible_causes"]:
                        st.markdown(f"* {cause}")
                    st.markdown("**What to check:**")
                    for check in entry["what_to_check"]:
                        st.markdown(f"* {check}")
                    st.markdown(f"**Risk if ignored:**\n\n{entry['risk_if_ignored']}")
                    if entry.get("reliability_details"):
                        with st.expander("Diagnostic Reliability Details", expanded=False):
                            st.markdown(entry["reliability_details"])
                    st.caption(SUGGESTION_DISCLAIMER)

        st.markdown("### Operational Time-Series")

        # Operational Time Series Plot (Cached PNG)
        ts_chart_bytes = build_drilldown_chart_png("timeseries", selected_loop, data_source, ldf)
        st.image(ts_chart_bytes, use_container_width=True)

        st.markdown("### Detailed Diagnostic Results")

        # 4 Diagnostic Panels
        col1, col2 = st.columns(2)

        # Panel 1: D#1 Setpoint Tracking
        with col1:
            st.markdown("#### 1. Setpoint Tracking (Sensor Drift)")
            st.caption("Checks if the sensor reading is slowly decalibrating and drifting away from the target over time.")
            d1_data = res["d1"]
            d1_flag_str = "FLAGGED (Drift Detected)" if res["d1_flag"] else "OK (Normal Tracking)"
            badge_class = "badge-red" if res["d1_flag"] else "badge-green"

            render_html(f"Status: <span class='{badge_class}'>{d1_flag_str}</span>")
            st.markdown(
                textwrap.dedent(
                    f"""
                    * **Linear Fit R2:** `{d1_data['r2_linear']:.4f}` (Threshold: `>= 0.50`)
                    * **Error Drift Rate:** `{d1_data['error_slope']:.6f}` units per step
                    * **Mean Tracking Offset:** `{d1_data['mean_error']:.3f}` units
                    * **Overall Error RMSE:** `{d1_data['rmse_error']:.3f}` units
                    """
                )
            )

        # Panel 2: D#2 Oscillation Detection
        with col2:
            st.markdown("#### 2. Oscillation & Noise Detection")
            st.caption("Checks if the loop is swinging back and forth rhythmically or jittering with high-frequency noise.")
            d2_data = res["d2"]
            d2_flag_str = "FLAGGED (Oscillation / Noise)" if res["d2_flag"] else "OK (Stable Regulation)"
            badge_class = "badge-red" if res["d2_flag"] else "badge-green"

            render_html(f"Status: <span class='{badge_class}'>{d2_flag_str}</span>")
            st.markdown(
                textwrap.dedent(
                    f"""
                    * **Autocorrelation Score:** `{d2_data['acf_osc_score']:.4f}`
                    * **Dominant Period:** `{d2_data['dominant_period']:.1f}` steps per cycle
                    * **Zero-Crossing Rate:** `{d2_data['zero_crossing_rate']:.2f}` sign changes per 100 steps
                    * **Detrended Oscillation Amplitude:** `{d2_data['osc_amplitude']:.3f}` units
                    """
                )
            )

        col3, col4 = st.columns(2)

        # Panel 3: D#3 Valve Stiction
        with col3:
            st.markdown("#### 3. Valve Stiction Detection")
            st.caption("Checks if the valve sticks in place and suddenly slips forward rather than moving smoothly.")
            d3_data = res["d3"]
            if res.get("chatter_p2"):
                stict_msg = "FLAGGED (Period-2 Actuator Chatter)"
            elif res["d3_flag"]:
                stict_msg = "FLAGGED (Valve Stiction Limit Cycle)"
            else:
                stict_msg = "OK (Smooth Valve Travel)"
            badge_class = "badge-red" if res["d3_flag"] else "badge-green"

            render_html(f"Status: <span class='{badge_class}'>{stict_msg}</span>")
            corr_val = d3_data.get("op_pv_corr_filt", np.nan)
            corr_display = f"`{corr_val:.4f}`" if not np.isnan(corr_val) else "N/A"
            st.markdown(
                textwrap.dedent(
                    f"""
                    * **Filtered OP-PV Correlation:** {corr_display} (Threshold: `< 0.50` stiction gap)
                    * **Oscillation Gate:** `{'Passed' if d3_data.get('oscillating') else 'No limit cycle'}`
                    * **Noise Gate:** `{'Passed' if d3_data.get('noise_excluded') else 'Exceeded noise ceiling'}`
                    * **Period-2 Chatter Flag:** `{'Yes (alternating limit cycle)' if res.get('chatter_p2') else 'No'}`
                    """
                )
            )

        # Panel 4: D#4 Harris Performance Index
        with col4:
            st.markdown("#### 4. Harris Performance Index")
            st.caption("Grades regulatory control quality against the theoretical minimum achievable variance.")
            d4_data = res["d4"]
            d4_flag_str = "FLAGGED (Degraded Regulation)" if res["d4_flag"] else "OK (Adequate Control)"
            badge_class = "badge-red" if res["d4_flag"] else "badge-green"

            render_html(f"Status: <span class='{badge_class}'>{d4_flag_str}</span>")
            ih_val = d4_data.get("harris_index", np.nan)
            ih_display = f"`{ih_val:.4f}`" if not np.isnan(ih_val) else "N/A"
            st.markdown(
                textwrap.dedent(
                    f"""
                    * **Harris Index I_H(d={res['harris_delay']}):** {ih_display} (Scale: 0.0 poor, 1.0 ideal)
                    * **Best AR Model Order:** `{d4_data.get('ar_order', 'N/A')}`
                    * **Total Error Variance:** `{d4_data.get('var_total', 0.0):.4f}`
                    * **Minimum Achievable Variance:** `{d4_data.get('var_mv', 0.0):.4f}`
                    """
                )
            )

        # Stiction Phase Plane Plot (Cached PNG)
        st.markdown("### Valve Movement Phase Plane")
        st.caption("Scatter plot of valve output increment magnitude versus process variable increment magnitude.")

        phase_chart_bytes = build_drilldown_chart_png("phase_plane", selected_loop, data_source, ldf)
        st.image(phase_chart_bytes)

        st.caption(
            "Interpretation: A healthy linear loop exhibits diagonal scatter (valve moves cause immediate process changes). "
            "A sticky valve shows decoupled L-shaped points (valve command ramps while process is stuck, followed by sudden process jump)."
        )

        # Advanced Expander for Threshold Inspection
        with st.expander("Active Thresholds Summary (Read-Only)", expanded=False):
            thresh_dict = get_thresholds_summary(ruleset=active_ruleset, harris_delay=res["harris_delay"])
            for k, v in thresh_dict.items():
                st.markdown(f"* **{k}:** `{v}`")


# -----------------------------------------------------------------------------
# PAGE 3: ABOUT AND LIMITS
# -----------------------------------------------------------------------------
elif nav_choice == "About & Limits":
    st.title("About Loop Guard & Honest Limitations")

    st.markdown("### What is Loop Guard?")
    st.markdown(
        textwrap.dedent(
            """
            Imagine driving a car on cruise control set to 60 mph:
            * If your speedometer slowly decalibrates, the car might speed up to 75 mph while still claiming it is doing 60.
            * If your accelerator pedal gets sticky, the engine lags behind your foot and then suddenly jerks forward.
            * If the computer reacts too aggressively, the car constantly surges faster and slower in an endless wave.
            * If you are driving on a bumpy road, you want a metric telling you how much fuel is being wasted compared to ideal driving.

            In industrial plants (chemical refineries, oil platforms, pharmaceutical facilities), thousands of automated controllers 
            called **PID loops** perform this exact cruise control job. They throttle valves and pumps to keep temperatures, pressures, 
            and flows steady.

            **Loop Guard** acts as an automated, passive health monitor. It analyzes routine operational data already recorded by plant historians 
            (target setpoint, actual sensor reading, valve command) to catch equipment and tuning faults before they lead to emergency shutdowns 
            or spoiled product.
            """
        )
    )

    st.markdown("---")
    st.markdown("### Honest Limitations")
    st.markdown(
        textwrap.dedent(
            """
            These limitations are directly supported by experimental evaluation on the project datasets:

            1. **Simulated Healthy Loops Are Cleaner Than Real Plants:**  
               Synthetic healthy loops had zero measurement noise, making baseline tracking artificially near-zero. On the industrial Tennessee Eastman Process (TEP) benchmark, healthy reactor temperature naturally crosses zero ~52 times per 100 steps with lag-1 autocorrelation of -0.15, proving that real plant loops are noise-dominated.

            2. **Valve Stiction Detection Misses in Heavy Noise and Alarms Under Load Disturbances:**  
               Under heavy sensor noise (noise standard deviation 0.50), stiction recall drops to 50% to 70% because random zero crossings exceed the noise ceiling. When colored load disturbances wander slowly, healthy valves drift to compensate, producing false stiction alarms (4 to 5 out of 5 healthy loops).

            3. **Fault 4 Was Completely Undetected by All PV Diagnostics:**  
               TEP Fault 4 (a step change in cooling water inlet temperature) was missed by every PV-based diagnostic (0/50 runs). Because the closed-loop PI controller successfully absorbed the disturbance by shifting valve position by +6.9 standard deviations, the temperature PV showed no change. Detecting disturbances absorbed by control action requires monitoring valve position shifts, not tracking error alone.

            4. **Harris Index is Coarse and Biased on Finite Data Records:**  
               On 500-sample records, finite-sample estimation bias shifts the calculated Harris Index upward by approximately +0.05 with a sample standard deviation of 0.09. It functions as a relative performance health band rather than an exact regulator index. Furthermore, pure white measurement noise drives the index to 1.0, making pure noise look like ideal control.

            5. **Only One TEP Loop Was Evaluated:**  
               Industrial benchmark evaluation was conducted exclusively on the Reactor Temperature loop (cooling water valve `xmv_10` and temperature `xmeas_9`). Multi-loop plant-wide interactions across the remaining 11 control valves were not evaluated.

            6. **TEP Rules Required Hand Adaptation:**  
               Unmodified synthetic thresholds did not transfer out-of-the-box to the industrial benchmark due to baseline noise shifts. The reported benchmark rules had to be calibrated on development runs 1 to 50 before evaluating report runs 51 to 100.

            7. **No Real Plant Data Evaluated Yet:**  
               Loop Guard has so far been evaluated on synthetic transfer-function simulations and the simulated Tennessee Eastman Process, not on live telemetry from an operating refinery or chemical plant.
            """
        )
    )
