"""
Test suite verifying button navigation in Streamlit app.
Confirms clicking Inspect transitions to drill down without StreamlitAPIException.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_inspect_navigation_flow():
    app_path = ROOT / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception, f"Exception on launch: {at.exception}"

    # 1. Navigate to Overview
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception, f"Exception on navigating to Overview: {at.exception}"

    # 2. Inspect healthy loop (Simulated Loop 01)
    btn_01 = at.button(key="btn_Simulated Loop 01")
    assert btn_01 is not None, "Button for Simulated Loop 01 not found"
    btn_01.click().run()
    assert not at.exception, f"Exception on inspecting Simulated Loop 01: {at.exception}"
    assert at.session_state.nav_radio == "Loop Drill-down"
    assert at.session_state.selected_loop_name == "Simulated Loop 01"

    # 3. Return to Overview and inspect stiction loop (Simulated Loop 38)
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    btn_38 = at.button(key="btn_Simulated Loop 38")
    assert btn_38 is not None, "Button for Simulated Loop 38 not found"
    btn_38.click().run()
    assert not at.exception, f"Exception on inspecting Simulated Loop 38: {at.exception}"
    assert at.session_state.nav_radio == "Loop Drill-down"
    assert at.session_state.selected_loop_name == "Simulated Loop 38"

    # 4. Return to Overview and inspect oscillation loop (Simulated Loop 43)
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    btn_43 = at.button(key="btn_Simulated Loop 43")
    assert btn_43 is not None, "Button for Simulated Loop 43 not found"
    btn_43.click().run()
    assert not at.exception, f"Exception on inspecting Simulated Loop 43: {at.exception}"
    assert at.session_state.nav_radio == "Loop Drill-down"
    assert at.session_state.selected_loop_name == "Simulated Loop 43"
