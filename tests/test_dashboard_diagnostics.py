"""
Pytest suite for Streamlit dashboard diagnostics and demo execution.
"""

import pytest
from app.helpers import (
    load_simulated_demo_data,
    load_tep_demo_data,
    evaluate_loop_diagnostics,
    generate_plain_headline,
    get_card_summary,
    get_fault_actions,
)


def test_simulated_demo_loops_execution():
    sim_loops = load_simulated_demo_data()
    assert len(sim_loops) == 5, f"Expected 5 simulated demo loops, got {len(sim_loops)}"
    for loop_name, ldf in sim_loops.items():
        res = evaluate_loop_diagnostics(ldf, ruleset="simulation")
        assert "overall_status" in res
        assert "flag_count" in res
        headline = generate_plain_headline(res)
        assert headline
        card_summary = get_card_summary(res, loop_name)
        assert card_summary
        assert "possible cause: possible" not in card_summary.lower()
        actions = get_fault_actions(res)
        assert len(actions) > 0
        for _, entry in actions:
            assert entry["title"]
            assert entry["what_it_means"]
            assert entry["reliability_short"]
            assert entry["reliability_details"]


def test_combined_stiction_oscillation_arbitration():
    res_combined = {
        "flag_count": 2,
        "d1_flag": False,
        "d2_flag": True,
        "d2_osc_flag": True,
        "d2_noise_flag": False,
        "d3_flag": True,
        "chatter_p2": False,
        "d4_flag": False,
    }
    actions = get_fault_actions(res_combined)
    assert any(k in ("stiction", "chatter") for k, _ in actions)
    assert any(k == "oscillation" for k, _ in actions)


def test_tep_demo_loops_execution_if_present():
    tep_loops, err = load_tep_demo_data()
    if err is None and tep_loops is not None:
        assert len(tep_loops) > 0
        for loop_name, ldf in tep_loops.items():
            res = evaluate_loop_diagnostics(ldf, ruleset="tep")
            actions = get_fault_actions(res)
            assert len(actions) > 0
