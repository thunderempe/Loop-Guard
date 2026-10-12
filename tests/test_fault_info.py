"""
Pytest suite for app/fault_info.py and helpers.py fault routing.
Verifies key completeness, absence of dashes, and diagnostic mapping.
"""

import pytest
from app.fault_info import FAULT_KNOWLEDGE_BASE
from app.helpers import get_fault_actions

REQUIRED_KEYS = [
    "title",
    "what_it_means",
    "jargon_explained",
    "reliability_short",
    "reliability_details",
    "possible_causes",
    "what_to_check",
    "risk_if_ignored",
]

CORE_ENTRIES = [
    "healthy",
    "sensor_drift",
    "valve_stiction",
    "valve_chatter",
    "oscillation",
    "noise",
    "performance",
]


def test_core_entries_completeness():
    for name in CORE_ENTRIES:
        assert name in FAULT_KNOWLEDGE_BASE, f"Missing core entry: {name}"
        entry = FAULT_KNOWLEDGE_BASE[name]
        for key in REQUIRED_KEYS:
            assert key in entry, f"Entry {name} missing key: {key}"
            assert entry[key] is not None, f"Entry {name} key {key} is None"
            if isinstance(entry[key], list):
                assert len(entry[key]) > 0, f"Entry {name} list {key} is empty"
            elif isinstance(entry[key], str):
                assert len(entry[key].strip()) > 0, f"Entry {name} str {key} is empty"


def test_absence_of_dashes():
    dash_characters = ["-", "—", "–"]
    dash_failures = []
    for name, entry in FAULT_KNOWLEDGE_BASE.items():
        for k, v in entry.items():
            if isinstance(v, str):
                for d in dash_characters:
                    if d in v:
                        dash_failures.append(f"Entry {name}[{k}] contains '{d}'")
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    if isinstance(item, str):
                        for d in dash_characters:
                            if d in item:
                                dash_failures.append(f"Entry {name}[{k}][{idx}] contains '{d}'")
    assert not dash_failures, f"Dashes detected: {dash_failures}"


def test_reliability_disclaimer_wording():
    disclaimer_phrase = "tested on real plant data"
    for name in CORE_ENTRIES:
        short_text = FAULT_KNOWLEDGE_BASE[name]["reliability_short"]
        details_text = FAULT_KNOWLEDGE_BASE[name]["reliability_details"]
        assert disclaimer_phrase in short_text, f"{name} short reliability missing disclaimer"
        assert disclaimer_phrase in details_text, f"{name} detailed reliability missing disclaimer"


def test_action_routing_healthy():
    res = {"flag_count": 0, "d1_flag": False, "d2_flag": False, "d3_flag": False, "d4_flag": False}
    actions = get_fault_actions(res)
    assert len(actions) == 1
    assert actions[0][0] == "healthy"


def test_action_routing_individual_flags():
    cases = [
        ("drift", {"flag_count": 1, "d1_flag": True, "d2_flag": False, "d3_flag": False, "d4_flag": False}),
        ("stiction", {"flag_count": 1, "d1_flag": False, "d2_flag": False, "d3_flag": True, "chatter_p2": False, "d4_flag": False}),
        ("chatter", {"flag_count": 1, "d1_flag": False, "d2_flag": False, "d3_flag": True, "chatter_p2": True, "d4_flag": False}),
        ("noise", {"flag_count": 1, "d1_flag": False, "d2_flag": True, "d2_noise_flag": True, "d2_osc_flag": False, "d3_flag": False, "d4_flag": False}),
        ("oscillation", {"flag_count": 1, "d1_flag": False, "d2_flag": True, "d2_noise_flag": False, "d2_osc_flag": True, "d3_flag": False, "d4_flag": False}),
        ("performance", {"flag_count": 1, "d1_flag": False, "d2_flag": False, "d3_flag": False, "d4_flag": True}),
    ]
    for expected_key, res in cases:
        actions = get_fault_actions(res)
        assert any(k == expected_key for k, _ in actions), f"Expected {expected_key} in {actions}"
