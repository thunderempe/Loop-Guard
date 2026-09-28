"""
Loop Guard - PID Control Loop Fault Diagnostics package.
"""

from src.simulator import (
    FOPDTProcess,
    PIDController,
    SimpleDeadbandValve,
    ValveStictionSJ,
    SimulationResult,
    simulate,
)
from src.faults import (
    inject_sensor_drift,
    inject_valve_stiction_v1,
    inject_valve_stiction_v2,
    inject_oscillatory_tuning,
    inject_noise,
)

__version__ = "0.1.0"
__all__ = [
    "FOPDTProcess",
    "PIDController",
    "SimpleDeadbandValve",
    "ValveStictionSJ",
    "SimulationResult",
    "simulate",
    "inject_sensor_drift",
    "inject_valve_stiction_v1",
    "inject_valve_stiction_v2",
    "inject_oscillatory_tuning",
    "inject_noise",
]
