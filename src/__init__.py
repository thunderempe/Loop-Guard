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
from src.diagnostics import (
    compute_setpoint_tracking,
    compute_setpoint_tracking_batch,
    compute_oscillation_detection,
    compute_oscillation_detection_batch,
    compute_stiction_detection,
    compute_stiction_detection_batch,
    compute_stiction_detection_v2,
    compute_stiction_detection_v2_batch,
    compute_harris_index,
    compute_harris_index_batch,
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
    "compute_setpoint_tracking",
    "compute_setpoint_tracking_batch",
    "compute_oscillation_detection",
    "compute_oscillation_detection_batch",
    "compute_stiction_detection",
    "compute_stiction_detection_batch",
    "compute_stiction_detection_v2",
    "compute_stiction_detection_v2_batch",
    "compute_harris_index",
    "compute_harris_index_batch",
]
