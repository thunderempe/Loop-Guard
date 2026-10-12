"""
Loop Guard Fault Knowledge Base
Provides plain language explanations, root causes, technician checks,
and honest diagnostic reliability notes based on empirical testing.
This file contains text data only so user interface code can read from it cleanly.
"""

from typing import Dict, Any

FAULT_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "title": "Healthy Operation",
        "what_it_means": (
            "None of the four automated diagnostic checks raised an alert for this control loop. "
            "Tracking error and valve movement remained within expected normal boundaries during this record."
        ),
        "jargon_explained": (
            "A diagnostic check is an automated calculation that inspects one specific aspect of control behavior."
        ),
        "reliability_short": (
            "No flags means these four specific checks found nothing abnormal, but this is based on simulations and one TEP loop and has not been tested on real plant data."
        ),
        "reliability_details": (
            "In clean simulations, healthy loops produced 0 false alarms on drift and performance. "
            "On the Tennessee Eastman Process benchmark, adapted rules produced 1 false alarm across 50 healthy runs. "
            "However, Fault 4 was not detected by any check. "
            "No flags means these four checks found nothing, not a guarantee that the physical loop is free of problems. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "possible_causes": [
            "Normal regulatory operation within acceptable performance limits.",
        ],
        "what_to_check": [
            "No corrective maintenance is required at this time based on these checks.",
            "Maintain scheduled routine inspections for valve packing lubrication and instrument air filters.",
            "Verify that the loop remains in automated control and that calibration audit logs are current.",
        ],
        "risk_if_ignored": (
            "Assuming a loop will remain permanently healthy without routine checks risks missing gradual mechanical wear or unmeasured baseline shifts."
        ),
    },
    "sensor_drift": {
        "title": "Sensor Drift",
        "what_it_means": (
            "The sensor measurement is slowly pulling away from the target setpoint over time. "
            "This could indicate sensor decalibration or a real slow physical change in process operating conditions."
        ),
        "jargon_explained": (
            "Sensor measurement means the numerical reading from the physical instrument. "
            "Setpoint means the desired target value for the process."
        ),
        "reliability_short": (
            "In simulations, this check separated drift with zero false alarms, but these numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "reliability_details": (
            "The dashboard flags drift when linear fit R2 is at or above 0.50. "
            "In validation benchmarks across 60 simulated loops, all sensor drift loops scored R2 between 0.68 and 0.95 (mean 0.85), while non drift loops stayed below 0.09, separating drift by a margin between 21 and 33 times with 0 false alarms. "
            "The stiction exclusion gate uses R2 below 0.40. "
            "However, this check assumes stationary setpoints, misses sudden step jumps, and cannot distinguish an instrument drift from a genuine slow process change. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Possible sensor probe coating or physical mineral buildup distorting measurement contact.",
            "Possible aging electronic transmitter components losing factory calibration over time.",
            "Possible real slow process change occurring in the plant that the controller is struggling to track.",
        ],
        "possible_causes": [
            "Possible sensor probe coating or physical mineral buildup distorting measurement contact.",
            "Possible aging electronic transmitter components losing factory calibration over time.",
            "Possible real slow process change occurring in the plant that the controller is struggling to track.",
        ],
        "what_to_check": [
            "Confirm with an independent calibrated measurement before attempting any recalibration.",
            "Inspect the sensor probe tip for chemical fouling, product buildup, or physical corrosion.",
            "Reperform zero and span calibration only after confirming the instrument has drifted rather than the process.",
        ],
        "risk_if_ignored": (
            "Product quality will quietly drift outside technical specifications, leading to off spec batches or wasted raw materials."
        ),
    },
    "valve_stiction": {
        "title": "Possible Valve Stiction",
        "what_it_means": (
            "The mechanical valve appears to stick in place and then jump forward suddenly once actuator pressure overcomes friction. "
            "This sticky stop and start motion prevents the valve from making smooth gradual corrections and can induce cycling."
        ),
        "jargon_explained": (
            "Stiction means static friction that prevents mechanical parts from sliding smoothly. "
            "Valve command means the control signal sent to reposition the valve."
        ),
        "reliability_short": (
            "Stiction detection weakens under heavy noise and can falsely flag healthy loops under disturbances, based on tests from simulations and one TEP loop that have not been tested on real plant data."
        ),
        "reliability_details": (
            "In clean simulations, stiction detection achieved 19 out of 20 recall on design sets and 9 out of 10 on fresh seed sets, with 1 false alarm on healthy loop 2. "
            "In low noise, recall was 10 out of 10 on design sets and 9 out of 10 on fresh sets, with 1 false alarm on healthy loop 2. "
            "In moderate noise, recall dropped to 7 out of 10 on design sets and 5 out of 10 on fresh sets. "
            "Under colored load disturbances, healthy loops were falsely flagged as stiction in 5 out of 5 design loops and 4 out of 5 fresh loops due to slow wandering. "
            "On the Tennessee Eastman Process benchmark, standard stiction detection v2 detected 0 out of 50 runs on Fault 14 and 2 out of 50 runs on Fault 11. "
            "Fault 14 was only caught by the period two chatter rule, which was hand set on one loop. "
            "Fault 4 was not detected by any check. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Possible excessive mechanical friction from an overtightened valve stem packing seal.",
            "Possible product residue or dried process fluid coating the valve plug and seat.",
            "Possible worn actuator guides or inadequate instrument air supply pressure.",
        ],
        "possible_causes": [
            "Possible excessive mechanical friction from an overtightened valve stem packing seal.",
            "Possible product residue or dried process fluid coating the valve plug and seat.",
            "Possible worn actuator guides or inadequate instrument air supply pressure.",
        ],
        "technician_checks": [
            "Obtain operator permission, put the loop in manual mode, and follow plant procedure before inspecting or stroking the valve.",
            "Packing adjustment by maintenance only to prevent valve stem leaks or over tightening.",
            "Check actuator instrument air supply pressure at the positioner to verify adequate operating force.",
        ],
        "what_to_check": [
            "Obtain operator permission, put the loop in manual mode, and follow plant procedure before inspecting or stroking the valve.",
            "Packing adjustment by maintenance only to prevent valve stem leaks or over tightening.",
            "Check actuator instrument air supply pressure at the positioner to verify adequate operating force.",
        ],
        "risk_if_ignored": (
            "Sticky valves force continuous process cycling, accelerate actuator wear, and increase product variability."
        ),
    },
    "valve_chatter": {
        "title": "Rapid Valve Chatter",
        "what_it_means": (
            "The valve position is rapidly alternating back and forth at every single measurement step. "
            "This rapid jumping indicates severe mechanical binding or an overly sensitive positioner loop."
        ),
        "jargon_explained": (
            "Actuator chatter means rapid back and forth jumping of the valve stem between adjacent sample steps. "
            "Positioner means the local device that moves the valve stem to match the controller command."
        ),
        "reliability_short": (
            "This period two chatter rule caught all fifty Fault 14 test runs on one TEP benchmark loop, but it was hand set and has not been tested on real plant data."
        ),
        "reliability_details": (
            "The period two chatter rule detected 50 out of 50 runs of Fault 14 on report runs 51 to 100 with 0 false alarms across 50 healthy runs. "
            "However, this rule was hand set specifically for the Tennessee Eastman Process Reactor Temperature loop. "
            "Standard stiction detection detected 0 out of 50 on Fault 14. "
            "Fault 4 was not detected by any check. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Possible severe valve stem friction creating an alternating limit cycle.",
            "Possible digital valve positioner deadband set too tight for rapid process dynamics.",
            "Possible mechanical play or loose linkage between the actuator and positioner arm.",
        ],
        "possible_causes": [
            "Possible severe valve stem friction creating an alternating limit cycle.",
            "Possible digital valve positioner deadband set too tight for rapid process dynamics.",
            "Possible mechanical play or loose linkage between the actuator and positioner arm.",
        ],
        "technician_checks": [
            "Obtain operator permission, put the loop in manual mode, and follow plant procedure before inspecting the valve linkage.",
            "Packing adjustment by maintenance only to prevent valve stem damage.",
            "Inspect digital positioner configuration for error logs, cycle counts, and travel deadband settings.",
        ],
        "what_to_check": [
            "Obtain operator permission, put the loop in manual mode, and follow plant procedure before inspecting the valve linkage.",
            "Packing adjustment by maintenance only to prevent valve stem damage.",
            "Inspect digital positioner configuration for error logs, cycle counts, and travel deadband settings.",
        ],
        "risk_if_ignored": (
            "Extreme chatter can destroy valve packings and internal positioner linkages within days or weeks of continuous operation."
        ),
    },
    "oscillation": {
        "title": "Oscillation detected, cause unclear",
        "what_it_means": (
            "The process variable is repeatedly swinging back and forth across the setpoint target. "
            "This wave pattern suggests aggressive controller tuning, an external cyclic disturbance, or valve stiction."
        ),
        "jargon_explained": (
            "Controller tuning means the mathematical gain and reset settings that decide how forcefully the automatic controller reacts to errors."
        ),
        "reliability_short": (
            "Autocorrelation detects cyclic swings but overlaps between tuning and stiction, based on tests from simulations and one TEP loop that have not been tested on real plant data."
        ),
        "reliability_details": (
            "Sliding window autocorrelation detected 50 out of 50 oscillatory runs on the Tennessee Eastman Process benchmark with 1 false alarm in 50 healthy runs. "
            "However, the autocorrelation score overlaps between tuning and stiction and only the dominant period separates them, and noise degrades the zero crossing rate. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Likely controller gain set too high or reset time set too short for current process dynamics.",
            "Possible external oscillating disturbance entering this loop from upstream equipment.",
            "Possible mechanical valve stiction, which can also induce limit cycle oscillation.",
        ],
        "possible_causes": [
            "Likely controller gain set too high or reset time set too short for current process dynamics.",
            "Possible external oscillating disturbance entering this loop from upstream equipment.",
            "Possible mechanical valve stiction, which can also induce limit cycle oscillation.",
        ],
        "technician_checks": [
            "Check for mechanical valve stiction first, because sticky valves can also cause oscillation.",
            "Inspect upstream and neighboring loops to determine if an external oscillating disturbance is entering this loop.",
            "Lower controller gain or increase integral reset time to soften control response after obtaining operator agreement.",
        ],
        "what_to_check": [
            "Check for mechanical valve stiction first, because sticky valves can also cause oscillation.",
            "Inspect upstream and neighboring loops to determine if an external oscillating disturbance is entering this loop.",
            "Lower controller gain or increase integral reset time to soften control response after obtaining operator agreement.",
        ],
        "risk_if_ignored": (
            "Continuous swings accelerate mechanical valve wear, waste utility energy, and can destabilize downstream equipment."
        ),
    },
    "noise": {
        "title": "High Frequency Measurement Noise",
        "what_it_means": (
            "The measurement signal contains rapid fluctuations rather than genuine process movement. "
            "This rapid jitter can cause excessive valve movement and mechanical wear if the controller reacts to every tiny jump."
        ),
        "jargon_explained": (
            "Signal noise means rapid random fluctuations in measurement readings caused by electrical interference or fluid turbulence. Derivative action means the controller setting that reacts to the rate of error change."
        ),
        "reliability_short": (
            "Zero crossing rate flags high frequency jitter in clean data but breaks down under moderate noise, based on tests from simulations and one TEP loop that have not been tested on real plant data."
        ),
        "reliability_details": (
            "The zero crossing rate is defined in code as the count of sign changes between consecutive samples of the detrended error (error after subtracting the linear regression line), scaled to sign changes per 100 steps. "
            "In clean simulations, a threshold above 30 crossings per 100 steps separated noise cleanly. "
            "In moderate noise simulations, healthy loops naturally crossed zero 22 to 41 times per 100 steps, degrading separation. "
            "On the Tennessee Eastman Process benchmark, healthy reactor temperature has a natural baseline zero crossing rate of about 52 per 100 steps, which trips fixed noise thresholds. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Possible electrical interference or improper grounding on the instrument signal cable.",
            "Possible high fluid turbulence or vortex shedding near the sensor probe tip inside the process line.",
            "Possible failing transmitter electronics or loose terminal screws inside the field junction box.",
        ],
        "possible_causes": [
            "Possible electrical interference or improper grounding on the instrument signal cable.",
            "Possible high fluid turbulence or vortex shedding near the sensor probe tip inside the process line.",
            "Possible failing transmitter electronics or loose terminal screws inside the field junction box.",
        ],
        "technician_checks": [
            "Inspect signal cable shielding to confirm that the shield drain wire is grounded at one end only.",
            "Reduce controller derivative action to prevent the controller from amplifying high frequency measurement jitter.",
            "Increase analog input filtering to smooth jitter, noting that more filtering adds process lag.",
        ],
        "what_to_check": [
            "Inspect signal cable shielding to confirm that the shield drain wire is grounded at one end only.",
            "Reduce controller derivative action to prevent the controller from amplifying high frequency measurement jitter.",
            "Increase analog input filtering to smooth jitter, noting that more filtering adds process lag.",
        ],
        "risk_if_ignored": (
            "Continuous measurement noise can cause rapid valve cycling and premature actuator wear, while masking underlying process changes."
        ),
    },
    "performance": {
        "title": "Degraded Control Performance",
        "what_it_means": (
            "The control loop produces substantially more error variance than the theoretical minimum variance benchmark. "
            "This indicates that tracking variability is higher than necessary for the given physical process delay."
        ),
        "jargon_explained": (
            "Minimum variance means the smallest possible error spread that an ideal theoretical controller could achieve given physical process delay."
        ),
        "reliability_short": (
            "Harris performance index can flag healthy loops under colored load disturbances where scores average 0.38, based on tests from simulations and one TEP loop that have not been tested on real plant data."
        ),
        "reliability_details": (
            "The Harris index calculation passed 8 out of 8 theoretical known answer unit tests. "
            "On segmented records with about 340 usable samples after transient blanking from a 500 step run, finite sample regression bias shifts the estimated index upward by about 0.05 with a standard deviation of about 0.09 (true index 0.47). "
            "On unmasked 500 sample records, the bias is about 0.02 with a standard deviation of about 0.07. "
            "It functions as a coarse relative indicator rather than an exact metric. "
            "Pure white noise drives the score to 1.0, making noise look like ideal control. "
            "Healthy loops averaged about 0.38 under colored load disturbances, which is below the 0.50 flag line, so Diagnostic 4 can flag healthy loops when disturbances are present, and the oscillatory tuning range overlaps healthy. "
            "Fault 4 was not detected by any check. "
            "These numbers come from simulations and one TEP loop and have not been tested on real plant data."
        ),
        "likely_causes": [
            "Likely sluggish controller tuning that reacts too slowly to routine load disturbances.",
            "Possible continuous unmeasured load disturbances buffeting the process.",
            "Possible secondary consequence of an undetected physical problem or uncompensated transport delay.",
        ],
        "possible_causes": [
            "Likely sluggish controller tuning that reacts too slowly to routine load disturbances.",
            "Possible continuous unmeasured load disturbances buffeting the process.",
            "Possible secondary consequence of an undetected physical problem or uncompensated transport delay.",
        ],
        "technician_checks": [
            "Inspect the loop for physical sensor drift or valve stiction first, as poor performance is often a side effect of hardware issues.",
            "Evaluate process settling time after a routine setpoint step to assess whether controller tuning is overly sluggish.",
            "Verify process dead time and time constant estimates against design specifications.",
        ],
        "what_to_check": [
            "Inspect the loop for physical sensor drift or valve stiction first, as poor performance is often a side effect of hardware issues.",
            "Evaluate process settling time after a routine setpoint step to assess whether controller tuning is overly sluggish.",
            "Verify process dead time and time constant estimates against design specifications.",
        ],
        "risk_if_ignored": (
            "Operating with degraded efficiency increases utility energy consumption, widens quality spreads, and lowers plant productivity."
        ),
    },
}

# Aliases for backward compatibility and alternate lookup keys
FAULT_KNOWLEDGE_BASE["drift"] = FAULT_KNOWLEDGE_BASE["sensor_drift"]
FAULT_KNOWLEDGE_BASE["stiction"] = FAULT_KNOWLEDGE_BASE["valve_stiction"]
FAULT_KNOWLEDGE_BASE["chatter"] = FAULT_KNOWLEDGE_BASE["valve_chatter"]
FAULT_KNOWLEDGE_BASE["poor_performance"] = FAULT_KNOWLEDGE_BASE["performance"]
FAULT_KNOWLEDGE_BASE["oscillatory_tuning"] = FAULT_KNOWLEDGE_BASE["oscillation"]
FAULT_KNOWLEDGE_BASE["measurement_noise"] = FAULT_KNOWLEDGE_BASE["noise"]
