"""
Loop Guard - Fault Injection Module
====================================
Contains four distinct fault injection functions for PID control loops:
1. inject_sensor_drift: Linear sensor drift adding a progressive bias to PV.
2. inject_valve_stiction_v1: 1-parameter dead-band valve model causing limit cycles.
3. inject_oscillatory_tuning: Overly aggressive PID tuning causing sustained oscillations.
4. inject_noise: Configurable measurement noise injected into PV.

Each function accepts a clean/healthy simulation run and returns a faulty version.
"""

from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.simulator import (
    SimulationResult,
    FOPDTProcess,
    PIDController,
    SimpleDeadbandValve,
    ValveStictionSJ,
    simulate,
)


def _to_sim_result(run: Union[SimulationResult, dict, pd.DataFrame]) -> SimulationResult:
    """Helper to convert input run into a SimulationResult container."""
    if isinstance(run, SimulationResult):
        return run

    if isinstance(run, pd.DataFrame):
        sp = run["SP"].to_numpy(dtype=float)
        pv = run["PV"].to_numpy(dtype=float)
        op = run["OP"].to_numpy(dtype=float)
        t = run["timestep"].to_numpy(dtype=float) if "timestep" in run.columns else np.arange(len(sp), dtype=float)
        return SimulationResult(sp=sp, pv=pv, op=op, time=t)

    if isinstance(run, dict):
        sp = np.asarray(run.get("SP", run.get("sp")), dtype=float)
        pv = np.asarray(run.get("PV", run.get("pv")), dtype=float)
        op = np.asarray(run.get("OP", run.get("op")), dtype=float)
        t = run.get("time", run.get("t"))
        time_arr = np.asarray(t, dtype=float) if t is not None else np.arange(len(sp), dtype=float)
        dt = float(run.get("dt", 1.0))
        process_params = run.get("process_params", {})
        pid_params = run.get("pid_params", {})
        metadata = run.get("metadata", {})
        return SimulationResult(
            sp=sp,
            pv=pv,
            op=op,
            time=time_arr,
            dt=dt,
            process_params=process_params,
            pid_params=pid_params,
            metadata=metadata,
        )

    raise TypeError(f"Unsupported run type: {type(run)}. Expected SimulationResult, dict, or DataFrame.")


def inject_sensor_drift(
    clean_run: Union[SimulationResult, dict, pd.DataFrame],
    drift_rate: float = 0.05,
    start_step: int = 100,
) -> SimulationResult:
    """
    Inject sensor drift fault:
    Re-runs the closed-loop simulation with accumulating sensor drift wired
    into the feedback path. Sensed PV = true_PV + drift is fed to the controller,
    causing OP to respond to erroneous feedback and driving the true PV away
    from the setpoint.

    The returned SimulationResult preserves the TRUE process output in the PV
    column, while OP reflects the controller's active reaction to the sensor bias.

    Parameters
    ----------
    clean_run : SimulationResult, dict, or DataFrame
        The baseline healthy simulation run.
    drift_rate : float, default=0.05
        Rate of sensor drift per second (or timestep).
    start_step : int, default=100
        Timestep index at which the sensor drift begins.

    Returns
    -------
    SimulationResult
        Faulty simulation run with true PV and reacting OP.
    """
    run = _to_sim_result(clean_run)

    # If the clean run has the generative process & PID parameters, simulate closed-loop sensor drift
    if run.process_params and run.pid_params:
        proc = FOPDTProcess(
            K=run.process_params.get("K", 1.0),
            tau=run.process_params.get("tau", 15.0),
            theta=run.process_params.get("theta", 2.0),
            dt=run.dt,
            initial_pv=run.process_params.get("initial_pv", run.pv[0]),
        )
        ctrl = PIDController(
            Kp=run.pid_params.get("Kp", 1.2),
            Ki=run.pid_params.get("Ki", 0.08),
            Kd=run.pid_params.get("Kd", 0.2),
            setpoint=run.sp[0],
            dt=run.dt,
            op_min=run.pid_params.get("op_min", 0.0),
            op_max=run.pid_params.get("op_max", 100.0),
            op_bias=run.pid_params.get("op_bias", 50.0),
        )

        res = simulate(
            process=proc,
            controller=ctrl,
            sp_profile=run.sp,
            dt=run.dt,
            sensor_drift_rate=drift_rate,
            sensor_drift_start_step=start_step,
        )
        res.metadata.update({
            "fault_type": "sensor_drift",
            "drift_rate": drift_rate,
            "start_step": start_step,
        })
        return res

    # Fallback if raw signals were passed without model parameters
    pv_faulty = np.copy(run.pv)
    n_steps = len(pv_faulty)

    if start_step < n_steps:
        drift_length = n_steps - start_step
        # Linear drift accumulation over time
        drift = drift_rate * np.arange(drift_length, dtype=float) * run.dt
        pv_faulty[start_step:] += drift

    metadata = dict(run.metadata)
    metadata.update({
        "fault_type": "sensor_drift",
        "drift_rate": drift_rate,
        "start_step": start_step,
    })

    return SimulationResult(
        sp=np.copy(run.sp),
        pv=pv_faulty,
        op=np.copy(run.op),
        time=np.copy(run.time),
        dt=run.dt,
        process_params=dict(run.process_params),
        pid_params=dict(run.pid_params),
        metadata=metadata,
    )


def inject_valve_stiction_v1(
    clean_run: Union[SimulationResult, dict, pd.DataFrame],
    deadband: float = 3.0,
) -> SimulationResult:
    """
    Inject valve stiction fault (simple 1-parameter dead-band model):
    The valve does not move until the controller output changes by more
    than the threshold deadband amount, then moves normally.

    When process and PID parameters are available from the clean run,
    re-runs the simulation with the valve deadband active in the closed
    loop to generate realistic stiction limit cycles. Otherwise, applies
    the deadband transformation directly to the controller output.

    Parameters
    ----------
    clean_run : SimulationResult, dict, or DataFrame
        The baseline healthy simulation run.
    deadband : float, default=3.0
        Threshold change in controller output required to move the valve.

    Returns
    -------
    SimulationResult
        Faulty simulation run exhibiting valve stiction dynamics.
    """
    run = _to_sim_result(clean_run)

    # If the clean run has the generative process & PID parameters, simulate closed-loop stiction
    if run.process_params and run.pid_params:
        proc = FOPDTProcess(
            K=run.process_params.get("K", 1.0),
            tau=run.process_params.get("tau", 15.0),
            theta=run.process_params.get("theta", 2.0),
            dt=run.dt,
            initial_pv=run.process_params.get("initial_pv", run.pv[0]),
        )
        ctrl = PIDController(
            Kp=run.pid_params.get("Kp", 1.2),
            Ki=run.pid_params.get("Ki", 0.08),
            Kd=run.pid_params.get("Kd", 0.2),
            setpoint=run.sp[0],
            dt=run.dt,
            op_min=run.pid_params.get("op_min", 0.0),
            op_max=run.pid_params.get("op_max", 100.0),
            op_bias=run.pid_params.get("op_bias", 50.0),
        )

        res = simulate(
            process=proc,
            controller=ctrl,
            sp_profile=run.sp,
            dt=run.dt,
            valve_deadband=deadband,
        )
        res.metadata.update({
            "fault_type": "valve_stiction",
            "deadband": deadband,
        })
        return res

    # Fallback if raw signals were passed: apply deadband filter directly
    valve = SimpleDeadbandValve(deadband=deadband, initial_pos=run.op[0])
    op_faulty = np.zeros_like(run.op)
    for i, u in enumerate(run.op):
        op_faulty[i] = valve.step(u)

    metadata = dict(run.metadata)
    metadata.update({
        "fault_type": "valve_stiction",
        "deadband": deadband,
    })

    return SimulationResult(
        sp=np.copy(run.sp),
        pv=np.copy(run.pv),
        op=op_faulty,
        time=np.copy(run.time),
        dt=run.dt,
        process_params=dict(run.process_params),
        pid_params=dict(run.pid_params),
        metadata=metadata,
    )


def inject_valve_stiction_v2(
    clean_run: Union[SimulationResult, dict, pd.DataFrame],
    S: float = 3.0,
    J: float = 1.5,
) -> SimulationResult:
    """
    Inject valve stiction fault (Choudhury two-parameter S/J model):
    S is the deadband + stickband threshold that must be overcome before movement.
    J is the sudden slip-jump dynamic release once static friction is broken.

    Re-runs simulate() closed-loop with ValveStictionSJ active between controller
    and process to produce realistic stick-slip limit cycles and stair-stepping.

    Parameters
    ----------
    clean_run : SimulationResult, dict, or DataFrame
        The baseline healthy simulation run.
    S : float, default=3.0
        Stickband + deadband threshold (must be > 0).
    J : float, default=1.5
        Slip jump magnitude upon release (0 <= J <= S).

    Returns
    -------
    SimulationResult
        Faulty simulation run exhibiting Choudhury S/J stiction dynamics.
    """
    run = _to_sim_result(clean_run)

    # If the clean run has generative process & PID parameters, re-simulate closed loop
    if run.process_params and run.pid_params:
        proc = FOPDTProcess(
            K=run.process_params.get("K", 1.0),
            tau=run.process_params.get("tau", 15.0),
            theta=run.process_params.get("theta", 2.0),
            dt=run.dt,
            initial_pv=run.process_params.get("initial_pv", run.pv[0]),
        )
        ctrl = PIDController(
            Kp=run.pid_params.get("Kp", 1.2),
            Ki=run.pid_params.get("Ki", 0.08),
            Kd=run.pid_params.get("Kd", 0.2),
            setpoint=run.sp[0],
            dt=run.dt,
            op_min=run.pid_params.get("op_min", 0.0),
            op_max=run.pid_params.get("op_max", 100.0),
            op_bias=run.pid_params.get("op_bias", 50.0),
        )

        res = simulate(
            process=proc,
            controller=ctrl,
            sp_profile=run.sp,
            dt=run.dt,
            stiction_S=S,
            stiction_J=J,
        )
        res.metadata.update({
            "fault_type": "valve_stiction_v2",
            "stiction_S": S,
            "stiction_J": J,
        })
        return res

    # Fallback if raw signals were passed without model parameters
    valve = ValveStictionSJ(S=S, J=J, initial_pos=run.op[0])
    op_faulty = np.zeros_like(run.op)
    for i, u in enumerate(run.op):
        op_faulty[i] = valve.step(u)

    metadata = dict(run.metadata)
    metadata.update({
        "fault_type": "valve_stiction_v2",
        "stiction_S": S,
        "stiction_J": J,
    })

    return SimulationResult(
        sp=np.copy(run.sp),
        pv=np.copy(run.pv),
        op=op_faulty,
        time=np.copy(run.time),
        dt=run.dt,
        process_params=dict(run.process_params),
        pid_params=dict(run.pid_params),
        metadata=metadata,
    )


def inject_oscillatory_tuning(
    clean_run: Union[SimulationResult, dict, pd.DataFrame],
    kp_factor: float = 3.2,
    ki_factor: float = 2.5,
    custom_kp: Optional[float] = None,
    custom_ki: Optional[float] = None,
) -> SimulationResult:
    """
    Inject oscillatory tuning fault:
    Simulates poor/aggressive PID tuning that causes sustained oscillation
    in the loop by increasing Kp and/or Ki beyond the stability margin.

    Parameters
    ----------
    clean_run : SimulationResult, dict, or DataFrame
        The baseline healthy simulation run.
    kp_factor : float, default=3.2
        Multiplier for proportional gain Kp (making controller overly aggressive).
    ki_factor : float, default=2.5
        Multiplier for integral gain Ki (speeding up integration and destabilizing loop).
    custom_kp : float, optional
        Explicit aggressive Kp value. If provided, overrides kp_factor.
    custom_ki : float, optional
        Explicit aggressive Ki value. If provided, overrides ki_factor.

    Returns
    -------
    SimulationResult
        Faulty simulation run with sustained oscillatory behavior.
    """
    run = _to_sim_result(clean_run)

    # Base process parameters
    p_params = run.process_params if run.process_params else {"K": 1.0, "tau": 15.0, "theta": 2.0}
    c_params = run.pid_params if run.pid_params else {"Kp": 1.2, "Ki": 0.08, "Kd": 0.2, "op_bias": 50.0}

    proc = FOPDTProcess(
        K=p_params.get("K", 1.0),
        tau=p_params.get("tau", 15.0),
        theta=p_params.get("theta", 2.0),
        dt=run.dt,
        initial_pv=p_params.get("initial_pv", run.pv[0]),
    )

    # Scale controller gains aggressively to trigger sustained oscillations
    base_kp = c_params.get("Kp", 1.2)
    base_ki = c_params.get("Ki", 0.08)

    aggressive_kp = custom_kp if custom_kp is not None else base_kp * kp_factor
    aggressive_ki = custom_ki if custom_ki is not None else base_ki * ki_factor
    kd = c_params.get("Kd", 0.0)

    ctrl = PIDController(
        Kp=aggressive_kp,
        Ki=aggressive_ki,
        Kd=kd,
        setpoint=run.sp[0],
        dt=run.dt,
        op_min=c_params.get("op_min", 0.0),
        op_max=c_params.get("op_max", 100.0),
        op_bias=c_params.get("op_bias", 50.0),
    )

    res = simulate(
        process=proc,
        controller=ctrl,
        sp_profile=run.sp,
        dt=run.dt,
        valve_deadband=0.0,
    )
    res.metadata.update({
        "fault_type": "oscillatory_tuning",
        "aggressive_Kp": aggressive_kp,
        "aggressive_Ki": aggressive_ki,
        "kp_factor": kp_factor,
        "ki_factor": ki_factor,
    })
    return res


def inject_noise(
    clean_run: Union[SimulationResult, dict, pd.DataFrame],
    noise_std: float = 2.0,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Inject measurement noise fault:
    Re-runs the closed-loop simulation with configurable-amplitude random
    Gaussian noise added to the measured PV signal before it reaches the
    controller at each timestep.

    The true underlying process state remains governed solely by process physics
    (driven by valve movement), while the measured feedback signal seen by the
    controller and recorded in the dataset contains the measurement noise jitter.

    Parameters
    ----------
    clean_run : SimulationResult, dict, or DataFrame
        The baseline healthy simulation run.
    noise_std : float, default=2.0
        Standard deviation of Gaussian noise to add to the measured PV.
    seed : int, optional
        Random seed for reproducible noise generation.

    Returns
    -------
    SimulationResult
        Faulty simulation run with noisy PV measurement and responding OP.
    """
    run = _to_sim_result(clean_run)

    # If the clean run has generative process & PID parameters, re-simulate closed loop
    if run.process_params and run.pid_params:
        proc = FOPDTProcess(
            K=run.process_params.get("K", 1.0),
            tau=run.process_params.get("tau", 15.0),
            theta=run.process_params.get("theta", 2.0),
            dt=run.dt,
            initial_pv=run.process_params.get("initial_pv", run.pv[0]),
        )
        ctrl = PIDController(
            Kp=run.pid_params.get("Kp", 1.2),
            Ki=run.pid_params.get("Ki", 0.08),
            Kd=run.pid_params.get("Kd", 0.2),
            setpoint=run.sp[0],
            dt=run.dt,
            op_min=run.pid_params.get("op_min", 0.0),
            op_max=run.pid_params.get("op_max", 100.0),
            op_bias=run.pid_params.get("op_bias", 50.0),
        )

        res = simulate(
            process=proc,
            controller=ctrl,
            sp_profile=run.sp,
            dt=run.dt,
            measurement_noise_std=noise_std,
            noise_seed=seed,
        )
        res.metadata.update({
            "fault_type": "noise",
            "noise_std": noise_std,
            "seed": seed,
        })
        return res

    # Fallback if raw signals were passed without model parameters
    rng = np.random.RandomState(seed)
    pv_noisy = np.copy(run.pv) + rng.normal(loc=0.0, scale=noise_std, size=len(run.pv))

    metadata = dict(run.metadata)
    metadata.update({
        "fault_type": "noise",
        "noise_std": noise_std,
        "seed": seed,
    })

    return SimulationResult(
        sp=np.copy(run.sp),
        pv=pv_noisy,
        op=np.copy(run.op),
        time=np.copy(run.time),
        dt=run.dt,
        process_params=dict(run.process_params),
        pid_params=dict(run.pid_params),
        metadata=metadata,
    )
