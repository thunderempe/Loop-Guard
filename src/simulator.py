"""
Loop Guard - PID Control Loop Simulator Module
==============================================
Provides:
1. FOPDTProcess: First-Order Plus Dead-Time process model.
2. PIDController: Discrete position-form PID controller with anti-windup.
3. ValveModel: Valve dead-band model for valve stiction simulation.
4. SimulationResult: Container for time-series outputs (SP, PV, OP, time).
5. simulate(): Closed-loop simulation loop function.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd


class FOPDTProcess:
    """
    First-Order Plus Dead-Time (FOPDT) transfer function model:
        G(s) = (K / (tau * s + 1)) * exp(-theta * s)

    In discrete time with sample period dt:
        y[k] = alpha * y[k-1] + K * (1 - alpha) * u[k - delay_steps]
    where:
        alpha = exp(-dt / tau)
        delay_steps = round(theta / dt)
    """

    def __init__(
        self,
        K: float = 1.0,
        tau: float = 10.0,
        theta: float = 2.0,
        dt: float = 1.0,
        initial_pv: float = 0.0,
    ):
        """
        Parameters
        ----------
        K : float
            Process steady-state gain (change in PV / change in OP).
        tau : float
            Process time constant (seconds). Must be > 0.
        theta : float
            Process apparent dead time / time delay (seconds). Must be >= 0.
        dt : float
            Simulation sample time (seconds).
        initial_pv : float
            Initial process variable value.
        """
        if tau <= 0:
            raise ValueError("Time constant tau must be positive.")
        if theta < 0:
            raise ValueError("Dead time theta must be non-negative.")
        if dt <= 0:
            raise ValueError("Sampling interval dt must be positive.")

        self.K = float(K)
        self.tau = float(tau)
        self.theta = float(theta)
        self.dt = float(dt)
        self.initial_pv = float(initial_pv)

        # Discrete-time pole and gain coefficient
        self.alpha = float(np.exp(-self.dt / self.tau))
        self.beta = self.K * (1.0 - self.alpha)

        # Delay buffer length (number of samples to delay)
        self.delay_steps = int(np.round(self.theta / self.dt))

        self.reset(self.initial_pv)

    def reset(self, initial_pv: Optional[float] = None) -> None:
        """Reset internal process state and delay buffer."""
        if initial_pv is not None:
            self.initial_pv = float(initial_pv)

        self.pv = float(self.initial_pv)

        # Initialize delay buffer with the steady-state equivalent input
        # At steady state: PV = K * u  =>  u = PV / K (if K != 0)
        u_init = self.initial_pv / self.K if self.K != 0 else 0.0
        buffer_len = max(1, self.delay_steps + 1)
        self._delay_buffer = [u_init] * buffer_len

    def step(self, u: float) -> float:
        """
        Advance the process by one sample step.

        Parameters
        ----------
        u : float
            Manipulated variable / valve position entering the process.

        Returns
        -------
        float
            Updated process variable (PV).
        """
        # Append current input to the delay buffer
        self._delay_buffer.append(float(u))

        # Retrieve the delayed input u(t - theta)
        u_delayed = self._delay_buffer.pop(0)

        # First-order difference equation update
        self.pv = self.alpha * self.pv + self.beta * u_delayed
        return self.pv


class PIDController:
    """
    Standard discrete position-form PID controller:
        e[k] = SP[k] - PV[k]
        P[k] = Kp * e[k]
        I[k] = I[k-1] + Ki * e[k] * dt
        D[k] = Kd * (e[k] - e[k-1]) / dt
        OP[k] = clip(P[k] + I[k] + D[k] + op_bias, op_min, op_max)

    Includes anti-windup clamping to prevent integrator saturation.
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.1,
        Kd: float = 0.0,
        setpoint: float = 50.0,
        dt: float = 1.0,
        op_min: float = 0.0,
        op_max: float = 100.0,
        op_bias: float = 0.0,
    ):
        """
        Parameters
        ----------
        Kp : float
            Proportional gain.
        Ki : float
            Integral gain (1/sec).
        Kd : float
            Derivative gain (sec).
        setpoint : float
            Default setpoint value.
        dt : float
            Controller sampling interval (sec).
        op_min : float
            Minimum controller output saturation limit (e.g. 0% valve).
        op_max : float
            Maximum controller output saturation limit (e.g. 100% valve).
        op_bias : float
            Manual reset / steady-state controller output offset.
        """
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.setpoint = float(setpoint)
        self.dt = float(dt)
        self.op_min = float(op_min)
        self.op_max = float(op_max)
        self.op_bias = float(op_bias)

        self.reset()

    def reset(self) -> None:
        """Reset internal integrator and previous error history."""
        self.integral = 0.0
        self.prev_error: Optional[float] = None
        self.last_op = self.op_bias

    def step(self, pv: float, sp: Optional[float] = None) -> float:
        """
        Compute the controller output OP for the current sample step.

        Parameters
        ----------
        pv : float
            Current measured process variable.
        sp : float, optional
            Current setpoint. If None, uses default self.setpoint.

        Returns
        -------
        float
            Controller output (OP), bounded within [op_min, op_max].
        """
        if sp is not None:
            self.setpoint = float(sp)

        error = self.setpoint - float(pv)

        # Proportional term
        p_term = self.Kp * error

        # Integral term candidate
        i_term_candidate = self.integral + self.Ki * error * self.dt

        # Derivative term (on error)
        if self.prev_error is None:
            d_term = 0.0
        else:
            d_term = self.Kd * (error - self.prev_error) / self.dt

        # Total unconstrained controller output
        op_unconstrained = p_term + i_term_candidate + d_term + self.op_bias

        # Clamping output within operational limits [op_min, op_max]
        op_clamped = float(np.clip(op_unconstrained, self.op_min, self.op_max))

        # Anti-windup clamping: only accumulate integral if not saturated
        # or if error pushes the controller out of saturation
        is_saturated_high = (op_unconstrained > self.op_max) and (error > 0)
        is_saturated_low = (op_unconstrained < self.op_min) and (error < 0)

        if not (is_saturated_high or is_saturated_low):
            self.integral = i_term_candidate

        # Update history
        self.prev_error = error
        self.last_op = op_clamped
        return op_clamped


class SimpleDeadbandValve:
    """
    Simple one-parameter dead-band valve model:
    The valve position does not move until the controller output (OP)
    changes by more than a specified threshold from the current valve position,
    after which it follows the controller output.
    """

    def __init__(self, deadband: float = 0.0, initial_pos: float = 0.0):
        """
        Parameters
        ----------
        deadband : float
            Threshold change in OP required to overcome static dead-band.
            If deadband <= 0, the valve responds linearly.
        initial_pos : float
            Initial position of the valve stem.
        """
        self.deadband = max(0.0, float(deadband))
        self.position = float(initial_pos)

    def reset(self, initial_pos: float = 0.0) -> None:
        """Reset valve position."""
        self.position = float(initial_pos)

    def step(self, op: float) -> float:
        """
        Update valve position given current controller output.

        Parameters
        ----------
        op : float
            Controller output demand signal.

        Returns
        -------
        float
            Actual valve stem position entering the process.
        """
        op = float(op)
        if self.deadband <= 0.0:
            self.position = op
            return self.position

        delta = op - self.position
        # If change exceeds deadband threshold, move valve
        if abs(delta) >= self.deadband:
            self.position = op
        # Otherwise, the valve sticks at its previous position
        return self.position


class ValveStictionSJ:
    """
    Choudhury two-parameter (S, J) valve stiction model:
    Choudhury, Thornhill, Shah (2005) "Modelling valve stiction", Control Engineering Practice.

    Parameters
    ----------
    S : float
        Stickband + Deadband (apparent stiction threshold). Cumulative change
        in controller output (OP) required to overcome static friction before
        the valve stem breaks free and moves.
    J : float
        Slip-jump. Magnitude of sudden dynamic slip release when static
        friction is overcome (0 <= J <= S).
    initial_pos : float
        Initial position of the valve stem.
    """

    def __init__(self, S: float = 0.0, J: float = 0.0, initial_pos: float = 0.0):
        self.S = max(0.0, float(S))
        self.J = max(0.0, float(J))
        self.reset(initial_pos)

    def reset(self, initial_pos: float = 0.0) -> None:
        """Reset valve position and stiction state history."""
        self.position = float(initial_pos)
        self.x_ss = float(initial_pos)       # Controller output at last stick position
        self.u_prev = float(initial_pos)     # Controller output at previous sample
        self.moving = False                  # True if currently in moving/slip phase
        self.direction = 0                   # Movement direction (+1 or -1)

    def step(self, op: float) -> float:
        """
        Compute actual valve position given controller demand op.

        Parameters
        ----------
        op : float
            Commanded valve position (controller output OP).

        Returns
        -------
        float
            Actual valve stem position entering the process.
        """
        u = float(op)
        if self.S <= 0.0:
            self.position = u
            self.u_prev = u
            return self.position

        delta_u = u - self.u_prev

        if not self.moving:
            # Valve is currently stuck at self.position
            delta_stick = u - self.x_ss
            if abs(delta_stick) > self.S:
                # Static friction overcome! Slip jump occurs
                self.direction = 1 if delta_stick > 0 else -1
                self.moving = True
                excess = abs(delta_stick) - self.S
                jump = self.J + excess
                new_pos = self.position + self.direction * jump
                # Do not overshoot commanded position
                if self.direction > 0:
                    new_pos = min(new_pos, u)
                else:
                    new_pos = max(new_pos, u)
                self.position = new_pos
        else:
            # Valve is currently moving
            if delta_u * self.direction <= 0:
                # Velocity halted or direction reversed -> valve sticks
                self.moving = False
                self.x_ss = self.u_prev
                # Check if reverse motion in this same step already exceeds S
                delta_stick = u - self.x_ss
                if abs(delta_stick) > self.S:
                    self.direction = 1 if delta_stick > 0 else -1
                    self.moving = True
                    excess = abs(delta_stick) - self.S
                    jump = self.J + excess
                    new_pos = self.position + self.direction * jump
                    if self.direction > 0:
                        new_pos = min(new_pos, u)
                    else:
                        new_pos = max(new_pos, u)
                    self.position = new_pos
            else:
                # Continuing in the same direction -> valve moves with input
                new_pos = self.position + delta_u
                if self.direction > 0:
                    new_pos = min(new_pos, u)
                else:
                    new_pos = max(new_pos, u)
                self.position = new_pos

        self.u_prev = u
        return self.position


@dataclass
class SimulationResult:
    """
    Container for simulation run outputs and metadata.
    Supports attribute access (run.sp), dict access (run['SP']),
    and conversion to pandas DataFrame.
    """
    sp: np.ndarray
    pv: np.ndarray
    op: np.ndarray
    time: np.ndarray
    dt: float = 1.0
    process_params: Dict[str, Any] = field(default_factory=dict)
    pid_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        normalized = key.lower()
        if normalized in ("sp", "setpoint"):
            return self.sp
        elif normalized in ("pv", "process_variable"):
            return self.pv
        elif normalized in ("op", "controller_output", "mv"):
            return self.op
        elif normalized in ("time", "t"):
            return self.time
        raise KeyError(f"Invalid signal key '{key}'. Use 'SP', 'PV', 'OP', or 'time'.")

    def to_dataframe(self, loop_id: Union[int, str] = 0, fault_type: str = "healthy") -> pd.DataFrame:
        """Convert simulation time-series into a standardized pandas DataFrame."""
        n_steps = len(self.time)
        return pd.DataFrame({
            "loop_id": loop_id,
            "timestep": np.arange(n_steps, dtype=int),
            "SP": self.sp,
            "PV": self.pv,
            "OP": self.op,
            "fault_type": fault_type,
        })


def simulate(
    process: Optional[FOPDTProcess] = None,
    controller: Optional[PIDController] = None,
    sp_profile: Optional[Union[np.ndarray, list, float]] = None,
    n_steps: int = 500,
    dt: float = 1.0,
    initial_pv: Optional[float] = None,
    valve_deadband: float = 0.0,
    stiction_S: float = 0.0,
    stiction_J: float = 0.0,
    valve_model: Optional[Any] = None,
    sensor_drift_rate: float = 0.0,
    sensor_drift_start_step: int = 0,
    measurement_noise_std: float = 0.0,
    noise_seed: Optional[int] = None,
    load_disturbance_std: float = 0.0,
    load_disturbance_seed: Optional[int] = None,
    load_disturbance_phi: float = 0.0,
) -> SimulationResult:
    """
    Run closed-loop PID simulation for N timesteps.

    Parameters
    ----------
    process : FOPDTProcess, optional
        Process model instance. If None, default FOPDT is instantiated.
    controller : PIDController, optional
        PID controller instance. If None, default controller is instantiated.
    sp_profile : array-like or float, optional
        Setpoint trajectory. If float or None, builds a constant or step profile.
    n_steps : int, default=500
        Number of simulation timesteps (used if sp_profile length is not set).
    dt : float, default=1.0
        Simulation time step (sec).
    initial_pv : float, optional
        Starting process variable value. If None, uses process.initial_pv.
    valve_deadband : float, default=0.0
        Deadband threshold for simple stiction model (v1). 0.0 means ideal valve.
    stiction_S : float, default=0.0
        Choudhury stiction stickband/deadband parameter S (v2).
    stiction_J : float, default=0.0
        Choudhury stiction slip-jump parameter J (v2).
    valve_model : Any, optional
        Explicit valve model instance (e.g. ValveStictionSJ). Overrides S, J, and deadband.
    sensor_drift_rate : float, default=0.0
        Rate of sensor drift per second added to measured PV before feedback.
    sensor_drift_start_step : int, default=0
        Timestep index at which sensor drift begins accumulating.
    measurement_noise_std : float, default=0.0
        Standard deviation of Gaussian sensor measurement noise.
    noise_seed : int, optional
        Random seed for reproducible measurement noise generation.
    load_disturbance_std : float, default=0.0
        Standard deviation of unmeasured stochastic load disturbance entering process input.
    load_disturbance_seed : int, optional
        Random seed for reproducible load disturbance generation.
    load_disturbance_phi : float, default=0.0
        AR(1) coefficient for colored load disturbance. When > 0.0,
        dist[k] = phi * dist[k-1] + w[k], with w[k] ~ N(0, load_disturbance_std^2).

    Returns
    -------
    SimulationResult
        Object containing SP, PV, OP, time arrays and config metadata.
    """
    # 1. Setup Process
    if process is None:
        process = FOPDTProcess(K=1.0, tau=15.0, theta=2.0, dt=dt, initial_pv=50.0)
    else:
        process.dt = dt

    if initial_pv is not None:
        process.reset(initial_pv)
    else:
        process.reset()

    # 2. Setup Controller
    if controller is None:
        controller = PIDController(Kp=1.2, Ki=0.08, Kd=0.2, setpoint=50.0, dt=dt, op_bias=50.0)
    else:
        controller.dt = dt
    controller.reset()

    # 3. Setup Setpoint Profile
    if sp_profile is None:
        # Default profile: starts at 50, steps to 60 at 25% duration, steps to 45 at 65%
        sp_arr = np.full(n_steps, 50.0, dtype=float)
        step1 = int(n_steps * 0.25)
        step2 = int(n_steps * 0.65)
        sp_arr[step1:step2] = 60.0
        sp_arr[step2:] = 45.0
    elif np.isscalar(sp_profile):
        sp_arr = np.full(n_steps, float(sp_profile), dtype=float)
    else:
        sp_arr = np.asarray(sp_profile, dtype=float)
        n_steps = len(sp_arr)

    # 4. Setup Valve Model
    init_valve_pos = controller.op_bias
    if valve_model is not None:
        valve = valve_model
        valve.reset(init_valve_pos)
    elif stiction_S > 0.0:
        valve = ValveStictionSJ(S=stiction_S, J=stiction_J, initial_pos=init_valve_pos)
    else:
        valve = SimpleDeadbandValve(deadband=valve_deadband, initial_pos=init_valve_pos)

    # Noise and Disturbance RNG setup
    noise_rng = np.random.RandomState(noise_seed) if measurement_noise_std > 0.0 else None
    dist_rng = np.random.RandomState(load_disturbance_seed) if load_disturbance_std > 0.0 else None

    # 5. Pre-allocate arrays
    sp_out = np.zeros(n_steps, dtype=float)
    pv_out = np.zeros(n_steps, dtype=float)
    op_out = np.zeros(n_steps, dtype=float)
    time_out = np.arange(n_steps, dtype=float) * dt

    # 6. Simulation Loop
    current_pv = process.pv
    dist_state = 0.0
    for k in range(n_steps):
        current_sp = sp_arr[k]

        # Sensed PV feedback with accumulating sensor drift or measurement noise if active
        measured_pv = current_pv
        if sensor_drift_rate != 0.0 and k >= sensor_drift_start_step:
            measured_pv += sensor_drift_rate * (k - sensor_drift_start_step) * dt
        noise_val = 0.0
        if noise_rng is not None:
            noise_val = float(noise_rng.normal(loc=0.0, scale=measurement_noise_std))
            measured_pv += noise_val

        # Controller calculates OP based on feedback from sensor
        op = controller.step(pv=measured_pv, sp=current_sp)

        # Valve applies deadband or Choudhury S/J stiction to controller demand
        valve_pos = valve.step(op)

        # Process input with optional unmeasured load disturbance
        proc_input = valve_pos
        if dist_rng is not None:
            w = float(dist_rng.normal(loc=0.0, scale=load_disturbance_std))
            if load_disturbance_phi != 0.0:
                dist_state = load_disturbance_phi * dist_state + w
                proc_input += dist_state
            else:
                proc_input += w

        # Process responds to valve position and disturbance
        next_pv = process.step(proc_input)

        # Record signals (records physical PV with measurement noise jitter)
        sp_out[k] = current_sp
        pv_out[k] = current_pv + noise_val
        op_out[k] = op

        # Advance feedback state
        current_pv = next_pv

    return SimulationResult(
        sp=sp_out,
        pv=pv_out,
        op=op_out,
        time=time_out,
        dt=dt,
        process_params={
            "K": process.K,
            "tau": process.tau,
            "theta": process.theta,
            "initial_pv": process.initial_pv,
        },
        pid_params={
            "Kp": controller.Kp,
            "Ki": controller.Ki,
            "Kd": controller.Kd,
            "op_min": controller.op_min,
            "op_max": controller.op_max,
            "op_bias": controller.op_bias,
        },
        metadata={
            "valve_deadband": valve_deadband,
            "stiction_S": stiction_S,
            "stiction_J": stiction_J,
            "measurement_noise_std": measurement_noise_std,
            "n_steps": n_steps,
        },
    )
