"""
Dataset Generation Script
=========================
Generates simulated PID control loop runs with diverse process conditions
and fault scenarios:
- healthy
- sensor_drift
- valve_stiction_v1 (simple 1-parameter dead-band)
- valve_stiction_v2 (Choudhury 2-parameter S/J model)
- oscillatory_tuning
- noise

Saves output as a labeled dataset to: data/simulated/labeled_loops.csv
Columns: loop_id, timestep, SP, PV, OP, fault_type
"""

import sys
from pathlib import Path

# Add project root to sys.path so src imports reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.simulator import FOPDTProcess, PIDController, simulate
from src.faults import (
    inject_sensor_drift,
    inject_valve_stiction_v1,
    inject_valve_stiction_v2,
    inject_oscillatory_tuning,
    inject_noise,
)


def create_sp_profile(n_steps: int = 500, base_sp: float = 50.0, seed: int = 42) -> np.ndarray:
    """Create a setpoint profile with realistic step transitions."""
    rng = np.random.RandomState(seed)
    sp = np.full(n_steps, base_sp, dtype=float)

    # Add 1 or 2 setpoint step changes
    t1 = int(n_steps * 0.25)
    t2 = int(n_steps * 0.65)
    step1_delta = rng.choice([-10.0, -5.0, 8.0, 12.0, 15.0])
    step2_delta = rng.choice([-12.0, -8.0, 5.0, 10.0])

    sp[t1:t2] = base_sp + step1_delta
    sp[t2:] = base_sp + step1_delta + step2_delta
    return sp


def main():
    print("=" * 70)
    print("Loop Guard - PID Control Loop Dataset Generator")
    print("=" * 70)

    n_steps_per_loop = 500
    loops_per_class = 10
    dt = 1.0

    all_dataframes = []
    current_loop_id = 1

    # Base process variation pool
    process_configs = [
        {"K": 1.0, "tau": 12.0, "theta": 2.0, "base_sp": 50.0, "Kp": 1.2, "Ki": 0.08, "Kd": 0.15},
        {"K": 1.5, "tau": 18.0, "theta": 3.0, "base_sp": 60.0, "Kp": 0.8, "Ki": 0.05, "Kd": 0.20},
        {"K": 0.8, "tau": 10.0, "theta": 1.5, "base_sp": 40.0, "Kp": 1.5, "Ki": 0.12, "Kd": 0.10},
        {"K": 1.2, "tau": 22.0, "theta": 4.0, "base_sp": 55.0, "Kp": 0.9, "Ki": 0.04, "Kd": 0.25},
        {"K": 1.0, "tau": 15.0, "theta": 2.5, "base_sp": 50.0, "Kp": 1.1, "Ki": 0.07, "Kd": 0.18},
    ]

    # -------------------------------------------------------------
    # 1. Healthy Loops
    # -------------------------------------------------------------
    print("\n[1/6] Simulating healthy loops...")
    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=100 + i)

        run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="healthy")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # 2. Sensor Drift Fault Loops
    # -------------------------------------------------------------
    print("[2/6] Simulating sensor drift fault loops...")
    drift_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.05, 0.035, 0.045]
    start_steps = [50, 75, 100, 120, 150, 80, 100, 130, 90, 110]

    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=200 + i)
        clean_run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)

        faulty_run = inject_sensor_drift(
            clean_run=clean_run,
            drift_rate=drift_rates[i],
            start_step=start_steps[i],
        )
        df_loop = faulty_run.to_dataframe(loop_id=current_loop_id, fault_type="sensor_drift")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # 3. Valve Stiction v1 (Deadband) Fault Loops
    # -------------------------------------------------------------
    print("[3/6] Simulating valve stiction v1 (deadband) fault loops...")
    deadbands = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 2.8, 3.8]

    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=300 + i)
        clean_run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)

        faulty_run = inject_valve_stiction_v1(
            clean_run=clean_run,
            deadband=deadbands[i],
        )
        df_loop = faulty_run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v1")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # 4. Valve Stiction v2 (Choudhury S/J) Fault Loops
    # -------------------------------------------------------------
    print("[4/6] Simulating valve stiction v2 (Choudhury S/J) fault loops...")
    S_params = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 3.2, 4.2, 5.5, 6.0]
    J_params = [1.2, 1.5, 1.8, 2.0, 2.5, 2.0, 1.6, 2.8, 3.0, 3.5]

    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=350 + i)
        clean_run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)

        faulty_run = inject_valve_stiction_v2(
            clean_run=clean_run,
            S=S_params[i],
            J=J_params[i],
        )
        df_loop = faulty_run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v2")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # 5. Oscillatory Tuning Fault Loops
    # -------------------------------------------------------------
    print("[5/6] Simulating oscillatory tuning fault loops...")
    kp_factors = [2.8, 3.2, 3.6, 4.0, 4.5, 3.0, 3.5, 3.8, 4.2, 4.8]
    ki_factors = [2.0, 2.5, 2.8, 3.2, 3.5, 2.2, 2.6, 3.0, 3.3, 3.8]

    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=400 + i)
        clean_run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)

        faulty_run = inject_oscillatory_tuning(
            clean_run=clean_run,
            kp_factor=kp_factors[i],
            ki_factor=ki_factors[i],
        )
        df_loop = faulty_run.to_dataframe(loop_id=current_loop_id, fault_type="oscillatory_tuning")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # 6. Noise Fault Loops
    # -------------------------------------------------------------
    print("[6/6] Simulating noise fault loops...")
    noise_stds = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 1.8, 2.2, 2.8]

    for i in range(loops_per_class):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"],
            Ki=cfg["Ki"],
            Kd=cfg["Kd"],
            setpoint=cfg["base_sp"],
            dt=dt,
            op_bias=cfg["base_sp"] / cfg["K"],
        )
        sp_profile = create_sp_profile(n_steps=n_steps_per_loop, base_sp=cfg["base_sp"], seed=500 + i)
        clean_run = simulate(process=proc, controller=ctrl, sp_profile=sp_profile, dt=dt)

        faulty_run = inject_noise(
            clean_run=clean_run,
            noise_std=noise_stds[i],
            seed=600 + i,
        )
        df_loop = faulty_run.to_dataframe(loop_id=current_loop_id, fault_type="noise")
        all_dataframes.append(df_loop)
        current_loop_id += 1

    # -------------------------------------------------------------
    # Concatenate and save dataset
    # -------------------------------------------------------------
    dataset = pd.concat(all_dataframes, ignore_index=True)

    output_dir = PROJECT_ROOT / "data" / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "labeled_loops.csv"

    dataset.to_csv(output_path, index=False)
    print(f"\nSaved labeled dataset to: {output_path}")

    # -------------------------------------------------------------
    # Output Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total rows generated: {len(dataset):,}")
    print(f"Total loops generated: {dataset['loop_id'].nunique()}")
    print(f"Timesteps per loop: {n_steps_per_loop}")
    print(f"Columns: {list(dataset.columns)}")
    print("\nLoop counts by fault type:")
    loop_summary = dataset.groupby("fault_type")["loop_id"].nunique().reset_index()
    loop_summary.columns = ["Fault Type", "Number of Loops"]
    print(loop_summary.to_string(index=False))

    print("\nRow counts by fault type:")
    row_summary = dataset["fault_type"].value_counts().reset_index()
    row_summary.columns = ["Fault Type", "Row Count"]
    print(row_summary.to_string(index=False))

    print("\nFirst 5 rows preview:")
    print(dataset.head())
    print("\nLast 5 rows preview:")
    print(dataset.tail())


if __name__ == "__main__":
    main()
