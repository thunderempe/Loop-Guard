"""
Generate Disturbance-Dominated Dataset
======================================
Builds data/simulated/labeled_loops_disturb.csv:
  - 30 loops (5 per class, 6 classes)
  - Seed base: 4000
  - Small measurement noise: 0.05
  - Colored load disturbance: AR(1) phi=0.9, innovation std=0.15 (at process input)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FOPDTProcess, PIDController, simulate

def create_sp_profile(n_steps: int = 500, base_sp: float = 50.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sp = np.full(n_steps, base_sp, dtype=float)
    t1 = int(n_steps * 0.25)
    t2 = int(n_steps * 0.65)
    step1_delta = rng.choice([-10.0, -5.0, 8.0, 12.0, 15.0])
    step2_delta = rng.choice([-12.0, -8.0, 5.0, 10.0])
    sp[t1:t2] = base_sp + step1_delta
    sp[t2:] = base_sp + step1_delta + step2_delta
    return sp

def generate_disturb_dataset():
    n_steps = 500
    dt = 1.0
    seed_base = 4000
    meas_std = 0.05
    dist_std = 0.15
    dist_phi = 0.9

    process_configs = [
        {"K": 1.0, "tau": 12.0, "theta": 2.0, "base_sp": 50.0, "Kp": 1.2, "Ki": 0.08, "Kd": 0.15},
        {"K": 1.5, "tau": 18.0, "theta": 3.0, "base_sp": 60.0, "Kp": 0.8, "Ki": 0.05, "Kd": 0.20},
        {"K": 0.8, "tau": 10.0, "theta": 1.5, "base_sp": 40.0, "Kp": 1.5, "Ki": 0.12, "Kd": 0.10},
        {"K": 1.2, "tau": 22.0, "theta": 4.0, "base_sp": 55.0, "Kp": 0.9, "Ki": 0.04, "Kd": 0.25},
        {"K": 1.0, "tau": 15.0, "theta": 2.5, "base_sp": 50.0, "Kp": 1.1, "Ki": 0.07, "Kd": 0.18},
    ]

    drift_rates = [0.02, 0.03, 0.04, 0.05, 0.06]
    start_steps = [50, 75, 100, 120, 150]
    deadbands = [1.5, 2.0, 2.5, 3.0, 3.5]
    S_params = [2.5, 3.0, 3.5, 4.0, 4.5]
    J_params = [1.2, 1.5, 1.8, 2.0, 2.5]
    kp_factors = [2.8, 3.2, 3.6, 4.0, 4.5]
    ki_factors = [2.0, 2.5, 2.8, 3.2, 3.5]
    noise_fault_stds = [0.8, 1.2, 1.6, 2.0, 2.5]

    all_dfs = []
    current_loop_id = 1

    # 1. Healthy
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 10 + i)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            measurement_noise_std=meas_std, noise_seed=seed_base + 110 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 210 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="healthy")
        all_dfs.append(df_loop)
        current_loop_id += 1

    # 2. Sensor Drift
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 20 + i)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            sensor_drift_rate=drift_rates[i], sensor_drift_start_step=start_steps[i],
            measurement_noise_std=meas_std, noise_seed=seed_base + 120 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 220 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="sensor_drift")
        all_dfs.append(df_loop)
        current_loop_id += 1

    # 3. Valve Stiction v1
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 30 + i)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            valve_deadband=deadbands[i],
            measurement_noise_std=meas_std, noise_seed=seed_base + 130 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 230 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v1")
        all_dfs.append(df_loop)
        current_loop_id += 1

    # 4. Valve Stiction v2
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 40 + i)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            stiction_S=S_params[i], stiction_J=J_params[i],
            measurement_noise_std=meas_std, noise_seed=seed_base + 140 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 240 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v2")
        all_dfs.append(df_loop)
        current_loop_id += 1

    # 5. Oscillatory Tuning
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(
            Kp=cfg["Kp"] * kp_factors[i], Ki=cfg["Ki"] * ki_factors[i], Kd=cfg["Kd"],
            setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"]
        )
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 50 + i)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            measurement_noise_std=meas_std, noise_seed=seed_base + 150 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 250 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="oscillatory_tuning")
        all_dfs.append(df_loop)
        current_loop_id += 1

    # 6. Noise Fault
    for i in range(5):
        cfg = process_configs[i % len(process_configs)]
        proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
        ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
        sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + 60 + i)
        total_noise_std = np.sqrt(meas_std**2 + noise_fault_stds[i]**2)
        run = simulate(
            process=proc, controller=ctrl, sp_profile=sp, dt=dt,
            measurement_noise_std=total_noise_std, noise_seed=seed_base + 160 + i,
            load_disturbance_std=dist_std, load_disturbance_seed=seed_base + 260 + i,
            load_disturbance_phi=dist_phi,
        )
        df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="noise")
        all_dfs.append(df_loop)
        current_loop_id += 1

    dataset = pd.concat(all_dfs, ignore_index=True)
    out_path = PROJECT_ROOT / "data" / "simulated" / "labeled_loops_disturb.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {dataset.shape[0]:,} rows, {dataset['loop_id'].nunique()} loops to {out_path}")
    return dataset

if __name__ == "__main__":
    generate_disturb_dataset()
