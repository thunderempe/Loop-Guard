"""
Generate Noisy Stress Test Dataset
==================================
Generates data/simulated/labeled_loops_noisy.csv with:
- Two noise levels: LOW (meas=0.15, dist=0.20) and MODERATE (meas=0.50, dist=0.80)
- 5 loops per class per level (6 classes = 30 loops per level, 60 loops total)
- Seed offset +3000
- Baseline measurement noise AND load disturbance applied to ALL classes including healthy
- Column: noise_level ('LOW' or 'MODERATE')
- Target: data/simulated/labeled_loops_noisy.csv (does NOT overwrite labeled_loops.csv)
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

def generate_noisy_dataset():
    n_steps = 500
    dt = 1.0
    current_loop_id = 1

    process_configs = [
        {"K": 1.0, "tau": 12.0, "theta": 2.0, "base_sp": 50.0, "Kp": 1.2, "Ki": 0.08, "Kd": 0.15},
        {"K": 1.5, "tau": 18.0, "theta": 3.0, "base_sp": 60.0, "Kp": 0.8, "Ki": 0.05, "Kd": 0.20},
        {"K": 0.8, "tau": 10.0, "theta": 1.5, "base_sp": 40.0, "Kp": 1.5, "Ki": 0.12, "Kd": 0.10},
        {"K": 1.2, "tau": 22.0, "theta": 4.0, "base_sp": 55.0, "Kp": 0.9, "Ki": 0.04, "Kd": 0.25},
        {"K": 1.0, "tau": 15.0, "theta": 2.5, "base_sp": 50.0, "Kp": 1.1, "Ki": 0.07, "Kd": 0.18},
    ]

    noise_levels = [
        ("LOW", 0.15, 0.20),
        ("MODERATE", 0.50, 0.80),
    ]

    # Fault parameters for 5 loops
    drift_rates = [0.02, 0.03, 0.04, 0.05, 0.06]
    start_steps = [50, 75, 100, 120, 150]
    deadbands = [1.5, 2.0, 2.5, 3.0, 3.5]
    S_params = [2.5, 3.0, 3.5, 4.0, 4.5]
    J_params = [1.2, 1.5, 1.8, 2.0, 2.5]
    kp_factors = [2.8, 3.2, 3.6, 4.0, 4.5]
    ki_factors = [2.0, 2.5, 2.8, 3.2, 3.5]
    noise_fault_stds = [0.8, 1.2, 1.6, 2.0, 2.5]

    all_dfs = []

    seed_base = 3000
    for level_name, meas_std, load_std in noise_levels:
        level_offset = 0 if level_name == "LOW" else 500

        # 1. Healthy
        for i in range(5):
            cfg = process_configs[i % len(process_configs)]
            proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
            ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 10 + i)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                measurement_noise_std=meas_std, noise_seed=seed_base + level_offset + 110 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 210 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="healthy")
            df_loop["noise_level"] = level_name
            all_dfs.append(df_loop)
            current_loop_id += 1

        # 2. Sensor Drift
        for i in range(5):
            cfg = process_configs[i % len(process_configs)]
            proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
            ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 20 + i)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                sensor_drift_rate=drift_rates[i], sensor_drift_start_step=start_steps[i],
                measurement_noise_std=meas_std, noise_seed=seed_base + level_offset + 120 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 220 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="sensor_drift")
            df_loop["noise_level"] = level_name
            all_dfs.append(df_loop)
            current_loop_id += 1

        # 3. Valve Stiction v1
        for i in range(5):
            cfg = process_configs[i % len(process_configs)]
            proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
            ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 30 + i)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                valve_deadband=deadbands[i],
                measurement_noise_std=meas_std, noise_seed=seed_base + level_offset + 130 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 230 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v1")
            df_loop["noise_level"] = level_name
            all_dfs.append(df_loop)
            current_loop_id += 1

        # 4. Valve Stiction v2
        for i in range(5):
            cfg = process_configs[i % len(process_configs)]
            proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
            ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 40 + i)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                stiction_S=S_params[i], stiction_J=J_params[i],
                measurement_noise_std=meas_std, noise_seed=seed_base + level_offset + 140 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 240 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="valve_stiction_v2")
            df_loop["noise_level"] = level_name
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
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 50 + i)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                measurement_noise_std=meas_std, noise_seed=seed_base + level_offset + 150 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 250 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="oscillatory_tuning")
            df_loop["noise_level"] = level_name
            all_dfs.append(df_loop)
            current_loop_id += 1

        # 6. Noise Fault
        for i in range(5):
            cfg = process_configs[i % len(process_configs)]
            proc = FOPDTProcess(K=cfg["K"], tau=cfg["tau"], theta=cfg["theta"], dt=dt, initial_pv=cfg["base_sp"])
            ctrl = PIDController(Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"], setpoint=cfg["base_sp"], dt=dt, op_bias=cfg["base_sp"]/cfg["K"])
            sp = create_sp_profile(n_steps, cfg["base_sp"], seed=seed_base + level_offset + 60 + i)
            # Combine baseline meas noise + fault noise in quadrature
            total_noise_std = np.sqrt(meas_std**2 + noise_fault_stds[i]**2)
            run = simulate(
                process=proc, controller=ctrl, sp_profile=sp, dt=dt,
                measurement_noise_std=total_noise_std, noise_seed=seed_base + level_offset + 160 + i,
                load_disturbance_std=load_std, load_disturbance_seed=seed_base + level_offset + 260 + i,
            )
            df_loop = run.to_dataframe(loop_id=current_loop_id, fault_type="noise")
            df_loop["noise_level"] = level_name
            all_dfs.append(df_loop)
            current_loop_id += 1

    dataset = pd.concat(all_dfs, ignore_index=True)
    out_path = PROJECT_ROOT / "data" / "simulated" / "labeled_loops_noisy.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {dataset.shape[0]:,} rows, {dataset['loop_id'].nunique()} loops to {out_path}")
    return dataset

if __name__ == "__main__":
    generate_noisy_dataset()
