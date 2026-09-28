"""
Sanity check script for Loop Guard data assets:
1. Tennessee Eastman Process (TEP) benchmark data loading via pyreadr.
2. Simulated labeled loops dataset (data/simulated/labeled_loops.csv) validation across
   all fault categories: healthy, sensor_drift, valve_stiction_v1, valve_stiction_v2,
   oscillatory_tuning, and noise.
"""

from pathlib import Path
import pyreadr
import pandas as pd


def check_tep_data(project_root: Path) -> bool:
    print("=" * 70)
    print("CHECK 1: Tennessee Eastman Process (TEP) Data")
    print("=" * 70)
    data_file = project_root / "data" / "tep" / "TEP_FaultFree_Training.RData"
    print(f"Loading data from: {data_file}")

    if not data_file.exists():
        print(f"[FAIL] Data file not found at: {data_file}")
        return False

    result = pyreadr.read_r(str(data_file))
    keys = list(result.keys())
    print(f"Objects loaded from RData: {keys}")

    df_name = keys[0]
    df: pd.DataFrame = result[df_name]
    print(f"Extracted object: '{df_name}', shape: {df.shape} (rows: {df.shape[0]:,}, cols: {df.shape[1]})")

    expected_col_count = 55
    expected_first_three = ["faultNumber", "simulationRun", "sample"]

    actual_col_count = df.shape[1]
    actual_first_three = list(df.columns[:3])

    is_count_valid = (actual_col_count == expected_col_count)
    is_first_three_valid = (actual_first_three == expected_first_three)

    print(f"  - Total columns: {actual_col_count} (Expected: {expected_col_count}) -> {'PASS' if is_count_valid else 'FAIL'}")
    print(f"  - First 3 columns: {actual_first_three} -> {'PASS' if is_first_three_valid else 'FAIL'}")

    return is_count_valid and is_first_three_valid


def check_simulated_dataset(project_root: Path) -> bool:
    print("\n" + "=" * 70)
    print("CHECK 2: Simulated Labeled Control Loops Dataset")
    print("=" * 70)
    sim_file = project_root / "data" / "simulated" / "labeled_loops.csv"
    print(f"Loading data from: {sim_file}")

    if not sim_file.exists():
        print(f"[FAIL] Simulated dataset not found at: {sim_file}")
        return False

    df = pd.read_csv(sim_file)
    print(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns across {df['loop_id'].nunique()} loops")

    all_passed = True

    # 1. Column names check
    expected_columns = ["loop_id", "timestep", "SP", "PV", "OP", "fault_type"]
    cols_match = list(df.columns) == expected_columns
    print(f"  - Column structure check: {list(df.columns)} -> {'PASS' if cols_match else 'FAIL'}")
    all_passed = all_passed and cols_match

    # 2. Null values check
    null_count = df.isna().sum().sum()
    print(f"  - Missing/NaN values check: {null_count} NaNs -> {'PASS' if null_count == 0 else 'FAIL'}")
    all_passed = all_passed and (null_count == 0)

    # 3. Fault types check
    expected_faults = {
        "healthy",
        "sensor_drift",
        "valve_stiction_v1",
        "valve_stiction_v2",
        "oscillatory_tuning",
        "noise",
    }
    actual_faults = set(df["fault_type"].unique())
    faults_match = expected_faults.issubset(actual_faults)
    print(f"  - Fault classes coverage check: {sorted(list(actual_faults))} -> {'PASS' if faults_match else 'FAIL'}")
    all_passed = all_passed and faults_match

    # 4. Breakdown report
    print("\n  Row & Loop Count Breakdown by Fault Type:")
    breakdown = df.groupby("fault_type").agg(
        num_loops=("loop_id", "nunique"),
        total_rows=("timestep", "count"),
        min_pv=("PV", "min"),
        max_pv=("PV", "max"),
        min_op=("OP", "min"),
        max_op=("OP", "max"),
    ).reset_index()

    for _, row in breakdown.iterrows():
        print(f"    * {row['fault_type']:<20}: {row['num_loops']:>2} loops, {row['total_rows']:>5} rows | PV: [{row['min_pv']:.1f}, {row['max_pv']:.1f}] | OP: [{row['min_op']:.1f}, {row['max_op']:.1f}]")

    return all_passed


def main():
    project_root = Path(__file__).resolve().parent.parent

    tep_ok = check_tep_data(project_root)
    sim_ok = check_simulated_dataset(project_root)

    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)
    print(f"  - TEP Benchmark Data: {'PASS' if tep_ok else 'FAIL'}")
    print(f"  - Simulated Loops Data: {'PASS' if sim_ok else 'FAIL'}")

    if tep_ok and sim_ok:
        print("\n[SUCCESS] All data assets verified and passed sanity checks!")
    else:
        print("\n[WARNING] Sanity check encountered failures.")
        assert tep_ok and sim_ok, "One or more sanity checks failed."


if __name__ == "__main__":
    main()
