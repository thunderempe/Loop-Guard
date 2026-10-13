"""
Extract a compact, committed TEP demonstration subset for the Reactor Temperature loop.

Source:
    Tennessee Eastman Process simulation dataset by Rieth et al. (2017),
    Harvard Dataverse: https://doi.org/10.7910/DVN/6C3JR1

Purpose:
    Enables fresh clones to run the Streamlit dashboard TEP demonstration mode
    out of the box without requiring the full ~2 GB raw .RData download.

Subset contents:
    - Reactor Temperature loop signals:
        xmeas_9: Reactor cooling temperature (PV)
        xmv_10:  Reactor cooling water flow valve position (OP)
    - Fault conditions:
        faultNumber 0:  Fault-free baseline (Healthy)
        faultNumber 4:  Cooling water inlet temperature step
        faultNumber 11: Cooling water inlet temperature oscillation
        faultNumber 14: Cooling water valve sticking (stiction)
    - Runs: Simulation runs 51 to 55 (report evaluation split, samples 21 to 500)
    - Output size: < 100 KB gzipped CSV (well under the 1 MB repository budget)
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FF_PATH = ROOT / "scratch" / "tep_cache_ff_train.pkl"
FAULTY_PATH = ROOT / "scratch" / "tep_faulty_train_subset.csv.gz"
OUT_DIR = ROOT / "data" / "tep_demo"
OUT_FILE = OUT_DIR / "tep_reactor_temp_demo.csv.gz"

SELECTED_RUNS = [51, 52, 53, 54, 55]
SELECTED_FAULTS = [4, 11, 14]
COLS = ["faultNumber", "simulationRun", "sample", "xmeas_9", "xmv_10"]


def extract_subset() -> Path:
    if not FF_PATH.exists() or not FAULTY_PATH.exists():
        raise FileNotFoundError(
            f"Source scratch files missing. Run scratch/tep_01_faultfree_structure.py "
            f"and scratch/tep_05_faulty_train_extract.py first."
        )

    print(f"Reading fault-free source: {FF_PATH}")
    df_ff = pd.read_pickle(FF_PATH)

    print(f"Reading faulty source: {FAULTY_PATH}")
    df_faulty = pd.read_csv(FAULTY_PATH)
    df_faulty["sample"] = df_faulty["sample"].astype(int)

    # Filter healthy subset
    sub_ff = df_ff[
        (df_ff["simulationRun"].isin(SELECTED_RUNS)) &
        (df_ff["sample"] >= 21)
    ][COLS].copy()

    # Filter faulty subset
    sub_faulty = df_faulty[
        (df_faulty["faultNumber"].isin(SELECTED_FAULTS)) &
        (df_faulty["simulationRun"].isin(SELECTED_RUNS)) &
        (df_faulty["sample"] >= 21)
    ][COLS].copy()

    combined = pd.concat([sub_ff, sub_faulty], ignore_index=True)
    combined = combined.sort_values(["faultNumber", "simulationRun", "sample"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_FILE, index=False, compression="gzip")

    file_size_bytes = OUT_FILE.stat().st_size
    print(f"Saved: {OUT_FILE}")
    print(f"Total rows: {len(combined):,}")
    print(f"File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)")
    assert file_size_bytes < 1024 * 1024, "Demo file must be under 1 MB"

    return OUT_FILE


if __name__ == "__main__":
    extract_subset()
