"""
Runner to execute all validation scripts and extract exact numbers for README.md
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r'D:\Loop Gaurd')
scripts = [
    'notebooks/02_validate_setpoint_tracking.py',
    'notebooks/03_validate_oscillation_detection.py',
    'notebooks/04_validate_stiction_detection.py',
    'notebooks/05_validate_harris_index.py',
    'notebooks/06_test_harris_known_answer.py',
    'notebooks/08_validate_noisy_stress_test.py',
    'notebooks/10_validate_disturb_dataset.py',
    'notebooks/11_validate_stiction_v2.py',
    'notebooks/13_evaluate_fresh2000.py',
]

with open(PROJECT_ROOT / 'scratch/all_validation_runs.txt', 'w', encoding='utf-8') as out_f:
    for script in scripts:
        print(f"Running {script}...")
        out_f.write(f"\n{'='*80}\nRUNNING: {script}\n{'='*80}\n")
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        out_f.write(f"RETURNCODE: {res.returncode}\n")
        out_f.write("STDOUT:\n")
        out_f.write(res.stdout)
        if res.stderr:
            out_f.write("\nSTDERR:\n")
            out_f.write(res.stderr)
        print(f"Finished {script} (code {res.returncode})")

print("All scripts finished! Results saved to scratch/all_validation_runs.txt")
