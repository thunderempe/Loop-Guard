# Tennessee Eastman Process Demo Data Subset

### Source
* Benchmark: Tennessee Eastman Process simulation dataset published by Rieth et al. (2017).
* Repository: Harvard Dataverse
* Identifier: https://doi.org/10.7910/DVN/6C3JR1

### Description
This directory contains a compact, derived demonstration subset created specifically for Loop Guard. It allows immediate execution of the Streamlit dashboard demonstration mode without downloading the full multi gigabyte raw benchmark files.

### Extracted Signals and Loop
* Process Loop: Reactor Temperature control loop
* Process Variable (PV): `xmeas_9` (Reactor temperature)
* Controller Output (OP): `xmv_10` (Reactor cooling water flow command)
* Setpoint: Grand mean nominal value 120.40

### Extracted Runs and Conditions
* Runs included: Simulation runs 51 to 55 from the report evaluation partition (samples 21 to 500 per run)
* Operating conditions:
  1. Fault free baseline (`faultNumber` 0): Healthy operation
  2. Fault 4: Cooling water inlet temperature step
  3. Fault 11: Cooling water inlet temperature oscillation
  4. Fault 14: Cooling water valve sticking (stiction)

### Note
This dataset is a derived subset intended strictly for demonstration and testing. For full evaluation across all 100 runs and 12 process loops, follow the reproduction instructions in the main repository documentation.
