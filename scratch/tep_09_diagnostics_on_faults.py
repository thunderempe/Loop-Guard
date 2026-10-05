"""Inspect diagnostic outputs from scratch/tep_fault_effects.pkl to identify ground truth candidates."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_pickle(ROOT / "scratch" / "tep_fault_effects.pkl")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

print("=== REACTOR TEMPERATURE LOOP ACROSS ALL 20 FAULTS ===")
sub_rt = df[df.loop == "Reactor_Temp"].sort_values("pv_var_ratio", ascending=False)
print(sub_rt[["fault", "pv_z_shift", "pv_var_ratio", "op_z_shift", "op_var_ratio", "d1_r2_mean", "d2_acf_mean", "d2_period_med", "d3_flag_frac", "d4_ih_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print("\n=== HIGH OSCILLATION CANDIDATES (D2 acf > 0.40) ===")
sub_osc = df[df.d2_acf_mean > 0.40].sort_values("d2_acf_mean", ascending=False)
print(sub_osc[["fault", "loop", "d2_acf_mean", "d2_period_med", "pv_var_ratio", "op_var_ratio", "d4_ih_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print("\n=== HIGH D1 DRIFT CANDIDATES (d1_r2_max > 0.30 or d1_r2_mean > 0.15) ===")
sub_d1 = df[df.d1_r2_max > 0.20].sort_values("d1_r2_mean", ascending=False)
print(sub_d1[["fault", "loop", "pv_z_shift", "d1_r2_mean", "d1_r2_max", "op_z_shift"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print("\n=== STICTION FLAG FRAC > 0 on REACTOR_TEMP, SEP_LEVEL, STRIP_LEVEL, STRIP_TEMP ===")
sub_stict = df[(df.d3_flag_frac > 0) & (df.loop.isin(["Reactor_Temp", "Sep_Level", "Strip_Level", "Strip_Temp"]))]
if len(sub_stict) > 0:
    print(sub_stict[["fault", "loop", "d3_flag_frac", "d2_acf_mean", "pv_var_ratio"]].to_string(index=False))
else:
    print("None flagged as stiction on continuous loops!")

print("\n=== HARRIS INDEX DEGRADATION (d4_ih_mean < 0.85 on continuous loops) ===")
sub_d4 = df[(df.d4_ih_mean < 0.85) & (df.loop.isin(["Reactor_Temp", "Sep_Level", "Strip_Level", "Strip_Temp"]))].sort_values("d4_ih_mean")
print(sub_d4[["fault", "loop", "d4_ih_mean", "pv_var_ratio", "op_var_ratio", "d2_acf_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
