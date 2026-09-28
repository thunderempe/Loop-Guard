"""
Unit Test: Known-Answer Validation for compute_harris_index
===========================================================
Validates compute_harris_index against exact theoretical closed-form
minimum variance control benchmarks for AR(1) processes:

Theory
------
For an AR(1) error process:
    e(t) = phi * e(t-1) + a(t),  where a(t) ~ i.i.d. N(0, sigma_a^2)
with loop delay d, the impulse response weights are:
    f_k = phi^k,  for k = 0, 1, 2, ...
The minimum achievable variance over horizon d is:
    sigma_MV^2 = sigma_a^2 * sum_{k=0}^{d-1} f_k^2
               = sigma_a^2 * sum_{k=0}^{d-1} phi^(2k)
               = sigma_a^2 * (1 - phi^(2d)) / (1 - phi^2)
Since the total stationary variance of an AR(1) process is:
    sigma_e^2 = sigma_a^2 / (1 - phi^2)
the theoretical Harris Performance Index is exact:
    I_H = sigma_MV^2 / sigma_e^2 = 1 - phi^(2d)

Tests
-----
1. phi=0.9, d=3 at N=20000 samples (Tolerance +/-0.02).
2. phi in [0.5, 0.7, 0.9, 0.95] at d=3 (expected 1 - phi^6).
3. phi=0.9 at d=2 and d=4 (expected 1 - phi^4 and 1 - phi^8).
4. White noise (phi=0.0) at any d (expected 1.0).
5. Realistic length: N=500, phi=0.9, d=3 over 200 random seeds (mean, std, +/-0.1 accuracy).
6. AR order selection check (AIC mode for AR(1) data should be 1).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.diagnostics import compute_harris_index


def generate_synthetic_loop(n_steps: int, phi: float, seed: int = 42, burn_in: int = 500) -> pd.DataFrame:
    """Generate a synthetic loop DataFrame with AR(1) error and constant SP."""
    rng = np.random.RandomState(seed)
    total_len = n_steps + burn_in
    a = rng.randn(total_len)
    e = np.zeros(total_len, dtype=float)
    for t in range(1, total_len):
        e[t] = phi * e[t - 1] + a[t]
    e_valid = e[burn_in:]

    base_sp = 50.0
    return pd.DataFrame({
        "loop_id": 1,
        "timestep": np.arange(n_steps, dtype=int),
        "SP": np.full(n_steps, base_sp, dtype=float),
        "PV": base_sp + e_valid,
        "OP": np.full(n_steps, base_sp, dtype=float),
        "fault_type": "synthetic_ar1",
    })


def main():
    print("=" * 82)
    print("KNOWN-ANSWER UNIT TESTS: compute_harris_index()")
    print("Theoretical benchmark for AR(1): I_H = 1 - phi^(2d)")
    print("=" * 82)

    all_passed = True
    test_results = []

    # -------------------------------------------------------------------------
    # Test 1: phi=0.9, d=3 at N=20000 (Tolerance +/- 0.02)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Asymptotic convergence: phi=0.9, d=3, N=20,000 (tol = +/-0.02)")
    df1 = generate_synthetic_loop(20000, phi=0.9, seed=42)
    res1 = compute_harris_index(df1, delay=3)
    exp1 = 1.0 - 0.9**6  # 0.468559
    meas1 = res1["harris_index"]
    err1 = abs(meas1 - exp1)
    pass1 = err1 <= 0.02
    all_passed = all_passed and pass1

    print(f"  Expected   : {exp1:.4f} (1 - 0.9^6)")
    print(f"  Measured   : {meas1:.4f}")
    print(f"  Abs Error  : {err1:.4f} (tolerance <= 0.0200)")
    print(f"  Order (AIC): {res1['ar_order']}")
    print(f"  Result     : {'PASS' if pass1 else 'FAIL'}")
    test_results.append(("Test 1: phi=0.9, d=3, N=20k", exp1, meas1, err1, "PASS" if pass1 else "FAIL"))

    # -------------------------------------------------------------------------
    # Test 2: phi in [0.5, 0.7, 0.9, 0.95], d=3, N=20,000
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Sensitivity to correlation parameter phi (d=3, N=20,000)")
    print(f"  {'phi':<6} {'Expected':>10} {'Measured':>10} {'Abs Error':>12} {'Order':>8} {'Status':>8}")
    print("  " + "-" * 58)
    for phi in [0.5, 0.7, 0.9, 0.95]:
        df2 = generate_synthetic_loop(20000, phi=phi, seed=42)
        res2 = compute_harris_index(df2, delay=3)
        exp2 = 1.0 - phi**6
        meas2 = res2["harris_index"]
        err2 = abs(meas2 - exp2)
        p2 = err2 <= 0.025
        all_passed = all_passed and p2
        status = "PASS" if p2 else "FAIL"
        print(f"  {phi:<6.2f} {exp2:>10.4f} {meas2:>10.4f} {err2:>12.4f} {res2['ar_order']:>8d} {status:>8}")
        test_results.append((f"Test 2: phi={phi:.2f}, d=3", exp2, meas2, err2, status))

    # -------------------------------------------------------------------------
    # Test 3: phi=0.9, d in [2, 4], N=20,000
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Sensitivity to process delay d (phi=0.9, N=20,000)")
    print(f"  {'delay':<6} {'Expected':>10} {'Measured':>10} {'Abs Error':>12} {'Order':>8} {'Status':>8}")
    print("  " + "-" * 58)
    for d in [2, 4]:
        df3 = generate_synthetic_loop(20000, phi=0.9, seed=42)
        res3 = compute_harris_index(df3, delay=d)
        exp3 = 1.0 - 0.9**(2 * d)
        meas3 = res3["harris_index"]
        err3 = abs(meas3 - exp3)
        p3 = err3 <= 0.025
        all_passed = all_passed and p3
        status = "PASS" if p3 else "FAIL"
        print(f"  d={d:<4d} {exp3:>10.4f} {meas3:>10.4f} {err3:>12.4f} {res3['ar_order']:>8d} {status:>8}")
        test_results.append((f"Test 3: phi=0.9, d={d}", exp3, meas3, err3, status))

    # -------------------------------------------------------------------------
    # Test 4: White noise (phi=0.0), any d, N=20,000
    # -------------------------------------------------------------------------
    print("\n[TEST 4] White noise baseline: phi=0.0, d=3, N=20,000")
    df4 = generate_synthetic_loop(20000, phi=0.0, seed=789)
    res4 = compute_harris_index(df4, delay=3)
    exp4 = 1.0000
    meas4 = res4["harris_index"]
    err4 = abs(meas4 - exp4)
    pass4 = err4 <= 0.005
    all_passed = all_passed and pass4

    print(f"  Expected   : {exp4:.4f}")
    print(f"  Measured   : {meas4:.4f}")
    print(f"  Abs Error  : {err4:.4f}")
    print(f"  Order (AIC): {res4['ar_order']}")
    print(f"  Result     : {'PASS' if pass4 else 'FAIL'}")
    test_results.append(("Test 4: White noise (phi=0)", exp4, meas4, err4, "PASS" if pass4 else "FAIL"))

    # -------------------------------------------------------------------------
    # Test 5: Realistic sample length (N=500, phi=0.9, d=3, 200 random seeds)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Realistic length distribution: N=500, phi=0.9, d=3 (200 seeds)")
    n_seeds = 200
    meas5_list = []
    orders5_list = []
    samples_retained = []

    for s in range(n_seeds):
        df5 = generate_synthetic_loop(500, phi=0.9, seed=s)
        res5 = compute_harris_index(df5, delay=3)
        meas5_list.append(res5["harris_index"])
        orders5_list.append(res5["ar_order"])
        samples_retained.append(res5["n_samples_used"])

    meas5 = np.array(meas5_list)
    exp5 = 1.0 - 0.9**6  # 0.4686
    mean5 = float(np.mean(meas5))
    std5 = float(np.std(meas5))
    within_01 = float(np.mean(np.abs(meas5 - exp5) <= 0.1))
    unique_samples = np.unique(samples_retained)
    masking_ok = (len(unique_samples) == 1 and unique_samples[0] == 470)
    pass5 = abs(mean5 - exp5) <= 0.03 and within_01 >= 0.75 and masking_ok
    all_passed = all_passed and pass5

    print(f"  Theoretical truth : {exp5:.4f}")
    print(f"  Monte Carlo Mean  : {mean5:.4f} (bias = {mean5 - exp5:+.4f})")
    print(f"  Monte Carlo Std   : {std5:.4f}")
    print(f"  Min / Max         : [{np.min(meas5):.4f}, {np.max(meas5):.4f}]")
    print(f"  Within +/-0.10    : {within_01:.1%} (fraction of runs in [{exp5-0.1:.4f}, {exp5+0.1:.4f}])")
    print(f"  Samples retained  : {unique_samples[0]} steps per loop (500 - 30 startup settle = 470)")
    print(f"  Masking on const-SP: {'OK (no false setpoint triggers)' if masking_ok else 'DISTORTED'}")
    print(f"  Result            : {'PASS' if pass5 else 'FAIL'}")
    test_results.append(("Test 5: N=500 Monte Carlo (mean)", exp5, mean5, abs(mean5 - exp5), "PASS" if pass5 else "FAIL"))

    # -------------------------------------------------------------------------
    # Test 6: AR order selected by AIC for AR(1) data
    # -------------------------------------------------------------------------
    print("\n[TEST 6] AR order chosen by AIC for pure AR(1) data (N=500, 200 seeds)")
    order_counts = pd.Series(orders5_list).value_counts().sort_index()
    mode_order = int(order_counts.idxmax())
    frac_order1 = float(order_counts.get(1, 0) / n_seeds)
    frac_order1_to_3 = float(sum(order_counts.get(k, 0) for k in [1, 2, 3]) / n_seeds)
    pass6 = (mode_order == 1) and (frac_order1 >= 0.50)
    all_passed = all_passed and pass6

    print(f"  Mode order chosen : {mode_order} (selected in {frac_order1:.1%} of runs)")
    print(f"  Orders 1, 2, or 3 : {frac_order1_to_3:.1%} of runs")
    print(f"  Order frequency breakdown:")
    for order_val, cnt in order_counts.items():
        bar = "#" * int(cnt // 3)
        print(f"    p={order_val:<2d}: {cnt:>3d} runs ({cnt / n_seeds:>5.1%})  {bar}")
    print(f"  Result            : {'PASS' if pass6 else 'FAIL'}")
    test_results.append(("Test 6: AIC Order Selection (mode=1)", 1.0, float(mode_order), float(abs(mode_order - 1)), "PASS" if pass6 else "FAIL"))

    # -------------------------------------------------------------------------
    # Test 7: Segmented path & transient blanking (N=500, d=3, 200 seeds)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Segmented regulatory path & transient blanking (N=500, phi=0.9, d=3, 200 seeds)")
    exp7 = 1.0 - 0.9**6  # 0.4686

    def gen_segmented_loop(n_steps=500, phi=0.9, seed=42, burn_in=500, add_transient=False):
        rng = np.random.RandomState(seed)
        tot = n_steps + burn_in
        a = rng.randn(tot)
        e = np.zeros(tot, dtype=float)
        for t in range(1, tot):
            e[t] = phi * e[t - 1] + a[t]
        e_valid = e[burn_in:]

        sp = np.full(n_steps, 50.0, dtype=float)
        sp[125:325] = 60.0
        sp[325:] = 55.0

        pv = sp + e_valid
        if add_transient:
            trans = np.zeros(n_steps, dtype=float)
            for t in range(125, n_steps):
                trans[t] += 5.0 * np.exp(-(t - 125) / 15.0)
            for t in range(325, n_steps):
                trans[t] += -5.0 * np.exp(-(t - 325) / 15.0)
            pv = pv + trans

        return pd.DataFrame({
            "loop_id": 1,
            "timestep": np.arange(n_steps, dtype=int),
            "SP": sp,
            "PV": pv,
            "OP": np.full(n_steps, 50.0, dtype=float),
            "fault_type": "segmented_ar1",
        })

    meas7_no_list = []
    meas7_tr_list = []
    samples7_list = []

    for s in range(n_seeds):
        df_no = gen_segmented_loop(500, phi=0.9, seed=s, add_transient=False)
        r_no = compute_harris_index(df_no, delay=3)
        meas7_no_list.append(r_no["harris_index"])
        samples7_list.append(r_no["n_samples_used"])

        df_tr = gen_segmented_loop(500, phi=0.9, seed=s, add_transient=True)
        r_tr = compute_harris_index(df_tr, delay=3)
        meas7_tr_list.append(r_tr["harris_index"])

    meas7_no = np.array(meas7_no_list)
    meas7_tr = np.array(meas7_tr_list)
    mean7_no, std7_no = float(np.mean(meas7_no)), float(np.std(meas7_no))
    mean7_tr, std7_tr = float(np.mean(meas7_tr)), float(np.std(meas7_tr))
    frac7_no = float(np.mean(np.abs(meas7_no - exp7) <= 0.1))
    frac7_tr = float(np.mean(np.abs(meas7_tr - exp7) <= 0.1))

    diff_transient = abs(mean7_tr - mean7_no)
    pass7_no = abs(mean7_no - exp7) <= 0.06
    pass7_tr = diff_transient <= 0.01
    pass7 = pass7_no and pass7_tr
    all_passed = all_passed and pass7

    print(f"  Theoretical truth         : {exp7:.4f}")
    print(f"  No Transient: Mean={mean7_no:.4f} (bias={mean7_no - exp7:+.4f}), Std={std7_no:.4f}, Within +/-0.10: {frac7_no:.1%}")
    print(f"  With Transient: Mean={mean7_tr:.4f} (bias={mean7_tr - exp7:+.4f}), Std={std7_tr:.4f}, Within +/-0.10: {frac7_tr:.1%}")
    print(f"  Difference due to transient: {diff_transient:.6f} (matches no-transient: {'YES' if diff_transient <= 0.01 else 'NO'})")
    print(f"  Samples retained per loop : {np.unique(samples7_list)[0]} steps (500 - 30 startup - 2x65 step settles = 340)")
    print(f"  Comparison vs Test 5 (470 steps, const-SP):")
    print(f"    Test 5 (const SP, 470 steps): Mean={mean5:.4f}, Std={std5:.4f}, +/-0.10 fraction={within_01:.1%}")
    print(f"    Test 7 (segmented, 340 steps): Mean={mean7_no:.4f}, Std={std7_no:.4f}, +/-0.10 fraction={frac7_no:.1%}")
    print(f"    Result is slightly noisier due to fewer unmasked samples (340 vs 470), as theoretically expected.")
    print(f"  Result                    : {'PASS' if pass7 else 'FAIL'}")
    test_results.append(("Test 7a: Segmented no transient", exp7, mean7_no, abs(mean7_no - exp7), "PASS" if pass7_no else "FAIL"))
    test_results.append(("Test 7b: Segmented with transient", exp7, mean7_tr, abs(mean7_tr - exp7), "PASS" if pass7_tr else "FAIL"))

    # -------------------------------------------------------------------------
    # Test 8: Higher order AR(2) errors: a1=1.2, a2=-0.5 at d=3 and d=5
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Higher AR order: AR(2) with coefficients (1.2, -0.5)")
    a1, a2 = 1.2, -0.5
    f_weights = [1.0, a1]
    for k in range(2, 250):
        f_weights.append(a1 * f_weights[-1] + a2 * f_weights[-2])
    f_arr = np.array(f_weights)

    sum_sq_inf = float(np.sum(f_arr[:200]**2))
    exp8_d3 = float(np.sum(f_arr[:3]**2) / sum_sq_inf)  # ~0.8974
    exp8_d5 = float(np.sum(f_arr[:5]**2) / sum_sq_inf)  # ~0.9799

    def gen_ar2_loop(n_steps, a1=1.2, a2=-0.5, seed=42, burn_in=500):
        rng = np.random.RandomState(seed)
        tot = n_steps + burn_in
        a = rng.randn(tot)
        e = np.zeros(tot, dtype=float)
        for t in range(2, tot):
            e[t] = a1 * e[t - 1] + a2 * e[t - 2] + a[t]
        e_valid = e[burn_in:]
        return pd.DataFrame({
            "loop_id": 1,
            "timestep": np.arange(n_steps, dtype=int),
            "SP": np.full(n_steps, 50.0, dtype=float),
            "PV": 50.0 + e_valid,
            "OP": np.full(n_steps, 50.0, dtype=float),
            "fault_type": "synthetic_ar2",
        })

    # Asymptotic N=20000 tests
    df8_20k = gen_ar2_loop(20000, a1=a1, a2=a2, seed=42)
    res8_d3_20k = compute_harris_index(df8_20k, delay=3)
    res8_d5_20k = compute_harris_index(df8_20k, delay=5)
    meas8_d3_20k = res8_d3_20k["harris_index"]
    meas8_d5_20k = res8_d5_20k["harris_index"]
    err8_d3_20k = abs(meas8_d3_20k - exp8_d3)
    err8_d5_20k = abs(meas8_d5_20k - exp8_d5)
    pass8_d3_20k = err8_d3_20k <= 0.02
    pass8_d5_20k = err8_d5_20k <= 0.02
    all_passed = all_passed and pass8_d3_20k and pass8_d5_20k

    print(f"  Asymptotic (N=20,000):")
    print(f"    d=3: Expected={exp8_d3:.4f}, Measured={meas8_d3_20k:.4f}, Abs Error={err8_d3_20k:.4f} (tol <= 0.02) -> {'PASS' if pass8_d3_20k else 'FAIL'}")
    print(f"    d=5: Expected={exp8_d5:.4f}, Measured={meas8_d5_20k:.4f}, Abs Error={err8_d5_20k:.4f} (tol <= 0.02) -> {'PASS' if pass8_d5_20k else 'FAIL'}")
    test_results.append(("Test 8a: AR(2), N=20k, d=3", exp8_d3, meas8_d3_20k, err8_d3_20k, "PASS" if pass8_d3_20k else "FAIL"))
    test_results.append(("Test 8b: AR(2), N=20k, d=5", exp8_d5, meas8_d5_20k, err8_d5_20k, "PASS" if pass8_d5_20k else "FAIL"))

    # Finite sample N=500 over 200 seeds
    meas8_d3_500_list = []
    meas8_d5_500_list = []
    for s in range(n_seeds):
        df8_500 = gen_ar2_loop(500, a1=a1, a2=a2, seed=s)
        meas8_d3_500_list.append(compute_harris_index(df8_500, delay=3)["harris_index"])
        meas8_d5_500_list.append(compute_harris_index(df8_500, delay=5)["harris_index"])

    meas8_d3_500 = np.array(meas8_d3_500_list)
    meas8_d5_500 = np.array(meas8_d5_500_list)
    mean8_d3, std8_d3 = float(np.mean(meas8_d3_500)), float(np.std(meas8_d3_500))
    mean8_d5, std8_d5 = float(np.mean(meas8_d5_500)), float(np.std(meas8_d5_500))
    pass8_d3_500 = abs(mean8_d3 - exp8_d3) <= 0.03
    pass8_d5_500 = abs(mean8_d5 - exp8_d5) <= 0.03
    all_passed = all_passed and pass8_d3_500 and pass8_d5_500

    print(f"  Finite sample (N=500, 200 seeds):")
    print(f"    d=3: Expected={exp8_d3:.4f}, Mean={mean8_d3:.4f} (bias={mean8_d3 - exp8_d3:+.4f}), Std={std8_d3:.4f} -> {'PASS' if pass8_d3_500 else 'FAIL'}")
    print(f"    d=5: Expected={exp8_d5:.4f}, Mean={mean8_d5:.4f} (bias={mean8_d5 - exp8_d5:+.4f}), Std={std8_d5:.4f} -> {'PASS' if pass8_d5_500 else 'FAIL'}")
    test_results.append(("Test 8c: AR(2), N=500, d=3 (mean)", exp8_d3, mean8_d3, abs(mean8_d3 - exp8_d3), "PASS" if pass8_d3_500 else "FAIL"))
    test_results.append(("Test 8d: AR(2), N=500, d=5 (mean)", exp8_d5, mean8_d5, abs(mean8_d5 - exp8_d5), "PASS" if pass8_d5_500 else "FAIL"))

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("PASS / FAIL SUMMARY TABLE")
    print("=" * 82)
    print(f"  {'Test Name':<38} {'Expected':>10} {'Measured':>10} {'Abs Error':>10} {'Status':>8}")
    print("  " + "-" * 80)
    for name, exp_v, meas_v, err_v, stat in test_results:
        print(f"  {name:<38} {exp_v:>10.4f} {meas_v:>10.4f} {err_v:>10.4f} {stat:>8}")

    print("\n" + "=" * 82)
    print(f"OVERALL STATUS: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
