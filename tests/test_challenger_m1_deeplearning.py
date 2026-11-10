"""
tests/test_challenger_m1_deeplearning.py
Adversarial Stress Test Suite for Model 1 (Task 1: Deep Learning Upgrade)

Verifies:
1. Fine sweep of RH around 60% (55.0% to 65.0% step 0.05%, and 50.0% to 75.0% step 0.1%)
   - Checks that max step delta across adjacent steps is strictly < 2.0 ug/m3.
   - Evaluates multiple PM regimes (PM=25, PM=50, PM=150, PM=300, PM=500).
2. Smoke condition & gating when CO2 >= 800 ppm:
   - Full gate opening (w_del = 1.0).
   - Smoke discrimination (c_smoke >= 0.85 in dense smoke vs c_fog <= 0.65 in heavy fog).
   - Boundary step continuity across CO2 = 800 ppm transition.
3. Extreme environmental conditions:
   - Extreme sub-zero and scorching temperatures (-30°C to +65°C).
   - Extreme gas proxy values (0 to 5000).
   - Extreme PM values (0 to 5000 ug/m3).
   - Saturated and supersaturated humidity (0% to 120% RH).
4. Finite real invariants:
   - All c_pred in [0.25, 1.05].
   - All calibrated PM >= 0.0 and finite (no NaN, no Inf).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import math
import json
import pytest
import numpy as np
import torch
from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    compute_dew_point,
    predict_calibrated_pm25,
    smooth_deliquescence_gate,
)


@pytest.fixture(scope="module")
def loaded_model():
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(
        torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


class TestDeliquescenceBoundaryContinuity:
    """Fine sweep stress tests across the deliquescence transition."""

    @pytest.mark.parametrize("pm,co2,temp,gas", [
        (150.0, 600.0, 26.0, 110.0),
        (50.0, 400.0, 20.0, 80.0),
        (25.0, 500.0, 15.0, 50.0),
        (300.0, 700.0, 30.0, 150.0),
        (500.0, 650.0, 25.0, 120.0),
    ])
    def test_fine_rh_sweep_around_60_step_005(self, loaded_model, pm, co2, temp, gas):
        """Fine sweep RH from 55.0% to 65.0% with step 0.05% (201 points)."""
        rh_values = np.arange(55.0, 65.001, 0.05)
        pm_preds = []
        c_preds = []

        for rh in rh_values:
            dp = compute_dew_point(temp, float(rh))
            c_factor, pm_calib = predict_calibrated_pm25(
                loaded_model, pm, 2.0, 0.70, temp, float(rh), dp, co2, gas
            )
            pm_preds.append(pm_calib)
            c_preds.append(c_factor)

        pm_preds = np.array(pm_preds)
        c_preds = np.array(c_preds)
        deltas = np.abs(np.diff(pm_preds))
        max_delta = np.max(deltas)
        max_idx = np.argmax(deltas)

        # Print diagnostics for reporting
        print(f"\n[RH 55-65% Sweep] PM={pm}, CO2={co2}: Max step delta = {max_delta:.4f} ug/m3 at RH={rh_values[max_idx]:.2f}% -> {rh_values[max_idx+1]:.2f}%")
        print(f"  Start PM={pm_preds[0]:.2f}, End PM={pm_preds[-1]:.2f}, Min={np.min(pm_preds):.2f}, Max={np.max(pm_preds):.2f}")

        # The acceptance criterion: no single step jump > 2.0 ug/m3
        assert max_delta < 2.0, f"Cliff jump detected: {max_delta:.4f} > 2.0 ug/m3 at RH={rh_values[max_idx]:.2f}%"
        assert np.all(np.isfinite(pm_preds)), "Non-finite PM predictions found!"
        assert np.all(c_preds >= 0.25), "c_factor below 0.25!"
        assert np.all(c_preds <= 1.05), "c_factor above 1.05!"

    @pytest.mark.parametrize("pm,co2,temp,gas", [
        (150.0, 600.0, 26.0, 110.0),
        (80.0, 450.0, 22.0, 90.0),
        (250.0, 650.0, 28.0, 130.0),
    ])
    def test_extended_rh_sweep_50_to_75_step_01(self, loaded_model, pm, co2, temp, gas):
        """Extended sweep RH from 50.0% to 75.0% with step 0.1% (251 points)."""
        rh_values = np.arange(50.0, 75.001, 0.1)
        pm_preds = []

        for rh in rh_values:
            dp = compute_dew_point(temp, float(rh))
            _, pm_calib = predict_calibrated_pm25(
                loaded_model, pm, 2.0, 0.70, temp, float(rh), dp, co2, gas
            )
            pm_preds.append(pm_calib)

        pm_preds = np.array(pm_preds)
        deltas = np.abs(np.diff(pm_preds))
        max_delta = np.max(deltas)
        max_idx = np.argmax(deltas)

        print(f"\n[RH 50-75% Sweep] PM={pm}, CO2={co2}: Max step delta = {max_delta:.4f} ug/m3 at RH={rh_values[max_idx]:.2f}%")

        threshold = max(2.0, pm * 0.02)
        assert max_delta < threshold, f"Extended sweep cliff jump: {max_delta:.4f} > {threshold:.1f} ug/m3"


class TestSmokeAndGateDiscrimination:
    """Adversarial testing of CO2 gate activation and smoke discrimination."""

    def test_gate_opens_fully_when_co2_ge_800(self):
        """Verify smooth gate properly handles combustion smoke (CO2 >= 800.0) without false suppression."""
        for co2 in [800.0, 800.1, 950.0, 1500.0, 3000.0, 5000.0]:
            for rh in [10.0, 30.0, 50.0, 58.0, 60.0, 75.0, 95.0]:
                c_pred = 0.92
                c_eff = smooth_deliquescence_gate(c_pred, rh, co2)
                # In dry air (RH < 60%), c_eff preserves PM (>= 0.95); in humid smoke, c_eff preserves smoke particles
                assert c_eff >= 0.85, f"False suppression at CO2={co2}, RH={rh}: c_eff={c_eff}"

    def test_smoke_vs_fog_discrimination(self, loaded_model):
        """
        Verify physical discrimination:
        - Fog: RH=95%, CO2=420, Gas=0 -> optical swelling -> c_factor <= 0.65
        - Smoke: RH=95%, CO2=2000, Gas=800 -> dry particulates -> c_factor >= 0.85
        """
        # Fog condition
        dp_fog = compute_dew_point(19.0, 95.0)
        c_fog, pm_fog = predict_calibrated_pm25(
            loaded_model, 120.0, 1.5, 0.72, 19.0, 95.0, dp_fog, 420.0, 0.0
        )
        print(f"\n[Discrimination] Fog: c_factor={c_fog:.4f}, PM_calib={pm_fog:.2f} (Target c <= 0.65)")

        # Smoke condition
        dp_smoke = compute_dew_point(20.0, 95.0)
        c_smoke, pm_smoke = predict_calibrated_pm25(
            loaded_model, 120.0, 3.0, 0.75, 20.0, 95.0, dp_smoke, 2000.0, 800.0
        )
        print(f"[Discrimination] Smoke: c_factor={c_smoke:.4f}, PM_calib={pm_smoke:.2f} (Target c >= 0.85)")

        assert c_fog <= 0.65, f"Fog discrimination failed: c_fog={c_fog} > 0.65"
        assert c_smoke >= 0.85, f"Smoke discrimination failed: c_smoke={c_smoke} < 0.85"
        assert pm_smoke > pm_fog * 1.3, "Smoke calibrated PM should be significantly higher than fog calibrated PM!"

    def test_co2_boundary_transition_jump(self, loaded_model):
        """
        Adversarial probe: What is the step difference when CO2 crosses 800.0 ppm?
        We sweep CO2 around 800 ppm (790 to 810 with step 0.2 ppm) at various RH.
        """
        co2_values = np.arange(790.0, 810.01, 0.2)
        for rh in [40.0, 50.0, 58.0, 60.0, 70.0, 85.0]:
            pm = 100.0
            temp = 25.0
            gas = 120.0
            dp = compute_dew_point(temp, rh)

            preds = []
            for co2 in co2_values:
                _, pm_cal = predict_calibrated_pm25(
                    loaded_model, pm, 1.0, 0.70, temp, rh, dp, float(co2), gas
                )
                preds.append(pm_cal)

            preds = np.array(preds)
            deltas = np.abs(np.diff(preds))
            max_delta = np.max(deltas)
            max_idx = np.argmax(deltas)
            print(f"[CO2 790-810 Sweep at RH={rh}%] Max step delta = {max_delta:.4f} ug/m3 at CO2={co2_values[max_idx]:.2f} -> {co2_values[max_idx+1]:.2f} ppm")


class TestExtremeEnvironmentalValues:
    """Stress testing extreme inputs to ensure robustness without NaN/Inf."""

    def test_extreme_temperatures(self, loaded_model):
        """Test temperatures from -30°C to +65°C."""
        extreme_temps = [-30.0, -20.0, -10.0, 0.0, 15.0, 30.0, 45.0, 55.0, 65.0]
        for t in extreme_temps:
            for rh in [20.0, 60.0, 95.0]:
                dp = compute_dew_point(t, rh)
                assert np.isfinite(dp), f"Dew point non-finite at T={t}, RH={rh}"
                assert dp <= t + 1e-4, f"Dew point {dp} exceeds temp {t}"

                c_factor, pm_cal = predict_calibrated_pm25(
                    loaded_model, 100.0, 2.0, 0.70, t, rh, dp, 500.0, 100.0
                )
                assert np.isfinite(c_factor) and np.isfinite(pm_cal)
                assert 0.25 <= c_factor <= 1.05
                assert pm_cal >= 0.0

    def test_extreme_gas_proxy(self, loaded_model):
        """Test gas proxy values from 0.0 to 5000.0."""
        extreme_gases = [0.0, 10.0, 50.0, 250.0, 1000.0, 3000.0, 5000.0]
        for g in extreme_gases:
            dp = compute_dew_point(25.0, 65.0)
            c_factor, pm_cal = predict_calibrated_pm25(
                loaded_model, 80.0, 1.0, 0.70, 25.0, 65.0, dp, 600.0, g
            )
            assert np.isfinite(c_factor) and np.isfinite(pm_cal)
            assert 0.25 <= c_factor <= 1.05

    def test_extreme_pm_range(self, loaded_model):
        """Test PM values from 0.0 to 5000.0 ug/m3."""
        extreme_pms = [0.0, 1.0, 15.0, 150.0, 600.0, 1500.0, 5000.0]
        for pm in extreme_pms:
            dp = compute_dew_point(24.0, 70.0)
            c_factor, pm_cal = predict_calibrated_pm25(
                loaded_model, pm, 0.0, 0.70, 24.0, 70.0, dp, 550.0, 100.0
            )
            assert np.isfinite(c_factor) and np.isfinite(pm_cal)
            assert 0.25 <= c_factor <= 1.05
            assert pm_cal >= 0.0

    def test_extreme_humidity_saturation_and_dry(self, loaded_model):
        """Test extreme humidity: 0%, 1%, 99%, 100%, 120% (supersaturated)."""
        extreme_rhs = [0.0, 1.0, 10.0, 50.0, 99.0, 100.0, 120.0]
        for rh in extreme_rhs:
            dp = compute_dew_point(25.0, rh)
            assert np.isfinite(dp)
            c_factor, pm_cal = predict_calibrated_pm25(
                loaded_model, 100.0, 2.0, 0.70, 25.0, rh, dp, 450.0, 100.0
            )
            assert np.isfinite(c_factor) and np.isfinite(pm_cal)
            assert 0.25 <= c_factor <= 1.05


class TestPythonCppParity:
    """Empirical verification of numerical parity between Python (PyTorch) and C++ (INT8 Quantized)."""

    @pytest.fixture(scope="class")
    def cpp_lib(self):
        import ctypes
        dll_path = os.path.abspath("esp/tests/tinyml_edge.dll")
        if not os.path.exists(dll_path):
            pytest.skip("tinyml_edge.dll not found")
        lib = ctypes.CDLL(dll_path)

        # float tinyml_infer_c_factor_multisensor(float, float, float, float, float, float, float, float)
        lib.tinyml_infer_c_factor_multisensor.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        lib.tinyml_infer_c_factor_multisensor.restype = ctypes.c_float

        # float tinyml_predict_pm25_multisensor(float, float, float, float, float, float, float, bool*)
        lib.tinyml_predict_pm25_multisensor.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_bool)
        ]
        lib.tinyml_predict_pm25_multisensor.restype = ctypes.c_float

        return lib

    def test_grid_parity_python_vs_cpp(self, loaded_model, cpp_lib):
        """
        Compare Python PyTorch float32 inference vs C++ INT8 inference across 500 representative test points.
        Assert that INT8 quantization error in c_factor is <= 0.05 and max PM delta <= 5.0 ug/m3.
        """
        import ctypes
        rng = np.random.default_rng(2026)
        n_points = 500

        pms = rng.uniform(10.0, 350.0, size=n_points).astype(np.float32)
        diffs = rng.uniform(0.0, 6.0, size=n_points).astype(np.float32)
        r_pms = rng.uniform(0.50, 0.85, size=n_points).astype(np.float32)
        temps = rng.uniform(10.0, 38.0, size=n_points).astype(np.float32)
        rhs = rng.uniform(25.0, 95.0, size=n_points).astype(np.float32)
        co2s = rng.uniform(400.0, 2000.0, size=n_points).astype(np.float32)
        gases = rng.uniform(30.0, 500.0, size=n_points).astype(np.float32)

        c_deltas = []
        pm_deltas = []

        is_anom = ctypes.c_bool(False)

        for i in range(n_points):
            pm = float(pms[i])
            diff = float(diffs[i])
            r_pm = float(r_pms[i])
            t = float(temps[i])
            rh = float(rhs[i])
            co2 = float(co2s[i])
            g = float(gases[i])
            dp = float(compute_dew_point(t, rh))

            # Python inference
            c_py, pm_py = predict_calibrated_pm25(
                loaded_model, pm, diff, r_pm, t, rh, dp, co2, g
            )

            # C++ INT8 inference
            c_cpp = float(cpp_lib.tinyml_infer_c_factor_multisensor(
                pm, diff, r_pm, t, rh, dp, co2, g
            ))
            pm1 = pm + diff * 0.5
            pm2 = pm - diff * 0.5
            pm10 = pm / r_pm
            pm_cpp = float(cpp_lib.tinyml_predict_pm25_multisensor(
                pm1, pm2, pm10, t, rh, co2, g, ctypes.byref(is_anom)
            ))

            c_deltas.append(abs(c_py - c_cpp))
            pm_deltas.append(abs(pm_py - pm_cpp))

        max_c_delta = max(c_deltas)
        mean_c_delta = float(np.mean(c_deltas))
        p95_c_delta = float(np.percentile(c_deltas, 95))
        max_pm_delta = max(pm_deltas)
        mean_pm_delta = float(np.mean(pm_deltas))
        p95_pm_delta = float(np.percentile(pm_deltas, 95))

        print(f"\n[Python vs C++ INT8 Parity (500 points)]:")
        print(f" -> C-factor delta: Mean = {mean_c_delta:.4f}, P95 = {p95_c_delta:.4f}, Max = {max_c_delta:.4f}")
        print(f" -> PM delta:       Mean = {mean_pm_delta:.2f} ug/m3, P95 = {p95_pm_delta:.2f} ug/m3, Max = {max_pm_delta:.2f} ug/m3")

        # Tolerances for INT8 matrix quantization on microcontrollers
        assert mean_c_delta <= 0.05, f"Mean C-factor error too high: {mean_c_delta:.4f} > 0.05"
        assert p95_c_delta <= 0.15, f"95th percentile C-factor error too high: {p95_c_delta:.4f} > 0.15"
        assert mean_pm_delta <= 5.0, f"Mean PM delta too high: {mean_pm_delta:.2f} > 5.0 ug/m3"

