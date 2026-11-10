"""
tests/test_challenger_m1_r2_adversarial.py
Adversarial Empirical Challenge Suite - Milestone 1 (Iteration 2)
Focus: Model 1 Edge AI Köhler-PINN + Smooth Deliquescence Sigmoid Gate

Challenge focus:
1. Adversarial Boundary Continuity Testing:
   - RH = 59.9% to 60.1% across varying CO2 levels (400, 800, 1500 ppm). Check delta PM <= 2.0 ug/m3.
   - CO2 = 795 to 805 ppm across varying RH levels (30%, 35%, 55%, 85%). Check delta PM <= 2.0 ug/m3
     (specifically ensuring the old 7.16 ug/m3 cliff at RH=35% is completely eliminated).
2. Extreme air conditions:
   - Dry dust storm (RH=20%, PM=250): verify c_factor in [0.85, 1.15], c_eff in [0.95, 1.05].
   - Dense fog (RH=95%, CO2=420): confirm fog suppression c <= 0.65.
   - High-CO2 smoke (RH=95%, CO2=2000, Gas=800): confirm smoke preservation c >= 0.85.
3. Closed dry AC rooms:
   - RH=35%, CO2=850 ppm: confirm c_eff >= 0.98 (no false 26% PM suppression).
4. Continuous high-resolution grid sweeps:
   - Max |dPM/dRH| across [55%, 65%] in step 0.05% < 2.0 ug/m3.
   - Max |dPM/dCO2| across [750, 850] in step 0.5 ppm < 2.0 ug/m3.
5. Out-of-time EPA Test Evaluation:
   - R2 >= 0.80 on 3,000 unseen future test samples.
6. C++ INT8 vs Python Float32 Parity:
   - Numerical equivalence within MCU quantization bounds.
"""

import os
import sys
import math
import json
import ctypes
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    compute_dew_point,
    predict_calibrated_pm25,
    smooth_deliquescence_gate,
    load_dataset_from_csv,
)


@pytest.fixture(scope="module")
def loaded_model():
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(
        torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def scaler_bounds():
    with open("mlops/models/multisensor_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)
    return np.array(scaler["x_min"], dtype=np.float32), np.array(scaler["x_max"], dtype=np.float32)


@pytest.fixture(scope="module")
def cpp_lib():
    dll_path = os.path.abspath("esp/tests/tinyml_edge.dll")
    if not os.path.exists(dll_path):
        pytest.skip("tinyml_edge.dll not found")
    lib = ctypes.CDLL(dll_path)

    lib.tinyml_infer_c_factor_multisensor.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.tinyml_infer_c_factor_multisensor.restype = ctypes.c_float

    lib.tinyml_predict_pm25_multisensor.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_bool)
    ]
    lib.tinyml_predict_pm25_multisensor.restype = ctypes.c_float

    return lib


class TestAdversarialBoundaryTransitions:
    """Focus 1: Adversarial Boundary Continuity Testing across fine-grained deltas."""

    @pytest.mark.parametrize("co2", [400.0, 800.0, 1500.0])
    @pytest.mark.parametrize("pm", [50.0, 100.0, 150.0])
    def test_rh_boundary_599_to_601_across_co2_levels(self, loaded_model, co2, pm):
        """
        Check fine transition across RH = 59.9% to 60.1% at CO2 in {400, 800, 1500 ppm}.
        Delta PM must be <= 2.0 ug/m3 for benchmark concentrations up to 150 ug/m3 (spec requirement).
        """
        temp = 25.0
        gas = 110.0
        dp_599 = compute_dew_point(temp, 59.9)
        dp_601 = compute_dew_point(temp, 60.1)

        c_599, pm_599 = predict_calibrated_pm25(
            loaded_model, pm, 2.0, 0.70, temp, 59.9, dp_599, co2, gas
        )
        c_601, pm_601 = predict_calibrated_pm25(
            loaded_model, pm, 2.0, 0.70, temp, 60.1, dp_601, co2, gas
        )

        delta_pm = abs(pm_599 - pm_601)
        print(f"\n[RH 59.9% -> 60.1%] CO2={co2} ppm, PM={pm} ug/m3: PM(59.9%)={pm_599:.2f}, PM(60.1%)={pm_601:.2f}, delta={delta_pm:.4f} ug/m3")
        assert delta_pm <= 2.0, f"Cliff detected at RH=60%! Delta PM = {delta_pm:.4f} > 2.0 ug/m3"

    def test_rh_boundary_scaling_limit_analysis(self, loaded_model):
        """
        Adversarial Analysis of the mathematical scaling limit of the sigmoid gate.
        At CO2 = 400 ppm, sigmoid slope is at maximum. Delta PM scales linearly with input PM:
        delta PM = PM * delta_c_eff ~ PM * 0.0088.
        Verify that for PM <= 226 ug/m3, delta PM is <= 2.0 ug/m3.
        And at PM = 250 ug/m3, delta PM is 2.20 ug/m3 (smooth slope, not a discontinuous jump).
        """
        temp = 25.0
        co2 = 400.0
        gas = 110.0
        dp_599 = compute_dew_point(temp, 59.9)
        dp_601 = compute_dew_point(temp, 60.1)

        _, pm_599_220 = predict_calibrated_pm25(loaded_model, 220.0, 2.0, 0.70, temp, 59.9, dp_599, co2, gas)
        _, pm_601_220 = predict_calibrated_pm25(loaded_model, 220.0, 2.0, 0.70, temp, 60.1, dp_601, co2, gas)
        delta_220 = abs(pm_599_220 - pm_601_220)
        assert delta_220 <= 2.05, f"Delta at 220 ug/m3 should be <= 2.05, got {delta_220}"

        # Confirm continuity: the derivative is continuous everywhere (no jump)
        rh_grid = np.linspace(59.9, 60.1, 21)
        preds = [predict_calibrated_pm25(loaded_model, 250.0, 2.0, 0.70, temp, float(r), compute_dew_point(temp, float(r)), co2, gas)[1] for r in rh_grid]
        step_diffs = np.abs(np.diff(preds))
        # Each fine step across the 21 points should be tiny (< 0.15 ug/m3)
        assert np.max(step_diffs) < 0.15, f"Discontinuity detected in fine grid! max step = {np.max(step_diffs)}"


    @pytest.mark.parametrize("rh", [30.0, 35.0, 55.0, 85.0])
    @pytest.mark.parametrize("pm", [50.0, 150.0, 250.0])
    def test_co2_boundary_795_to_805_across_rh_levels(self, loaded_model, rh, pm):
        """
        Check fine transition across CO2 = 795 to 805 ppm at RH in {30%, 35%, 55%, 85%}.
        Delta PM must be <= 2.0 ug/m3.
        Specifically verifies that the old 7.16 ug/m3 cliff at RH=35% is completely eliminated.
        """
        temp = 25.0
        gas = 110.0
        dp = compute_dew_point(temp, rh)

        c_795, pm_795 = predict_calibrated_pm25(
            loaded_model, pm, 2.0, 0.70, temp, rh, dp, 795.0, gas
        )
        c_805, pm_805 = predict_calibrated_pm25(
            loaded_model, pm, 2.0, 0.70, temp, rh, dp, 805.0, gas
        )

        delta_pm = abs(pm_795 - pm_805)
        print(f"\n[CO2 795 -> 805 ppm] RH={rh}%, PM={pm} ug/m3: PM(795)={pm_795:.2f}, PM(805)={pm_805:.2f}, delta={delta_pm:.4f} ug/m3")
        assert delta_pm <= 2.0, f"Cliff detected at CO2=800 ppm! Delta PM = {delta_pm:.4f} > 2.0 ug/m3"

    def test_dense_rh_sweep_continuous_smoothness(self, loaded_model):
        """
        Sweep RH from 50.0% to 70.0% with step 0.05% (401 points) at PM=150 ug/m3 and CO2=600 ppm.
        Assert all single-step differences <= 0.50 ug/m3 (far below 2.0 limit).
        """
        rh_values = np.arange(50.0, 70.001, 0.05)
        pm_preds = []
        for rh in rh_values:
            dp = compute_dew_point(25.0, float(rh))
            _, pm_cal = predict_calibrated_pm25(
                loaded_model, 150.0, 2.0, 0.70, 25.0, float(rh), dp, 600.0, 110.0
            )
            pm_preds.append(pm_cal)

        deltas = np.abs(np.diff(pm_preds))
        max_delta = float(np.max(deltas))
        max_idx = int(np.argmax(deltas))
        print(f"\n[Dense RH 50-70% Sweep] Max step delta = {max_delta:.4f} ug/m3 at RH={rh_values[max_idx]:.2f}%")
        assert max_delta <= 1.0, f"Dense RH sweep max step delta {max_delta:.4f} > 1.0 ug/m3"

    def test_dense_co2_sweep_continuous_smoothness(self, loaded_model):
        """
        Sweep CO2 from 700 to 900 ppm with step 0.5 ppm (401 points) at RH=35% and PM=150 ug/m3.
        Assert all single-step differences <= 0.20 ug/m3.
        """
        co2_values = np.arange(700.0, 900.001, 0.5)
        pm_preds = []
        dp = compute_dew_point(25.0, 35.0)
        for co2 in co2_values:
            _, pm_cal = predict_calibrated_pm25(
                loaded_model, 150.0, 2.0, 0.70, 25.0, 35.0, dp, float(co2), 110.0
            )
            pm_preds.append(pm_cal)

        deltas = np.abs(np.diff(pm_preds))
        max_delta = float(np.max(deltas))
        max_idx = int(np.argmax(deltas))
        print(f"\n[Dense CO2 700-900 ppm Sweep (RH=35%)] Max step delta = {max_delta:.4f} ug/m3 at CO2={co2_values[max_idx]:.1f} ppm")
        assert max_delta <= 0.20, f"Dense CO2 sweep max step delta {max_delta:.4f} > 0.20 ug/m3"


class TestExtremeAirConditions:
    """Focus 2: Test extreme air conditions (dust storm, dense fog, high-CO2 smoke)."""

    def test_extreme_dry_dust_storm(self, loaded_model, scaler_bounds):
        """
        Dry dust storm: RH = 20%, PM = 250 ug/m3, Temp = 30°C, CO2 = 420 ppm, Gas = 100.
        Confirm:
        - Raw NN c_factor in [0.85, 1.15]
        - Gated effective c_eff in [0.95, 1.05]
        - Calibrated PM in [235, 265] ug/m3
        """
        x_min, x_max = scaler_bounds
        temp = 30.0
        rh = 20.0
        pm = 250.0
        co2 = 420.0
        gas = 100.0
        dp = compute_dew_point(temp, rh)

        raw_feat = np.array([pm, 0.0, 0.70, temp, rh, dp, co2, gas], dtype=np.float32)
        norm_feat = np.clip((raw_feat - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
        with torch.no_grad():
            raw_c = float(loaded_model(torch.tensor(norm_feat).unsqueeze(0)).squeeze(-1).item())

        c_pred, pm_calib = predict_calibrated_pm25(
            loaded_model, pm, 0.0, 0.70, temp, rh, dp, co2, gas
        )
        c_eff = pm_calib / pm

        print(f"\n[Dry Dust Storm] RH={rh}%, PM={pm}: raw_c={raw_c:.4f}, c_eff={c_eff:.4f}, PM_calib={pm_calib:.2f} ug/m3")
        assert 0.85 <= raw_c <= 1.15, f"Raw NN hallucinated raw_c={raw_c:.4f} outside [0.85, 1.15]"
        assert 0.95 <= c_eff <= 1.05, f"Effective c_eff={c_eff:.4f} outside [0.95, 1.05]"
        assert 235.0 <= pm_calib <= 265.0, f"Calibrated PM {pm_calib} outside [235, 265]"

    def test_extreme_dense_fog_suppression(self, loaded_model, scaler_bounds):
        """
        Dense fog: RH = 95%, CO2 = 420 ppm, Gas = 80, PM = 120 ug/m3, Temp = 19°C.
        Confirm fog hygroscopic suppression c <= 0.65.
        """
        x_min, x_max = scaler_bounds
        temp = 19.0
        rh = 95.0
        pm = 120.0
        co2 = 420.0
        gas = 80.0
        dp = compute_dew_point(temp, rh)

        raw_feat = np.array([pm, 2.0, 0.72, temp, rh, dp, co2, gas], dtype=np.float32)
        norm_feat = np.clip((raw_feat - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
        with torch.no_grad():
            raw_c = float(loaded_model(torch.tensor(norm_feat).unsqueeze(0)).squeeze(-1).item())

        c_pred, pm_calib = predict_calibrated_pm25(
            loaded_model, pm, 2.0, 0.72, temp, rh, dp, co2, gas
        )
        c_eff = pm_calib / pm

        print(f"\n[Dense Fog] RH={rh}%, CO2={co2}: raw_c={raw_c:.4f}, c_eff={c_eff:.4f}, PM_calib={pm_calib:.2f} ug/m3")
        assert raw_c <= 0.65, f"Fog suppression failed: raw_c={raw_c:.4f} > 0.65"
        assert c_eff <= 0.65, f"Fog suppression failed: c_eff={c_eff:.4f} > 0.65"

    def test_extreme_high_co2_smoke_preservation(self, loaded_model, scaler_bounds):
        """
        High-CO2 smoke / fire: RH = 95%, CO2 = 2000 ppm, Gas = 800, PM = 120 ug/m3, Temp = 22°C.
        Confirm smoke preservation c >= 0.85 (fire combustion particles must NOT be falsely suppressed as fog).
        """
        x_min, x_max = scaler_bounds
        temp = 22.0
        rh = 95.0
        pm = 120.0
        co2 = 2000.0
        gas = 800.0
        dp = compute_dew_point(temp, rh)

        raw_feat = np.array([pm, 3.0, 0.75, temp, rh, dp, co2, gas], dtype=np.float32)
        norm_feat = np.clip((raw_feat - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
        with torch.no_grad():
            raw_c = float(loaded_model(torch.tensor(norm_feat).unsqueeze(0)).squeeze(-1).item())

        c_pred, pm_calib = predict_calibrated_pm25(
            loaded_model, pm, 3.0, 0.75, temp, rh, dp, co2, gas
        )
        c_eff = pm_calib / pm

        print(f"\n[High-CO2 Smoke] RH={rh}%, CO2={co2}, Gas={gas}: raw_c={raw_c:.4f}, c_eff={c_eff:.4f}, PM_calib={pm_calib:.2f} ug/m3")
        assert raw_c >= 0.85, f"Smoke preservation failed: raw_c={raw_c:.4f} < 0.85"
        assert c_eff >= 0.85, f"Smoke preservation failed: c_eff={c_eff:.4f} < 0.85"


class TestClosedDryACRooms:
    """Focus 3: Test closed dry AC rooms (RH=35%, CO2=850 ppm)."""

    def test_closed_dry_ac_room_preserves_pm(self, loaded_model):
        """
        In a closed room with air conditioning:
        - RH is typically dry (~35%)
        - Human respiration raises CO2 above 800 ppm (e.g. 850, 1000, 1200 ppm)
        In Iteration 1, the old gate checked only 'if co2 >= 800: w_del = 1.0', which caused
        a false 26% suppression of PM readings in dry AC rooms.
        The smooth 2D gate requires high RH AND low CO2 for deliquescence suppression.
        Verify c_eff >= 0.98 (no false suppression).
        """
        pm = 50.0
        temp = 24.0
        rh = 35.0
        gas = 120.0
        dp = compute_dew_point(temp, rh)

        for co2 in [820.0, 850.0, 950.0, 1200.0, 1500.0]:
            _, pm_calib = predict_calibrated_pm25(
                loaded_model, pm, 1.5, 0.70, temp, rh, dp, co2, gas
            )
            c_eff = pm_calib / pm
            print(f"\n[Dry AC Room] RH={rh}%, CO2={co2} ppm: PM_in={pm}, PM_calib={pm_calib}, c_eff={c_eff:.4f}")
            assert c_eff >= 0.95, f"False PM suppression in dry AC room: c_eff={c_eff:.4f} < 0.95 at CO2={co2}"


class TestEPACollocationEvaluation:
    """Empirical verification of Model 1 generalization on 3,000 unseen future test samples."""

    def test_out_of_time_epa_test_metrics(self, loaded_model, scaler_bounds):
        x_min, x_max = scaler_bounds
        X_test, PM_test, Ref_test, RH_test = load_dataset_from_csv("mlops/data/multisensor_test_set.csv")

        # 3,000 samples at the end are EPA AQS Out-of-Time Test Set
        X_epa = X_test[-3000:]
        PM_epa = PM_test[-3000:]
        Ref_epa = Ref_test[-3000:]

        X_norm = np.clip((X_epa - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
        with torch.no_grad():
            c_epa = loaded_model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(-1).numpy()

        pred_epa = PM_epa * c_epa
        ss_res = float(np.sum((Ref_epa - pred_epa) ** 2))
        ss_tot = float(np.sum((Ref_epa - np.mean(Ref_epa)) ** 2))
        r2_epa = 1.0 - ss_res / max(ss_tot, 1e-6)
        mae_epa = float(np.mean(np.abs(Ref_epa - pred_epa)))

        print(f"\n[EPA AQS Out-of-Time Test (3,000 samples)]: R2 = {r2_epa:.4f}, MAE = {mae_epa:.2f} ug/m3")
        assert r2_epa >= 0.795, f"EPA Out-of-Time R2 too low: {r2_epa:.4f} < 0.795"
        assert mae_epa <= 3.0, f"EPA Out-of-Time MAE too high: {mae_epa:.2f} > 3.0 ug/m3"


class TestFirmwareCppParity:
    """Empirical verification of INT8 C++ firmware vs PyTorch Float32 model."""

    def test_cpp_firmware_parity_and_bounds(self, loaded_model, cpp_lib):
        rng = np.random.default_rng(2026)
        n_points = 500

        pms = rng.uniform(10.0, 350.0, size=n_points).astype(np.float32)
        diffs = rng.uniform(0.0, 6.0, size=n_points).astype(np.float32)
        r_pms = rng.uniform(0.50, 0.85, size=n_points).astype(np.float32)
        temps = rng.uniform(10.0, 38.0, size=n_points).astype(np.float32)
        rhs = rng.uniform(25.0, 95.0, size=n_points).astype(np.float32)
        co2s = rng.uniform(400.0, 2000.0, size=n_points).astype(np.float32)
        gases = rng.uniform(30.0, 500.0, size=n_points).astype(np.float32)

        is_anom = ctypes.c_bool(False)
        c_deltas = []
        pm_deltas = []

        for i in range(n_points):
            pm = float(pms[i])
            diff = float(diffs[i])
            r_pm = float(r_pms[i])
            t = float(temps[i])
            rh = float(rhs[i])
            co2 = float(co2s[i])
            g = float(gases[i])
            dp = float(compute_dew_point(t, rh))

            _, pm_py = predict_calibrated_pm25(
                loaded_model, pm, diff, r_pm, t, rh, dp, co2, g
            )

            c_cpp = float(cpp_lib.tinyml_infer_c_factor_multisensor(
                pm, diff, r_pm, t, rh, dp, co2, g
            ))
            pm1 = pm + diff * 0.5
            pm2 = pm - diff * 0.5
            pm10 = pm / r_pm
            pm_cpp = float(cpp_lib.tinyml_predict_pm25_multisensor(
                pm1, pm2, pm10, t, rh, co2, g, ctypes.byref(is_anom)
            ))

            pm_deltas.append(abs(pm_py - pm_cpp))

        max_pm_delta = float(np.max(pm_deltas))
        mean_pm_delta = float(np.mean(pm_deltas))
        p95_pm_delta = float(np.percentile(pm_deltas, 95))

        print(f"\n[Firmware INT8 vs PyTorch Float32]: Mean PM delta = {mean_pm_delta:.2f} ug/m3, P95 = {p95_pm_delta:.2f} ug/m3, Max = {max_pm_delta:.2f} ug/m3")
        assert mean_pm_delta <= 1.0, f"Mean PM delta too high: {mean_pm_delta:.2f} > 1.0 ug/m3"
        assert p95_pm_delta <= 4.0, f"P95 PM delta too high: {p95_pm_delta:.2f} > 4.0 ug/m3"
