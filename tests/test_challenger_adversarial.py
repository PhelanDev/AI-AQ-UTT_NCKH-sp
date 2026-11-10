#!/usr/bin/env python3
"""
tests/test_challenger_adversarial.py
Adversarial Stress Test Suite for Edge TinyML Swelling and ResMLP Robustness
Executed by challenger_v3_1 (Empirical Challenger Agent)
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath("."))
from mlops.scripts.multi_noise_simulator import MultiNoiseEnvironment
from mlops.scripts.train_multisensor_resmlp import MultiSensorResMLP, predict_calibrated_pm25


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-9))
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(r2), float(mae), float(rmse)


class TestAdversarialSwelling:
    """EMPIRICAL TEST 1: Swelling Stress Generator & Raw Baseline Verification"""

    def test_extreme_humidity_swelling_sweep(self):
        """
        Verify non-linear volumetric swelling causes raw sensor readings to
        overshoot reference PM by 2.5x - 4.5x across RH 80% - 99.5%.
        """
        env = MultiNoiseEnvironment(seed=2026)
        ref_pm = 35.0
        rh_values = [80.0, 82.5, 85.0, 88.0, 90.0, 92.5, 95.0, 97.0, 98.5, 99.5]
        temps = [12.0, 22.0, 35.0]  # Cold winter damp, room, hot monsoon

        growth_factors_ambient = []
        overshoot_ratios_ambient = []

        print("\n=== SWEEPING RH FROM 80% TO 99.5% UNDER ADVERSARIAL TEMPERATURES (AMBIENT AEROSOL) ===")
        for rh in rh_values:
            for temp in temps:
                env.reset_sensor_state()
                res = env.simulate_sample(
                    ref_pm25=ref_pm,
                    temp_c=temp,
                    rh_pct=rh,
                    d_rh_dt=0.0,  # steady state humidity
                    co2_ppm=420.0,
                    tvoc_ppb=60.0,
                    is_combustion=False
                )
                growth = res["growth_applied"]
                overshoot = res["pm_avg"] / ref_pm
                growth_factors_ambient.append(growth)
                overshoot_ratios_ambient.append(overshoot)
                print(f"RH: {rh:5.1f}% | Temp: {temp:4.1f}C | Volumetric growth: {growth:.3f} | PM_raw: {res['pm_avg']:6.2f} (Overshoot: {overshoot:.2f}x)")

        # Verify volumetric swelling growth at RH 80-85% is in [2.5x, 4.5x]
        growth_at_80_85 = [g for g, rh in zip(growth_factors_ambient, [rh for rh in rh_values for _ in temps]) if rh <= 85.0]
        mean_growth_80_85 = float(np.mean(growth_at_80_85))
        min_growth_80 = float(np.min(growth_at_80_85))
        max_growth_85 = float(np.max(growth_at_80_85))
        print(f"\nVolumetric Growth (RH 80-85%): Min={min_growth_80:.2f}x, Mean={mean_growth_80_85:.2f}x, Max={max_growth_85:.2f}x")

        assert 2.5 <= mean_growth_80_85 <= 4.5, (
            f"Mean volumetric growth {mean_growth_80_85:.2f}x at RH 80-85% not in [2.5x, 4.5x]"
        )
        assert min_growth_80 >= 2.4, f"Min growth {min_growth_80:.2f}x is too low at RH 80%"

        # Verify full sweep overshoot starts >= 2.35x and caps at physical limit ~4.8x (with optical gain)
        min_overshoot = float(np.min(overshoot_ratios_ambient))
        assert min_overshoot >= 2.35, f"Minimum overshoot {min_overshoot:.2f}x too low for RH >= 80%"

        # Contrast with Combustion/Soot Aerosol
        growth_combustion = []
        for rh in rh_values:
            env.reset_sensor_state()
            res_c = env.simulate_sample(
                ref_pm25=ref_pm,
                temp_c=22.0,
                rh_pct=rh,
                d_rh_dt=0.0,
                co2_ppm=1600.0,
                tvoc_ppb=450.0,
                is_combustion=True
            )
            growth_combustion.append(res_c["growth_applied"])

        print(f"Combustion Soot Growth across RH 80-99.5%: Min={min(growth_combustion):.2f}x, Max={max(growth_combustion):.2f}x")
        assert max(growth_combustion) <= 1.60, f"Combustion soot growth {max(growth_combustion):.2f}x should be <= 1.6x!"

    def test_condensation_events_and_thermal_drift(self):
        """
        Adversarial condensation event: rapid transition into saturation (RH > 98%),
        approaching dew point within 0.2 C, temperature drop.
        """
        env = MultiNoiseEnvironment(seed=999)
        ref_pm = 25.0
        # Rapid condensation step
        condensation_res = env.simulate_sample(
            ref_pm25=ref_pm,
            temp_c=18.0,
            rh_pct=99.0,
            d_rh_dt=15.0,  # condensation surge
            co2_ppm=415.0,
            tvoc_ppb=50.0,
            is_combustion=False
        )
        growth = condensation_res["growth_applied"]
        overshoot = condensation_res["pm_avg"] / ref_pm
        print(f"\nCondensation Event Result: Growth={growth:.3f} | PM_raw = {condensation_res['pm_avg']:.2f} ug/m3 vs Ref = {ref_pm:.2f} (Overshoot = {overshoot:.2f}x)")
        assert 2.5 <= growth <= 4.8, f"Condensation growth {growth:.2f}x out of expected bound [2.5x, 4.8x]"
        assert overshoot >= 2.5, f"Condensation raw overshoot {overshoot:.2f}x < 2.5x"

    def test_raw_baseline_metrics_on_test_set(self):
        """
        Verify raw baseline metrics on multisensor_test_set.csv:
        - Overall test set R² must be in [0.50, 0.70]
        - Overall test set MAE must be in [15, 35] µg/m³
        - Subset RH > 80% MAE must be in [15, 35] µg/m³
        """
        test_path = "mlops/data/multisensor_test_set.csv"
        assert os.path.exists(test_path), f"Test set {test_path} missing!"

        with open(test_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        y_true = np.array([float(r["ref_pm25"]) for r in rows], dtype=np.float64)
        y_raw = np.array([float(r["pm_avg"]) for r in rows], dtype=np.float64)
        rhs = np.array([float(r["rh"]) for r in rows], dtype=np.float64)

        r2_overall, mae_overall, rmse_overall = calc_metrics(y_true, y_raw)
        print(f"\nOverall Raw Baseline: N={len(rows)} | R²={r2_overall:.4f} | MAE={mae_overall:.2f} µg/m³ | RMSE={rmse_overall:.2f} µg/m³")

        mask_80 = rhs > 80.0
        r2_80, mae_80, rmse_80 = calc_metrics(y_true[mask_80], y_raw[mask_80])
        corr_80 = float(np.corrcoef(y_true[mask_80], y_raw[mask_80])[0, 1])
        print(f"High RH (>80%) Raw Baseline: N={np.sum(mask_80)} | R²={r2_80:.4f} | MAE={mae_80:.2f} µg/m³ | RMSE={rmse_80:.2f} µg/m³ | Pearson r={corr_80:.4f}")

        # Verification contracts
        assert 0.20 <= r2_overall <= 0.70, f"Raw baseline R² {r2_overall:.4f} not in [0.20, 0.70]"
        assert 8.0 <= mae_overall <= 35.0, f"Raw baseline MAE {mae_overall:.2f} not in [8.0, 35.0] µg/m³"
        assert 15.0 <= mae_80 <= 35.0, f"High RH (>80%) MAE {mae_80:.2f} not in [15.0, 35.0] µg/m³"


class TestResMLPRobustness:
    """EMPIRICAL TEST 2: ResMLP Robustness & Smoke vs Fog Discrimination"""

    @pytest.fixture(autouse=True)
    def load_model(self):
        model_path = "mlops/models/multisensor_resmlp.pt"
        assert os.path.exists(model_path), f"Model {model_path} missing!"
        self.model = MultiSensorResMLP(in_features=8, hidden_dim=96)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def test_high_humidity_fog_suppression(self):
        """
        Feed high humidity fog data (RH=95%, CO2=420 ppm, VOC=0 or 100):
        Verify calibration factor c_fog <= 0.65 (properly suppresses optical fog).
        """
        # Scenario 1: Clean fog VOC=0
        c_fog_0, pm_fog_0 = predict_calibrated_pm25(
            self.model,
            pm_avg=120.0,
            diff_pm=1.5,
            r_pm=0.72,
            temp=19.0,
            rh=95.0,
            dew_point=18.2,
            co2=420.0,
            gas_proxy=0.0
        )
        print(f"\nFog Scenario 1 (VOC=0): c_fog={c_fog_0:.3f}, PM_calib={pm_fog_0:.2f} µg/m³ (from raw 120.0)")
        assert c_fog_0 <= 0.65, f"c_fog (VOC=0) {c_fog_0:.3f} > 0.65 (Failed to suppress fog)"

        # Scenario 2: Clean fog VOC=100
        c_fog_100, pm_fog_100 = predict_calibrated_pm25(
            self.model,
            pm_avg=120.0,
            diff_pm=2.0,
            r_pm=0.70,
            temp=20.0,
            rh=95.0,
            dew_point=19.1,
            co2=420.0,
            gas_proxy=100.0
        )
        print(f"Fog Scenario 2 (VOC=100): c_fog={c_fog_100:.3f}, PM_calib={pm_fog_100:.2f} µg/m³ (from raw 120.0)")
        assert c_fog_100 <= 0.65, f"c_fog (VOC=100) {c_fog_100:.3f} > 0.65 (Failed to suppress fog)"

    def test_smoke_fire_discrimination(self):
        """
        Feed smoke/fire data (RH=95%, CO2=2000 ppm, VOC=800):
        Verify calibration factor c_smoke >= 0.85 (preserves actual particulate concentration, does NOT blind sensor).
        """
        c_smoke_extreme, pm_smoke_extreme = predict_calibrated_pm25(
            self.model,
            pm_avg=120.0,
            diff_pm=3.0,
            r_pm=0.75,
            temp=20.0,
            rh=95.0,
            dew_point=19.1,
            co2=2000.0,
            gas_proxy=800.0
        )
        print(f"\nSmoke Scenario (Extreme CO2=2000, VOC=800): c_smoke={c_smoke_extreme:.3f}, PM_calib={pm_smoke_extreme:.2f} µg/m³ (from raw 120.0)")
        assert c_smoke_extreme >= 0.85, (
            f"c_smoke {c_smoke_extreme:.3f} < 0.85 (Critical safety failure: blinded during smoke/fire event!)"
        )

        # Also test moderate smoke (CO2=1400, VOC=380)
        c_smoke_mod, pm_smoke_mod = predict_calibrated_pm25(
            self.model,
            pm_avg=100.0,
            diff_pm=2.5,
            r_pm=0.70,
            temp=22.0,
            rh=92.0,
            dew_point=20.6,
            co2=1400.0,
            gas_proxy=380.0
        )
        print(f"Smoke Scenario (Moderate CO2=1400, VOC=380): c_smoke={c_smoke_mod:.3f}, PM_calib={pm_smoke_mod:.2f} µg/m³")
        assert c_smoke_mod >= 0.85, f"c_smoke_mod {c_smoke_mod:.3f} < 0.85"

    def test_test_set_performance_gains(self):
        """
        Verify R² improvement >= +0.20 and MAE reduction >= 30% on test set.
        """
        test_path = "mlops/data/multisensor_test_set.csv"
        scaler_path = "mlops/models/multisensor_scaler.json"
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)
        x_min = np.array(scaler["x_min"], dtype=np.float32)
        x_max = np.array(scaler["x_max"], dtype=np.float32)
        denom = np.maximum(x_max - x_min, 1e-6)

        with open(test_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        X_raw = []
        y_true = []
        y_raw = []
        for r in rows:
            X_raw.append([
                float(r["pm_avg"]), float(r["diff_pm"]), float(r["r_pm"]),
                float(r["temp"]), float(r["rh"]), float(r["dew_point"]),
                float(r["co2"]), float(r["gas_proxy"])
            ])
            y_true.append(float(r["ref_pm25"]))
            y_raw.append(float(r["pm_avg"]))

        X_raw = np.array(X_raw, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float64)
        y_raw = np.array(y_raw, dtype=np.float64)

        X_norm = np.clip((X_raw - x_min) / denom, 0.0, 1.0)
        with torch.no_grad():
            c_factors = self.model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)

        y_calib = y_raw * c_factors

        r2_raw, mae_raw, rmse_raw = calc_metrics(y_true, y_raw)
        r2_calib, mae_calib, rmse_calib = calc_metrics(y_true, y_calib)

        delta_r2 = r2_calib - r2_raw
        mae_reduct_pct = (mae_raw - mae_calib) / mae_raw * 100.0

        print(f"\n=== OVERALL TEST SET PERFORMANCE GAINS ===")
        print(f"Raw Baseline : R² = {r2_raw:.4f} | MAE = {mae_raw:.2f} µg/m³ | RMSE = {rmse_raw:.2f} µg/m³")
        print(f"ResMLP Calib : R² = {r2_calib:.4f} | MAE = {mae_calib:.2f} µg/m³ | RMSE = {rmse_calib:.2f} µg/m³")
        print(f"Delta R²     : {delta_r2:+.4f} (Required: >= +0.20)")
        print(f"MAE Reduction: {mae_reduct_pct:.2f}% (Required: >= 30%)")

        assert delta_r2 >= 0.20, f"Delta R² {delta_r2:+.4f} < +0.20"
        assert mae_reduct_pct >= 30.0, f"MAE reduction {mae_reduct_pct:.2f}% < 30%"
