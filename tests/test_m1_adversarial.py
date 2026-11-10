"""
tests/test_m1_adversarial.py
Adversarial Stress-Testing Harness for Milestone 1 (M1): Multi-Station Simulation & Data Pipeline.

Empirical verification of:
1. Numerical boundary conditions:
   - Zero wind speed (u=0, v=0)
   - Gale/storm wind (V > 25 m/s)
   - 100% RH saturated fog / supersaturation
   - Extreme negative temperatures (-15°C to -30°C)
   - Hyper PM spikes (> 1000 to 10,000 ug/m3)
2. Long horizon numerical stability:
   - 10-day continuous simulation (1440 steps)
   - Ornstein-Uhlenbeck wind process boundedness and stationary variance
   - Gaussian puff memory boundedness (no buffer leak / OOM)
3. Strict finite real number invariants:
   - All generated values, features, and tensors are finite (no NaN, no Inf)
   - TinyML edge calibration physics factor C strictly in [0.25, 1.05]
4. Data leakage stress:
   - Strict chronological ordering: max(train) < min(val) and max(val) < min(test)
   - Zero temporal overlap across sliding window DataLoader batches
   - Scaler statistics strictly fitted on train partition
5. Bug reproduction & edge case mining:
   - AQI breakpoint gap defect: continuous PM2.5 in (35.4, 35.5) dropping to AQI=0
   - Fire emission rate 1,000,000x jump cliff at 1000.0 threshold
   - Calibrator failure to broadcast scalar temp/rh with array PM
"""

import math
import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pi as ac
from pi.config import (
    AtmosphericConstants,
    NUM_FEATURES,
    NUM_STATIONS,
    STATIONS,
)
from pi.data_pipeline import (
    AirQualityDataPipeline,
    calc_dew_point_magnus_tetens,
    calc_vn_aqi_pm25,
    calibrate_pm25_edge,
    create_dataloaders,
    filter_dual_pm_sensor,
)
from pi.simulator import (
    AnomalyScenario,
    AnomalyType,
    AtmosphericSimulator,
    GaussianPuff,
    HygroscopicSwellingModel,
    WindFieldOU,
)

class TestNumericalBoundaries:
    def test_zero_wind_speed_stability(self):
        """Stress test zero initial wind speed (u=0, v=0, mean_speed=0)."""
        wind = WindFieldOU(mean_speed=0.0, seed=42)
        wind.u = 0.0
        wind.v = 0.0
        wind.u_mean = 0.0
        wind.v_mean = 0.0

        for _ in range(50):
            u, v, spd, brg = wind.step(dt=60.0)
            assert np.isfinite(u)
            assert np.isfinite(v)
            assert np.isfinite(spd)
            assert np.isfinite(brg)

            assert spd >= AtmosphericConstants.MIN_WIND_SPEED
            assert 0.0 <= brg < 360.0

    def test_gale_storm_wind_stability(self):
        """Stress test gale/storm conditions with mean wind speed V > 25 m/s."""
        sim = AtmosphericSimulator(dt_s=600.0, seed=123)
        sim.wind_model.mean_speed = 32.0
        rad = math.radians(45.0)
        sim.wind_model.u_mean = 32.0 * math.sin(rad)
        sim.wind_model.v_mean = 32.0 * math.cos(rad)
        sim.wind_model.u = sim.wind_model.u_mean
        sim.wind_model.v = sim.wind_model.v_mean

        res = sim.run_simulation(duration_days=1.0, sample_interval_s=600.0)
        speeds = res["wind_speed"]
        assert np.min(speeds) > 25.0, "Wind speed failed to maintain storm regime"
        assert np.all(np.isfinite(speeds))
        assert np.all(np.isfinite(res["pm25_true"]))
        assert np.all(np.isfinite(res["pm25_raw_1"]))

    def test_saturated_fog_100_percent_rh(self):
        """Stress test 100% and supersaturated 105% relative humidity."""

        temp = 25.0
        dew_100 = calc_dew_point_magnus_tetens(temp, 100.0)
        assert math.isclose(dew_100, temp, abs_tol=1e-3), f"At 100% RH, dew point should equal temp: {dew_100}"

        dew_105 = calc_dew_point_magnus_tetens(temp, 105.0)
        assert np.isfinite(dew_105)

        swelling = HygroscopicSwellingModel()
        gf_100 = swelling.growth_factor(100.0)
        assert np.isfinite(gf_100)
        assert gf_100 <= AtmosphericConstants.HANEL_MAX_GROWTH_FACTOR

        pm25_c, c_final, gate, dew, is_anom = calibrate_pm25_edge(100.0, 95.0, temp, 100.0)
        assert np.isfinite(pm25_c)
        assert 0.25 <= c_final <= 1.05
        assert 0.95 <= gate <= 1.0

    def test_negative_temperatures(self):
        """Stress test sub-zero winter temperatures (-15°C to -30°C)."""
        temps = [-15.0, -25.0, -30.0]
        rhs = [40.0, 80.0, 95.0]

        for t in temps:
            for rh in rhs:
                dew = calc_dew_point_magnus_tetens(t, rh)
                assert np.isfinite(dew)
                assert dew <= t, "Dew point must not exceed ambient temperature"

                pm25_c, c_final, gate, _, _ = calibrate_pm25_edge(45.0, 42.0, t, rh)
                assert np.isfinite(pm25_c)
                assert 0.25 <= c_final <= 1.05

    def test_hyper_pm_spikes(self):
        """Stress test hyper-extreme PM spikes (1000 to 10,000 ug/m3)."""
        extreme_pms = [1000.0, 2500.0, 5000.0, 10000.0]

        for pm in extreme_pms:

            pm_avg, is_anom = filter_dual_pm_sensor(pm, pm + 500.0)
            assert is_anom is True
            assert pm_avg == pm, "Dual filter must select min value on mismatch"

            pm25_c, c_final, _, _, _ = calibrate_pm25_edge(pm, pm, 22.0, 80.0)
            assert np.isfinite(pm25_c)
            assert 0.25 <= c_final <= 1.05
            assert math.isclose(pm25_c, pm * c_final, rel_tol=1e-5)

            aqi = calc_vn_aqi_pm25(pm25_c)
            if pm25_c > 500.4:
                assert aqi == 500.0, f"Expected capped AQI 500.0 for PM2.5={pm25_c}, got {aqi}"
            else:
                assert 300.0 <= aqi <= 500.0

class TestLongHorizonStability:
    def test_10_day_continuous_simulation(self):
        """Run 10-day continuous simulation to prove buffer stability and zero OOM."""
        sim = AtmosphericSimulator(dt_s=600.0, seed=42)
        res = sim.run_simulation(duration_days=10.0, sample_interval_s=600.0)

        total_steps = len(res["time_s"])
        assert total_steps == 1440

        assert len(sim.active_puffs) <= sim.max_puffs
        assert len(sim.active_puffs) > 0

        ws = res["wind_speed"]
        assert np.all(ws >= AtmosphericConstants.MIN_WIND_SPEED)
        assert np.max(ws) < 20.0, f"OU process drifted unexpectedly high: {np.max(ws)}"
        assert math.isclose(np.mean(ws), 2.5, abs_tol=1.0)

        pm25 = res["pm25_true"]
        assert np.all(np.isfinite(pm25))
        assert np.all(pm25 >= 1.0)
        assert np.all(pm25 < 1500.0), f"PM2.5 exploded to unrealistic levels: max={np.max(pm25)}"

class TestFiniteRealInvariants:
    def test_pipeline_dataset_finite_real_and_c_bounded(self):
        """Assert all features in generated dataset are finite and C in [0.25, 1.05]."""
        pipeline = AirQualityDataPipeline()
        dataset = pipeline.generate_dataset(duration_days=3.0, sample_interval_s=600.0, seed=2026)

        features = dataset["features"]
        assert features.shape[1] == NUM_STATIONS
        assert features.shape[2] == NUM_FEATURES
        assert np.all(np.isfinite(features)), "Non-finite values found in dataset features"

        pm25_calib = dataset["pm25_calib"]
        pm25_true = dataset["pm25_true"]
        assert np.all(np.isfinite(pm25_calib))
        assert np.all(pm25_calib >= 0.0)

        temp_mat = np.tile(dataset["temp_c"][:, None], (1, NUM_STATIONS))
        rh_mat = np.tile(dataset["rh_percent"][:, None], (1, NUM_STATIONS))
        _, c_factors, _, _, _ = pipeline.calibrator.calibrate(
            pm25_true, pm25_true, temp_mat, rh_mat
        )
        assert np.all(c_factors >= 0.25), f"C factor below lower bound 0.25: min={np.min(c_factors)}"
        assert np.all(c_factors <= 1.05), f"C factor above upper bound 1.05: max={np.max(c_factors)}"

class TestDataLeakageAndChronologicalIntegrity:
    def test_temporal_splits_strictly_monotonic_and_disjoint(self):
        """Assert zero temporal data leakage between Train, Val, and Test partitions."""
        pipeline = AirQualityDataPipeline()
        dataset = pipeline.generate_dataset(duration_days=5.0, sample_interval_s=600.0, seed=99)

        train_dict, val_dict, test_dict = pipeline.chronological_split(
            dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        train_t = train_dict["timestamps"]
        val_t = val_dict["timestamps"]
        test_t = test_dict["timestamps"]

        assert np.all(np.diff(train_t) > 0)
        assert np.all(np.diff(val_t) > 0)
        assert np.all(np.diff(test_t) > 0)

        assert train_t[-1] < val_t[0], "Temporal leakage: Train overlaps Val!"
        assert val_t[-1] < test_t[0], "Temporal leakage: Val overlaps Test!"

    def test_sliding_window_dataloader_batch_temporal_isolation(self):
        """Assert sliding window DataLoaders maintain strict temporal disjointness."""
        pipeline = AirQualityDataPipeline()
        dataset = pipeline.generate_dataset(duration_days=6.0, sample_interval_s=600.0, seed=123)

        train_loader, val_loader, test_loader, meta = create_dataloaders(
            dataset, batch_size=16, seq_len_in=24, seq_len_out=24
        )

        train_timestamps = [b["timestamp"].numpy() for b in train_loader]
        train_timestamps = np.concatenate(train_timestamps)

        val_timestamps = [b["timestamp"].numpy() for b in val_loader]
        val_timestamps = np.concatenate(val_timestamps)

        test_timestamps = [b["timestamp"].numpy() for b in test_loader]
        test_timestamps = np.concatenate(test_timestamps)

        assert np.max(train_timestamps) < np.min(val_timestamps)
        assert np.max(val_timestamps) < np.min(test_timestamps)

    def test_scaler_fitted_on_train_only(self):
        """Assert scaler statistics are computed strictly on train set."""
        pipeline = AirQualityDataPipeline()
        dataset = pipeline.generate_dataset(duration_days=4.0, sample_interval_s=600.0, seed=42)

        train_d, val_d, test_d = pipeline.chronological_split(dataset)

        _, _, _, meta1 = pipeline.fit_transform_scaler(
            train_d["features"], val_d["features"], test_d["features"]
        )

        corrupted_test = test_d["features"] * 1000.0
        _, _, _, meta2 = pipeline.fit_transform_scaler(
            train_d["features"], val_d["features"], corrupted_test
        )

        assert np.allclose(meta1["mean"], meta2["mean"])
        assert np.allclose(meta1["std"], meta2["std"])

class TestEmpiricalBugReproductions:
    def test_defect_aqi_breakpoint_gap(self):
        """VERIFY FIX: calc_vn_aqi_pm25 does NOT drop to 0.0 in breakpoint gaps.

        EPA / QCVN breakpoints now have contiguous intervals.
        Continuous PM2.5 readings falling in gaps interpolate smoothly without dropping to 0.0.
        """
        gap_points = [12.05, 35.45, 55.45, 150.45, 250.45]
        for p in gap_points:
            aqi_val = calc_vn_aqi_pm25(p)
            assert aqi_val > 0.0, f"Expected non-zero continuous AQI for {p}, got {aqi_val}"

        aqi_35_45 = calc_vn_aqi_pm25(35.45)
        assert np.isclose(aqi_35_45, 100.125, atol=0.05)
        assert round(aqi_35_45, 1) == 100.1

    def test_defect_fire_emission_scale_cliff(self):
        """VERIFY FIX: Fire emission rate scaling is continuous without a 1,000,000x cliff."""
        sim = AtmosphericSimulator(dt_s=60.0)
        a_999 = AnomalyScenario(
            anomaly_type=AnomalyType.FIRE,
            start_time_s=0.0,
            duration_s=60.0,
            emission_rate_pm25=999.0,
        )
        a_1001 = AnomalyScenario(
            anomaly_type=AnomalyType.FIRE,
            start_time_s=0.0,
            duration_s=60.0,
            emission_rate_pm25=1001.0,
        )

        sim.anomalies = [a_999]
        sim._emit_anomalies(0.0)
        puff_999 = sim.active_puffs.pop()

        sim.anomalies = [a_1001]
        sim._emit_anomalies(0.0)
        puff_1001 = sim.active_puffs.pop()

        ratio = puff_999.mass_pm25 / puff_1001.mass_pm25

        assert np.isclose(ratio, 999.0 / 1001.0, rtol=1e-3), f"Expected continuous ratio, got {ratio}"

    def test_defect_calibrator_fails_to_broadcast_scalar_temp_rh(self):
        """VERIFY FIX: calibrate_pm25_edge cleanly broadcasts 2D PM arrays against scalar temp/rh."""
        pm1 = np.ones((10, 8), dtype=np.float32) * 35.0
        pm2 = np.ones((10, 8), dtype=np.float32) * 35.0
        temp = 25.0
        rh = 75.0

        pm_calib, c_final, gate, dew, is_anom = calibrate_pm25_edge(pm1, pm2, temp, rh)
        assert pm_calib.shape == (10, 8)
        assert c_final.shape == (10, 8)
        assert gate.shape == (10, 8)
        assert dew.shape == (10, 8)
        assert is_anom.shape == (10, 8)
        assert np.all(pm_calib > 0.0)
