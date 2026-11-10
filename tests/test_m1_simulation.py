"""
tests/test_m1_simulation.py
Unit tests for Milestone 1 (M1): Multi-Station Spatial-Temporal Simulation & Data Pipeline.
Covers:
- Configuration, coordinates, distance and adjacency matrices.
- Atmospheric physics: OU wind field, Diurnal PBLH, traffic emissions, Hänel swelling.
- 3 Realistic anomaly scenarios (Fire downwind propagation, Sensor drift, Micro-burst).
- Tier 1 TinyML ResMLP19k calibration (Dew point, Dual sensor filter, Humidity gate, Bounded C).
- Chronological dataset splitting (Zero temporal leakage).
- PyTorch Dataset & DataLoader sliding window tensor construction.
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
    ADJACENCY_MATRIX,
    DISTANCE_MATRIX,
    FEATURE_NAMES,
    NUM_FEATURES,
    NUM_STATIONS,
    STATION_COORDS_UTM,
    STATION_COORDS_WGS84,
    STATION_IDS,
    STATION_NAMES,
    STATIONS,
    AtmosphericConstants,
    compute_distance_matrix,
    compute_spatial_adjacency_matrix,
    wgs84_to_utm48n,
)
from pi.data_pipeline import (
    AirQualityDataPipeline,
    RobustResMLP19k,
    SpatialTemporalAirQualityDataset,
    TinyMLEdgeCalibrator,
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
    DiurnalPBLH,
    GaussianPuff,
    HygroscopicSwellingModel,
    TrafficEmissionModel,
    WindFieldOU,
)

class TestConfig:
    def test_stations_count_and_attributes(self):
        """Verify all 8 Thanh Xuan stations exist with required attributes."""
        assert len(STATIONS) == 8
        assert len(STATION_IDS) == 8
        assert len(STATION_NAMES) == 8
        assert NUM_STATIONS == 8

        expected_ids = [
            "STATION_01", "STATION_02", "STATION_03", "STATION_04",
            "STATION_05", "STATION_06", "STATION_07", "STATION_08"
        ]
        assert STATION_IDS == expected_ids

        for station in STATIONS:
            assert station.lat > 20.95 and station.lat < 21.05
            assert station.lon > 105.75 and station.lon < 105.85
            assert station.utm_x > 580000.0 and station.utm_x < 590000.0
            assert station.utm_y > 2315000.0 and station.utm_y < 2330000.0
            assert station.alt_m >= 10.0 and station.alt_m <= 30.0

    def test_wgs84_to_utm48n(self):
        """Verify coordinate conversion from WGS84 to UTM Zone 48N."""

        ux, uy = wgs84_to_utm48n(20.9930, 105.8080)
        assert math.isclose(ux, 583980.48, abs_tol=5.0)
        assert math.isclose(uy, 2321585.08, abs_tol=5.0)

    def test_distance_matrix_properties(self):
        """Verify properties of the distance matrix D_ij."""
        assert DISTANCE_MATRIX.shape == (8, 8)

        assert np.allclose(DISTANCE_MATRIX, DISTANCE_MATRIX.T)

        assert np.allclose(np.diag(DISTANCE_MATRIX), 0.0)

        off_diag = DISTANCE_MATRIX[~np.eye(8, dtype=bool)]
        assert np.all(off_diag > 300.0)
        assert np.all(off_diag < 4000.0)

    def test_adjacency_matrix_properties(self):
        """Verify Gaussian kernel spatial adjacency matrix A_ij = exp(-D_ij^2 / (2*sigma^2))."""
        assert ADJACENCY_MATRIX.shape == (8, 8)

        assert np.allclose(ADJACENCY_MATRIX, ADJACENCY_MATRIX.T)

        assert np.allclose(np.diag(ADJACENCY_MATRIX), 1.0)

        assert np.all((ADJACENCY_MATRIX > 0.0) & (ADJACENCY_MATRIX <= 1.0))

        # Kiểm tra cặp trạm lân cận gần nhau nhất (STATION_02 CV Nhân Chính & STATION_04 Nguyễn Tuân)
        assert ADJACENCY_MATRIX[1, 3] > 0.90

        assert ADJACENCY_MATRIX[2, 5] < 0.05

    def test_atmospheric_constants(self):
        """Verify standard atmospheric constants and 11 feature names."""
        assert AtmosphericConstants.LAMBDA_LOSS == 1.5e-5
        assert AtmosphericConstants.PBLH_MIN == 250.0
        assert AtmosphericConstants.PBLH_MAX == 1600.0
        assert AtmosphericConstants.RUSH_HOUR_MORNING_PEAK == 8.0
        assert AtmosphericConstants.RUSH_HOUR_EVENING_PEAK == 18.0
        assert AtmosphericConstants.SIGMA_SPATIAL_KERNEL == 1200.0
        assert len(FEATURE_NAMES) == 11
        assert NUM_FEATURES == 11
        assert "pm25_calib" in FEATURE_NAMES
        assert "wind_u" in FEATURE_NAMES
        assert "hour_sin" in FEATURE_NAMES

class TestAtmosphericPhysics:
    def test_wind_field_ou_process(self):
        """Verify Ornstein-Uhlenbeck stochastic wind process behavior."""
        wind = WindFieldOU(mean_speed=2.5, prevailing_bearing_deg=45.0, seed=123)
        speeds = []
        bearings = []
        for _ in range(100):
            u, v, spd, brg = wind.step(dt=60.0)
            speeds.append(spd)
            bearings.append(brg)
            assert spd >= AtmosphericConstants.MIN_WIND_SPEED
            assert 0.0 <= brg < 360.0

        assert math.isclose(np.mean(speeds), 2.5, abs_tol=0.6)

    def test_diurnal_pblh_and_meteorology(self):
        """Verify diurnal PBLH thermal inversion and temperature/RH cycle."""
        pblh_model = DiurnalPBLH(h_min=250.0, h_max=1600.0)

        pblh_night, temp_night, rh_night = pblh_model.get_state(time_s=3.0 * 3600.0)
        assert pblh_night == 250.0
        assert rh_night > 75.0

        pblh_day, temp_day, rh_day = pblh_model.get_state(time_s=14.0 * 3600.0)
        assert pblh_day > 1400.0
        assert temp_day > temp_night
        assert rh_day < rh_night

    def test_traffic_emission_bimodal_profile(self):
        """Verify bimodal rush-hour peaks at 08:00 and 18:00."""
        traffic = TrafficEmissionModel(base_emission=15.0, morn_peak_emission=75.0, eve_peak_emission=85.0)

        q_night = traffic.get_emission_rate(time_s=2.0 * 3600.0)
        q_morn = traffic.get_emission_rate(time_s=8.0 * 3600.0)
        q_noon = traffic.get_emission_rate(time_s=13.0 * 3600.0)
        q_eve = traffic.get_emission_rate(time_s=18.0 * 3600.0)

        assert q_morn > q_night * 4.0
        assert q_eve > q_night * 4.0
        assert q_morn > q_noon
        assert q_eve > q_noon

        q_weekday_morn = traffic.get_emission_rate(time_s=8.0 * 3600.0)
        q_weekend_morn = traffic.get_emission_rate(time_s=5 * 86400.0 + 8.0 * 3600.0)
        assert math.isclose(q_weekend_morn, q_weekday_morn * 0.65, rel_tol=1e-3)

    def test_hygroscopic_swelling_model(self):
        """Verify Hänel optical swelling of particle diameter under RH > 75%."""
        swelling = HygroscopicSwellingModel(ah=0.28, bh=0.92)

        assert swelling.growth_factor(rh_percent=45.0) == 1.0
        assert swelling.growth_factor(rh_percent=60.0) == 1.0

        f_75 = swelling.growth_factor(rh_percent=75.0)
        f_85 = swelling.growth_factor(rh_percent=85.0)
        f_95 = swelling.growth_factor(rh_percent=95.0)

        assert f_75 > 1.5
        assert f_85 > 2.0
        assert f_95 > 3.0
        assert f_95 <= AtmosphericConstants.HANEL_MAX_GROWTH_FACTOR

    def test_gaussian_puff_advection_decay(self):
        """Verify puff advection, dispersion growth, and exponential decay."""
        puff = GaussianPuff(
            x=1000.0, y=1000.0, release_time_s=0.0,
            mass_pm25=1e6, mass_pm10=1.5e6, mass_co2=5e6, mass_voc=2e6,
            initial_sigma=20.0
        )

        initial_mass = puff.mass_pm25

        puff.advance(u=2.0, v=0.0, dt=100.0, lambda_loss=1.5e-5)
        assert math.isclose(puff.x, 1200.0, rel_tol=1e-4)
        assert math.isclose(puff.y, 1000.0, rel_tol=1e-4)
        assert puff.mass_pm25 < initial_mass

        sigma = puff.get_dispersion_sigma(current_time_s=100.0, wind_speed=2.0)
        assert sigma > puff.initial_sigma

class TestAnomalyScenarios:
    def test_type_a_fire_downwind_propagation(self):
        """Type A (Fire): Large combustion source dispatches downwind across stations with realistic lag."""
        st0 = STATIONS[1]
        st1 = STATIONS[3]
        rad = math.atan2(st1.utm_x - st0.utm_x, st1.utm_y - st0.utm_y)

        fire_x = st0.utm_x - 150.0 * math.sin(rad)
        fire_y = st0.utm_y - 150.0 * math.cos(rad)

        sim = AtmosphericSimulator(dt_s=60.0, seed=42)

        sim.wind_model.u_mean = 2.5 * math.sin(rad)
        sim.wind_model.v_mean = 2.5 * math.cos(rad)
        sim.wind_model.u = sim.wind_model.u_mean
        sim.wind_model.v = sim.wind_model.v_mean

        fire = AnomalyScenario(
            anomaly_type=AnomalyType.FIRE,
            start_time_s=600.0,
            duration_s=3600.0,
            source_coords_utm=(fire_x, fire_y),
            emission_rate_pm25=25.0,
            emission_rate_voc=15.0,
            emission_rate_co2=35.0
        )
        sim.add_anomaly(fire)

        data = sim.run_simulation(duration_days=0.08, sample_interval_s=60.0)

        pm_max_st0 = data["pm25_true"][:, 1].max()
        pm_max_st1 = data["pm25_true"][:, 3].max()
        pm_max_st2 = data["pm25_true"][:, 2].max()

        assert pm_max_st0 > 150.0, f"ST0 should receive fire plume, got {pm_max_st0}"
        assert pm_max_st1 > 100.0, f"ST1 downwind should receive fire plume, got {pm_max_st1}"

        assert pm_max_st2 < 55.0, f"Transverse ST2 should not receive plume, got {pm_max_st2}"

        flags_st0 = data["anomaly_flags"][:, 1]
        assert np.any(flags_st0 == AnomalyType.FIRE)

    def test_type_b_sensor_drift(self):
        """Type B (Sensor Drift): Single station exhibits monotonic upward drift while others are unaffected."""
        sim = AtmosphericSimulator(dt_s=300.0, seed=42)
        drift = AnomalyScenario(
            anomaly_type=AnomalyType.SENSOR_DRIFT,
            start_time_s=3600.0,
            duration_s=86400.0,
            target_station_id="STATION_04",
            drift_slope_per_day=20.0
        )
        sim.add_anomaly(drift)
        data = sim.run_simulation(duration_days=1.0, sample_interval_s=300.0)

        flags = data["anomaly_flags"]

        assert np.all(flags[:, :3] == 0)
        assert np.all(flags[:, 4:] == 0)
        assert np.sum(flags[:, 3] == AnomalyType.SENSOR_DRIFT) > 200

        raw_st3 = data["pm25_raw_1"][:, 3]
        raw_st0 = data["pm25_raw_1"][:, 0]

        assert (raw_st3[-1] - raw_st3[0]) > (raw_st0[-1] - raw_st0[0]) + 10.0

    def test_type_c_hyper_local_micro_burst(self):
        """Type C (Micro-Burst): Short transient spike on 1 station, zero propagation."""
        sim = AtmosphericSimulator(dt_s=60.0, seed=42)
        burst = AnomalyScenario(
            anomaly_type=AnomalyType.MICRO_BURST,
            start_time_s=1800.0,
            duration_s=900.0,
            target_station_id="STATION_03",
            emission_rate_pm25=150.0
        )
        sim.add_anomaly(burst)
        data = sim.run_simulation(duration_days=0.08, sample_interval_s=60.0)

        flags = data["anomaly_flags"]

        assert np.all(flags[:, :2] == 0)
        assert np.all(flags[:, 3:] == 0)
        burst_steps = np.sum(flags[:, 2] == AnomalyType.MICRO_BURST)
        assert 10 <= burst_steps <= 25

        pm_st2 = data["pm25_true"][:, 2]
        max_idx = np.argmax(pm_st2)
        assert pm_st2[max_idx] > 140.0
        assert pm_st2[0] < 50.0
        assert pm_st2[-1] < 50.0

class TestTinyMLEdgeCalibration:
    def test_magnus_tetens_dew_point(self):
        """Verify Magnus-Tetens approximation formula."""

        dp = calc_dew_point_magnus_tetens(25.0, 80.0)
        assert math.isclose(dp, 21.33, abs_tol=0.2)

        dp2 = calc_dew_point_magnus_tetens(30.0, 50.0)
        assert math.isclose(dp2, 18.45, abs_tol=0.2)

        dps = calc_dew_point_magnus_tetens(np.array([25.0, 30.0]), np.array([80.0, 50.0]))
        assert len(dps) == 2
        assert math.isclose(dps[0], 21.33, abs_tol=0.2)

    def test_filter_dual_pm_sensor(self):
        """Verify dual PMS7003 sensor difference filter."""

        avg, is_anom = filter_dual_pm_sensor(40.0, 44.0)
        assert avg == 42.0
        assert is_anom is False

        avg_anom, is_anom_flag = filter_dual_pm_sensor(35.0, 75.0)
        assert avg_anom == 35.0
        assert is_anom_flag is True

    def test_resmlp_calibration_bound_and_humidity_gate(self):
        """Verify RobustResMLP19k bounded output C in [0.25, 1.05] and humidity gate."""
        calibrator = TinyMLEdgeCalibrator()

        for rh in [20.0, 55.0, 85.0, 95.0]:
            for pm in [10.0, 80.0, 250.0]:
                pm_calib, c_final, gate, dp, _ = calibrator.calibrate(
                    pm1_raw=pm, pm2_raw=pm, temp_c=25.0, rh_percent=rh
                )
                assert 0.25 <= c_final <= 1.05
                assert 0.0 <= gate <= 1.0
                assert pm_calib >= 0.0

        _, c_dry, gate_dry, _, _ = calibrator.calibrate(50.0, 50.0, 28.0, 35.0)
        assert gate_dry < 0.10
        assert math.isclose(c_dry, 0.95, abs_tol=0.08)

    def test_swelling_correction_under_high_rh(self):
        """Verify that high RH optical swelling is corrected back down near ground truth."""
        calibrator = TinyMLEdgeCalibrator()

        pm_calib, c_final, gate, _, _ = calibrator.calibrate(
            pm1_raw=115.0, pm2_raw=112.0, temp_c=22.0, rh_percent=90.0
        )
        assert gate > 0.98
        assert c_final < 0.50

        assert 30.0 <= pm_calib <= 55.0

    def test_vn_aqi_calculation(self):
        """Verify AQI calculation matches official breakpoints and interpolates continuously."""
        assert calc_vn_aqi_pm25(12.0) == 50.0
        assert calc_vn_aqi_pm25(35.4) == 100.0
        assert calc_vn_aqi_pm25(55.4) == 150.0
        assert calc_vn_aqi_pm25(150.4) == 200.0
        assert calc_vn_aqi_pm25(250.4) == 300.0
        assert calc_vn_aqi_pm25(500.4) == 500.0

        aqi_35_45 = calc_vn_aqi_pm25(35.45)
        assert np.isclose(aqi_35_45, 100.125, atol=0.05)
        assert round(aqi_35_45, 1) == 100.1

    def test_edge_calibration_broadcasting(self):
        """Verify calibrate_pm25_edge handles 2D PM arrays with scalar or 1D temp/rh."""
        pm1 = np.ones((10, 8), dtype=np.float32) * 35.0
        pm2 = np.ones((10, 8), dtype=np.float32) * 35.0
        pm_calib, c_final, gate, dew, is_anom = calibrate_pm25_edge(pm1, pm2, 25.0, 75.0)
        assert pm_calib.shape == (10, 8)
        assert c_final.shape == (10, 8)
        assert gate.shape == (10, 8)
        assert dew.shape == (10, 8)
        assert is_anom.shape == (10, 8)
        assert np.all(pm_calib > 0.0)

class TestDataPipelineAndPyTorch:
    def test_pipeline_generate_dataset(self):
        """Verify full simulation dataset generation with all 11 features."""
        pipeline = AirQualityDataPipeline()
        data = pipeline.generate_dataset(duration_days=1.0, sample_interval_s=600.0, seed=42)

        assert data["features"].shape == (144, 8, 11)
        assert data["pm25_calib"].shape == (144, 8)
        assert data["aqi"].shape == (144, 8)
        assert data["timestamps"].shape == (144,)
        assert data["anomaly_flags"].shape == (144, 8)

        assert not np.isnan(data["features"]).any()
        assert not np.isinf(data["features"]).any()

    def test_chronological_split_anti_leakage(self):
        """Verify chronological split guarantees 70/15/15 ratio with strictly zero temporal leakage."""
        pipeline = AirQualityDataPipeline()
        data = pipeline.generate_dataset(duration_days=3.0, sample_interval_s=600.0, seed=42)
        total_steps = len(data["timestamps"])

        train_dict, val_dict, test_dict = pipeline.chronological_split(
            data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        n_train = len(train_dict["timestamps"])
        n_val = len(val_dict["timestamps"])
        n_test = len(test_dict["timestamps"])

        assert n_train + n_val + n_test == total_steps
        assert math.isclose(n_train / total_steps, 0.70, abs_tol=0.01)
        assert math.isclose(n_val / total_steps, 0.15, abs_tol=0.01)
        assert math.isclose(n_test / total_steps, 0.15, abs_tol=0.01)

        assert train_dict["timestamps"][-1] < val_dict["timestamps"][0]
        assert val_dict["timestamps"][-1] < test_dict["timestamps"][0]

    def test_standardization_fitted_on_train_only(self):
        """Verify StandardScaler is computed strictly from training split to avoid data leakage."""
        pipeline = AirQualityDataPipeline()
        data = pipeline.generate_dataset(duration_days=2.0, sample_interval_s=600.0, seed=42)
        train_dict, val_dict, test_dict = pipeline.chronological_split(data)

        train_norm, val_norm, test_norm, meta = pipeline.fit_transform_scaler(
            train_dict["features"], val_dict["features"], test_dict["features"]
        )

        train_mean = np.mean(train_norm, axis=(0, 1))
        train_std = np.std(train_norm, axis=(0, 1))
        assert np.allclose(train_mean, 0.0, atol=1e-3)
        assert np.allclose(train_std, 1.0, atol=1e-3)

    def test_sliding_window_dataset_and_loader(self):
        """Verify PyTorch Dataset and DataLoader tensor dimensions."""
        pipeline = AirQualityDataPipeline()
        data = pipeline.generate_dataset(duration_days=2.0, sample_interval_s=600.0, seed=42)

        train_loader, val_loader, test_loader, meta = create_dataloaders(
            data, batch_size=8, seq_len_in=24, seq_len_out=12, normalize=True
        )

        assert meta["num_stations"] == 8
        assert meta["num_features"] == 11
        assert meta["seq_len_in"] == 24
        assert meta["seq_len_out"] == 12

        batch = next(iter(train_loader))
        x = batch["x"]
        y = batch["y"]

        assert x.shape == (8, 24, 8, 11)
        assert y.shape == (8, 12, 8, 2)
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()

    def test_simulation_reproducibility(self):
        """Verify that identical seeds produce identical simulation datasets."""
        p1 = AirQualityDataPipeline()
        p2 = AirQualityDataPipeline()

        d1 = p1.generate_dataset(duration_days=1.0, sample_interval_s=600.0, seed=999)
        d2 = p2.generate_dataset(duration_days=1.0, sample_interval_s=600.0, seed=999)

        assert np.allclose(d1["features"], d2["features"])
        assert np.allclose(d1["pm25_calib"], d2["pm25_calib"])
        assert np.all(d1["anomaly_flags"] == d2["anomaly_flags"])
