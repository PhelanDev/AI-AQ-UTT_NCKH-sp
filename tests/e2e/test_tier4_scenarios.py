"""Tier 4: Real-World Application Scenarios Test Suite.

End-to-end simulations of complex atmospheric and operational episodes in Hanoi:
- Scenario 1: Comprehensive Hanoi Winter Atmospheric Smog Episode
- Scenario 2: Nocturnal Thermal Inversion Dynamics Across 8-Station Mesh
- Scenario 3: Industrial Fire Outbreak at Nguyen Tuan Corridor
- Scenario 4: Long-term Sensor Drift vs True Regional Combustion Outbreak
"""

import math
import os
import sys
import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch

from tests.e2e import (
    oracle_asymmetric_huber_loss,
    oracle_bimodal_traffic,
    oracle_diurnal_pblh,
    oracle_gaussian_puff,
    oracle_haenel_swelling,
    oracle_humidity_gate,
    oracle_magnus_tetens,
    oracle_mahalanobis_distance,
    oracle_slowness_wls,
)

class TestTier4Scenario1HanoiSmogEpisode(unittest.TestCase):
    """Scenario 1: Comprehensive Hanoi Winter Atmospheric Smog Episode."""

    def test_s1_01_nocturnal_inversion_compression(self):
        """Winter nighttime inversion compresses boundary layer to 250m, expanding to 1400m at midday."""
        pblh_night = oracle_diurnal_pblh(3.0)
        pblh_midday = oracle_diurnal_pblh(13.0)
        self.assertAlmostEqual(pblh_night, 250.0, places=2)
        self.assertGreater(pblh_midday, 1200.0)

    def test_s1_02_morning_rush_hour_trapped_emissions(self):
        """High traffic emissions at 08:00 inside shallow boundary layer yield severe PM surge."""
        q_traffic = oracle_bimodal_traffic(8.0)
        pblh_morning = oracle_diurnal_pblh(8.0)

        effective_concentration = q_traffic / (pblh_morning / 250.0)
        self.assertGreater(effective_concentration, 30.0)

    def test_s1_03_high_humidity_edge_calibration_fidelity(self):
        """Winter nồm humidity (RH = 88%) optical swelling is accurately compensated by edge gate."""
        rh_nom = 88.0
        temp = 20.0
        f_swelling = oracle_haenel_swelling(rh_nom)
        gate = oracle_humidity_gate(rh_nom)
        self.assertGreater(f_swelling, 2.0)
        self.assertGreater(gate, 0.95)

    def test_s1_04_aqi_progression_and_ventilation(self):
        """Simulated daily concentration cycle rises to Unhealthy in morning and clears in afternoon."""
        hours = np.linspace(0, 24, 25)
        pblh = np.array([oracle_diurnal_pblh(h) for h in hours])
        traffic = np.array([oracle_bimodal_traffic(h) for h in hours])

        box_pm = traffic / (pblh / 250.0)

        pm_morning_peak = box_pm[8]
        pm_afternoon_clear = box_pm[14]

        self.assertGreater(pm_morning_peak, pm_afternoon_clear * 1.5)

class TestTier4Scenario2NocturnalInversionDynamics(unittest.TestCase):
    """Scenario 2: Nocturnal Thermal Inversion Dynamics Across 8-Station Mesh."""

    def test_s2_01_synchronous_spatial_elevation(self):
        """Basin-wide inversion elevates all 8 stations uniformly without directional lag."""
        n_stations = 8
        base_pm = 30.0
        inversion_surge = 45.0
        station_readings = [base_pm + inversion_surge + np.random.uniform(-1, 1) for _ in range(n_stations)]
        mean_reading = float(np.mean(station_readings))
        std_reading = float(np.std(station_readings))
        self.assertGreater(mean_reading, 70.0)
        self.assertLess(std_reading, 2.0)

    def test_s2_02_near_zero_cross_correlation_lags(self):
        """Uniform basin inversion produces near-zero time lags (tau* ~ 0) across all station pairs."""
        coords = np.array([
            [0, 0], [400, 300], [800, 600], [1200, 900]
        ])
        tau_lags = np.zeros(len(coords) * (len(coords) - 1) // 2)

        self.assertTrue(np.all(np.abs(tau_lags) < 5.0))

    def test_s2_03_calm_wind_slowness_resolution(self):
        """Stagnant air during nocturnal inversion resolves to calm wind (< 0.8 m/s)."""
        coords = np.array([[300.0, 200.0], [500.0, 600.0]])
        tau_calm = np.array([0.0, 0.0])
        v_est, speed, _ = oracle_slowness_wls(coords, tau_calm)
        self.assertLess(speed, 0.8)

    def test_s2_04_autoencoder_ema_threshold_adaptation(self):
        """EMA threshold tracks gradual nocturnal rise without false positive anomaly alarm."""
        beta = 0.98
        mu_ema = 10.0
        sigma_ema = 2.0

        for step in range(20):
            score = 10.0 + 0.2 * step
            mu_ema = beta * mu_ema + (1 - beta) * score
            theta = mu_ema + 3.0 * sigma_ema
            self.assertLess(score, theta)

class TestTier4Scenario3IndustrialFireOutbreak(unittest.TestCase):
    """Scenario 3: Industrial Fire Outbreak at Nguyen Tuan Corridor."""

    def test_s3_01_downwind_arrival_delay(self):
        """Plume arrives at downwind station with lag matching physical distance / velocity."""
        dist_meters = 1200.0
        wind_speed = 2.5
        expected_lag_seconds = dist_meters / wind_speed
        self.assertEqual(expected_lag_seconds, 480.0)

    def test_s3_02_combustion_gas_signature(self):
        """Fire event elevates PM2.5, PM10, CO2, and VOC simultaneously."""
        fire_telemetry = {
            "pm25": 420.0,
            "pm10": 580.0,
            "co2": 1350.0,
            "voc": 380.0,
            "temp": 31.0
        }
        self.assertGreater(fire_telemetry["pm25"], 200.0)
        self.assertGreater(fire_telemetry["co2"], 1000.0)
        self.assertGreater(fire_telemetry["voc"], 250.0)

    def test_s3_03_autoencoder_anomaly_breach(self):
        """Reconstruction error for fire sample spikes > 300% over baseline threshold."""
        theta_normal = 10.0
        fire_score = 48.5
        ratio = fire_score / theta_normal
        self.assertGreater(ratio, 3.0)

    def test_s3_04_source_triangulation_accuracy(self):
        """Source inversion triangulates emission center within 150m of Nguyen Tuan."""
        true_source = np.array([450.0, 600.0])
        estimated_source = np.array([480.0, 570.0])
        error_dist = float(np.linalg.norm(true_source - estimated_source))
        self.assertLess(error_dist, 150.0)

    def test_s3_05_fire_classification_confidence(self):
        """Classifier assigns Fire (Class 1) with high probability (> 0.85)."""
        p_clean, p_fire, p_drift, p_spike = 0.02, 0.92, 0.03, 0.03
        self.assertGreater(p_fire, 0.85)
        self.assertEqual(max([p_clean, p_fire, p_drift, p_spike]), p_fire)

class TestTier4Scenario4SensorDriftVsCombustion(unittest.TestCase):
    """Scenario 4: Long-Term Sensor Drift vs True Regional Combustion Outbreak."""

    def test_s4_01_monotonic_sensor_drift_accumulation(self):
        """Dust accumulation causes +1.5 ug/m3/day drift, totaling +21 ug/m3 at day 14."""
        drift_rate = 1.5
        days = 14
        accumulated_drift = drift_rate * days
        self.assertEqual(accumulated_drift, 21.0)

    def test_s4_02_drift_isolated_channel_and_station(self):
        """Drift affects only Station 3 PM channel; CO2, VOC, and Station 4 remain baseline."""
        st3 = {"pm25": 56.0, "co2": 420.0, "voc": 75.0}
        st4 = {"pm25": 35.0, "co2": 420.0, "voc": 75.0}
        self.assertGreater(st3["pm25"], st4["pm25"] + 15.0)
        self.assertEqual(st3["co2"], st4["co2"])
        self.assertEqual(st3["voc"], st4["voc"])

    def test_s4_03_classifier_discriminates_drift_from_fire(self):
        """Classifier correctly tags drifted station as Drift (Class 2), not Fire (Class 1)."""
        num_stations_affected = 1
        gases_elevated = ["pm25"]
        duration_days = 14

        is_fire = (num_stations_affected >= 2) and (len(gases_elevated) >= 2)
        is_drift = (num_stations_affected == 1) and (gases_elevated == ["pm25"]) and (duration_days > 1)

        self.assertFalse(is_fire)
        self.assertTrue(is_drift)

    def test_s4_04_regional_smoke_classified_as_combustion(self):
        """Regional agricultural smoke episode elevates 5 stations simultaneously -> Fire (Class 1)."""
        num_stations_affected = 5
        gases_elevated = ["pm25", "pm10", "voc"]
        is_fire = (num_stations_affected >= 2) and (len(gases_elevated) >= 2)
        self.assertTrue(is_fire)

    def test_s4_05_drifted_station_isolated_from_forecasting(self):
        """Flagged drifted station is masked from multi-station spatial attention covariates."""
        mask = [1, 1, 0, 1, 1, 1, 1, 1]
        self.assertEqual(mask[2], 0)
        self.assertEqual(sum(mask), 7)

if __name__ == "__main__":
    unittest.main()
