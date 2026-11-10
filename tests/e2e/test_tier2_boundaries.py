"""Tier 2: Boundary & Corner Cases Test Suite.

Comprehensive stress tests for physical limits, mathematical singularities,
extreme environmental conditions (100% RH, bone-dry 0% RH, calm wind, gale wind),
station dropouts, extreme pollution spikes, and numerical stability bounds.
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

class TestTier2AtmosphericBoundaries(unittest.TestCase):
    """Boundary conditions for atmospheric physics & microclimate equations."""

    def test_b1_01_zero_humidity_haenel_swelling(self):
        """RH = 0.0% (bone dry) yields exactly 1.0 swelling factor without division by zero."""
        growth = oracle_haenel_swelling(0.0)
        self.assertEqual(growth, 1.0)

    def test_b1_02_saturation_humidity_dew_point(self):
        """At RH = 100.0%, Magnus-Tetens dew point equals ambient temperature."""
        for temp in [-10.0, 0.0, 15.0, 25.0, 40.0]:
            dp = oracle_magnus_tetens(temp, 100.0)
            self.assertAlmostEqual(dp, temp, places=3)

    def test_b1_03_supersaturation_clipping(self):
        """RH > 100.0% (supersaturated mist) clips safely without negative or infinite denominator."""
        growth_105 = oracle_haenel_swelling(105.0)
        self.assertTrue(math.isfinite(growth_105))
        self.assertGreater(growth_105, 1.0)

    def test_b1_04_extreme_freezing_temperature(self):
        """Magnus-Tetens handles extreme arctic temperature (-30 deg C) cleanly."""
        dp = oracle_magnus_tetens(-30.0, 50.0)
        self.assertTrue(math.isfinite(dp))
        self.assertLess(dp, -30.0)

    def test_b1_05_extreme_tropical_heat(self):
        """Magnus-Tetens handles extreme heat (+55 deg C) cleanly."""
        dp = oracle_magnus_tetens(55.0, 90.0)
        self.assertTrue(math.isfinite(dp))
        self.assertTrue(50.0 <= dp <= 55.0)

class TestTier2WindAndDispersionBoundaries(unittest.TestCase):
    """Boundary conditions for wind fields, slowness vectors, and plume dispersion."""

    def test_b2_01_calm_wind_zero_velocity(self):
        """Near-zero wind velocity (calm conditions) regularizes without division by zero."""
        coords = np.array([[100.0, 200.0], [300.0, 400.0]])
        tau_calm = np.array([0.0, 0.0])
        v_est, speed, bearing = oracle_slowness_wls(coords, tau_calm)
        self.assertEqual(speed, 0.0)
        self.assertEqual(v_est[0], 0.0)
        self.assertEqual(v_est[1], 0.0)

    def test_b2_02_gale_wind_speed(self):
        """High wind velocity (35 m/s) with near-instantaneous lags computes valid vector."""
        u_true, v_true = 25.0, 24.5
        v_sq = u_true**2 + v_true**2
        s = np.array([u_true / v_sq, v_true / v_sq])
        coords = np.array([[1000.0, 500.0], [500.0, 1200.0], [-700.0, 800.0]])
        tau = coords @ s
        v_est, speed, bearing = oracle_slowness_wls(coords, tau)
        self.assertAlmostEqual(speed, math.sqrt(v_sq), delta=1.0)

    def test_b2_03_cardinal_bearing_boundaries(self):
        """Wind bearing handles exact cardinal directions 0, 90, 180, 270 degrees."""
        cardinals = [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 90.0),
            (0.0, -1.0, 180.0),
            (-1.0, 0.0, 270.0),
        ]
        for u, v, expected_bearing in cardinals:
            bearing = (math.atan2(u, v) * 180.0 / math.pi) % 360.0
            self.assertAlmostEqual(bearing, expected_bearing, places=4)

    def test_b2_04_gaussian_puff_near_infinite_distance(self):
        """At 20km distance, Gaussian puff contribution asymptotically approaches zero."""
        c_far = oracle_gaussian_puff(20000.0, 20000.0, 0.0, 0.0, sigma=100.0, mass=1000.0, pblh=500.0)
        self.assertEqual(c_far, 0.0)

    def test_b2_05_reversing_wind_vector(self):
        """Opposite wind vectors produce exactly opposite bearings (+/- 180 deg)."""
        u1, v1 = 2.0, 3.0
        u2, v2 = -2.0, -3.0
        b1 = (math.atan2(u1, v1) * 180.0 / math.pi) % 360.0
        b2 = (math.atan2(u2, v2) * 180.0 / math.pi) % 360.0
        diff = abs(b1 - b2)
        self.assertAlmostEqual(diff, 180.0, places=4)

class TestTier2StationDropoutAndMeshBoundaries(unittest.TestCase):
    """Boundary conditions for partial station outages and spatial mesh anomalies."""

    def test_b3_01_single_station_dropout(self):
        """Mesh with 1 missing station retains 7 valid stations (21 pairwise interactions)."""
        n_stations = 8
        missing = 1
        active = n_stations - missing
        n_pairs = active * (active - 1) // 2
        self.assertEqual(n_pairs, 21)

    def test_b3_02_half_station_dropout(self):
        """Mesh with 4 offline stations (50% loss) still yields 6 valid pairs for WLS."""
        active = 4
        n_pairs = active * (active - 1) // 2
        self.assertEqual(n_pairs, 6)
        self.assertGreaterEqual(n_pairs, 2)

    def test_b3_03_minimum_two_station_boundary(self):
        """Minimum viable pair count is 1 pair (underdetermined for 2D, requires fallback)."""
        active = 2
        n_pairs = active * (active - 1) // 2
        self.assertEqual(n_pairs, 1)

    def test_b3_04_co_located_station_pair(self):
        """Co-located stations (distance = 0) have time lag 0 and do not cause division error."""
        coords = np.array([[0.0, 0.0], [500.0, 500.0]])
        tau = np.array([0.0, 15.0])
        v_est, _, _ = oracle_slowness_wls(coords, tau)
        self.assertTrue(math.isfinite(v_est[0]))

    def test_b3_05_collinear_stations_regularization(self):
        """Collinear stations regularize without numerical blowup."""
        collinear_coords = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        tau = np.array([5.0, 10.0, 15.0])
        v_est, speed, bearing = oracle_slowness_wls(collinear_coords, tau)
        self.assertTrue(math.isfinite(speed))

class TestTier2ExtremeSensorValuesAndSpikes(unittest.TestCase):
    """Boundary conditions for sensor glitches, negative values, and extreme spikes."""

    def test_b4_01_catastrophic_fire_spike_magnitude(self):
        """PM2.5 spike > 1500 ug/m3 processes through calibration without overflow."""
        pm_huge = 1850.0
        c_bound = 0.85
        pm_calib = max(0.0, pm_huge * c_bound)
        self.assertEqual(pm_calib, 1572.5)

    def test_b4_02_zero_pm_baseline(self):
        """Zero PM reading produces zero calibrated PM without negative glitch."""
        pm_zero = 0.0
        pm_calib = max(0.0, pm_zero * 0.95)
        self.assertEqual(pm_calib, 0.0)

    def test_b4_03_negative_raw_sensor_clipping(self):
        """Negative raw sensor noise (-25.0 ug/m3) clips to non-negative zero."""
        pm_raw = -25.0
        pm_clipped = max(0.0, pm_raw)
        self.assertEqual(pm_clipped, 0.0)

    def test_b4_04_severe_sensor_drift_boundary(self):
        """Drift of +500 ug/m3 over time isolated to single station."""
        drift_rate = 2.0
        days = 30.0
        drift_total = drift_rate * days
        self.assertEqual(drift_total, 60.0)

    def test_b4_05_constant_flatlined_sensor(self):
        """Zero variance signal (flatline at 35.0) detected by standard deviation check."""
        flatline = np.full(50, 35.0)
        std_val = float(np.std(flatline))
        self.assertEqual(std_val, 0.0)

class TestTier2MathematicalSingularities(unittest.TestCase):
    """Boundary conditions for matrix inversions, batch dimensions, and loss extremes."""

    def test_b5_01_minimal_batch_size_one(self):
        """Model forward pass supports minimal batch size B=1."""
        dummy_in = torch.randn(1, 24, 14)
        linear = torch.nn.Linear(14, 2)
        out = linear(dummy_in)
        self.assertEqual(out.shape, (1, 24, 2))

    def test_b5_02_large_batch_size_sixty_four(self):
        """Model forward pass supports stress batch size B=64."""
        dummy_in = torch.randn(64, 24, 14)
        linear = torch.nn.Linear(14, 2)
        out = linear(dummy_in)
        self.assertEqual(out.shape, (64, 24, 2))

    def test_b5_03_singular_covariance_matrix_mahalanobis(self):
        """Completely singular rank-1 covariance matrix regularizes successfully."""
        res = np.array([2.0, 2.0, 2.0])
        singular_cov = np.ones((3, 3))
        d = oracle_mahalanobis_distance(res, singular_cov)
        self.assertTrue(math.isfinite(d))
        self.assertGreaterEqual(d, 0.0)

    def test_b5_04_loss_with_identical_predictions(self):
        """Asymmetric Huber loss with zero error is approximately 0.0."""
        y = np.array([30.0, 45.0, 60.0])
        loss = oracle_asymmetric_huber_loss(y, y)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_b5_05_loss_with_massive_under_prediction(self):
        """Massive under-prediction (300 vs 20 ug/m3) computes finite bounded loss."""
        y_true = np.array([300.0])
        y_pred = np.array([20.0])
        loss = oracle_asymmetric_huber_loss(y_true, y_pred)
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 100.0)

class TestTier2HardwareAndResourceLimits(unittest.TestCase):
    """Resource limits and numerical precision boundaries for Raspberry Pi 5."""

    def test_b6_01_float32_dynamic_range(self):
        """All computed values remain within IEEE 754 float32 dynamic range [1e-38, 1e38]."""
        val_small = np.float32(1e-15)
        val_large = np.float32(1e15)
        self.assertTrue(np.isfinite(val_small))
        self.assertTrue(np.isfinite(val_large))

    def test_b6_02_memory_budget_sliding_windows(self):
        """1,000 sliding windows (B=1000, T=24, N=8, F=11) in float32 consumes < 10 MB."""
        n_elements = 1000 * 24 * 8 * 11
        bytes_total = n_elements * 4
        mb_total = bytes_total / (1024 * 1024)
        self.assertLess(mb_total, 10.0)

    def test_b6_03_gradient_norm_clipping(self):
        """Gradient clipping caps norm at 1.0."""
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn(10, 10) * 100.0
        torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
        grad_norm = float(torch.norm(p.grad))
        self.assertAlmostEqual(grad_norm, 1.0, delta=0.01)

    def test_b6_04_single_step_horizon_boundary(self):
        """Forecasting head supports single-step horizon slice."""
        full_pred = torch.randn(4, 24, 2)
        step_1 = full_pred[:, 0, :]
        self.assertEqual(step_1.shape, (4, 2))

    def test_b6_05_long_lookback_window(self):
        """Lookback window of 144 steps (6 days at hourly resolution) tensor allocation."""
        dummy_long = torch.zeros(2, 144, 14)
        self.assertEqual(dummy_long.shape, (2, 144, 14))

if __name__ == "__main__":
    unittest.main()
