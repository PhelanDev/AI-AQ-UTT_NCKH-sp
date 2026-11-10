"""
Empirical Multi-Seed & Adversarial Stress Test Suite
Milestone 4 Iteration 2 (Challenger M4 R2)

Author: challenger_m4_r2_1 (Empirical Challenger)
Target: Comprehensive stress testing across 10 random seeds on Forecaster,
Autoencoder, Spatial Cross-Correlation, and Numerical Stability.
"""

import math
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np
import torch

from pi.config import STATION_IDS, STATION_COORDS_UTM
from pi.spatial_temporal_engine import compute_cross_correlation_matrix, SpatialTemporalEngine
from pi.central_engine import CentralAIEngine
from pi.forecasting_engine import DilatedTCNBiGRUForecaster
from pi.anomaly_engine import MultiStationDeepAutoencoder
from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    predict_calibrated_pm25,
    compute_dew_point,
)

SEEDS = [42, 101, 2024, 777, 1337, 9999, 12345, 54321, 88888, 314159]

class TestMultiSeedSpatialCorrelation(unittest.TestCase):
    """Stress test spatial correlation matrix calculation across 10 seeds and multiple T windows."""

    def test_spatial_matrix_contracts_multi_seed(self):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for T in [8, 15, 24, 60, 120]:
                # Generate random synthetic station concentrations
                base = rng.uniform(20.0, 80.0, size=(T, 1))
                noise = rng.normal(0.0, 5.0, size=(T, 8))
                data = np.clip(base + noise, 0.0, 500.0)

                R, tau = compute_cross_correlation_matrix(data, max_lag=min(30, T // 2 - 1), dt_s=30.0)

                # Contract 1: Shape is (8, 8)
                self.assertEqual(R.shape, (8, 8), f"Seed {seed}, T={T}: R shape invalid")
                self.assertEqual(tau.shape, (8, 8), f"Seed {seed}, T={T}: tau shape invalid")

                # Contract 2: No NaNs or Infs
                self.assertFalse(np.isnan(R).any(), f"Seed {seed}, T={T}: NaN in R")
                self.assertFalse(np.isinf(R).any(), f"Seed {seed}, T={T}: Inf in R")
                self.assertFalse(np.isnan(tau).any(), f"Seed {seed}, T={T}: NaN in tau")
                self.assertFalse(np.isinf(tau).any(), f"Seed {seed}, T={T}: Inf in tau")

                # Contract 3: Diagonal R_ii == 1.0, tau_ii == 0.0
                np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-5, err_msg=f"Seed {seed}, T={T}: diag(R) != 1")
                np.testing.assert_allclose(np.diag(tau), 0.0, atol=1e-5, err_msg=f"Seed {seed}, T={T}: diag(tau) != 0")

                # Contract 4: Range [-1.0, 1.0]
                self.assertTrue(np.all(R >= -1.0 - 1e-6) and np.all(R <= 1.0 + 1e-6),
                                f"Seed {seed}, T={T}: R out of [-1, 1] range: min={R.min()}, max={R.max()}")

                # Contract 5: Symmetry of R: R_ij == R_ji
                np.testing.assert_allclose(R, R.T, atol=1e-5, err_msg=f"Seed {seed}, T={T}: R not symmetric")

                # Contract 6: Anti-symmetry of tau: tau_ij == -tau_ji
                np.testing.assert_allclose(tau, -tau.T, atol=1e-5, err_msg=f"Seed {seed}, T={T}: tau not anti-symmetric")

    def test_spatial_matrix_constant_and_zero_variance(self):
        """Zero variance or perfectly constant data must not cause division by zero or NaNs."""
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            T = 30
            # Station 0 is completely constant (std=0)
            data = rng.uniform(20.0, 60.0, size=(T, 8))
            data[:, 0] = 45.0
            data[:, 3] = 0.0  # Station 3 is all zeros

            R, tau = compute_cross_correlation_matrix(data, max_lag=10, dt_s=30.0)
            self.assertFalse(np.isnan(R).any(), f"Seed {seed}: NaN in constant test R")
            self.assertFalse(np.isinf(R).any(), f"Seed {seed}: Inf in constant test R")
            self.assertFalse(np.isnan(tau).any(), f"Seed {seed}: NaN in constant test tau")


class TestMultiSeedForecaster(unittest.TestCase):
    """Stress test central forecaster inference across 10 random seeds."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CentralAIEngine()

    def test_forecaster_quantiles_and_bounds_multi_seed(self):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # Generate 24 steps of 8-station telemetry
            base_pm = rng.uniform(15.0, 120.0)
            hist = np.clip(base_pm + rng.normal(0, 8.0, size=(24, 8)), 5.0, 300.0).astype(np.float32)

            curr_48 = np.zeros((8, 6), dtype=np.float32)
            for s in range(8):
                pm25 = float(hist[-1, s])
                pm10 = pm25 * rng.uniform(1.1, 1.6)
                temp = rng.uniform(18.0, 32.0)
                rh = rng.uniform(40.0, 95.0)
                co2 = rng.uniform(420.0, 1200.0)
                voc = rng.uniform(100.0, 450.0)
                curr_48[s] = [pm25, pm10, temp, rh, co2, voc]

            res = self.engine.process_telemetry_frame(hist, curr_48.ravel())

            # Verification of forecast outputs
            fc = res.get("forecast_24h")
            self.assertIsNotNone(fc, f"Seed {seed}: forecast_24h missing")

            p10 = np.array(fc["pm25_p10"])
            p50 = np.array(fc["pm25_pred"])
            p90 = np.array(fc["pm25_p90"])

            self.assertEqual(len(p10), 24, f"Seed {seed}: p10 length != 24")
            self.assertEqual(len(p50), 24, f"Seed {seed}: p50 length != 24")
            self.assertEqual(len(p90), 24, f"Seed {seed}: p90 length != 24")

            # Check no NaNs or Infs
            self.assertFalse(np.isnan(p10).any() or np.isinf(p10).any(), f"Seed {seed}: invalid p10")
            self.assertFalse(np.isnan(p50).any() or np.isinf(p50).any(), f"Seed {seed}: invalid p50")
            self.assertFalse(np.isnan(p90).any() or np.isinf(p90).any(), f"Seed {seed}: invalid p90")

            # Check Quantile Ordering: P10 <= P50 <= P90 (allowing floating point eps)
            diff_50_10 = p50 - p10
            diff_90_50 = p90 - p50
            self.assertTrue(np.all(diff_50_10 >= -1e-4), f"Seed {seed}: P50 < P10 violated: {diff_50_10.min()}")
            self.assertTrue(np.all(diff_90_50 >= -1e-4), f"Seed {seed}: P90 < P50 violated: {diff_90_50.min()}")

            # Non-negativity of physical concentrations
            self.assertTrue(np.all(p10 >= 0.0), f"Seed {seed}: P10 negative: {p10.min()}")
            self.assertTrue(np.all(p50 >= 0.0), f"Seed {seed}: P50 negative: {p50.min()}")
            self.assertTrue(np.all(p90 >= 0.0), f"Seed {seed}: P90 negative: {p90.min()}")

            # Finite upper bound check (no blowup to > 1000)
            self.assertTrue(np.all(p90 <= 1000.0), f"Seed {seed}: P90 blown up > 1000: {p90.max()}")

    def test_forecaster_surge_collapse_multi_seed(self):
        """Surge 20 -> 180 ug/m3 must consistently predict [145, 220] ug/m3 across diverse noisy surge trajectories."""
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # Create a 24-step trajectory surging up to 180 with varying noise
            base_surge = np.linspace(20.0, 180.0, 24, dtype=np.float32)
            noise = rng.normal(0.0, 1.5, size=(24, 8)).astype(np.float32)
            hist_surge = np.clip(base_surge[:, None] + noise, 10.0, 250.0)

            curr_48 = np.tile(np.array([180.0, 235.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
            res = self.engine.process_telemetry_frame(hist_surge, curr_48)

            pred_1h = res["forecast"]["pm25_pred"][0]
            self.assertTrue(145.0 <= pred_1h <= 220.0,
                            f"Seed {seed}: Surge collapse at +1h: {pred_1h:.2f} ug/m3 (expected 145-220)")


class TestMultiSeedAutoencoder(unittest.TestCase):
    """Stress test central autoencoder reconstruction and anomaly detection across 10 random seeds."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CentralAIEngine()

    def test_autoencoder_clean_background_stability(self):
        """Natural variations (+-8%) must NOT trigger false anomalies across 10 seeds."""
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            hist_clean = np.full((24, 8), 42.0, dtype=np.float32) + rng.normal(0, 2.5, size=(24, 8)).astype(np.float32)
            curr_clean = np.zeros((8, 6), dtype=np.float32)
            for i in range(8):
                curr_clean[i] = [
                    42.0 + rng.normal(0, 3.0),
                    56.0 + rng.normal(0, 4.0),
                    25.0 + rng.normal(0, 0.7),
                    72.0 + rng.normal(0, 2.5),
                    2.0,
                    110.0,
                ]

            res = self.engine.process_telemetry_frame(hist_clean, curr_clean.ravel())
            anomaly_info = res["anomaly"]

            # Must not be an anomaly
            self.assertFalse(anomaly_info["is_anomaly"],
                             f"Seed {seed}: False positive on clean background (score={anomaly_info['anomaly_score']})")
            self.assertFalse(np.isnan(anomaly_info["anomaly_score"]))
            self.assertFalse(np.isinf(anomaly_info["anomaly_score"]))

    def test_autoencoder_sensor_spike_detection(self):
        """A single station jumping +150 ug/m3 MUST trigger anomaly detection across all 10 seeds."""
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            hist = np.full((24, 8), 45.0, dtype=np.float32) + rng.normal(0, 2.0, size=(24, 8)).astype(np.float32)
            curr = np.zeros((8, 6), dtype=np.float32)
            for i in range(8):
                curr[i] = [45.0, 60.0, 26.0, 70.0, 2.0, 115.0]

            # Measure baseline clean score
            res_clean = self.engine.process_telemetry_frame(hist, curr.ravel())
            clean_score = res_clean["anomaly"]["anomaly_score"]

            # Pick a random station to spike
            fault_station = rng.integers(0, 8)
            curr[fault_station, 0] = 275.0  # +230 spike
            curr[fault_station, 1] = 340.0

            res = self.engine.process_telemetry_frame(hist, curr.ravel())
            anomaly_info = res["anomaly"]

            self.assertTrue(anomaly_info["is_anomaly"],
                            f"Seed {seed}: Failed to detect spike at station {fault_station}")
            self.assertGreater(anomaly_info["anomaly_score"], clean_score * 2.5,
                               f"Seed {seed}: Anomaly score ({anomaly_info['anomaly_score']}) not > 2.5x clean ({clean_score})")


class TestNumericalStabilityAdversarial(unittest.TestCase):
    """Stress test boundary and adversarial inputs: NaNs, Infs, extreme values."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CentralAIEngine()

    def test_nan_inf_injection_graceful_handling(self):
        """Injecting NaNs and Infs into telemetry must be sanitized without crash or propagating NaNs."""
        for seed in SEEDS[:5]:
            rng = np.random.default_rng(seed)
            hist = rng.uniform(20.0, 100.0, size=(24, 8)).astype(np.float32)
            curr_48 = rng.uniform(20.0, 100.0, size=48).astype(np.float32)

            # Inject NaNs and Infs
            hist[5, 2] = np.nan
            hist[12, 4] = np.inf
            hist[20, 0] = -np.inf
            curr_48[3] = np.nan
            curr_48[15] = np.inf

            res = self.engine.process_telemetry_frame(hist, curr_48)

            # Verify forecast has no NaNs
            fc = res["forecast_24h"]
            self.assertFalse(np.isnan(fc["pm25_pred"]).any(), "NaN propagated to forecast")
            self.assertFalse(np.isinf(fc["pm25_pred"]).any(), "Inf propagated to forecast")
            self.assertFalse(np.isnan(res["anomaly"]["anomaly_score"]), "NaN in anomaly score")

    def test_edge_ai_deliquescence_boundary_multi_seed(self):
        """Verify Smooth Deliquescence Gate continuity across sweep RH 58-62% for multiple CO2 and PM levels."""
        model = MultiSensorResMLP(in_features=8, hidden_dim=96)
        if os.path.exists("mlops/models/multisensor_resmlp.pt"):
            model.load_state_dict(torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True))
        model.eval()

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            pm_val = rng.uniform(80.0, 250.0)
            co2_val = rng.uniform(500.0, 950.0)
            temp = rng.uniform(20.0, 30.0)

            # Continuous sweep across RH = 58.0% to 62.0% in 40 steps
            rh_sweep = np.linspace(58.0, 62.0, 40)
            pm_calibrated = []
            for rh in rh_sweep:
                dp = compute_dew_point(temp, rh)
                c_eff, pm_out = predict_calibrated_pm25(
                    model, pm_val, pm_val * 0.02, 0.70, temp, rh, dp, co2_val, 110.0
                )
                pm_calibrated.append(pm_out)

            pm_calibrated = np.array(pm_calibrated)
            diffs = np.abs(np.diff(pm_calibrated))
            max_step = float(np.max(diffs))

            # The maximum step between consecutive 0.1% RH points must be < 0.85 ug/m3 (strictly continuous for PM up to 250 ug/m3)
            self.assertLess(max_step, 0.85,
                            f"Seed {seed}: Jump of {max_step:.4f} ug/m3 in smooth gate (pm={pm_val:.1f}, co2={co2_val:.1f})")


if __name__ == "__main__":
    unittest.main()
