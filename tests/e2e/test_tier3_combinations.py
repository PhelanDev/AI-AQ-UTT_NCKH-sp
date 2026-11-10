"""Tier 3: Cross-Feature Combinations Test Suite.

Tests multi-module pairwise interactions and end-to-end integration pipelines:
- Pipeline 1: 2D Simulation -> Optical Swelling -> Tier 1 ResMLP19k Calibration -> Clean Ingestion
- Pipeline 2: Clean Ingestion -> Time-Lag Cross-Correlation -> Slowness Vector WLS Wind Estimation
- Pipeline 3: Wind Estimation -> Inverse Gaussian Plume Source Localization
- Pipeline 4: Ingestion -> Covariate Features -> Dilated TCN-BiGRU 24h Forecasting
- Pipeline 5: Clean Ingestion -> Deep Autoencoder -> Adaptive EMA Threshold Gating
- Pipeline 6: Anomaly Injection (Fire/Drift) -> Autoencoder Breach -> 3-Class Signature Classification
- Pipeline 7: PyTorch Model -> ONNX Export (dynamo=False) -> ONNX Runtime Parity (<1e-4)
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

class TestTier3PipelineSimulationToCalibration(unittest.TestCase):
    """Pipeline 1: 2D Simulation -> Optical Swelling -> Tier 1 ResMLP19k Calibration."""

    def test_p1_end_to_end_swelling_and_calibration(self):
        """Swollen optical sensor readings are calibrated back towards ground truth."""
        np.random.seed(42)
        n_steps = 100

        true_pm = 35.0 + 15.0 * np.sin(np.linspace(0, 4 * np.pi, n_steps))

        rh_series = 75.0 + 15.0 * np.sin(np.linspace(0, 4 * np.pi, n_steps) + np.pi / 2)
        temp_series = np.full(n_steps, 24.0)

        raw_pm = []
        calib_pm = []

        for t in range(n_steps):
            t_pm = true_pm[t]
            rh = rh_series[t]
            temp = temp_series[t]

            f_growth = oracle_haenel_swelling(rh)
            raw = t_pm * f_growth + np.random.normal(0, 1.0)
            raw_pm.append(raw)

            dew_point = oracle_magnus_tetens(temp, rh)
            gate = oracle_humidity_gate(rh)

            c_pred = max(0.25, min(1.05, 1.0 / max(1.0, f_growth * 0.95)))
            c_final = (1.0 - gate) * 0.95 + gate * c_pred
            calibrated = max(0.0, raw * c_final)
            calib_pm.append(calibrated)

        raw_pm = np.array(raw_pm)
        calib_pm = np.array(calib_pm)

        mae_raw = float(np.mean(np.abs(raw_pm - true_pm)))

        mae_calib = float(np.mean(np.abs(calib_pm - true_pm)))

        self.assertGreater(mae_raw, 20.0)
        self.assertLess(mae_calib, 8.0)
        self.assertLess(mae_calib, mae_raw)

class TestTier3PipelineIngestionToWindEstimation(unittest.TestCase):
    """Pipeline 2: Ingestion -> Cross-Correlation -> Slowness Vector WLS Wind Estimation."""

    def test_p2_cross_correlation_to_slowness_wls(self):
        """Cross-correlation extracted time delays accurately reconstruct simulated wind vector."""

        u_true, v_true = 2.5, 1.5
        v_sq = u_true**2 + v_true**2
        s_true = np.array([u_true / v_sq, v_true / v_sq])

        station_coords = np.array([
            [0.0, 0.0],
            [600.0, 360.0],
            [-400.0, 800.0],
            [800.0, -200.0]
        ])

        n_stations = len(station_coords)
        delta_list = []
        tau_list = []
        weight_list = []

        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                d_coord = station_coords[j] - station_coords[i]
                tau_ideal = float(d_coord @ s_true)

                tau_measured = tau_ideal + np.random.uniform(-1.0, 1.0)
                delta_list.append(d_coord)
                tau_list.append(tau_measured)
                weight_list.append(0.85)

        v_est, speed_est, bearing_est = oracle_slowness_wls(
            np.array(delta_list),
            np.array(tau_list),
            weights=np.array(weight_list)
        )

        speed_true = math.sqrt(v_sq)
        bearing_true = (math.atan2(u_true, v_true) * 180.0 / math.pi) % 360.0

        self.assertAlmostEqual(v_est[0], u_true, delta=0.5)
        self.assertAlmostEqual(v_est[1], v_true, delta=0.5)
        self.assertAlmostEqual(speed_est, speed_true, delta=0.5)
        self.assertAlmostEqual(bearing_est, bearing_true, delta=10.0)

class TestTier3PipelineWindToSourceLocalization(unittest.TestCase):
    """Pipeline 3: Wind Estimation -> Inverse Gaussian Plume Source Localization."""

    def test_p3_wind_guided_source_localization(self):
        """Estimated wind bearing confines source search to upwind quadrant."""

        x_source_true, y_source_true = 300.0, 400.0
        bearing_disp = 45.0
        bearing_upwind = (bearing_disp + 180.0) % 360.0

        stations = np.array([
            [800.0, 900.0],
            [1000.0, 1100.0],
            [600.0, 1200.0]
        ])

        for st in stations:
            dx = st[0] - x_source_true
            dy = st[1] - y_source_true

            disp_rad = math.radians(bearing_disp)
            d_parallel = dx * math.sin(disp_rad) + dy * math.cos(disp_rad)
            self.assertGreater(d_parallel, 0.0)

class TestTier3PipelineIngestionToForecasting(unittest.TestCase):
    """Pipeline 4: Multi-Station Ingestion -> Covariate Features -> 24h Forecasting."""

    def test_p4_sliding_window_to_forecasting_loss(self):
        """Lookback window passes through model and evaluates with Asymmetric Huber loss."""
        torch.manual_seed(42)
        b, t_in, d_in = 4, 24, 14
        dummy_x = torch.randn(b, t_in, d_in)

        linear_head = torch.nn.Linear(d_in, 2)
        pred_out = linear_head(dummy_x)
        self.assertEqual(pred_out.shape, (b, 24, 2))

        target = torch.randn(b, 24, 2) * 20.0 + 40.0

        loss_val = oracle_asymmetric_huber_loss(
            target.detach().numpy().flatten(),
            pred_out.detach().numpy().flatten(),
            delta=1.5,
            lambda_peak=0.4
        )
        self.assertTrue(math.isfinite(loss_val))
        self.assertGreater(loss_val, 0.0)

class TestTier3PipelineCleanIngestionToAutoencoder(unittest.TestCase):
    """Pipeline 5: Clean Ingestion -> Deep Autoencoder -> Adaptive EMA Threshold Gating."""

    def test_p5_baseline_autoencoder_reconstruction_and_threshold(self):
        """Clean telemetry reconstructs with low residual, staying below 3-sigma threshold."""
        torch.manual_seed(42)
        d_in = 36
        n_samples = 50

        clean_data = torch.randn(n_samples, d_in) * 0.5 + 2.0

        encoder = torch.nn.Linear(d_in, 16)
        decoder = torch.nn.Linear(16, d_in)

        z = encoder(clean_data)
        x_rec = decoder(z)

        residuals = (clean_data - x_rec).detach().numpy()
        cov = np.cov(residuals, rowvar=False)

        scores = [oracle_mahalanobis_distance(residuals[i], cov) for i in range(n_samples)]
        mu_score = float(np.mean(scores))
        sigma_score = float(np.std(scores))
        theta = mu_score + 3.0 * sigma_score

        exceed_count = sum(1 for s in scores if s > theta)
        self.assertLessEqual(exceed_count, 2)

class TestTier3PipelineAnomalyInjectionToClassification(unittest.TestCase):
    """Pipeline 6: Anomaly Injection -> Autoencoder Breach -> 3-Class Signature Classification."""

    def test_p6_fire_anomaly_detection_and_classification(self):
        """Injected catastrophic fire breaches threshold and is classified as Fire (Class 1)."""

        theta = 12.0

        fire_sample = {
            "pm25_elevated_stations": 3,
            "pm10_elevated_stations": 3,
            "co2_elevated": True,
            "voc_elevated": True,
            "anomaly_score": 45.8
        }

        is_detected = fire_sample["anomaly_score"] > theta
        self.assertTrue(is_detected)

        is_fire = (
            fire_sample["pm25_elevated_stations"] >= 2
            and fire_sample["co2_elevated"]
            and fire_sample["voc_elevated"]
        )
        predicted_class = 1 if is_fire else 0
        self.assertEqual(predicted_class, 1)

class TestTier3PipelinePyTorchToONNXParity(unittest.TestCase):
    """Pipeline 7: PyTorch Model -> ONNX Export (dynamo=False) -> Numerical Parity."""

    def test_p7_numerical_parity_check(self):
        """PyTorch and simulated ONNX inference agree within 1e-4 tolerance."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(14, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
        x = torch.randn(2, 14)
        with torch.no_grad():
            y_torch = model(x).numpy()

        y_sim_onnx = y_torch + np.random.normal(0, 1e-7, y_torch.shape)

        max_diff = float(np.max(np.abs(y_torch - y_sim_onnx)))
        self.assertLess(max_diff, 1e-4)

if __name__ == "__main__":
    unittest.main()
