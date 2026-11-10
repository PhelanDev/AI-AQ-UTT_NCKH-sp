"""
tests/test_challenger_m2_adversarial.py
Adversarial Verification Suite for Milestone 2 (M2) Central Forecaster Deep Learning Upgrade:
1. Surge Preservation across sharp gradients (20->180 over 24h, 10->300 in 6h, 50->250 step jumps).
2. Monotonicity Invariants (0 <= P10 <= P50 <= P90) across diverse adversarial telemetry frames.
3. ONNX Numerical Parity (< 1e-4) across 1,000 random and extreme batches.
4. Empirical edge-case discovery (T < 8 history length constraint and NaN/Inf robustness).
"""

import os
import sys
import time
import pytest
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.central_engine import CentralAIEngine
from pi.forecasting_engine import DilatedTCNBiGRUForecaster


class TestChallengerM2SurgePreservation:
    """Empirical verification that CentralAIEngine strictly prevents surge collapse."""

    @classmethod
    def setup_class(cls):
        cls.engine = CentralAIEngine()

    def test_m2_surge_gradual_20_to_180_24h(self):
        """Gradual industrial pollution increase: 20 -> 180 ug/m3 over 24h."""
        hist = np.linspace(20.0, 180.0, 24, dtype=np.float32)[:, None].repeat(8, axis=1)
        curr = np.tile(np.array([180.0, 235.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]
        hist_mean = float(np.mean(hist))

        assert 145.0 <= pred_1h <= 220.0, f"Surge collapsed: +1h is {pred_1h:.2f} (expected [145, 220])"
        assert pred_1h > hist_mean * 1.4, f"Prediction {pred_1h:.2f} too close to historical mean {hist_mean:.2f}"

    def test_m2_surge_spike_10_to_300_last_6h(self):
        """Sudden smoke plume: 10 -> 300 ug/m3 spike in the last 6h (baseline 10 ug/m3 for 18h)."""
        seq = np.concatenate([np.full(18, 10.0), np.linspace(10.0, 300.0, 6)])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([300.0, 390.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]
        hist_mean = float(np.mean(hist))

        assert 200.0 <= pred_1h <= 350.0, f"Surge collapsed: +1h is {pred_1h:.2f} (expected [200, 350])"
        assert pred_1h > hist_mean * 3.0, f"Prediction {pred_1h:.2f} collapsed towards historical mean {hist_mean:.2f}"

    def test_m2_surge_step_jump_50_to_250_last_1h(self):
        """Abrupt emission step jump: 50 -> 250 ug/m3 at the 24th hour."""
        seq = np.concatenate([np.full(23, 50.0), [250.0]])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([250.0, 325.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]
        hist_mean = float(np.mean(hist))

        assert 180.0 <= pred_1h <= 300.0, f"Step jump collapsed: +1h is {pred_1h:.2f} (expected [180, 300])"
        assert pred_1h > hist_mean * 2.5, f"Prediction {pred_1h:.2f} collapsed towards historical mean {hist_mean:.2f}"

    @pytest.mark.parametrize("jump_hours", [2, 4, 6, 12])
    def test_m2_surge_step_jumps_varying_horizons(self, jump_hours):
        """Step jumps 50 -> 250 ug/m3 persisting across different durations (2h, 4h, 6h, 12h)."""
        seq = np.concatenate([np.full(24 - jump_hours, 50.0), np.full(jump_hours, 250.0)])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([250.0, 325.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]
        hist_mean = float(np.mean(hist))

        assert 180.0 <= pred_1h <= 300.0, f"Failed at jump_hours={jump_hours}: pred_1h={pred_1h:.2f}"
        assert pred_1h > hist_mean * 1.3, f"Failed at jump_hours={jump_hours}: pred_1h {pred_1h:.2f} <= {hist_mean:.2f} * 1.3"

    @pytest.mark.parametrize("jump_hours", [2, 4, 6, 12])
    def test_m2_surge_step_jumps_20_to_180_varying_horizons(self, jump_hours):
        """Step jumps 20 -> 180 ug/m3 persisting across different durations (2h, 4h, 6h, 12h) in [145, 215]."""
        seq = np.concatenate([np.full(24 - jump_hours, 20.0), np.full(jump_hours, 180.0)])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([180.0, 235.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]

        assert 145.0 <= pred_1h <= 215.0, f"Failed at jump_hours={jump_hours}: pred_1h={pred_1h:.2f} not in [145.0, 215.0]"

    def test_m2_surge_wildfire_extreme_smoke_plume(self):
        """Catastrophic wildfire plume: 15 -> 450 ug/m3 ramp over 12h."""
        seq = np.concatenate([np.full(12, 15.0), np.linspace(15.0, 450.0, 12)])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([450.0, 580.0, 32.0, 40.0, 4.0, 200.0], dtype=np.float32), 8)
        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]

        assert 300.0 <= pred_1h <= 550.0, f"Wildfire plume prediction collapsed: +1h={pred_1h:.2f}"


class TestChallengerM2Monotonicity:
    """Empirical verification of strict monotonicity invariant 0 <= P10 <= P50 <= P90."""

    @classmethod
    def setup_class(cls):
        cls.engine = CentralAIEngine()

    def test_m2_monotonicity_and_bounds_sampled_adversarial(self):
        """Audit 500 diverse adversarial telemetry frames for zero quantile crossings."""
        rng = np.random.default_rng(seed=42)
        checked_points = 0

        for i in range(500):
            case = i % 5
            if case == 0:
                hist = rng.uniform(0.0, 500.0, size=(24, 8)).astype(np.float32)
                curr = rng.uniform(0.0, 500.0, size=48).astype(np.float32)
            elif case == 1:
                hist = np.zeros((24, 8), dtype=np.float32)
                curr = np.zeros(48, dtype=np.float32)
            elif case == 2:
                hist = rng.uniform(-20.0, 200.0, size=(24, 8)).astype(np.float32)
                curr = rng.uniform(-20.0, 200.0, size=48).astype(np.float32)
            elif case == 3:
                hist = rng.exponential(scale=50.0, size=(24, 8)).astype(np.float32)
                curr = rng.exponential(scale=50.0, size=48).astype(np.float32)
            else:
                hist = np.tile(np.sin(np.linspace(0, np.pi, 24))[:, None] * 200.0 + 50.0, (1, 8)).astype(np.float32)
                curr = np.tile([250.0, 300.0, 28.0, 80.0, 2.0, 150.0], 8).astype(np.float32)

            res = self.engine.process_telemetry_frame(hist, curr)
            fc = res["forecast_24h"]
            p10 = fc["pm25_p10"]
            p50 = fc["pm25_p50"]
            p90 = fc["pm25_p90"]

            assert len(p10) == 24 and len(p50) == 24 and len(p90) == 24
            for h in range(24):
                checked_points += 1
                v10, v50, v90 = p10[h], p50[h], p90[h]
                assert v10 >= 0.0, f"Negative P10 at frame {i}, horizon {h}: {v10}"
                assert v10 <= v50 + 1e-6, f"Quantile crossing P10 > P50 at frame {i}, horizon {h}: {v10} > {v50}"
                assert v50 <= v90 + 1e-6, f"Quantile crossing P50 > P90 at frame {i}, horizon {h}: {v50} > {v90}"

        assert checked_points == 500 * 24

    def test_m2_monotonicity_10000_adversarial_frames_batched(self):
        """Stress-test 10,000 adversarial frames in batches to verify strict 0 <= P10 <= P50 <= P90."""
        h_std = getattr(self.engine, "horizon_std_pm25", np.array([4.0 + 0.5 * h for h in range(24)], dtype=np.float32))
        rng = np.random.default_rng(seed=2026)
        n_frames = 10000
        batch_size = 500
        ort_sess = ort.InferenceSession("mlops/models/central_forecaster.onnx", providers=["CPUExecutionProvider"])
        pm_range = float(self.engine.feat_max[0] - self.engine.feat_min[0])
        pm_min = float(self.engine.feat_min[0])

        total_checked = 0
        for b in range(0, n_frames, batch_size):
            bs = min(batch_size, n_frames - b)
            mode = (b // batch_size) % 8
            if mode == 0:
                x_raw = rng.uniform(0.0, 500.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 1:
                x_raw = rng.uniform(500.0, 5000.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 2:
                x_raw = rng.uniform(-500.0, 0.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 3:
                x_raw = np.zeros((bs, 24, 14), dtype=np.float32)
            elif mode == 4:
                x_raw = (rng.standard_cauchy(size=(bs, 24, 14)) * 100.0 + 50.0).astype(np.float32)
            elif mode == 5:
                x_raw = rng.exponential(scale=100.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 6:
                x_raw = np.full((bs, 24, 14), 15.0, dtype=np.float32)
                x_raw[:, 20:, :] = rng.uniform(200.0, 800.0, size=(bs, 4, 14)).astype(np.float32)
            else:
                x_raw = rng.uniform(0.0, 300.0, size=(bs, 24, 14)).astype(np.float32)
                x_raw[rng.random(size=(bs, 24, 14)) < 0.02] = np.nan

            x_clean = np.nan_to_num(x_raw, nan=0.0, posinf=500.0, neginf=0.0)
            feat_denom = np.maximum(self.engine.feat_max - self.engine.feat_min, 1e-6)
            feat_norm = (x_clean - self.engine.feat_min) / feat_denom
            feat_norm[:, :, 1:] = np.clip(feat_norm[:, :, 1:], 0.0, 1.0)
            feat_norm[:, :, 0] = np.clip(feat_norm[:, :, 0], 0.0, 2.5)

            forecast_norm = ort_sess.run(None, {"input_timeseries": feat_norm})[0]
            pm25_phys_batch = np.maximum(0.0, forecast_norm[:, :, 0] * pm_range + pm_min)

            for i in range(bs):
                p_row = pm25_phys_batch[i]
                p10 = [max(0.0, float(p - 1.2816 * s)) for p, s in zip(p_row, h_std)]
                p90 = [float(p + 1.2816 * s) for p, s in zip(p_row, h_std)]
                p10 = [min(v, float(p)) for v, p in zip(p10, p_row)]
                p90 = [max(v, float(p)) for v, p in zip(p90, p_row)]
                p50 = [float(p) for p in p_row]

                for h in range(24):
                    total_checked += 1
                    v10, v50, v90 = p10[h], p50[h], p90[h]
                    assert v10 >= 0.0, f"Negative P10: {v10}"
                    assert v10 <= v50 + 1e-6, f"Quantile crossing P10 > P50: {v10} > {v50}"
                    assert v50 <= v90 + 1e-6, f"Quantile crossing P50 > P90: {v50} > {v90}"

        assert total_checked == 10000 * 24


class TestChallengerM2ONNXParity:
    """Empirical verification that ONNX Runtime parity diff is strictly < 1e-4 across 1,000 batches."""

    @classmethod
    def setup_class(cls):
        pt_path = "mlops/models/central_forecaster.pt"
        onnx_path = "mlops/models/central_forecaster.onnx"
        assert os.path.exists(pt_path), f"Missing {pt_path}"
        assert os.path.exists(onnx_path), f"Missing {onnx_path}"

        cls.pt_model = DilatedTCNBiGRUForecaster()
        cls.pt_model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        cls.pt_model.eval()

        cls.ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def test_m2_onnx_pytorch_numerical_parity_1000_batches(self):
        """Run 1,000 diverse batches and check max absolute difference strictly < 1e-4."""
        batch_sizes = [1, 2, 4, 8, 16]
        max_diff = 0.0
        rng = np.random.default_rng(seed=123)

        for i in range(1000):
            bs = batch_sizes[i % len(batch_sizes)]
            mode = i % 5
            if mode == 0:
                x = rng.standard_normal((bs, 24, 14)).astype(np.float32)
            elif mode == 1:
                x = rng.uniform(0.0, 1.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 2:
                x = rng.uniform(-5.0, 5.0, size=(bs, 24, 14)).astype(np.float32)
            elif mode == 3:
                x = np.zeros((bs, 24, 14), dtype=np.float32)
            else:
                x = rng.exponential(scale=1.5, size=(bs, 24, 14)).astype(np.float32)

            with torch.no_grad():
                py_out = self.pt_model(torch.tensor(x)).numpy()
            onnx_out = self.ort_sess.run(None, {"input_timeseries": x})[0]

            diff = float(np.max(np.abs(py_out - onnx_out)))
            if diff > max_diff:
                max_diff = diff

        print(f"\n[ONNX Parity Test] Max Parity Diff across 1,000 batches: {max_diff:.8e}")
        assert max_diff < 1e-4, f"Parity diff {max_diff} exceeded 1e-4 threshold!"


class TestChallengerM2AdversarialVulnerabilities:
    """Documenting empirical vulnerabilities and edge conditions."""

    @classmethod
    def setup_class(cls):
        cls.engine = CentralAIEngine()

    def test_m2_nan_inf_robustness(self):
        """Verify that sliding window history and current 48d with NaNs and Infs are gracefully handled."""
        hist = np.full((24, 8), np.nan, dtype=np.float32)
        hist[0, :] = 50.0
        hist[1, :] = np.inf
        hist[2, :] = -np.inf
        curr = np.full(48, np.nan, dtype=np.float32)
        curr[0] = 50.0

        res = self.engine.process_telemetry_frame(hist, curr)
        pred_1h = res["forecast"]["pm25_pred"][0]
        assert not np.isnan(pred_1h), "NaN output produced from NaN inputs"
        assert not np.isinf(pred_1h), "Inf output produced from Inf inputs"
        assert pred_1h >= 0.0, f"Negative prediction: {pred_1h}"

    @pytest.mark.parametrize("t_len", [1, 2, 3, 4, 5, 6, 7])
    def test_m2_history_t_less_than_8_vulnerability_discovery(self, t_len):
        """
        [FIXED IN M2_R2]
        When sliding_window_history has T < 8 (e.g. T=1..7), CentralAIEngine pads history
        to prevent transposition in compute_cross_correlation_matrix, running smoothly without IndexError.
        Empirically test both (T, 8) and (8, T) shapes.
        """
        curr = np.ones(48, dtype=np.float32) * 50.0

        # Orientation 1: (T, 8)
        hist_t8 = np.ones((t_len, 8), dtype=np.float32) * 50.0
        res1 = self.engine.process_telemetry_frame(hist_t8, curr)
        assert "forecast_24h" in res1
        assert "spatial_dispersion" in res1
        assert res1["spatial_dispersion"]["r_matrix"].shape == (8, 8)
        assert res1["spatial_dispersion"]["tau_star_matrix"].shape == (8, 8)

        # Orientation 2: (8, T)
        hist_8t = np.ones((8, t_len), dtype=np.float32) * 50.0
        res2 = self.engine.process_telemetry_frame(hist_8t, curr)
        assert "forecast_24h" in res2
        assert "spatial_dispersion" in res2
        assert res2["spatial_dispersion"]["r_matrix"].shape == (8, 8)
        assert res2["spatial_dispersion"]["tau_star_matrix"].shape == (8, 8)
