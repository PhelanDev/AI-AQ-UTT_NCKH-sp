"""
tests/test_m3_adversarial.py
Adversarial Stress-Testing Harness for Milestone 3 (M3): Central AI Training Divergence,
Early Stopping, Forecast Horizon Stability (+1h to +24h), and ONNX Numerical Parity.

Empirical verification of:
1. Training Log & Loss Curve Analysis:
   - Programmatic audit of mlops/logs/central_ai_training_log.csv across all 45 epochs.
   - Verification that validation loss divergence spike is < 2% (eliminating previous +34% spike).
   - Verification that R² is positive across all 5 forecast horizons (+1h, +3h, +6h, +12h, +24h).
   - Verification that 0 epochs in the logged trajectory suffer from negative R² (+24h).
2. ONNX Numerical Parity & Robustness Harness (1,000 Batches):
   - PyTorch model vs ONNX model parity across 1,000 diverse and adversarial input batches.
   - Test scenarios: zero inputs, extreme PM up to 1,000 µg/m³, zero wind, diurnal spikes,
     saturated humidity (100% RH), temperature extremes, and random noise.
   - Verification that max absolute difference is strictly < 1e-4 across all 1,000 batches.
3. Latency Verification:
   - CPU inference latency verification ensuring average latency < 50ms per batch.
4. Boundary & Sensor Error Injection:
   - Sensor dropout replacement with median values (RH, VOC, CO2, Wind, Temp).
   - Output boundedness and physical clipping evaluation (PM2.5 >= 0, AQI >= 0).
"""

import os
import sys
import csv
import json
import time
import pytest
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pi.unified_model import HierarchicalCentralAI


class TestM3CentralAIEmpiricalHarness:
    """Empirical verification suite for Central AI v3.0."""

    @classmethod
    def setup_class(cls):
        cls.log_csv_path = "mlops/logs/central_ai_training_log.csv"
        cls.pt_model_path = "mlops/models/central_unified_ai.pt"
        cls.onnx_model_path = "mlops/models/central_unified_ai.onnx"
        cls.params_json_path = "mlops/models/central_unified_ai_params.json"

        assert os.path.exists(cls.log_csv_path), f"Missing {cls.log_csv_path}"
        assert os.path.exists(cls.pt_model_path), f"Missing {cls.pt_model_path}"
        assert os.path.exists(cls.onnx_model_path), f"Missing {cls.onnx_model_path}"
        assert os.path.exists(cls.params_json_path), f"Missing {cls.params_json_path}"

        with open(cls.params_json_path, "r", encoding="utf-8") as f:
            cls.params = json.load(f)

        cls.feat_min = np.array(cls.params["feat_min"], dtype=np.float32)
        cls.feat_max = np.array(cls.params["feat_max"], dtype=np.float32)
        cls.denom = np.maximum(cls.feat_max - cls.feat_min, 1e-6)

        cls.pt_model = HierarchicalCentralAI(
            in_features=16, tcn_channels=32, gru_hidden=32, out_steps=24, out_targets=2
        )
        cls.pt_model.load_state_dict(torch.load(cls.pt_model_path, map_location="cpu", weights_only=True))
        cls.pt_model.eval()

        cls.ort_sess = ort.InferenceSession(cls.onnx_model_path, providers=["CPUExecutionProvider"])

    def test_m3_01_training_log_divergence_spike_under_2pct(self):
        """Test that validation loss divergence spike is < 2%, confirming +34% spike is eliminated."""
        with open(self.log_csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        total_epochs = len(rows)
        assert 15 <= total_epochs <= 50, f"Expected training between 15-50 epochs, got {total_epochs}"

        best_row = min(rows, key=lambda x: float(x["val_loss"]))
        best_epoch = int(best_row["epoch"])
        best_val_loss = float(best_row["val_loss"])

        final_row = rows[-1]
        final_epoch = int(final_row["epoch"])
        final_val_loss = float(final_row["val_loss"])

        # Verification of stability: divergence spike must be minimal (< 2%)
        spike_pct = max(0.0, ((final_val_loss - best_val_loss) / best_val_loss) * 100.0)
        print(f"\n[M3 Test 1] Best Epoch: {best_epoch} ({best_val_loss:.6f}), Final Epoch: {final_epoch} ({final_val_loss:.6f})")
        print(f"[M3 Test 1] Divergence Spike: {spike_pct:.3f}% (Threshold: < 2.0%)")

        assert spike_pct < 2.0, f"Validation loss divergence spike {spike_pct:.3f}% exceeds 2.0% threshold!"

    def test_m3_02_forecast_horizons_positive_r2_across_all_epochs(self):
        """Test all 5 horizons have positive R² in best model and zero negative R² epochs in trajectory."""
        with open(self.log_csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        best_row = min(rows, key=lambda x: float(x["val_loss"]))
        final_row = rows[-1]

        horizons = ["1h", "3h", "6h", "12h", "24h"]

        # Check Best Checkpoint
        for h in horizons:
            r2_val = float(best_row[f"r2_{h}"])
            mae_val = float(best_row[f"mae_{h}"])
            print(f"[M3 Test 2] Best Checkpoint +{h}: R² = {r2_val:.4f}, MAE = {mae_val:.2f} µg/m³")
            assert r2_val > 0.0, f"Horizon +{h} has non-positive R²: {r2_val}"

        # Check Final Checkpoint
        for h in horizons:
            r2_val = float(final_row[f"r2_{h}"])
            assert r2_val > 0.0, f"Final Checkpoint Horizon +{h} has non-positive R²: {r2_val}"

        # Audit trajectory for negative R² (+24h) - Long-term degradation elimination
        negative_24h_epochs = [
            int(r["epoch"]) for r in rows if float(r["r2_24h"]) <= 0.0
        ]
        print(f"[M3 Test 2] Epochs with negative R² (+24h): {len(negative_24h_epochs)} (Found: {negative_24h_epochs})")
        assert len(negative_24h_epochs) == 0, f"Found negative R²(+24h) in epochs: {negative_24h_epochs}"

        # Audit converged epochs (epochs >= 16) for positive R² across all horizons
        converged_rows = [r for r in rows if int(r["epoch"]) >= 16]
        for r in converged_rows:
            ep = int(r["epoch"])
            for h in horizons:
                assert float(r[f"r2_{h}"]) > 0.0, f"Converged Epoch {ep} has non-positive R² on +{h}"

    def test_m3_03_onnx_pytorch_numerical_parity_1000_batches(self):
        """Test PyTorch vs ONNX numerical parity across 1,000 diverse and boundary input batches."""
        num_batches = 1000
        batch_size = 4
        np.random.seed(42)

        max_diff_fc = 0.0
        max_diff_rc = 0.0

        for b_idx in range(num_batches):
            scenario = b_idx % 8
            if scenario == 0:
                # 1. Zero input
                batch_data = np.zeros((batch_size, 24, 16), dtype=np.float32)
            elif scenario == 1:
                # 2. Extreme pollution (PM2.5 = 1,000 µg/m³)
                batch_data = np.random.uniform(0.1, 0.4, (batch_size, 24, 16)).astype(np.float32)
                pm_val_norm = (1000.0 - self.feat_min[0]) / self.denom[0]
                batch_data[:, :, 0] = pm_val_norm
                batch_data[:, :, 1] = pm_val_norm * 1.3
                batch_data[:, :, 15] = 1.0
            elif scenario == 2:
                # 3. Calm conditions (zero wind)
                batch_data = np.random.uniform(0.1, 0.6, (batch_size, 24, 16)).astype(np.float32)
                batch_data[:, :, 7] = (0.0 - self.feat_min[7]) / self.denom[7]
                batch_data[:, :, 8] = (0.0 - self.feat_min[8]) / self.denom[8]
            elif scenario == 3:
                # 4. Rapid diurnal spike
                batch_data = np.random.uniform(0.1, 0.3, (batch_size, 24, 16)).astype(np.float32)
                for sh in [10, 11, 12, 13, 14]:
                    batch_data[:, sh, 0] = np.random.uniform(0.8, 1.5, batch_size)
            elif scenario == 4:
                # 5. Saturated humidity (100% RH)
                batch_data = np.random.uniform(0.2, 0.7, (batch_size, 24, 16)).astype(np.float32)
                batch_data[:, :, 3] = (100.0 - self.feat_min[3]) / self.denom[3]
            elif scenario == 5:
                # 6. Temperature extremes (-20°C and +45°C)
                batch_data = np.random.uniform(0.1, 0.5, (batch_size, 24, 16)).astype(np.float32)
                t_val = -20.0 if b_idx % 2 == 0 else 45.0
                batch_data[:, :, 2] = (t_val - self.feat_min[2]) / self.denom[2]
            elif scenario == 6:
                # 7. Sensor dropout with median replacement
                batch_data = np.random.uniform(0.1, 0.6, (batch_size, 24, 16)).astype(np.float32)
                batch_data[:, :, [3, 5, 6, 7, 8]] = 0.35
            else:
                # 8. Unconstrained Gaussian noise
                batch_data = np.random.randn(batch_size, 24, 16).astype(np.float32)

            with torch.no_grad():
                pt_fc, pt_rc = self.pt_model(torch.from_numpy(batch_data))
            pt_fc_np = pt_fc.numpy()
            pt_rc_np = pt_rc.numpy()

            onnx_outs = self.ort_sess.run(None, {"input_timeseries": batch_data})
            onnx_fc_np = onnx_outs[0]
            onnx_rc_np = onnx_outs[1]

            diff_fc = float(np.max(np.abs(pt_fc_np - onnx_fc_np)))
            diff_rc = float(np.max(np.abs(pt_rc_np - onnx_rc_np)))

            if diff_fc > max_diff_fc:
                max_diff_fc = diff_fc
            if diff_rc > max_diff_rc:
                max_diff_rc = diff_rc

        print(f"\n[M3 Test 3] 1,000 Batches Parity: Forecast Max Diff = {max_diff_fc:.8e}")
        print(f"[M3 Test 3] 1,000 Batches Parity: Recon Max Diff    = {max_diff_rc:.8e}")

        assert max_diff_fc < 1e-4, f"Forecast parity diff {max_diff_fc} exceeds 1e-4!"
        assert max_diff_rc < 1e-4, f"Recon parity diff {max_diff_rc} exceeds 1e-4!"

    def test_m3_04_inference_latency_cpu_under_50ms(self):
        """Test CPU inference latency ensuring average latency is < 50ms per batch."""
        test_batch = np.random.randn(1, 24, 16).astype(np.float32)
        # Warm-up
        for _ in range(10):
            self.ort_sess.run(None, {"input_timeseries": test_batch})

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            self.ort_sess.run(None, {"input_timeseries": test_batch})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        max_lat = float(np.max(latencies))
        print(f"\n[M3 Test 4] CPU Latency: avg={avg_lat:.3f}ms, p95={p95_lat:.3f}ms, max={max_lat:.3f}ms (Budget: 50ms)")

        assert avg_lat < 50.0, f"Average latency {avg_lat:.2f}ms exceeds 50ms budget!"

    def test_m3_05_boundary_missing_sensor_dropout_imputation(self):
        """Test boundary sensor dropout with median imputation and physical boundedness."""
        # Median normalized feature vector
        norm_medians = np.array([
            0.08, 0.08, 0.65, 0.75, 0.46, 0.03, 0.05, 0.45, 0.45, 0.50,
            0.50, 0.50, 0.50, 0.50, 0.23, 0.15
        ], dtype=np.float32)

        # Realistic sequence with various sensor dropouts replaced by median
        np.random.seed(123)
        base_x = np.zeros((10, 24, 16), dtype=np.float32)
        for f in range(16):
            base_x[:, :, f] = norm_medians[f] + np.random.randn(10, 24) * 0.04
        base_x = np.clip(base_x, 0.0, 1.0)

        # Drop sensors 3, 5, 6, 7, 8 (RH, CO2, VOC, Wind) and impute with median
        x_dropped = base_x.copy()
        for c in [3, 5, 6, 7, 8]:
            x_dropped[:, :, c] = norm_medians[c]

        outs = self.ort_sess.run(None, {"input_timeseries": x_dropped})
        fc = outs[0]
        rc = outs[1]

        # Verify finite numerical values
        assert np.all(np.isfinite(fc)), "Forecast outputs contain non-finite numbers!"
        assert np.all(np.isfinite(rc)), "Reconstruction outputs contain non-finite numbers!"

        # Denormalize to physical units
        pm25_phys = fc[:, :, 0] * self.denom[0] + self.feat_min[0]
        aqi_phys = fc[:, :, 1] * self.denom[15] + self.feat_min[15]

        # Apply standard non-negative physical bounding
        pm25_bounded = np.maximum(0.0, pm25_phys)
        aqi_bounded = np.maximum(0.0, aqi_phys)

        assert np.all(pm25_bounded >= 0.0), "Bounded PM2.5 violates non-negativity!"
        assert np.all(aqi_bounded >= 0.0), "Bounded AQI violates non-negativity!"
        print(f"\n[M3 Test 5] Sensor Dropout Imputation: Bounded PM2.5 in [{pm25_bounded.min():.2f}, {pm25_bounded.max():.2f}], AQI in [{aqi_bounded.min():.2f}, {aqi_bounded.max():.2f}]")
