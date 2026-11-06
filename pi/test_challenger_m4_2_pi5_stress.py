"""
CHALLENGER M4-2: Central AI Hardware Budget & Latency Drift Stress Harness
Adversarially tests Raspberry Pi 5 Central AI across 100 sequential frames:
- Forecaster ONNX P95 < 5.0 ms
- Autoencoder ONNX P95 < 2.0 ms
- Full Frame process_telemetry_frame P95 < 15.0 ms
- Memory stability (RSS growth, leak detection, tensor footprint < 100 MB)
- Latency stability (detecting drift between first 20 and last 20 frames)
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import psutil
except ImportError:
    psutil = None

from pi.central_engine import CentralAIEngine
from pi.benchmark_pi5 import create_cortex_a76_session_options, _resolve_model_path

def run_challenger_stress_test():
    print("=" * 80)
    print("CHALLENGER M4-2: CENTRAL AI HARDWARE BUDGET & SEQUENTIAL STRESS HARNESS")
    print("=" * 80)

    process = psutil.Process(os.getpid()) if psutil is not None else None
    base_rss_mb = float(process.memory_info().rss / (1024.0 * 1024.0)) if process else 0.0
    print(f"[*] Initial Process RSS: {base_rss_mb:.2f} MB")

    fc_onnx = _resolve_model_path("mlops/models/central_forecaster.onnx")
    ae_onnx = _resolve_model_path("mlops/models/central_autoencoder.onnx")

    assert os.path.exists(fc_onnx), f"Forecaster ONNX not found: {fc_onnx}"
    assert os.path.exists(ae_onnx), f"Autoencoder ONNX not found: {ae_onnx}"

    sess_opts = create_cortex_a76_session_options(num_threads=4)
    fc_session = ort.InferenceSession(fc_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    ae_session = ort.InferenceSession(ae_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # -------------------------------------------------------------------------
    # PART 1: Isolated ONNX Models Latency Verification (100 iterations each)
    # -------------------------------------------------------------------------
    print("\n[PART 1: Isolated ONNX Latency Profiling (100 iterations)]")
    
    # 1.1 Forecaster ONNX
    dummy_fc_in = np.random.randn(1, 24, 14).astype(np.float32)
    for _ in range(5):
        _ = fc_session.run(None, {"input_timeseries": dummy_fc_in})
    
    fc_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = fc_session.run(None, {"input_timeseries": dummy_fc_in})
        t1 = time.perf_counter()
        fc_latencies.append((t1 - t0) * 1000.0)
    
    fc_p95 = float(np.percentile(fc_latencies, 95))
    fc_mean = float(np.mean(fc_latencies))
    fc_max = float(np.max(fc_latencies))
    print(f" -> Forecaster ONNX: Mean={fc_mean:.3f} ms, P95={fc_p95:.3f} ms, Max={fc_max:.3f} ms")
    assert fc_p95 < 5.0, f"Forecaster P95 {fc_p95:.3f} ms exceeds 5.0 ms limit!"
    print(f"    PASS: Forecaster P95 {fc_p95:.3f} ms < 5.0 ms budget")

    # 1.2 Autoencoder ONNX
    dummy_ae_in = np.random.randn(1, 48).astype(np.float32)
    for _ in range(5):
        _ = ae_session.run(None, {"input_features_48d": dummy_ae_in})
    
    ae_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = ae_session.run(None, {"input_features_48d": dummy_ae_in})
        t1 = time.perf_counter()
        ae_latencies.append((t1 - t0) * 1000.0)
    
    ae_p95 = float(np.percentile(ae_latencies, 95))
    ae_mean = float(np.mean(ae_latencies))
    ae_max = float(np.max(ae_latencies))
    print(f" -> Autoencoder ONNX: Mean={ae_mean:.3f} ms, P95={ae_p95:.3f} ms, Max={ae_max:.3f} ms")
    assert ae_p95 < 2.0, f"Autoencoder P95 {ae_p95:.3f} ms exceeds 2.0 ms limit!"
    print(f"    PASS: Autoencoder P95 {ae_p95:.3f} ms < 2.0 ms budget")

    # -------------------------------------------------------------------------
    # PART 2: Full CentralAIEngine Continuous 100 Sequential Frames Stress Test
    # -------------------------------------------------------------------------
    print("\n[PART 2: Full CentralAIEngine 100 Sequential Telemetry Frames Stress Test]")
    engine = CentralAIEngine(use_onnx=True)
    engine.onnx_forecaster_session = fc_session
    engine.onnx_autoencoder_session = ae_session

    # Compute model tensor footprint
    fc_param_mb = sum(p.numel() * p.element_size() for p in engine.forecaster.parameters()) / (1024.0 * 1024.0)
    ae_param_mb = sum(p.numel() * p.element_size() for p in engine.autoencoder.parameters()) / (1024.0 * 1024.0)
    total_model_tensor_mb = fc_param_mb + ae_param_mb
    print(f" -> PyTorch / ONNX Model Parameter Footprint: {total_model_tensor_mb:.2f} MB")
    assert total_model_tensor_mb < 100.0, f"Model parameter footprint {total_model_tensor_mb:.2f} MB >= 100 MB"
    print(f"    PASS: Model tensor footprint {total_model_tensor_mb:.2f} MB is < 100.0 MB")

    # Warmup 5 frames
    t_idx = np.arange(24, dtype=np.float32)
    for w in range(5):
        base_curve = 50.0 + 5.0 * np.sin(2.0 * np.pi * (t_idx + w) / 24.0)
        hist = np.tile(base_curve[:, None], (1, 8)).astype(np.float32)
        curr = np.tile(np.array([50.0, 70.0, 25.0, 70.0, 600.0, 120.0], dtype=np.float32), 8)
        _ = engine.process_telemetry_frame(hist, curr)

    rss_before_100 = float(process.memory_info().rss / (1024.0 * 1024.0)) if process else 0.0
    print(f"[*] Pre-loop Memory RSS: {rss_before_100:.2f} MB")

    frame_latencies = []
    checkpoint_rss = {}

    TOTAL_FRAMES = 100
    for frame_id in range(TOTAL_FRAMES):
        # Dynamically evolving telemetry across time
        t_shift = frame_id * 0.1
        curve = 45.0 + 15.0 * np.sin(2.0 * np.pi * (t_idx + t_shift) / 24.0) + (frame_id % 7) * 2.0
        hist_frame = np.tile(curve[:, None], (1, 8)).astype(np.float32)

        # Vary current stations
        current_frame = np.tile(
            np.array([
                curve[-1],
                curve[-1] * 1.35,
                24.0 + 2.0 * np.sin(frame_id * 0.05),
                65.0 + 10.0 * np.cos(frame_id * 0.05),
                620.0 + (frame_id % 10) * 15.0,
                110.0 + (frame_id % 5) * 8.0,
            ], dtype=np.float32),
            8,
        )

        t_start = time.perf_counter()
        out = engine.process_telemetry_frame(hist_frame, current_frame)
        t_end = time.perf_counter()
        dur_ms = (t_end - t_start) * 1000.0
        frame_latencies.append(dur_ms)

        # Invariant checks per frame
        fc_res = out["forecast_24h"]
        p10 = fc_res["pm25_p10"]
        p50 = fc_res["pm25_p50"]
        p90 = fc_res["pm25_p90"]
        for h in range(24):
            assert p10[h] <= p50[h] + 1e-5, f"Frame {frame_id} Horizon {h}: P10 {p10[h]} > P50 {p50[h]}"
            assert p50[h] <= p90[h] + 1e-5, f"Frame {frame_id} Horizon {h}: P50 {p50[h]} > P90 {p90[h]}"
            assert not np.isnan(p50[h]), f"Frame {frame_id} Horizon {h}: NaN in P50!"
            assert not np.isinf(p50[h]), f"Frame {frame_id} Horizon {h}: Inf in P50!"

        anom_res = out["anomaly_diagnostics"]
        assert not np.isnan(anom_res["reconstruction_score"]), f"Frame {frame_id}: NaN in reconstruction score"

        if (frame_id + 1) in [25, 50, 75, 100] and process:
            cur_rss = float(process.memory_info().rss / (1024.0 * 1024.0))
            checkpoint_rss[frame_id + 1] = cur_rss

    rss_after_100 = float(process.memory_info().rss / (1024.0 * 1024.0)) if process else 0.0
    mem_delta_100 = max(0.0, rss_after_100 - rss_before_100)

    frame_latencies = np.array(frame_latencies)
    mean_full_frame = float(np.mean(frame_latencies))
    min_full_frame = float(np.min(frame_latencies))
    p50_full_frame = float(np.percentile(frame_latencies, 50))
    p95_full_frame = float(np.percentile(frame_latencies, 95))
    p99_full_frame = float(np.percentile(frame_latencies, 99))
    max_full_frame = float(np.max(frame_latencies))

    # Latency drift analysis: First 20 vs Last 20
    first_20_mean = float(np.mean(frame_latencies[:20]))
    last_20_mean = float(np.mean(frame_latencies[-20:]))
    latency_drift_pct = ((last_20_mean - first_20_mean) / first_20_mean) * 100.0

    print("\n" + "=" * 80)
    print("CHALLENGER M4-2 CENTRAL AI STRESS TEST RESULTS:")
    print("=" * 80)
    print(f" -> Total Frames Processed:    {TOTAL_FRAMES}")
    print(f" -> Full Frame Latency Mean:   {mean_full_frame:.2f} ms")
    print(f" -> Full Frame Latency P50:    {p50_full_frame:.2f} ms")
    print(f" -> Full Frame Latency P95:    {p95_full_frame:.2f} ms (Target: < 15.0 ms)")
    print(f" -> Full Frame Latency P99:    {p99_full_frame:.2f} ms")
    print(f" -> Full Frame Latency Max:    {max_full_frame:.2f} ms")
    print(f" -> First 20 Frames Mean:      {first_20_mean:.2f} ms")
    print(f" -> Last 20 Frames Mean:       {last_20_mean:.2f} ms")
    print(f" -> Latency Drift:             {latency_drift_pct:+.2f}%")
    print(f" -> Memory RSS Checkpoints:    {checkpoint_rss}")
    print(f" -> Memory RSS Delta (100 f):  {mem_delta_100:.2f} MB")
    print(f" -> Model Tensor Footprint:    {total_model_tensor_mb:.2f} MB (Target: < 100.0 MB)")

    failures = []
    if fc_p95 >= 5.0:
        failures.append(f"Forecaster P95 {fc_p95:.2f} ms >= 5.0 ms")
    if ae_p95 >= 2.0:
        failures.append(f"Autoencoder P95 {ae_p95:.2f} ms >= 2.0 ms")
    if p95_full_frame >= 15.0:
        failures.append(f"Full Frame P95 {p95_full_frame:.2f} ms >= 15.0 ms")
    if total_model_tensor_mb >= 100.0:
        failures.append(f"Model Tensors {total_model_tensor_mb:.2f} MB >= 100.0 MB")
    if mem_delta_100 > 50.0:
        failures.append(f"Severe Memory Leak Detected: RSS grew by {mem_delta_100:.2f} MB over 100 frames")

    print("\n[VERDICT EVALUATION]")
    if not failures:
        print("RESULT: ALL CENTRAL AI HARDWARE BUDGET & SEQUENTIAL STRESS CHECKS PASSED!")
        verdict = "APPROVE"
    else:
        print(f"RESULT: FAILED CHECKS: {failures}")
        verdict = "REJECT"

    summary_data = {
        "verdict": verdict,
        "forecaster_onnx_p95_ms": fc_p95,
        "autoencoder_onnx_p95_ms": ae_p95,
        "full_frame_p95_ms": p95_full_frame,
        "full_frame_mean_ms": mean_full_frame,
        "latency_drift_pct": latency_drift_pct,
        "memory_rss_delta_mb": mem_delta_100,
        "model_tensor_footprint_mb": total_model_tensor_mb,
        "checkpoint_rss_mb": checkpoint_rss,
        "failures": failures,
    }

    with open("pi/challenger_m4_2_stress_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data

if __name__ == "__main__":
    run_challenger_stress_test()
