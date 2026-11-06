import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np
from pi.central_engine import CentralAIEngine
from pi.benchmark_pi5 import create_cortex_a76_session_options, _resolve_model_path
import onnxruntime as ort

engine = CentralAIEngine(use_onnx=True)
sess_opts = create_cortex_a76_session_options(num_threads=4)
fc_onnx = _resolve_model_path("mlops/models/central_forecaster.onnx")
ae_onnx = _resolve_model_path("mlops/models/central_autoencoder.onnx")
engine.onnx_forecaster_session = ort.InferenceSession(fc_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
engine.onnx_autoencoder_session = ort.InferenceSession(ae_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])

t_idx = np.arange(24, dtype=np.float32)
base_curve = 55.0 + 3.0 * np.sin(2.0 * np.pi * t_idx / 24.0)
history_24x8 = np.tile(base_curve[:, None], (1, 8)).astype(np.float32)
current_48d = np.tile(np.array([55.0, 75.0, 26.0, 75.0, 680.0, 130.0], dtype=np.float32), 8)

# Profile 20 frames
times_spatial = []
times_feature_prep = []
times_forecaster = []
times_autoencoder = []
times_anomaly_classify = []
times_total = []

for _ in range(5):
    engine.process_telemetry_frame(history_24x8, current_48d)

for _ in range(20):
    t_tot0 = time.perf_counter()

    # 1. Spatial engine
    t0 = time.perf_counter()
    spatial_results = engine.spatial_engine.analyze(history_24x8, dt_s=30.0, max_lag=12)
    t1 = time.perf_counter()
    times_spatial.append((t1 - t0) * 1000.0)

    # 2. Forecaster prep & run
    t0 = time.perf_counter()
    dummy_feat_in = np.zeros((1, 24, 14), dtype=np.float32)
    forecast_norm = engine.onnx_forecaster_session.run(None, {"input_timeseries": dummy_feat_in})[0][0]
    t1 = time.perf_counter()
    times_forecaster.append((t1 - t0) * 1000.0)

    # 3. Autoencoder run
    t0 = time.perf_counter()
    dummy_ae = np.zeros((1, 48), dtype=np.float32)
    rec_ae = engine.onnx_autoencoder_session.run(None, {"input_features_48d": dummy_ae})[0]
    t1 = time.perf_counter()
    times_autoencoder.append((t1 - t0) * 1000.0)

    t_tot1 = time.perf_counter()
    # Entire process_telemetry_frame call
    t_call0 = time.perf_counter()
    _ = engine.process_telemetry_frame(history_24x8, current_48d)
    t_call1 = time.perf_counter()
    times_total.append((t_call1 - t_call0) * 1000.0)

print(f"Spatial Engine Mean:      {np.mean(times_spatial):.3f} ms (P95: {np.percentile(times_spatial, 95):.3f} ms)")
print(f"Forecaster ONNX Mean:     {np.mean(times_forecaster):.3f} ms (P95: {np.percentile(times_forecaster, 95):.3f} ms)")
print(f"Autoencoder ONNX Mean:    {np.mean(times_autoencoder):.3f} ms (P95: {np.percentile(times_autoencoder, 95):.3f} ms)")
print(f"Full Process Call Mean:   {np.mean(times_total):.3f} ms (P95: {np.percentile(times_total, 95):.3f} ms)")
print(f"Remainder (AQI, Python):  {np.mean(times_total) - np.mean(times_spatial) - np.mean(times_forecaster) - np.mean(times_autoencoder):.3f} ms")
