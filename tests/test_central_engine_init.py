import pytest
import numpy as np
from pi.central_engine import CentralAIEngine

def test_central_engine_init_and_ring_buffer():
    engine = CentralAIEngine(forecaster_weights=None, autoencoder_weights=None, unified_weights=None)
    assert hasattr(engine, "_env_ring_buffer"), "CentralAIEngine missing _env_ring_buffer"
    assert len(engine._env_ring_buffer) == 0
    history = np.zeros((24, 1), dtype=np.float32)
    current = np.array([35.0, 48.0, 25.0, 75.0, 420.0, 100.0], dtype=np.float32)
    res = engine.process_telemetry_frame(history, current)
    assert "forecast_24h" in res
    assert "anomaly" in res
