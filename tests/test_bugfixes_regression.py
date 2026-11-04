"""
Regression Test Suite for AI Models Architecture, Hallucination & Data Leakage Fixes (BUG-01 to BUG-07).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.central_engine import CentralAIEngine


def test_bug01_central_engine_physical_scale_and_clean_anomaly():
    """
    BUG-01: Verify CentralAIEngine.process_telemetry_frame() normalizes inputs,
    denormalizes 24h PM2.5 and AQI forecasts into physical units, and produces
    calibrated low anomaly scores (< 5.0, is_anomaly=False) on clean normal telemetry.
    """
    engine = CentralAIEngine()

    t_idx = np.arange(24, dtype=np.float32)
    base_curve = 55.0 + 3.0 * np.sin(2.0 * np.pi * t_idx / 24.0)
    history_24x8 = np.tile(base_curve[:, None], (1, 8)).astype(np.float32)
    history_8x24 = history_24x8.T.copy()

    # Standard 8-station 48D vector ([pm25, pm10, temp, rh, co2, voc] per station)
    current_48d_gas = np.tile(
        np.array([55.0, 75.0, 26.0, 75.0, 680.0, 130.0], dtype=np.float32), 8
    )
    res_std = engine.process_telemetry_frame(history_24x8, current_48d_gas)

    pm25_pred_24h = res_std["forecast_24h"]["pm25_pred"]
    aqi_pred_24h = res_std["forecast_24h"]["aqi_pred"]
    anom_score_std = res_std["anomaly_diagnostics"]["reconstruction_score"]
    is_anom_std = res_std["anomaly_diagnostics"]["is_anomaly"]

    assert all(10.0 <= p <= 200.0 for p in pm25_pred_24h), (
        f"Expected physical PM2.5 forecast in [10, 200] ug/m3, got {pm25_pred_24h[:5]} "
        f"(anom_score={anom_score_std:.2f})"
    )
    assert all(15.0 <= a <= 300.0 for a in aqi_pred_24h), (
        f"Expected physical AQI forecast in [15, 300], got {aqi_pred_24h[:5]}"
    )
    assert anom_score_std < 5.0, f"Expected clean anomaly score < 5.0, got {anom_score_std}"
    assert is_anom_std is False, f"Expected is_anomaly=False on clean telemetry, got {is_anom_std}"

    # Brief calling convention: (current_48d, history_8x24) with [pm2_5, pm10, temp, rh, wind_spd, wind_dir]
    current_48d_wind = np.tile(
        np.array([55.0, 95.0, 26.0, 75.0, 2.5, 110.0], dtype=np.float32), 8
    )
    result = engine.process_telemetry_frame(current_48d_wind, history_8x24)

    pm25_preds = result["forecast"]["pm25_6h_12h_24h"]
    aqi_preds = result["forecast"]["aqi_6h_12h_24h"]
    assert len(pm25_preds) == 3 and all(10.0 <= p <= 200.0 for p in pm25_preds), (
        f"pm25_6h_12h_24h out of physical range [10, 200]: {pm25_preds}"
    )
    assert len(aqi_preds) == 3 and all(15.0 <= a <= 300.0 for a in aqi_preds), (
        f"aqi_6h_12h_24h out of physical range [15, 300]: {aqi_preds}"
    )
    assert result["anomaly"]["anomaly_score"] < 5.0, (
        f"Expected anomaly_score < 5.0, got {result['anomaly']['anomaly_score']}"
    )
    assert result["anomaly"]["is_anomaly"] is False

    # Verify true extreme spike triggers is_anomaly == True
    spike_48d = current_48d_gas.copy()
    spike_48d[0:6] = np.array([450.0, 600.0, 32.0, 60.0, 3200.0, 480.0], dtype=np.float32)
    res_spike = engine.process_telemetry_frame(history_24x8, spike_48d)
    assert res_spike["anomaly"]["is_anomaly"] is True, (
        f"Expected extreme fire spike to trigger is_anomaly=True, score={res_spike['anomaly']['anomaly_score']}"
    )

