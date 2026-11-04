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


def test_bug02_edge_dry_dust_storm_no_hallucination():
    """
    BUG-02: Verify MultiSensorResMLP does NOT hallucinate c_factor ~ 0.511 on dry dust storm
    (PM=250, RH=25%, T=30C, CO2=420, VOC=100), both in predict_calibrated_pm25() and raw model output,
    and verify train_and_plot_multinoise.py does not use bogus linear dew point formula.
    """
    import json
    import torch
    from mlops.scripts.train_multisensor_resmlp import MultiSensorResMLP, predict_calibrated_pm25
    from pi.data_pipeline import calc_dew_point_magnus_tetens

    model_path = "mlops/models/multisensor_resmlp.pt"
    scaler_path = "mlops/models/multisensor_scaler.json"

    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    x_min = np.array(scaler["x_min"], dtype=np.float32)
    x_max = np.array(scaler["x_max"], dtype=np.float32)

    # 1. Check raw neural network output c_factor directly (proving NN learned dry-air physics)
    dp_dry = float(calc_dew_point_magnus_tetens(30.0, 25.0))
    raw_feat = np.array([250.0, 0.0, 0.70, 30.0, 25.0, dp_dry, 420.0, 100.0], dtype=np.float32)
    norm_feat = np.clip((raw_feat - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
    x_norm = torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        c_factor = float(model(x_norm).item())
    assert 0.85 <= c_factor <= 1.15, (
        f"Raw NN hallucinated c_factor={c_factor:.3f} outside [0.85, 1.15] on dry dust storm!"
    )

    # 2. Check predict_calibrated_pm25 under both calling conventions
    c_kw, pm_calib_kw = predict_calibrated_pm25(
        model, pm_avg=250.0, diff_pm=0.0, r_pm=0.70, temp=30.0, rh=25.0, dew_point=dp_dry, co2=420.0, gas_proxy=100.0
    )
    assert 225.0 <= pm_calib_kw <= 260.0, (
        f"Expected calibrated PM2.5 in [225.0, 260.0] (kw), got {pm_calib_kw}"
    )

    res_pos = predict_calibrated_pm25(model, 250.0, 250.0, 25.0, 30.0, 1008.0, 420.0, 100.0, x_min, x_max)
    pm_calib_pos = res_pos[1] if isinstance(res_pos, tuple) else float(res_pos)
    assert 225.0 <= pm_calib_pos <= 260.0, (
        f"Expected calibrated PM2.5 in [225.0, 260.0] on dry dust storm, got {pm_calib_pos}"
    )

    # 3. Verify train_and_plot_multinoise.py does not contain bogus linear dew point formula
    with open("mlops/scripts/train_and_plot_multinoise.py", "r", encoding="utf-8") as f:
        multinoise_src = f.read()
    assert "rh_val * 0.22" not in multinoise_src, (
        "train_and_plot_multinoise.py still contains bogus linear dew point 'rh_val * 0.22'"
    )


def test_bug03_data_pipeline_scaler_sync():
    """
    BUG-03: Verify TINYML_X_MIN and TINYML_X_MAX in pi/data_pipeline.py match
    mlops/models/robust_resmlp_19k_params.json.
    """
    import json
    from pi.data_pipeline import TINYML_X_MIN, TINYML_X_MAX

    with open("mlops/models/robust_resmlp_19k_params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    assert np.allclose(TINYML_X_MIN, params["x_min"], atol=1e-3), (
        f"TINYML_X_MIN {TINYML_X_MIN} != params['x_min'] {params['x_min']}"
    )
    assert np.allclose(TINYML_X_MAX, params["x_max"], atol=1e-3), (
        f"TINYML_X_MAX {TINYML_X_MAX} != params['x_max'] {params['x_max']}"
    )


