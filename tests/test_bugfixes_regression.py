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


def test_bug04_no_header_overwrite_and_strict_train_scaler(tmp_path):
    """
    BUG-04: Verify:
    1. quantize_19k_int8.py outputs to model_data_19k.h and never overwrites esp/include/model_data.h.
    2. train_robust_resmlp_19k.py does NOT use np.random.permutation for splitting, uses chronological
       split with BUFFER_HOURS = 48, excludes test split from Phase 1 pre-training, and fits x_min/x_max
       strictly on np.vstack([X_mega_train, X_tx_train]).
    3. train_central_forecaster.py and train_central_autoencoder.py compute feat_min/feat_max and
       min_48/max_48 strictly on the training slice (ts_train / train_raw) AFTER chronological splitting
       with a 48h buffer.
    """
    import csv
    from mlops.scripts.train_central_forecaster import prepare_time_series_data
    from mlops.scripts.train_central_autoencoder import prepare_48d_multistation_dataset

    # 1. Check quantize_19k_int8.py target header path and esp/include/model_data.h integrity
    with open("mlops/scripts/quantize_19k_int8.py", "r", encoding="utf-8") as f:
        quant19k_src = f.read()
    assert '"model_data_19k.h"' in quant19k_src, (
        "quantize_19k_int8.py must write to model_data_19k.h, not model_data.h"
    )
    assert 'os.path.join(base_dir, "..", "..", "esp", "include", "model_data.h")' not in quant19k_src, (
        "quantize_19k_int8.py still overwrites esp/include/model_data.h"
    )
    with open("esp/include/model_data.h", "r", encoding="utf-8") as f:
        header_src = f.read()
    assert "#define TINYML_INPUT_DIM 8" in header_src and "FC_IN_WEIGHT" in header_src, (
        "esp/include/model_data.h must remain the 8-feature MultiSensor header"
    )

    # 2. Check train_robust_resmlp_19k.py chronological split with 48h buffer & leakage-free Phase 1 / scaler
    with open("mlops/scripts/train_robust_resmlp_19k.py", "r", encoding="utf-8") as f:
        resmlp19k_src = f.read()
    assert "np.random.permutation" not in resmlp19k_src, (
        "train_robust_resmlp_19k.py must not use np.random.permutation on time-series data"
    )
    assert "BUFFER_HOURS = 48" in resmlp19k_src, (
        "train_robust_resmlp_19k.py must use BUFFER_HOURS = 48"
    )
    assert "X_tx[split_idx + BUFFER_HOURS:]" in resmlp19k_src, (
        "train_robust_resmlp_19k.py must slice X_tx_test after BUFFER_HOURS"
    )
    assert "np.vstack([X_mega_train, X_tx_train])" in resmlp19k_src, (
        "train_robust_resmlp_19k.py must compute x_min/x_max strictly on np.vstack([X_mega_train, X_tx_train])"
    )

    # 3. Check train_central_forecaster.py chronological split + buffer + train-only scaler
    with open("mlops/scripts/train_central_forecaster.py", "r", encoding="utf-8") as f:
        forecaster_src = f.read()
    assert "BUFFER_HOURS = 48" in forecaster_src, "train_central_forecaster.py missing BUFFER_HOURS = 48"
    assert "ts_train = ts_np[:split_raw]" in forecaster_src, "train_central_forecaster.py missing ts_train slice"
    assert "ts_train.min(axis=0)" in forecaster_src and "ts_train.max(axis=0)" in forecaster_src, (
        "train_central_forecaster.py must compute feat_min/feat_max strictly on ts_train"
    )

    # Behavioral check: outlier in validation tail must NOT leak into feat_max
    fake_hanoi_csv = tmp_path / "hanoi_leak_test.csv"
    with open(fake_hanoi_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PM25", "PM10", "Temperature", "Relative Humidity", "CO", "NO2", "Wind Speed", "Pressure"])
        for i in range(400):
            is_val_tail = i >= 360
            pm25_val = 9999.0 if is_val_tail else 50.0 + (i % 20)
            writer.writerow([pm25_val, 80.0, 25.0, 75.0, 100.0, 20.0, 2.5, 1012.0])

    fc_data = prepare_time_series_data(hanoi_path=str(fake_hanoi_csv), seq_in=24, seq_out=24)
    feat_min, feat_max = fc_data[-2], fc_data[-1]
    assert feat_max[0] < 100.0, (
        f"Data leakage detected in prepare_time_series_data: validation outlier leaked into feat_max[0]={feat_max[0]}"
    )

    # 4. Check train_central_autoencoder.py chronological split + buffer + train-only scaler
    with open("mlops/scripts/train_central_autoencoder.py", "r", encoding="utf-8") as f:
        ae_src = f.read()
    assert "BUFFER_HOURS = 48" in ae_src, "train_central_autoencoder.py missing BUFFER_HOURS = 48"
    assert "train_raw = arr_48[:split]" in ae_src, "train_central_autoencoder.py missing train_raw slice"
    assert "train_raw.min(axis=0)" in ae_src and "train_raw.max(axis=0)" in ae_src, (
        "train_central_autoencoder.py must compute min_48/max_48 strictly on train_raw"
    )

    # Behavioral check: outlier in validation tail must NOT leak into max_48
    fake_ms_csv = tmp_path / "ms_leak_test.csv"
    with open(fake_ms_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "station_id", "pm25", "pm10", "temperature_c", "relative_humidity", "co", "no2"])
        for t_idx in range(300):
            is_val_tail = t_idx >= 260
            pm25_val = 8888.0 if is_val_tail else 45.0 + (t_idx % 15)
            t_str = f"2025-01-{(t_idx // 24) + 1:02d}T{t_idx % 24:02d}:00"
            for s in range(1, 9):
                writer.writerow([t_str, f"STATION_{s:02d}", pm25_val, 70.0, 24.0, 72.0, 120.0, 18.0])

    ae_data = prepare_48d_multistation_dataset(csv_path=str(fake_ms_csv))
    min_48, max_48 = ae_data[-2], ae_data[-1]
    assert max_48[0] < 150.0, (
        f"Data leakage detected in prepare_48d_multistation_dataset: validation outlier leaked into max_48[0]={max_48[0]}"
    )


def test_bug05_no_synthetic_clock_wind_and_distinct_station_scalers():
    """
    BUG-05: Verify:
    1. mlops/scripts/train_unified_central_ai.py and mlops/scripts/train_central_forecaster.py
       do not contain deterministic clock wind direction formula '(idx * 15) % 360'.
    2. mlops/models/central_autoencoder_params.json has distinct min/max scalers for each
       of the 8 stations, i.e., len(np.unique(np.round(max_48.reshape(8, 6)[:, 0], 4))) == 8.
    """
    import json

    with open("mlops/scripts/train_unified_central_ai.py", "r", encoding="utf-8") as f:
        unified_src = f.read()
    assert "(idx * 15) % 360" not in unified_src, (
        "mlops/scripts/train_unified_central_ai.py still contains synthetic clock wind '(idx * 15) % 360'"
    )

    with open("mlops/scripts/train_central_forecaster.py", "r", encoding="utf-8") as f:
        forecaster_src = f.read()
    assert "(idx * 15) % 360" not in forecaster_src, (
        "mlops/scripts/train_central_forecaster.py still contains synthetic clock wind '(idx * 15) % 360'"
    )

    params_path = "mlops/models/central_autoencoder_params.json"
    assert os.path.exists(params_path), f"{params_path} does not exist"
    with open(params_path, "r", encoding="utf-8") as f:
        ae_params = json.load(f)

    max_48 = np.array(ae_params["max_48"], dtype=np.float32)
    assert max_48.shape == (48,), f"Expected max_48 of length 48, got {max_48.shape}"
    max_8x6 = max_48.reshape(8, 6)

    distinct_pm25 = np.unique(np.round(max_8x6[:, 0], 4))
    assert len(distinct_pm25) == 8, (
        f"Expected 8 distinct PM2.5 maximums across stations, got {len(distinct_pm25)}: {distinct_pm25}"
    )

    distinct_temp = np.unique(np.round(max_8x6[:, 2], 4))
    assert len(distinct_temp) == 8, (
        f"Expected 8 distinct temp maximums across stations, got {len(distinct_temp)}: {distinct_temp}"
    )

    distinct_rh = np.unique(np.round(max_8x6[:, 3], 4))
    assert len(distinct_rh) == 8, (
        f"Expected 8 distinct RH maximums across stations, got {len(distinct_rh)}: {distinct_rh}"
    )




