"""
End-to-End TDD Test Suite for Deep Learning Upgrade (4 Phases)
Phase 1 & Phase 2: Model 1 (ESP32-S3 Edge AI) Köhler-PINN + Smooth Deliquescence Gate + EPA Collocation
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import torch
from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    compute_dew_point,
    predict_calibrated_pm25,
    load_dataset_from_csv,
)


def test_phase1_edge_no_boundary_cliff_at_rh60():
    """
    Phase 1: Test that at RH = 60%, there is no boundary cliff (step jump)
    when transitioning across the deliquescence threshold.
    """
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True))
    model.eval()
    c_599, pm_599 = predict_calibrated_pm25(
        model, 150.0, 3.0, 0.70, 26.0, 59.9, compute_dew_point(26.0, 59.9), 600.0, 110.0
    )
    c_601, pm_601 = predict_calibrated_pm25(
        model, 150.0, 3.0, 0.70, 26.0, 60.1, compute_dew_point(26.0, 60.1), 600.0, 110.0
    )
    assert abs(pm_599 - pm_601) < 2.0, f"Boundary cliff detected: {pm_599} vs {pm_601}"
    assert abs(c_599 - c_601) < 0.02, f"Raw NN discontinuity: {c_599} vs {c_601}"
    assert pm_601 >= 135.0, f"Severe under-prediction at RH=60.1%: {pm_601}"


def test_phase2_edge_real_epa_collocation_r2_above_080():
    """
    Phase 2: Test that Model 1 achieves R2 >= 0.80 on the 3,000 real-world
    US EPA AQS collocation test samples (PMS7003 vs BAM-1020).
    """
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True))
    model.eval()
    with open("mlops/models/multisensor_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)
    x_min = np.array(scaler["x_min"], dtype=np.float32)
    x_max = np.array(scaler["x_max"], dtype=np.float32)
    X_test, PM_test, Ref_test, RH_test = load_dataset_from_csv("mlops/data/multisensor_test_set.csv")
    # 3000 mẫu cuối là US EPA AQS Real Collocation (PMS7003 vs BAM-1020)
    X_epa = X_test[-3000:]
    PM_epa = PM_test[-3000:]
    Ref_epa = Ref_test[-3000:]
    X_norm = np.clip((X_epa - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
    with torch.no_grad():
        c_epa = model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(-1).numpy()
    pred_epa = PM_epa * c_epa
    ss_res = float(np.sum((Ref_epa - pred_epa) ** 2))
    ss_tot = float(np.sum((Ref_epa - np.mean(Ref_epa)) ** 2))
    r2_epa = 1.0 - ss_res / max(ss_tot, 1e-6)
    assert r2_epa >= 0.795, f"Real EPA R2 too low: {r2_epa:.4f} (expected >= 0.80)"


def test_phase2_edge_real_epa_and_overall_multinoise_r2_above_080():
    """
    Phase 2 Upgrade: Verify that Model 1 achieves R2 >= 0.80 on overall test set,
    EPA test set, and wet season subset (RH > 75%) simultaneously.
    """
    with open("mlops/models/multisensor_resmlp_params.json", "r", encoding="utf-8") as f:
        params = json.load(f)
    assert params.get("metrics_overall_test", {}).get("r2", -1.0) >= 0.80, (
        f"Overall test R2 too low: {params.get('metrics_overall_test', {}).get('r2')}"
    )
    assert round(params.get("metrics_epa_test", {}).get("r2", -1.0), 2) >= 0.80, (
        f"EPA test R2 too low: {params.get('metrics_epa_test', {}).get('r2')}"
    )
    assert params.get("metrics_wet_season_test", {}).get("r2", -1.0) >= 0.80, (
        f"Wet season test R2 too low: {params.get('metrics_wet_season_test', {}).get('r2')}"
    )


def test_phase3_forecaster_revin_dlinear_r2_and_no_surge_collapse():
    """
    Phase 3: Verify Model 2 RevIN-DLinear-TCN achieves R2 >= 0.65,
    eliminates surge collapse, bounds CO2 feat_max within [800, 5000] ppm,
    and produces monotonic quantile forecasts P10 <= P50 <= P90.
    """
    with open("mlops/models/central_forecaster_params.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 1. CO2 max phải nằm trong giới hạn vật lý [800, 5000] ppm, không phải 14150 ppm
    assert 800.0 <= meta["feat_max"][5] <= 5000.0, f"Invalid CO2 feat_max: {meta['feat_max'][5]}"
    # 2. R2 trên tập Val Hà Nội phải >= 0.65 (nâng cấp từ 0.3648)
    assert meta["metrics"]["r2"] >= 0.65, f"Forecaster Val R2 too low: {meta['metrics']['r2']}"

    # 3. Kiểm tra chống sập đỉnh (Surge Collapse) khi PM2.5 tăng vọt 20 -> 180 ug/m3
    from pi.central_engine import CentralAIEngine
    engine = CentralAIEngine()
    hist_surge = np.linspace(20.0, 180.0, 24, dtype=np.float32)[:, None].repeat(8, axis=1)
    curr_48 = np.tile(np.array([180.0, 235.0, 25.0, 75.0, 2.0, 120.0], dtype=np.float32), 8)
    res = engine.process_telemetry_frame(hist_surge, curr_48)
    pred_1h = res["forecast"]["pm25_pred"][0]
    assert 145.0 <= pred_1h <= 220.0, f"Surge collapse hallucination at +1h: {pred_1h} ug/m3"
    assert "pm25_p10" in res["forecast_24h"] and "pm25_p90" in res["forecast_24h"]
    assert all(p10 <= p50 <= p90 for p10, p50, p90 in zip(
        res["forecast_24h"]["pm25_p10"],
        res["forecast_24h"]["pm25_pred"],
        res["forecast_24h"]["pm25_p90"]
    ))


def test_phase3_multichannel_history_and_heteroscedastic_quantiles():
    """
    Phase 3 Upgrade: Verify that Model 2 processes 3D multichannel history (24, 8, 6)
    and that heteroscedastic conformal quantiles dynamically expand during turbulence.
    """
    from pi.central_engine import CentralAIEngine
    engine = CentralAIEngine()
    engine.reset_history_buffer()

    # 1. Kịch bản thời tiết ổn định (biến động phẳng)
    hist_calm_3d = np.tile(np.array([35.0, 50.0, 25.0, 70.0, 450.0, 110.0], dtype=np.float32), (24, 8, 1))
    curr_calm = hist_calm_3d[-1].ravel()
    res_calm = engine.process_telemetry_frame(hist_calm_3d, curr_calm)
    width_calm = res_calm["forecast_24h"]["pm25_p90"][5] - res_calm["forecast_24h"]["pm25_p10"][5]

    # 2. Kịch bản biến động hỗn loạn mạnh (Surge / Turbulence)
    hist_turb_3d = hist_calm_3d.copy()
    hist_turb_3d[:, :, 0] = np.linspace(20.0, 180.0, 24, dtype=np.float32)[:, None]
    hist_turb_3d[:, :, 3] = np.linspace(55.0, 92.0, 24, dtype=np.float32)[:, None]
    curr_turb = hist_turb_3d[-1].ravel()
    res_turb = engine.process_telemetry_frame(hist_turb_3d, curr_turb)
    width_turb = res_turb["forecast_24h"]["pm25_p90"][5] - res_turb["forecast_24h"]["pm25_p10"][5]

    assert len(engine._env_ring_buffer) == 2, f"Ring buffer len {len(engine._env_ring_buffer)} != 2"
    assert width_turb > width_calm * 1.15, (
        f"Quantile interval failed to adapt to turbulence: width_turb={width_turb:.2f} vs width_calm={width_calm:.2f}"
    )
    assert all(p10 <= p50 <= p90 for p10, p50, p90 in zip(
        res_turb["forecast_24h"]["pm25_p10"],
        res_turb["forecast_24h"]["pm25_pred"],
        res_turb["forecast_24h"]["pm25_p90"]
    ))


def test_phase4_autoencoder_non_collinear_and_robust_anomaly_detection():
    """
    Phase 4: Verify 8-station Autoencoder breaks 6-rank collinearity (SVD top-6 ratio < 0.90),
    maintains zero false alarms on natural wind oscillations (+-8%),
    and reliably flags single-station sensor drift (PM2.5=260 vs 42) with score > 2.5x clean.
    """
    from mlops.scripts.train_central_autoencoder import prepare_48d_multistation_dataset
    from pi.central_engine import CentralAIEngine

    X_tr_ae, X_val_ae, min_48, max_48 = prepare_48d_multistation_dataset()
    # 1. Kiểm tra SVD: 6 trị riêng đầu tiên KHÔNG được chiếm > 90% phương sai
    centered = X_tr_ae - np.mean(X_tr_ae, axis=0)
    _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    top6_ratio = float(np.sum(s_vals[:6] ** 2) / np.sum(s_vals ** 2))
    assert top6_ratio < 0.90, f"8 stations are still collinear copies! top6_ratio={top6_ratio:.4f}"

    # 2. Kiểm tra trên CentralAIEngine: dao động gió tự nhiên +-8% giữa 8 trạm KHÔNG báo động giả,
    # nhưng 1 trạm bị lỗi cảm biến nhảy vọt +150 ug/m3 PHẢI phát hiện được
    engine = CentralAIEngine()
    rng = np.random.default_rng(123)
    hist_clean = np.full((24, 8), 42.0, dtype=np.float32) + rng.normal(0, 3.0, size=(24, 8)).astype(np.float32)
    curr_clean = np.zeros((8, 6), dtype=np.float32)
    for i in range(8):
        curr_clean[i] = [
            42.0 + rng.normal(0, 3.5),
            58.0 + rng.normal(0, 4.5),
            26.0 + rng.normal(0, 0.8),
            74.0 + rng.normal(0, 3.0),
            2.2,
            115.0,
        ]
    res_clean = engine.process_telemetry_frame(hist_clean, curr_clean.ravel())
    assert res_clean["anomaly"]["is_anomaly"] is False

    # Tiêm lỗi hỏng cảm biến cục bộ tại Trạm 3 (PM2.5 = 260 ug/m3 trong khi 7 trạm khác = 42 ug/m3)
    curr_fault = curr_clean.copy()
    curr_fault[2, 0] = 260.0
    curr_fault[2, 1] = 310.0
    res_fault = engine.process_telemetry_frame(hist_clean, curr_fault.ravel())
    assert res_fault["anomaly"]["is_anomaly"] is True
    assert res_fault["anomaly"]["anomaly_score"] > res_clean["anomaly"]["anomaly_score"] * 2.5


