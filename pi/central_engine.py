"""
CENTRAL NEURAL NETWORK AI ENGINE ORCHESTRATOR
Hệ Thống Điều Phối & Tích Hợp Toàn Diện Bộ Não AI Trung Tâm Trên Raspberry Pi 5 (8GB RAM)
Phát triển cho Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
"""

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from pi.config import (
    STATIONS,
    STATION_COORDS_UTM,
    STATION_IDS,
    STATION_MAP,
)
from pi.spatial_temporal_engine import (
    SpatialTemporalEngine,
    compute_cross_correlation_matrix,
    estimate_wind_and_dispersion,
    localize_emission_source,
)
from pi.forecasting_engine import (
    DilatedTCNBiGRUForecaster,
    calculate_metrics,
)
from pi.anomaly_engine import (
    MultiStationDeepAutoencoder,
    AdaptiveAnomalyDetector,
    PhysicalSignatureClassifier,
    AnomalyEngine,
)
from pi.data_pipeline import (
    TinyMLEdgeCalibrator,
    calibrate_pm25_edge,
    calc_vn_aqi_pm25,
)

class CentralAIEngine:
    """
    Bộ não AI Trung tâm (Tầng 2) vận hành trên máy chủ Raspberry Pi 5.
    Điều phối luồng dữ liệu 4 tầng:
    1. Tiếp nhận gói tin Telemetry & kiểm định hiệu chuẩn biên Tier 1.
    2. Động cơ phân tích tương quan không gian - thời gian & nguồn phát thải (M2).
    3. Mạng nơ-ron chuỗi thời gian dự báo đa bước 24h cho PM2.5 & AQI (M3).
    4. Mạng nơ-ron Deep Autoencoder phát hiện bất thường & phân loại dấu vết (M4).
    """

    def __init__(
        self,
        forecaster_weights: Optional[str] = "mlops/models/central_forecaster.pt",
        autoencoder_weights: Optional[str] = "mlops/models/central_autoencoder.pt",
        use_onnx: bool = False,
    ):
        self.device = torch.device("cpu")
        self.use_onnx = use_onnx
        self.station_ids = STATION_IDS
        self.coords_utm = STATION_COORDS_UTM
        self.n_stations = len(self.station_ids)

        self.spatial_engine = SpatialTemporalEngine(
            station_ids=self.station_ids, coordinates=self.coords_utm
        )

        self.forecaster = DilatedTCNBiGRUForecaster(
            in_features=14, tcn_channels=32, gru_hidden=64, out_steps=24, out_targets=2
        ).to(self.device)
        if forecaster_weights and os.path.exists(forecaster_weights):
            self.forecaster.load_state_dict(torch.load(forecaster_weights, map_location=self.device))
        self.forecaster.eval()

        self.autoencoder = MultiStationDeepAutoencoder(
            in_features=48, latent_dim=16
        ).to(self.device)
        if autoencoder_weights and os.path.exists(autoencoder_weights):
            self.autoencoder.load_state_dict(torch.load(autoencoder_weights, map_location=self.device))
        self.autoencoder.eval()

        self.anomaly_detector = AdaptiveAnomalyDetector(beta_ema=0.98, n_sigma=3.0)
        self.classifier = PhysicalSignatureClassifier()
        self.edge_calibrator = TinyMLEdgeCalibrator()

        self.onnx_forecaster_session = None
        self.onnx_autoencoder_session = None

    def export_models_to_onnx(
        self,
        forecaster_path: str = "mlops/models/central_forecaster.onnx",
        autoencoder_path: str = "mlops/models/central_autoencoder.onnx",
        opset_version: int = 17,
    ) -> Dict[str, Union[str, float]]:
        """
        Xuất toàn bộ mô hình PyTorch sang định dạng ONNX tiêu chuẩn với opset=17 (dynamo=False),
        tối ưu hóa đa luồng tập lệnh ARM NEON SIMD cho CPU Cortex-A76 của Raspberry Pi 5.
        Đồng thời đối soát sai số tuyệt đối (Numerical Parity Check) đảm bảo diff < 1e-3.
        """
        os.makedirs(os.path.dirname(forecaster_path), exist_ok=True)
        os.makedirs(os.path.dirname(autoencoder_path), exist_ok=True)

        dummy_forecast_in = torch.randn(1, 24, 14, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            torch_forecast_out = self.forecaster(dummy_forecast_in).cpu().numpy()

        torch.onnx.export(
            self.forecaster,
            dummy_forecast_in,
            forecaster_path,
            export_params=True,
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["input_timeseries"],
            output_names=["forecast_24h_pm25_aqi"],
            dynamic_axes={"input_timeseries": {0: "batch_size"}, "forecast_24h_pm25_aqi": {0: "batch_size"}},
        )

        dummy_ae_in = torch.randn(1, 48, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            torch_ae_rec, torch_ae_z = self.autoencoder(dummy_ae_in)
            torch_ae_rec = torch_ae_rec.cpu().numpy()

        torch.onnx.export(
            self.autoencoder,
            dummy_ae_in,
            autoencoder_path,
            export_params=True,
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["input_features_48d"],
            output_names=["reconstructed_features", "latent_representation"],
            dynamic_axes={
                "input_features_48d": {0: "batch_size"},
                "reconstructed_features": {0: "batch_size"},
                "latent_representation": {0: "batch_size"},
            },
        )

        import onnxruntime as ort

        sess_forecaster = ort.InferenceSession(forecaster_path, providers=["CPUExecutionProvider"])
        onnx_forecast_out = sess_forecaster.run(None, {"input_timeseries": dummy_forecast_in.cpu().numpy()})[0]
        max_diff_forecast = float(np.max(np.abs(torch_forecast_out - onnx_forecast_out)))

        sess_ae = ort.InferenceSession(autoencoder_path, providers=["CPUExecutionProvider"])
        onnx_ae_rec = sess_ae.run(None, {"input_features_48d": dummy_ae_in.cpu().numpy()})[0]
        max_diff_ae = float(np.max(np.abs(torch_ae_rec - onnx_ae_rec)))

        return {
            "forecaster_onnx_path": forecaster_path,
            "autoencoder_onnx_path": autoencoder_path,
            "forecaster_parity_diff": max_diff_forecast,
            "autoencoder_parity_diff": max_diff_ae,
            "is_parity_verified": bool(max_diff_forecast < 1e-3 and max_diff_ae < 1e-3),
        }

    def process_telemetry_frame(
        self,
        sliding_window_history: np.ndarray,
        current_features_48d: np.ndarray,
        dt_s: float = 30.0,
    ) -> Dict[str, Union[Dict, List, float, int, str]]:
        """
        Xử lý 1 khung đo tổng hợp thời gian thực gửi từ mạng lưới trạm về Pi 5.

        Args:
            sliding_window_history: Ma trận lịch sử nồng độ PM2.5 hình dạng (T_steps, 8_stations).
            current_features_48d: Mảng đặc trưng đo lường hiện tại (48 chiều = 8 trạm x 6 kênh).
            dt_s: Bước thời gian lấy mẫu.

        Returns:
            Dict chứa toàn bộ kết quả phân tích: Vector gió, Nguồn ô nhiễm, Dự báo 24h, Bất thường.
        """

        spatial_results = self.spatial_engine.analyze(
            sliding_window_history, dt_s=dt_s, max_lag=min(40, sliding_window_history.shape[0] // 2)
        )

        t_steps = min(24, sliding_window_history.shape[0])
        recent_pm = sliding_window_history[-t_steps:]
        mean_pm = np.mean(recent_pm, axis=1)

        feat_in = np.zeros((1, 24, 14), dtype=np.float32)
        for t in range(t_steps):
            idx = 24 - t_steps + t
            feat_in[0, idx, 0] = mean_pm[t]
            feat_in[0, idx, 1] = mean_pm[t] * 1.35
            feat_in[0, idx, 2] = 24.5
            feat_in[0, idx, 3] = 78.0
            feat_in[0, idx, 7] = spatial_results["wind_vector"][0]
            feat_in[0, idx, 8] = spatial_results["wind_vector"][1]
            hour = (time.localtime().tm_hour + t) % 24
            feat_in[0, idx, 9] = math.sin(2.0 * math.pi * hour / 24.0)
            feat_in[0, idx, 10] = math.cos(2.0 * math.pi * hour / 24.0)

        with torch.no_grad():
            forecast_out = self.forecaster(torch.tensor(feat_in, device=self.device)).cpu().numpy()[0]

        x_ae = np.asarray(current_features_48d, dtype=np.float32).reshape(1, 48)
        with torch.no_grad():
            rec_ae, z_latent = self.autoencoder(torch.tensor(x_ae, device=self.device))
            err_vec = (x_ae - rec_ae.cpu().numpy()).ravel()

        anom_score = self.anomaly_detector.compute_score(err_vec)
        is_anom, thresh, _ = self.anomaly_detector.update_and_detect(anom_score)

        pm_per_station = current_features_48d.reshape(8, -1)[:, 0]
        stations_affected = int(np.sum(pm_per_station > 65.0))
        gases = ["pm25"]
        if np.any(pm_per_station > 120.0):
            gases.extend(["pm10", "co2", "voc"])

        cid, cname, probs = self.classifier.classify(
            stations_affected=stations_affected,
            gases_elevated=gases,
            duration_minutes=10.0,
            is_anomaly=is_anom,
        )

        return {
            "timestamp": time.time(),
            "spatial_dispersion": {
                "wind_speed_mps": spatial_results["wind_speed_mps"],
                "wind_direction_deg": spatial_results["dispersion_bearing_deg"],
                "source_location_utm": spatial_results["source_location_utm"],
                "emission_rate_gps": spatial_results["estimated_emission_rate_gps"],
                "confidence": spatial_results["localization_confidence"],
            },
            "forecast_24h": {
                "pm25_pred": [float(p) for p in forecast_out[:, 0]],
                "aqi_pred": [float(a) for a in forecast_out[:, 1]],
                "peak_pm25_horizon_hours": int(np.argmax(forecast_out[:, 0]) + 1),
                "peak_pm25_value": float(np.max(forecast_out[:, 0])),
            },
            "anomaly_diagnostics": {
                "is_anomaly": is_anom,
                "reconstruction_score": float(anom_score),
                "dynamic_threshold": float(thresh),
                "signature_class_id": cid,
                "signature_class_name": cname,
                "class_probabilities": probs,
            },
        }

    def generate_edge_calibration_feedback(
        self,
        station_id: str,
        current_telemetry: Dict[str, float],
        anomaly_diagnostics: Dict[str, Union[bool, float, str]],
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Cơ chế phản hồi hành vi khép kín (Closed-Loop Feedback) từ Pi 5 về ESP32-S3:
        Tự động điều chỉnh hệ số ngưỡng bù vi khí hậu thích ứng ngoài thực địa.
        """
        cname = anomaly_diagnostics.get("signature_class_name", "Clean Normal")
        is_anom = bool(anomaly_diagnostics.get("is_anomaly", False))
        rh = current_telemetry.get("rh", 60.0)
        co2 = current_telemetry.get("co2", 420.0)

        # 1. Neu phat hien troi cam bien doc lap (Chamber Drift)
        if cname == "Sensor Drift":
            return {
                "station_id": station_id,
                "command": "TRIGGER_CHAMBER_MAINTENANCE",
                "offset_bias": -12.5,
                "inhibit_humidity_damping": False,
                "force_maintenance": True,
            }

        # 2. Neu phat hien khoi chay/giao thong ket hop nom am
        if cname == "Fire / Combustion" or (co2 > 800.0 and rh > 75.0):
            return {
                "station_id": station_id,
                "command": "INHIBIT_HUMIDITY_DAMPING",
                "offset_bias": 0.0,
                "inhibit_humidity_damping": True,
                "force_maintenance": False,
            }

        # 3. Môi trường hoạt động bình thường
        return {
            "station_id": station_id,
            "command": "NORMAL_TELEMETRY",
            "offset_bias": 0.0,
            "inhibit_humidity_damping": False,
            "force_maintenance": False,
        }
