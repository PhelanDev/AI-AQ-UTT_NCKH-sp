"""
CENTRAL NEURAL NETWORK AI ENGINE ORCHESTRATOR
Hệ Thống Điều Phối & Tích Hợp Toàn Diện Bộ Não AI Trung Tâm Trên Raspberry Pi 5 (8GB RAM)
Phát triển cho Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
"""

import json
import math
import os
import time
from collections import deque
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

        self.forecaster_weights = forecaster_weights
        self.autoencoder_weights = autoencoder_weights
        self._load_scaler_params()

        self.onnx_forecaster_session = None
        self.onnx_autoencoder_session = None
        self._env_ring_buffer: deque = deque(maxlen=24)

    def reset_history_buffer(self) -> None:
        """Đặt lại bộ đệm vòng lịch sử đa kênh."""
        self._env_ring_buffer.clear()

    def _load_scaler_params(self) -> None:
        """Nạp tham số chuẩn hóa Min-Max và phân phối sai số nền từ JSON."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        fc_params_path = "mlops/models/central_forecaster_params.json"
        if self.forecaster_weights and self.forecaster_weights.endswith(".pt"):
            fc_params_path = self.forecaster_weights[:-3] + "_params.json"
        if not os.path.exists(fc_params_path):
            alt_fc = os.path.join(repo_root, fc_params_path)
            if os.path.exists(alt_fc):
                fc_params_path = alt_fc

        self.feat_min = np.array(
            [1.0, 1.3, 8.6, 17.0, -5.6, 423.5, 97.0, -4.5, -4.8, -1.0, -1.0, 994.0, 2.0, 1.0],
            dtype=np.float32,
        )
        self.feat_max = np.array(
            [446.0, 604.0, 40.0, 100.0, 29.6, 1514.2, 1413.0, 4.2, 4.6, 1.0, 1.0, 1033.0, 496.0, 1.0],
            dtype=np.float32,
        )
        self.horizon_std_pm25 = np.array(
            [4.0 + 0.5 * h for h in range(24)], dtype=np.float32
        )
        if os.path.exists(fc_params_path):
            with open(fc_params_path, "r", encoding="utf-8") as f:
                fc_meta = json.load(f)
            if "feat_min" in fc_meta and "feat_max" in fc_meta:
                self.feat_min = np.array(fc_meta["feat_min"], dtype=np.float32)
                self.feat_max = np.array(fc_meta["feat_max"], dtype=np.float32)
            if "horizon_std_pm25" in fc_meta:
                self.horizon_std_pm25 = np.array(fc_meta["horizon_std_pm25"], dtype=np.float32)

        ae_params_path = "mlops/models/central_autoencoder_params.json"
        if self.autoencoder_weights and self.autoencoder_weights.endswith(".pt"):
            ae_params_path = self.autoencoder_weights[:-3] + "_params.json"
        if not os.path.exists(ae_params_path):
            alt_ae = os.path.join(repo_root, ae_params_path)
            if os.path.exists(alt_ae):
                ae_params_path = alt_ae

        self.ae_min_48 = np.tile(
            np.array([10.7, 10.7, 20.4, 43.0, 654.0, 103.9], dtype=np.float32), 8
        )
        self.ae_max_48 = np.tile(
            np.array([285.0, 290.6, 35.3, 100.0, 8673.6, 544.3], dtype=np.float32), 8
        )
        baseline_mean = 0.005
        baseline_std = 0.015
        if os.path.exists(ae_params_path):
            with open(ae_params_path, "r", encoding="utf-8") as f:
                ae_meta = json.load(f)
            if "min_48" in ae_meta and "max_48" in ae_meta:
                self.ae_min_48 = np.array(ae_meta["min_48"], dtype=np.float32)
                self.ae_max_48 = np.array(ae_meta["max_48"], dtype=np.float32)
            if "baseline_mse_mean" in ae_meta and "baseline_mse_std" in ae_meta:
                baseline_mean = float(ae_meta["baseline_mse_mean"])
                baseline_std = max(float(ae_meta["baseline_mse_std"]), 0.001)
            else:
                mean_err = ae_meta.get("mean_error", 0.005)
                std_err = ae_meta.get("std_error", 0.006)
                baseline_mean = float(np.mean(mean_err))
                baseline_std = max(float(np.mean(std_err)), 0.015)

        self.anomaly_detector.mu_ema = baseline_mean
        self.anomaly_detector.var_ema = baseline_std ** 2
        self.anomaly_detector.initialized = True
        self.anomaly_detector.error_history = [baseline_mean] * 20

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
            sliding_window_history: Ma trận lịch sử nồng độ PM2.5 hình dạng (T_steps, N_stations) hoặc (N_stations, T_steps).
            current_features_48d: Mảng đặc trưng đo lường hiện tại (hỗ trợ từ 1 trạm prototype lên mạng lưới mở rộng 48 chiều = 8 trạm x 6 kênh).
            dt_s: Bước thời gian lấy mẫu.

        Returns:
            Dict chứa toàn bộ kết quả phân tích: Vector gió, Nguồn ô nhiễm, Dự báo 24h, Bất thường.
        """
        arr_a = np.asarray(sliding_window_history, dtype=np.float32)
        arr_b = np.asarray(current_features_48d, dtype=np.float32)

        # Hỗ trợ tự động nhận diện thứ tự tham số nếu caller truyền (current_48d, history_8x24)
        if arr_a.size == 48 and (arr_a.ndim == 1 or 1 in arr_a.shape) and arr_b.ndim >= 2 and arr_b.size != 48:
            arr_a, arr_b = arr_b, arr_a

        curr_48d = np.nan_to_num(arr_b, nan=0.0, posinf=500.0, neginf=0.0).reshape(48)
        self._env_ring_buffer.append(curr_48d.copy())

        is_3d = False
        is_2d_14ch = False
        if arr_a.ndim == 3:
            is_3d = True
            if arr_a.shape[0] == self.n_stations and arr_a.shape[1] != self.n_stations:
                arr_a = np.transpose(arr_a, (1, 0, 2))
            hist_spatial = np.nan_to_num(arr_a[:, :, 0], nan=0.0, posinf=500.0, neginf=0.0)
            hist_arr = hist_spatial
        elif arr_a.ndim == 2 and arr_a.shape[1] == 14 and arr_a.shape[0] != self.n_stations:
            is_2d_14ch = True
            hist_spatial = np.tile(np.nan_to_num(arr_a[:, 0:1], nan=0.0, posinf=500.0, neginf=0.0), (1, self.n_stations))
            hist_arr = hist_spatial
        else:
            hist_arr = np.nan_to_num(arr_a, nan=0.0, posinf=500.0, neginf=0.0)
            if hist_arr.ndim == 2 and hist_arr.shape[0] == self.n_stations and hist_arr.shape[1] != self.n_stations:
                hist_arr = hist_arr.T
            hist_spatial = hist_arr

        # Xử lý Cold-Start (T < 8 bước thời gian): đệm lặp hàng đầu tiên đảm bảo phân tích không gian ổn định
        if hist_spatial.shape[0] < 8:
            pad_needed = 8 - hist_spatial.shape[0]
            if hist_spatial.shape[0] > 0:
                first_row = hist_spatial[0:1, :]
                hist_spatial = np.vstack([np.repeat(first_row, pad_needed, axis=0), hist_spatial])
            else:
                pm_default = curr_48d.reshape(self.n_stations, 6)[:, 0:1].T
                hist_spatial = np.repeat(pm_default, 8, axis=0)

        spatial_results = self.spatial_engine.analyze(
            hist_spatial, dt_s=dt_s, max_lag=min(40, max(1, hist_spatial.shape[0] // 2))
        )

        st_mat = curr_48d.reshape(self.n_stations, 6)
        pm_station_mean = float(np.mean(st_mat[:, 0]))
        pm10_station_mean = float(np.mean(st_mat[:, 1]))
        pm10_ratio = (
            float(np.clip(pm10_station_mean / pm_station_mean, 1.0, 2.5))
            if pm_station_mean > 1.0 and pm10_station_mean > 1.0
            else 1.35
        )

        temp_mean = float(np.mean(st_mat[:, 2]))
        temp_c = temp_mean if (-10.0 <= temp_mean <= 60.0 and temp_mean != 0.0) else 24.5

        rh_mean = float(np.mean(st_mat[:, 3]))
        rh_pct = float(np.clip(rh_mean, 5.0, 100.0)) if (5.0 <= rh_mean <= 105.0) else 78.0

        a_mt, b_mt = 17.27, 237.7
        alpha_mt = ((a_mt * temp_c) / (b_mt + temp_c)) + math.log(max(1.0, min(100.0, rh_pct)) / 100.0)
        dp_c = (b_mt * alpha_mt) / (a_mt - alpha_mt)

        col4_mean = float(np.mean(st_mat[:, 4]))
        col5_mean = float(np.mean(st_mat[:, 5]))
        if 0.0 < col4_mean < 100.0:
            wind_u = col4_mean * math.cos(math.radians(col5_mean))
            wind_v = col4_mean * math.sin(math.radians(col5_mean))
        else:
            wind_u = float(spatial_results["wind_vector"][0])
            wind_v = float(spatial_results["wind_vector"][1])

        t_steps = min(24, arr_a.shape[0] if (is_3d or is_2d_14ch) else hist_arr.shape[0])
        recent_pm = hist_spatial[-t_steps:]
        mean_pm = np.mean(recent_pm, axis=1)

        feat_in = np.zeros((1, 24, 14), dtype=np.float32)
        base_hour = time.localtime().tm_hour
        for t in range(t_steps):
            idx = 24 - t_steps + t
            hour = (base_hour + t) % 24
            if is_3d:
                step_3d = arr_a[-t_steps + t, :, :]
                pm_t = max(0.0, float(np.mean(step_3d[:, 0])))
                pm10_t = max(0.0, float(np.mean(step_3d[:, 1])))
                temp_t = float(np.mean(step_3d[:, 2]))
                rh_t = float(np.clip(np.mean(step_3d[:, 3]), 5.0, 100.0))
                alpha_t = ((a_mt * temp_t) / (b_mt + temp_t)) + math.log(max(1.0, min(100.0, rh_t)) / 100.0)
                dp_t = (b_mt * alpha_t) / (a_mt - alpha_t)
                c4_t = float(np.mean(step_3d[:, 4]))
                c5_t = float(np.mean(step_3d[:, 5]))
                if 0.0 < c4_t < 100.0:
                    wu_t = c4_t * math.cos(math.radians(c5_t))
                    wv_t = c4_t * math.sin(math.radians(c5_t))
                else:
                    wu_t, wv_t = wind_u, wind_v
                co2_t = c4_t if c4_t >= 150.0 else (pm_t * 18.0 + 410.0)
                voc_t = c5_t if c4_t >= 150.0 else (pm_t * 2.1 + 90.0)
                aqi_t = float(calc_vn_aqi_pm25(pm_t))

                feat_in[0, idx, 0] = pm_t
                feat_in[0, idx, 1] = pm10_t
                feat_in[0, idx, 2] = temp_t
                feat_in[0, idx, 3] = rh_t
                feat_in[0, idx, 4] = dp_t
                feat_in[0, idx, 5] = co2_t
                feat_in[0, idx, 6] = voc_t
                feat_in[0, idx, 7] = wu_t
                feat_in[0, idx, 8] = wv_t
                feat_in[0, idx, 9] = math.sin(2.0 * math.pi * hour / 24.0)
                feat_in[0, idx, 10] = math.cos(2.0 * math.pi * hour / 24.0)
                feat_in[0, idx, 11] = 1013.2
                feat_in[0, idx, 12] = aqi_t
                feat_in[0, idx, 13] = 1.0
            elif is_2d_14ch:
                feat_in[0, idx, :] = arr_a[-t_steps + t, :]
            else:
                pm_t = max(0.0, float(mean_pm[t]))
                co2_t = col4_mean if col4_mean >= 150.0 else (pm_t * 18.0 + 410.0)
                voc_t = col5_mean if col4_mean >= 150.0 else (pm_t * 2.1 + 90.0)
                aqi_t = float(calc_vn_aqi_pm25(pm_t))

                feat_in[0, idx, 0] = pm_t
                feat_in[0, idx, 1] = pm_t * pm10_ratio
                feat_in[0, idx, 2] = temp_c
                feat_in[0, idx, 3] = rh_pct
                feat_in[0, idx, 4] = dp_c
                feat_in[0, idx, 5] = co2_t
                feat_in[0, idx, 6] = voc_t
                feat_in[0, idx, 7] = wind_u
                feat_in[0, idx, 8] = wind_v
                feat_in[0, idx, 9] = math.sin(2.0 * math.pi * hour / 24.0)
                feat_in[0, idx, 10] = math.cos(2.0 * math.pi * hour / 24.0)
                feat_in[0, idx, 11] = 1013.2
                feat_in[0, idx, 12] = aqi_t
                feat_in[0, idx, 13] = 1.0

        if t_steps < 24 and t_steps > 0:
            feat_in[0, : 24 - t_steps, :] = feat_in[0, 24 - t_steps, :]

        feat_denom = np.maximum(self.feat_max - self.feat_min, 1e-6)
        feat_in_norm = (feat_in - self.feat_min) / feat_denom
        feat_in_norm[:, :, 1:] = np.clip(feat_in_norm[:, :, 1:], 0.0, 1.0)
        feat_in_norm[:, :, 0] = np.clip(feat_in_norm[:, :, 0], 0.0, 2.5)

        if self.use_onnx and self.onnx_forecaster_session is not None:
            forecast_norm = self.onnx_forecaster_session.run(
                None, {"input_timeseries": feat_in_norm}
            )[0][0]
        else:
            with torch.no_grad():
                forecast_norm = (
                    self.forecaster(torch.tensor(feat_in_norm, device=self.device))
                    .cpu()
                    .numpy()[0]
                )

        pm_range = float(self.feat_max[0] - self.feat_min[0])
        aqi_range = float(self.feat_max[12] - self.feat_min[12])
        pm25_phys = np.maximum(0.0, forecast_norm[:, 0] * pm_range + float(self.feat_min[0]))
        aqi_head = np.maximum(0.0, forecast_norm[:, 1] * aqi_range + float(self.feat_min[12]))
        aqi_from_pm = np.array([float(calc_vn_aqi_pm25(float(p))) for p in pm25_phys], dtype=np.float32)
        aqi_phys = 0.5 * aqi_head + 0.5 * aqi_from_pm

        x_ae_mat = st_mat.copy()
        if 0.0 <= col4_mean < 150.0:
            x_ae_mat[:, 4] = np.maximum(654.0, x_ae_mat[:, 0] * 14.0 + 420.0)
            x_ae_mat[:, 5] = np.maximum(105.0, x_ae_mat[:, 0] * 1.8 + 100.0)

        x_ae_flat = x_ae_mat.reshape(1, 48)
        ae_denom = np.maximum(self.ae_max_48 - self.ae_min_48, 1e-6)
        x_ae_norm = (x_ae_flat - self.ae_min_48) / ae_denom
        x_ae_norm = np.where(x_ae_flat >= 0.0, np.maximum(0.0, x_ae_norm), x_ae_norm)
        x_ae_norm = np.clip(x_ae_norm, -10.0, 10.0).astype(np.float32)

        if self.use_onnx and self.onnx_autoencoder_session is not None:
            rec_ae_np = self.onnx_autoencoder_session.run(
                None, {"input_features_48d": x_ae_norm}
            )[0]
        else:
            with torch.no_grad():
                rec_ae, _ = self.autoencoder(torch.tensor(x_ae_norm, device=self.device))
                rec_ae_np = rec_ae.cpu().numpy()

        err_vec = (x_ae_norm - rec_ae_np).ravel()

        anom_score = self.anomaly_detector.compute_score(err_vec)
        is_anom, thresh, _ = self.anomaly_detector.update_and_detect(anom_score)

        pm_per_station = st_mat[:, 0]
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

        # Co giãn khoảng tin cậy phân vị thích ứng động (Heteroscedastic Conformal Quantiles)
        sigma_window = float(np.std(mean_pm)) if len(mean_pm) > 1 else 0.0
        sigma_spatial = float(np.std(st_mat[:, 0])) if st_mat.shape[0] > 1 else 0.0
        scale_dynamic = float(np.clip(
            0.75 + 0.25 * (sigma_window + 0.5 * sigma_spatial) / 15.0,
            0.65, 2.5
        ))

        base_std = getattr(self, "horizon_std_pm25", np.array([4.0 + 0.5 * h for h in range(24)], dtype=np.float32))
        h_std = base_std * scale_dynamic
        pm25_p10 = [max(0.0, float(p - 1.2816 * std_h)) for p, std_h in zip(pm25_phys, h_std)]
        pm25_p90 = [float(p + 1.2816 * std_h) for p, std_h in zip(pm25_phys, h_std)]
        # Enforce strict monotonicity invariants: 0 <= P10 <= P50 <= P90
        pm25_p10 = [min(p10, float(p)) for p10, p in zip(pm25_p10, pm25_phys)]
        pm25_p90 = [max(p90, float(p)) for p90, p in zip(pm25_p90, pm25_phys)]

        pm25_list = [float(p) for p in pm25_phys]
        aqi_list = [float(a) for a in aqi_phys]
        pm25_6_12_24 = [pm25_list[5], pm25_list[11], pm25_list[23]]
        aqi_6_12_24 = [aqi_list[5], aqi_list[11], aqi_list[23]]

        return {
            "timestamp": time.time(),
            "spatial_dispersion": {
                "wind_speed_mps": spatial_results["wind_speed_mps"],
                "wind_direction_deg": spatial_results["dispersion_bearing_deg"],
                "source_location_utm": spatial_results["source_location_utm"],
                "emission_rate_gps": spatial_results["estimated_emission_rate_gps"],
                "confidence": spatial_results["localization_confidence"],
                "r_matrix": spatial_results["r_matrix"],
                "tau_star_matrix": spatial_results["tau_star_matrix"],
            },
            "forecast_24h": {
                "pm25_pred": pm25_list,
                "pm25_p50": pm25_list,
                "pm25_p10": pm25_p10,
                "pm25_p90": pm25_p90,
                "aqi_pred": aqi_list,
                "pm25_6h_12h_24h": pm25_6_12_24,
                "aqi_6h_12h_24h": aqi_6_12_24,
                "peak_pm25_horizon_hours": int(np.argmax(pm25_phys) + 1),
                "peak_pm25_value": float(np.max(pm25_phys)),
            },
            "forecast": {
                "pm25_6h_12h_24h": pm25_6_12_24,
                "aqi_6h_12h_24h": aqi_6_12_24,
                "pm25_pred": pm25_list,
                "pm25_p50": pm25_list,
                "pm25_p10": pm25_p10,
                "pm25_p90": pm25_p90,
                "aqi_pred": aqi_list,
            },
            "anomaly_diagnostics": {
                "is_anomaly": bool(is_anom),
                "reconstruction_score": float(anom_score),
                "anomaly_score": float(anom_score),
                "dynamic_threshold": float(thresh),
                "signature_class_id": cid,
                "signature_class_name": cname,
                "class_probabilities": probs,
            },
            "anomaly": {
                "is_anomaly": bool(is_anom),
                "anomaly_score": float(anom_score),
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
