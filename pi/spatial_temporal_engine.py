"""
SPATIAL-TEMPORAL DISPERSION & CROSS-CORRELATION ENGINE
Động cơ Phân tích Tương quan Không gian - Thời gian & Định vị Nguồn Ô nhiễm
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5 (8GB RAM)
Thuộc Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from pi.config import (
    STATIONS,
    STATION_COORDS_UTM,
    STATION_IDS,
    DISTANCE_MATRIX,
)

def compute_cross_correlation_matrix(
    data_matrix: np.ndarray,
    max_lag: int = 60,
    dt_s: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính toán ma trận tương quan chéo chuẩn hóa R_ij(tau) và trích xuất độ trễ tối ưu tau_ij*.

    Args:
        data_matrix: Mảng nồng độ shape (T, N) hoặc (N, T) với T là số bước thời gian, N là số trạm.
        max_lag: Độ trễ tối đa xét (số bước lấy mẫu, mặc định 60 bước = 30 phút với dt=30s).
        dt_s: Khoảng thời gian lấy mẫu (giây).

    Returns:
        R_matrix: Ma trận đỉnh tương quan chuẩn hóa shape (N, N), giá trị trong [-1.0, 1.0].
        tau_star_matrix: Ma trận độ trễ tối ưu (giây) shape (N, N), thỏa mãn tau_ji* = -tau_ij*.
    """
    arr = np.asarray(data_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data_matrix must be 2D, got shape {arr.shape}")

    # Đảm bảo mảng có shape (T, N_stations). Chỉ chuyển vị khi trục 0 là danh sách trạm và trục 1 không phải
    if arr.shape[0] == len(STATION_IDS) and arr.shape[1] != len(STATION_IDS):
        arr = arr.T
    elif arr.shape[0] < arr.shape[1] and arr.shape[1] != len(STATION_IDS) and arr.shape[0] <= len(STATION_IDS):
        arr = arr.T

    T, N = arr.shape
    max_lag = min(max_lag, max(1, T // 2 - 1))

    means = np.mean(arr, axis=0, keepdims=True)
    stds = np.std(arr, axis=0, keepdims=True)
    stds = np.where(stds < 1e-8, 1e-8, stds)
    norm_data = (arr - means) / stds

    lags = np.arange(-max_lag, max_lag + 1)
    num_lags = len(lags)
    zero_idx = max_lag

    all_corrs = np.zeros((num_lags, N, N), dtype=np.float64)

    # Vectorized computation for lag 0
    c0 = norm_data.T @ norm_data
    e0 = np.sum(norm_data**2, axis=0)
    denom0 = np.sqrt(e0[:, None] * e0[None, :]) + 1e-12
    all_corrs[zero_idx] = c0 / denom0

    # Vectorized computation for positive lags and transpose-symmetry for negative lags
    for k in range(1, max_lag + 1):
        x = norm_data[:T - k, :]
        y = norm_data[k:, :]
        ck = x.T @ y
        ex = np.sum(x**2, axis=0)
        ey = np.sum(y**2, axis=0)
        denom_k = np.sqrt(ex[:, None] * ey[None, :]) + 1e-12
        corr_k = ck / denom_k
        all_corrs[zero_idx + k] = corr_k
        all_corrs[zero_idx - k] = corr_k.T

    best_idx = np.argmax(all_corrs, axis=0)
    best_lags = lags[best_idx]
    max_corrs = np.take_along_axis(all_corrs, best_idx[None, :, :], axis=0)[0]
    max_corrs = np.clip(max_corrs, -1.0, 1.0)

    R_matrix = np.eye(N, dtype=np.float64)
    tau_star_matrix = np.zeros((N, N), dtype=np.float64)

    iu = np.triu_indices(N, k=1)
    r_vals = max_corrs[iu]
    tau_vals = best_lags[iu] * dt_s

    R_matrix[iu] = r_vals
    R_matrix[(iu[1], iu[0])] = r_vals
    tau_star_matrix[iu] = tau_vals
    tau_star_matrix[(iu[1], iu[0])] = -tau_vals

    return R_matrix, tau_star_matrix

def estimate_wind_and_dispersion(
    tau_star_matrix: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    r_matrix: Optional[np.ndarray] = None,
    r_thresh: float = 0.55,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Ước lượng Vector Vận tốc Gió và Hướng Phát tán bằng phương pháp Bình phương Tối thiểu có Trọng số (WLS)
    dựa trên Vector Nghịch đảo Vận tốc (Slowness Vector: s = v / ||v||^2).

    Phương trình WLS: Delta x_ij * s_x + Delta y_ij * s_y = tau_ij*

    Args:
        tau_star_matrix: Ma trận độ trễ (giây) shape (N, N).
        coordinates: Tọa độ UTM (x_east, y_north) shape (N, 2). Mặc định lấy theo lưới 8 trạm Thanh Xuân.
        r_matrix: Ma trận hệ số tương quan R_ij shape (N, N).
        r_thresh: Ngưỡng tương quan tối thiểu để đưa cặp trạm vào WLS.
        weights: Mảng trọng số tùy biến (tùy chọn).

    Returns:
        wind_vector: Vector gió [u, v] (m/s).
        wind_speed: Tốc độ gió ||v|| (m/s).
        bearing_deg: Hướng gió theo độ [0, 360) (0° = Bắc, 90° = Đông).
        r2_fit: Hệ số xác định R^2 của mô hình hồi quy WLS.
    """
    coords = coordinates if coordinates is not None else STATION_COORDS_UTM
    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]

    A_rows = []
    b_vals = []
    w_vals = []

    for i in range(N):
        for j in range(i + 1, N):
            tau = tau_star_matrix[i, j]
            corr = r_matrix[i, j] if r_matrix is not None else 1.0

            if r_matrix is not None and corr < r_thresh:
                continue

            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]

            A_rows.append([dx, dy])
            b_vals.append(tau)

            weight = (corr ** 2) if r_matrix is not None else 1.0
            w_vals.append(weight)

    if len(A_rows) < 2:

        return np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0, 0.0

    A = np.array(A_rows, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)

    if weights is not None and len(weights) == len(b):
        W = np.diag(weights)
    else:
        W = np.diag(w_vals)

    try:
        At_W = A.T @ W
        At_W_A = At_W @ A
        At_W_b = At_W @ b

        reg = 1e-9 * np.eye(2)
        s = np.linalg.solve(At_W_A + reg, At_W_b)

        s_norm_sq = float(s[0]**2 + s[1]**2)

        if s_norm_sq < 1e-12 or np.all(np.abs(b) < 1e-6):
            return np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0, 1.0

        v_u = float(s[0] / s_norm_sq)
        v_v = float(s[1] / s_norm_sq)

        speed = math.sqrt(v_u**2 + v_v**2)
        if speed > 25.0:
            scale = 25.0 / speed
            v_u *= scale
            v_v *= scale
            speed = 25.0

        wind_vector = np.array([v_u, v_v], dtype=np.float64)

        bearing = (math.atan2(v_u, v_v) * 180.0 / math.pi) % 360.0

        b_pred = A @ s
        ss_res = float(np.sum(w_vals * (b - b_pred) ** 2))
        b_mean = float(np.average(b, weights=w_vals))
        ss_tot = float(np.sum(w_vals * (b - b_mean) ** 2))

        r2_fit = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 1.0
        r2_fit = max(0.0, min(1.0, r2_fit))

        return wind_vector, speed, bearing, r2_fit

    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0, 0.0

def localize_emission_source(
    station_concentrations: np.ndarray,
    wind_vector: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    pblh: float = 600.0,
    search_radius_m: float = 2500.0,
    grid_resolution_m: float = 50.0,
) -> Tuple[float, float, float, float]:
    """
    Định vị tọa độ nguồn phát thải (x_source, y_source) và cường độ phát thải Q (ug/s)
    bằng phương pháp Đảo ngược Khói Gaussian (Inverse Gaussian Plume Inversion).

    Args:
        station_concentrations: Mảng nồng độ PM2.5 trung bình tại các trạm (ug/m3) shape (N,).
        wind_vector: Vector vận tốc gió [u, v] (m/s).
        coordinates: Tọa độ UTM các trạm shape (N, 2).
        pblh: Chiều cao tầng biên khí quyển (m).
        search_radius_m: Bán kính quét tìm nguồn (m).
        grid_resolution_m: Độ phân giải lưới tìm kiếm (m).

    Returns:
        x_source: Tọa độ UTM Easting của nguồn (m).
        y_source: Tọa độ UTM Northing của nguồn (m).
        q_emission: Cường độ phát thải ước lượng (g/s).
        confidence: Độ tin cậy ước tính trong [0.0, 1.0].
    """
    coords = coordinates if coordinates is not None else STATION_COORDS_UTM
    coords = np.asarray(coords, dtype=np.float64)
    concs = np.asarray(station_concentrations, dtype=np.float64)
    N = coords.shape[0]

    u_wind = float(wind_vector[0])
    v_wind = float(wind_vector[1])
    wind_speed = math.sqrt(u_wind**2 + v_wind**2)

    if wind_speed < 0.3:

        max_idx = int(np.argmax(concs))
        return float(coords[max_idx, 0]), float(coords[max_idx, 1]), 10.0, 0.50

    wind_dir_x = u_wind / wind_speed
    wind_dir_y = v_wind / wind_speed

    upwind_x = -wind_dir_x
    upwind_y = -wind_dir_y

    baseline = np.percentile(concs, 25)
    weights = np.maximum(0.0, concs - baseline)
    total_w = np.sum(weights)

    if total_w < 1e-4:
        center_x = float(np.mean(coords[:, 0]))
        center_y = float(np.mean(coords[:, 1]))
    else:
        center_x = float(np.sum(coords[:, 0] * weights) / total_w)
        center_y = float(np.sum(coords[:, 1] * weights) / total_w)

    xs = np.arange(center_x - search_radius_m, center_x + search_radius_m, grid_resolution_m)
    ys = np.arange(center_y - search_radius_m, center_y + search_radius_m, grid_resolution_m)

    best_residual = float("inf")
    best_x = center_x
    best_y = center_y
    best_q = 0.0

    p_y = 0.16
    p_z = 0.08

    for sx in xs[::2]:
        for sy in ys[::2]:

            vec_to_center_x = center_x - sx
            vec_to_center_y = center_y - sy
            dot_wind = vec_to_center_x * wind_dir_x + vec_to_center_y * wind_dir_y
            if dot_wind < 0:
                continue

            c_pred_unit = []
            c_obs = []

            for k in range(N):
                rx = coords[k, 0] - sx
                ry = coords[k, 1] - sy

                x_down = rx * wind_dir_x + ry * wind_dir_y
                if x_down <= 10.0:
                    continue

                y_cross = abs(rx * (-wind_dir_y) + ry * wind_dir_x)

                sigma_y = max(5.0, p_y * x_down)
                sigma_z = min(pblh, max(5.0, p_z * x_down))

                val = math.exp(- (y_cross ** 2) / (2.0 * sigma_y ** 2)) / (
                    math.pi * sigma_y * sigma_z * wind_speed + 1e-6
                )
                c_pred_unit.append(val)
                c_obs.append(max(0.0, concs[k] - baseline))

            if len(c_pred_unit) >= 2:
                pred_arr = np.array(c_pred_unit)
                obs_arr = np.array(c_obs)

                pred_sq_sum = np.sum(pred_arr ** 2)
                if pred_sq_sum > 1e-12:
                    q_opt = float(np.sum(obs_arr * pred_arr) / pred_sq_sum)
                    if q_opt > 0:
                        residual = float(np.sum((obs_arr - q_opt * pred_arr) ** 2))
                        if residual < best_residual:
                            best_residual = residual
                            best_x = float(sx)
                            best_y = float(sy)
                            best_q = q_opt / 1e6

    confidence = float(np.exp(-best_residual / (total_w + 1e-4)))
    confidence = max(0.20, min(0.95, confidence))

    return best_x, best_y, max(0.1, best_q), confidence

class SpatialTemporalEngine:
    """
    Động cơ Tương quan Không gian - Thời gian Đa trạm (Milestone 2).
    Chạy định kỳ trên Raspberry Pi 5 để phân tích luồng phát tán ô nhiễm.
    """

    def __init__(self, station_ids: Optional[List[str]] = None, coordinates: Optional[np.ndarray] = None):
        self.station_ids = station_ids if station_ids is not None else STATION_IDS
        self.coordinates = coordinates if coordinates is not None else STATION_COORDS_UTM
        self.n_stations = len(self.station_ids)

    def analyze(
        self,
        pm25_history: np.ndarray,
        dt_s: float = 30.0,
        max_lag: int = 60,
    ) -> Dict[str, Union[float, np.ndarray, List[Dict]]]:
        """
        Phân tích toàn diện: Tương quan trễ -> Vector gió -> Định vị nguồn phát tán.

        Args:
            pm25_history: Dữ liệu nồng độ chuỗi thời gian shape (T, N_stations).
            dt_s: Bước lấy mẫu (giây).
            max_lag: Độ trễ tối đa (bước).

        Returns:
            Dict chứa R_matrix, tau_star, wind_vector, wind_speed, bearing, source_loc.
        """
        R_mat, tau_mat = compute_cross_correlation_matrix(pm25_history, max_lag=max_lag, dt_s=dt_s)
        wind_vec, wind_speed, bearing, r2 = estimate_wind_and_dispersion(
            tau_mat, coordinates=self.coordinates, r_matrix=R_mat
        )

        current_concs = pm25_history[-1] if pm25_history.ndim == 2 else pm25_history
        src_x, src_y, q_rate, conf = localize_emission_source(
            current_concs, wind_vec, coordinates=self.coordinates
        )

        return {
            "r_matrix": R_mat,
            "tau_star_matrix": tau_mat,
            "wind_vector": wind_vec,
            "wind_speed_mps": float(wind_speed),
            "dispersion_bearing_deg": float(bearing),
            "wls_r2_fit": float(r2),
            "source_location_utm": [float(src_x), float(src_y)],
            "estimated_emission_rate_gps": float(q_rate),
            "localization_confidence": float(conf),
        }
