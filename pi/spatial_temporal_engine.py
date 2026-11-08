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
    STATION_COORDS_WGS84,
    STATION_IDS,
    STATION_NAMES,
    DISTANCE_MATRIX,
    TELEMETRY_INTERVAL_S,
)


def _vectorized_vn_aqi_pm25(pm25_arr: np.ndarray) -> np.ndarray:
    """Tính chỉ số VN-AQI vector hóa trên toàn mảng từ nồng độ PM2.5."""
    pm = np.maximum(0.0, np.asarray(pm25_arr, dtype=np.float32))
    aqi = np.zeros_like(pm, dtype=np.float32)
    m1 = pm <= 25.0
    m2 = (pm > 25.0) & (pm <= 50.0)
    m3 = (pm > 50.0) & (pm <= 80.0)
    m4 = (pm > 80.0) & (pm <= 150.0)
    m5 = (pm > 150.0) & (pm <= 250.0)
    m6 = pm > 250.0

    aqi[m1] = (pm[m1] / 25.0) * 50.0
    aqi[m2] = 50.0 + ((pm[m2] - 25.0) / 25.0) * 50.0
    aqi[m3] = 100.0 + ((pm[m3] - 50.0) / 30.0) * 50.0
    aqi[m4] = 150.0 + ((pm[m4] - 80.0) / 70.0) * 50.0
    aqi[m5] = 200.0 + ((pm[m5] - 150.0) / 100.0) * 100.0
    aqi[m6] = np.minimum(500.0, 300.0 + ((pm[m6] - 250.0) / 100.0) * 100.0)
    return aqi


def compute_cross_correlation_matrix(
    data_matrix: np.ndarray,
    max_lag: int = 60,
    dt_s: float = 900.0,
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
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"data_matrix must be 2D, got shape {arr.shape}")

    # Đảm bảo mảng có shape (T, N_stations). Chỉ chuyển vị khi trục 0 là danh sách trạm và trục 1 không phải
    if arr.shape[0] == len(STATION_IDS) and arr.shape[1] != len(STATION_IDS):
        arr = arr.T
    elif arr.shape[0] < arr.shape[1] and arr.shape[1] != len(STATION_IDS) and arr.shape[0] <= len(STATION_IDS):
        arr = arr.T

    T, N = arr.shape
    if N == 1:
        return np.array([[1.0]], dtype=np.float64), np.array([[0.0]], dtype=np.float64)

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
        coordinates: Tọa độ UTM (x_east, y_north) shape (N, 2). Mặc định lấy theo lưới trạm Prototype và tọa độ quy hoạch mở rộng.
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
    N = min(coords.shape[0], tau_star_matrix.shape[0])
    coords = coords[:N]

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
    """
    coords = coordinates if coordinates is not None else STATION_COORDS_UTM
    coords = np.asarray(coords, dtype=np.float64)
    concs = np.asarray(station_concentrations, dtype=np.float64).ravel()
    N = min(coords.shape[0], concs.size)
    coords = coords[:N]
    concs = concs[:N]

    if N == 1:
        return float(coords[0, 0]), float(coords[0, 1]), float(max(0.1, concs[0] * 0.25)), 0.85

    u_wind = float(wind_vector[0])
    v_wind = float(wind_vector[1])
    wind_speed = math.sqrt(u_wind**2 + v_wind**2)

    if wind_speed < 0.3:
        max_idx = int(np.argmax(concs))
        return float(coords[max_idx, 0]), float(coords[max_idx, 1]), 10.0, 0.50

    wind_dir_x = u_wind / wind_speed
    wind_dir_y = v_wind / wind_speed

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


def generate_web_heatmap(
    station_values: Union[np.ndarray, List[float]],
    coordinates_utm: Optional[np.ndarray] = None,
    coordinates_wgs84: Optional[np.ndarray] = None,
    station_ids: Optional[List[str]] = None,
    wind_vector: Optional[np.ndarray] = None,
    grid_size: int = 25,
    radius_m: float = 1000.0,
    buffer_m: float = 800.0,
) -> Dict[str, Union[int, float, List[Dict], np.ndarray, Dict]]:
    """
    Dựng Bản Đồ Nhiệt Web 2D co giãn tự động từ 1 trạm (N=1) đến N trạm thực tế (N >= 1).
    - Không bịa thêm trạm ảo: Có bao nhiêu trạm thực tế thì xuất đúng bấy nhiêu Marker.
    - Khi N = 1: Vẽ quầng nhiệt bán kính Gaussian Plume theo hướng gió quanh trạm đó kèm opacity_grid.
    - Khi N >= 2: Nội suy Anisotropic Kriging / IDW nối màu mượt giữa N trạm thật và hợp nhất vùng phủ opacity_grid.
    """
    vals = np.maximum(0.0, np.asarray(station_values, dtype=np.float64).ravel())
    N = max(1, vals.size)

    coords_u = np.asarray(
        coordinates_utm if coordinates_utm is not None else STATION_COORDS_UTM[:N],
        dtype=np.float64,
    )[:N]
    coords_w = np.asarray(
        coordinates_wgs84 if coordinates_wgs84 is not None else STATION_COORDS_WGS84[:N],
        dtype=np.float64,
    )[:N]
    s_ids = list(station_ids)[:N] if station_ids is not None else list(STATION_IDS[:N])

    # Khung nhìn bản đồ cố định theo vùng quan trắc Thanh Xuân để giao diện Web không bị nhảy khung
    ref_utm = STATION_COORDS_UTM
    x_min = float(min(np.min(ref_utm[:, 0]), np.min(coords_u[:, 0])) - buffer_m)
    x_max = float(max(np.max(ref_utm[:, 0]), np.max(coords_u[:, 0])) + buffer_m)
    y_min = float(min(np.min(ref_utm[:, 1]), np.min(coords_u[:, 1])) - buffer_m)
    y_max = float(max(np.max(ref_utm[:, 1]), np.max(coords_u[:, 1])) + buffer_m)

    ref_wgs = STATION_COORDS_WGS84
    lat_min = float(min(np.min(ref_wgs[:, 0]), np.min(coords_w[:, 0])) - 0.007)
    lat_max = float(max(np.max(ref_wgs[:, 0]), np.max(coords_w[:, 0])) + 0.007)
    lon_min = float(min(np.min(ref_wgs[:, 1]), np.min(coords_w[:, 1])) - 0.007)
    lon_max = float(max(np.max(ref_wgs[:, 1]), np.max(coords_w[:, 1])) + 0.007)

    x_axis = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    lat_axis = np.linspace(lat_min, lat_max, grid_size, dtype=np.float64)
    lon_axis = np.linspace(lon_min, lon_max, grid_size, dtype=np.float64)

    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)  # shape (G, G)

    w_vec = np.asarray(wind_vector if wind_vector is not None else [0.0, 0.0], dtype=np.float64).ravel()
    u_w = float(w_vec[0]) if w_vec.size >= 1 else 0.0
    v_w = float(w_vec[1]) if w_vec.size >= 2 else 0.0
    w_speed = math.sqrt(u_w * u_w + v_w * v_w)

    # Tính khoảng cách hiệu dụng có xét hướng gió (Anisotropic Wind-Plume Distance) cho từng trạm
    eff_dists = np.zeros((N, grid_size, grid_size), dtype=np.float64)
    station_alphas = np.zeros((N, grid_size, grid_size), dtype=np.float64)
    nearest_cells: List[Tuple[int, int]] = []

    for i in range(N):
        dx = mesh_x - coords_u[i, 0]
        dy = mesh_y - coords_u[i, 1]

        # Tìm ô lưới gần tọa độ trạm i nhất
        euclid_sq = dx * dx + dy * dy
        r_idx, c_idx = np.unravel_index(int(np.argmin(euclid_sq)), (grid_size, grid_size))
        nearest_cells.append((int(r_idx), int(c_idx)))

        if w_speed > 0.2:
            wx = u_w / w_speed
            wy = v_w / w_speed
            d_par = dx * wx + dy * wy
            d_perp = -dx * wy + dy * wx

            stretch_down = 1.0 + min(1.5, 0.25 * w_speed)
            compress_up = max(0.55, 1.0 - 0.08 * w_speed)
            scale_par = np.where(d_par >= 0.0, stretch_down, compress_up)
            d_eff = np.sqrt((d_par / scale_par) ** 2 + d_perp ** 2)
        else:
            d_eff = np.sqrt(euclid_sq)

        d_eff[r_idx, c_idx] = 0.0
        eff_dists[i] = d_eff
        alpha_i = np.exp(-(d_eff ** 2) / (2.0 * (radius_m ** 2)))
        alpha_i[r_idx, c_idx] = 1.0
        station_alphas[i] = alpha_i

    # Hợp nhất độ phủ mờ (opacity_grid) từ đúng N trạm đang hoạt động
    opacity_grid = 1.0 - np.prod(1.0 - station_alphas, axis=0)
    opacity_grid = np.clip(opacity_grid, 0.0, 1.0).astype(np.float32)

    # Tính trường nồng độ PM2.5 trên lưới 2D
    if N == 1:
        c0 = float(vals[0])
        c_bg = min(c0, 22.0)
        grid_pm25 = (station_alphas[0] * c0 + (1.0 - station_alphas[0]) * c_bg).astype(np.float32)
        r0, col0 = nearest_cells[0]
        grid_pm25[r0, col0] = float(c0)
    else:
        weights = 1.0 / ((eff_dists + 15.0) ** 2) + 1.5 * station_alphas  # shape (N, G, G)
        w_sum = np.sum(weights, axis=0) + 1e-12
        grid_pm25 = (np.sum(weights * vals[:, None, None], axis=0) / w_sum).astype(np.float32)
        for i, (ri, ci) in enumerate(nearest_cells):
            grid_pm25[ri, ci] = float(vals[i])

    grid_pm25 = np.maximum(0.0, grid_pm25)
    grid_aqi = _vectorized_vn_aqi_pm25(grid_pm25)

    # Tạo danh sách Marker thực tế (chỉ chứa đúng N trạm đang chạy)
    markers = []
    station_aqi = _vectorized_vn_aqi_pm25(vals)
    for i in range(N):
        sid = s_ids[i] if i < len(s_ids) else f"STATION_{i+1:02d}"
        sname = STATION_NAMES[i] if i < len(STATION_NAMES) else sid
        markers.append({
            "station_id": sid,
            "station_name": sname,
            "lat": float(coords_w[i, 0]),
            "lon": float(coords_w[i, 1]),
            "utm_x": float(coords_u[i, 0]),
            "utm_y": float(coords_u[i, 1]),
            "pm25": float(vals[i]),
            "aqi": float(station_aqi[i]),
        })

    return {
        "num_active_stations": int(N),
        "grid_size": int(grid_size),
        "markers": markers,
        "x_utm_axis": x_axis.astype(np.float32),
        "y_utm_axis": y_axis.astype(np.float32),
        "lat_axis": lat_axis.astype(np.float32),
        "lon_axis": lon_axis.astype(np.float32),
        "grid_pm25": grid_pm25,
        "grid_aqi": grid_aqi,
        "opacity_grid": opacity_grid,
    }


class SpatialTemporalEngine:
    """
    Động cơ Tương quan Không gian - Thời gian Đa trạm (Milestone 2 & V5.0 Dynamic 1-to-N).
    Chạy định kỳ trên Raspberry Pi 5 để phân tích luồng phát tán ô nhiễm và vẽ bản đồ Web.
    """

    def __init__(self, station_ids: Optional[List[str]] = None, coordinates: Optional[np.ndarray] = None):
        self.station_ids = station_ids if station_ids is not None else STATION_IDS
        self.coordinates = coordinates if coordinates is not None else STATION_COORDS_UTM
        self.n_stations = len(self.station_ids)

    def analyze(
        self,
        pm25_history: np.ndarray,
        dt_s: float = 900.0,
        max_lag: int = 60,
    ) -> Dict[str, Union[int, float, np.ndarray, List[Dict], Dict]]:
        """
        Phân tích toàn diện: Tương quan trễ -> Vector gió -> Định vị nguồn phát tán -> Bản đồ 2D.
        Hỗ trợ linh hoạt từ 1 trạm (N=1) đến N trạm.
        """
        hist = np.asarray(pm25_history, dtype=np.float64)
        if hist.ndim == 1:
            hist = hist[:, np.newaxis]

        R_mat, tau_mat = compute_cross_correlation_matrix(hist, max_lag=max_lag, dt_s=dt_s)
        active_n = R_mat.shape[0]
        active_coords = np.asarray(self.coordinates, dtype=np.float64)[:active_n]
        active_ids = list(self.station_ids)[:active_n]

        wind_vec, wind_speed, bearing, r2 = estimate_wind_and_dispersion(
            tau_mat, coordinates=active_coords, r_matrix=R_mat
        )

        if hist.shape[1] == active_n:
            current_concs = hist[-1, :]
        else:
            current_concs = hist[:, -1]

        src_x, src_y, q_rate, conf = localize_emission_source(
            current_concs, wind_vec, coordinates=active_coords
        )

        heatmap_2d = generate_web_heatmap(
            station_values=current_concs,
            coordinates_utm=active_coords,
            station_ids=active_ids,
            wind_vector=wind_vec,
            grid_size=25,
        )

        return {
            "num_active_stations": int(active_n),
            "r_matrix": R_mat,
            "tau_star_matrix": tau_mat,
            "wind_vector": wind_vec,
            "wind_speed_mps": float(wind_speed),
            "dispersion_bearing_deg": float(bearing),
            "wls_r2_fit": float(r2),
            "source_location_utm": [float(src_x), float(src_y)],
            "estimated_emission_rate_gps": float(q_rate),
            "localization_confidence": float(conf),
            "heatmap_2d": heatmap_2d,
        }

    def generate_pollution_heatmap(
        self,
        station_values: Union[np.ndarray, Dict[str, float], List[float]],
        grid_resolution: int = 20,
        nugget: float = 1.0,
        sill: Optional[float] = None,
        range_m: float = 1800.0,
        buffer_m: float = 500.0,
    ) -> Dict[str, Union[np.ndarray, Dict, float, int]]:
        """
        Dựng bản đồ nhiệt nồng độ bụi PM2.5 không gian 2D bằng Ordinary Kriging vector hóa NumPy.
        Mô hình Variogram: Exponential Variogram gamma(h) = c0 + c * (1 - exp(-h / a)).
        """
        import time
        t_start = time.perf_counter()

        # 1. Trích xuất vector giá trị PM2.5 tại N trạm
        if isinstance(station_values, dict):
            z = np.array([float(station_values.get(sid, 0.0)) for sid in self.station_ids], dtype=np.float64)
        else:
            arr = np.asarray(station_values, dtype=np.float64)
            if arr.ndim == 0:
                z = np.full(self.n_stations, float(arr), dtype=np.float64)
            elif arr.ndim == 1:
                if arr.size == self.n_stations:
                    z = arr.copy()
                elif arr.size == 48:
                    z = arr.reshape(self.n_stations, 6)[:, 0].copy()
                else:
                    z = np.resize(arr, self.n_stations)
            elif arr.ndim == 2:
                if arr.shape[1] == 6 and arr.shape[0] == self.n_stations:
                    z = arr[:, 0].copy()
                elif arr.shape[0] == self.n_stations:
                    z = arr[:, -1].copy()
                elif arr.shape[1] == self.n_stations:
                    z = arr[-1, :].copy()
                else:
                    z = arr.ravel()[:self.n_stations]
            else:
                z = arr.ravel()[:self.n_stations]

        z = np.maximum(0.0, z)

        coords = np.asarray(self.coordinates, dtype=np.float64)
        N = coords.shape[0]

        x_min, x_max = float(np.min(coords[:, 0]) - buffer_m), float(np.max(coords[:, 0]) + buffer_m)
        y_min, y_max = float(np.min(coords[:, 1]) - buffer_m), float(np.max(coords[:, 1]) + buffer_m)

        x_lin = np.linspace(x_min, x_max, grid_resolution, dtype=np.float64)
        y_lin = np.linspace(y_min, y_max, grid_resolution, dtype=np.float64)
        mesh_x, mesh_y = np.meshgrid(x_lin, y_lin)
        M = grid_resolution * grid_resolution
        points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])  # shape (M, 2)

        c0 = float(nugget)
        if sill is not None:
            c = float(sill)
        else:
            var_z = float(np.var(z))
            c = max(10.0, var_z if var_z > 0.0 else 25.0)
        a = max(100.0, float(range_m))

        diff_stations = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_stations = np.sqrt(np.sum(diff_stations**2, axis=-1))  # (N, N)

        gamma_stations = c0 + c * (1.0 - np.exp(-dist_stations / a))
        np.fill_diagonal(gamma_stations, 0.0)

        K = np.ones((N + 1, N + 1), dtype=np.float64)
        K[:N, :N] = gamma_stations
        K[N, N] = 0.0

        K_reg = K + 1e-9 * np.eye(N + 1)

        diff_grid = coords[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, M, 2)
        dist_grid = np.sqrt(np.sum(diff_grid**2, axis=-1))  # (N, M)

        gamma_grid = c0 + c * (1.0 - np.exp(-dist_grid / a))
        gamma_grid[dist_grid < 1e-6] = 0.0

        G = np.ones((N + 1, M), dtype=np.float64)
        G[:N, :] = gamma_grid

        try:
            Lambda = np.linalg.solve(K_reg, G)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(K_reg) @ G

        weights = Lambda[:N, :]  # (N, M)

        z_interp = z @ weights  # (M,)
        z_interp = np.maximum(0.0, z_interp)

        kriging_var = np.sum(Lambda * G, axis=0)
        kriging_var = np.maximum(0.0, kriging_var)

        heatmap = z_interp.reshape((grid_resolution, grid_resolution))
        variance_map = kriging_var.reshape((grid_resolution, grid_resolution))

        exec_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "grid_resolution": grid_resolution,
            "bounds": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "grid_x": x_lin,
            "grid_y": y_lin,
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "heatmap_pm25": heatmap,
            "kriging_variance": variance_map,
            "variance_map": variance_map,
            "min_pm25": float(np.min(heatmap)),
            "max_pm25": float(np.max(heatmap)),
            "mean_pm25": float(np.mean(heatmap)),
            "execution_time_ms": round(float(exec_ms), 3),
        }


def generate_pollution_heatmap(
    station_values: Union[np.ndarray, Dict[str, float], List[float]],
    grid_resolution: int = 20,
    coordinates: Optional[np.ndarray] = None,
    station_ids: Optional[List[str]] = None,
    nugget: float = 1.0,
    sill: Optional[float] = None,
    range_m: float = 1800.0,
    buffer_m: float = 500.0,
) -> Dict[str, Union[np.ndarray, Dict, float, int]]:
    """
    Hàm độc lập dựng bản đồ nhiệt nội suy không gian 2D bằng Ordinary Kriging.
    """
    engine = SpatialTemporalEngine(station_ids=station_ids, coordinates=coordinates)
    return engine.generate_pollution_heatmap(
        station_values=station_values,
        grid_resolution=grid_resolution,
        nugget=nugget,
        sill=sill,
        range_m=range_m,
        buffer_m=buffer_m,
    )
