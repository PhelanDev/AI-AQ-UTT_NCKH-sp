"""
SPATIAL-TEMPORAL DISPERSION & CROSS-CORRELATION ENGINE
Động cơ Phân tích Tương quan Không gian - Thời gian & Định vị Nguồn Ô nhiễm
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5 (8GB RAM)
Thuộc Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
Phiên bản V5.0: Hỗ trợ số lượng trạm động N >= 1 và Động cơ Bản đồ Nhiệt Thích ứng 2D.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from pi.config import (
    STATIONS,
    STATION_COORDS_UTM,
    STATION_COORDS_WGS84,
    STATION_IDS,
    STATION_NAMES,
    DISTANCE_MATRIX,
    TELEMETRY_INTERVAL_S,
    wgs84_to_utm48n,
)


def _vectorized_vn_aqi_pm25(pm25_arr: np.ndarray) -> np.ndarray:
    """Tính chỉ số VN-AQI vector hóa trên toàn mảng từ nồng độ PM2.5 theo QCVN."""
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
    Hỗ trợ linh hoạt từ N = 1 trạm đến N trạm bất kỳ.
    """
    arr = np.asarray(data_matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"data_matrix must be 2D, got shape {arr.shape}")

    # Đảm bảo mảng có shape (T, N_stations).
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 32:
        arr = arr.T
    elif arr.shape[0] == len(STATION_IDS) and arr.shape[1] > arr.shape[0]:
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

    c0 = norm_data.T @ norm_data
    e0 = np.sum(norm_data**2, axis=0)
    denom0 = np.sqrt(e0[:, None] * e0[None, :]) + 1e-12
    all_corrs[zero_idx] = c0 / denom0

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
    Ước lượng Vector Vận tốc Gió và Hướng Phát tán bằng phương pháp Bình phương Tối thiểu có Trọng số (WLS).
    """
    coords = coordinates if coordinates is not None else STATION_COORDS_UTM
    coords = np.asarray(coords, dtype=np.float64)
    N = min(coords.shape[0], tau_star_matrix.shape[0])
    coords = coords[:N]

    if N < 2:
        return np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0, 0.0

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

    xs = np.arange(center_x - search_radius_m, center_x + search_radius_m, max(100.0, grid_resolution_m))
    ys = np.arange(center_y - search_radius_m, center_y + search_radius_m, max(100.0, grid_resolution_m))
    grid_x, grid_y = np.meshgrid(xs, ys)
    gx = grid_x.ravel()
    gy = grid_y.ravel()

    # Lọc các điểm ngược hướng gió (Upwind filter)
    upwind = (center_x - gx) * wind_dir_x + (center_y - gy) * wind_dir_y >= 0
    gx = gx[upwind]
    gy = gy[upwind]

    p_y = 0.16
    p_z = 0.08

    # Vector hóa song song toàn bộ lưới (M x N)
    rx = coords[:N, 0][None, :] - gx[:, None]
    ry = coords[:N, 1][None, :] - gy[:, None]
    x_down = rx * wind_dir_x + ry * wind_dir_y
    valid = x_down > 10.0

    y_cross = np.abs(rx * (-wind_dir_y) + ry * wind_dir_x)
    sy_arr = np.maximum(5.0, p_y * x_down)
    sz_arr = np.minimum(pblh, np.maximum(5.0, p_z * x_down))

    pred = np.where(
        valid,
        np.exp(- (y_cross ** 2) / (2.0 * sy_arr ** 2)) / (math.pi * sy_arr * sz_arr * wind_speed + 1e-6),
        0.0
    )

    obs = np.maximum(0.0, concs[:N] - baseline)[None, :]
    pred_sq = np.sum(pred ** 2, axis=1)
    valid_pts = (np.sum(valid, axis=1) >= 2) & (pred_sq > 1e-12)

    if np.any(valid_pts):
        q_opts = np.zeros(gx.shape[0])
        q_opts[valid_pts] = np.sum(obs * pred, axis=1)[valid_pts] / pred_sq[valid_pts]
        valid_q = valid_pts & (q_opts > 0)
        residuals = np.full(gx.shape[0], np.inf)
        residuals[valid_q] = np.sum((obs - q_opts[valid_q, None] * pred[valid_q]) ** 2, axis=1)

        best_idx = np.argmin(residuals)
        best_x = float(gx[best_idx])
        best_y = float(gy[best_idx])
        best_q = float(q_opts[best_idx] / 1e6)
        best_residual = float(residuals[best_idx])
    else:
        best_x, best_y, best_q, best_residual = center_x, center_y, 0.0, float("inf")

    confidence = float(np.exp(-best_residual / (total_w + 1e-4)))
    confidence = max(0.20, min(0.95, confidence))

    return best_x, best_y, max(0.1, best_q), confidence


# Bộ nhớ đệm trọng số Ordinary Kriging tĩnh tối ưu hóa chu kỳ thời gian thực
_KRIGING_WEIGHTS_CACHE: Dict[Tuple[int, int, float, float], np.ndarray] = {}


def build_adaptive_spatial_heatmap(
    station_values: Union[np.ndarray, List[float], float],
    coordinates: Optional[np.ndarray] = None,
    wind_vector: Optional[Union[np.ndarray, List[float]]] = None,
    grid_size: int = 25,
    radius_m: float = 1000.0,
    buffer_m: float = 800.0,
    station_ids: Optional[List[str]] = None,
    nugget: float = 0.5,
    range_m: float = 2000.0,
    sill: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Dựng bản đồ nhiệt không gian 2D thích ứng theo số lượng trạm động N >= 1 (Kế hoạch V5.0).
    - N = 1: mode 'single_station_plume' (Quầng khói Gaussian định hướng gió, opacity giảm dần theo bán kính).
    - N = 2: mode 'dual_station_idw' (Nội suy khoảng cách nghịch đảo Anisotropic, nối màu mượt giữa 2 trạm).
    - N >= 3: mode 'multistation_ordinary_kriging' (Ordinary Kriging vector hóa tiền tính toán K^-1, < 2ms, zero NaN/Inf).
    Hỗ trợ đầu vào 1 mốc thời gian (N,) hoặc đa mốc thời gian (N, M).
    """
    t_start = time.perf_counter()

    raw_vals = np.asarray(station_values, dtype=np.float64)
    is_multi_step = False
    M_steps = 1

    if raw_vals.ndim == 0:
        vals = raw_vals.reshape(1)
        N = 1
    elif raw_vals.ndim == 1:
        vals = np.maximum(0.0, raw_vals)
        N = vals.size
    elif raw_vals.ndim == 2:
        if raw_vals.shape[1] > 1:
            is_multi_step = True
            N, M_steps = raw_vals.shape
            vals = np.maximum(0.0, raw_vals)  # shape (N, M)
        else:
            vals = np.maximum(0.0, raw_vals.ravel())
            N = vals.size
    else:
        vals = np.maximum(0.0, raw_vals.reshape(raw_vals.shape[0], -1))
        N, M_steps = vals.shape
        is_multi_step = M_steps > 1

    # Chuẩn hóa tọa độ không gian (UTM và WGS84)
    if coordinates is not None:
        coords_raw = np.asarray(coordinates, dtype=np.float64)
        if coords_raw.ndim == 1:
            coords_raw = coords_raw.reshape(1, -1)
        N_c = min(N, coords_raw.shape[0])
        coords_in = coords_raw[:N_c]
        if coords_in[0, 0] <= 90.0 and coords_in[0, 1] <= 180.0:
            coords_u = np.array([wgs84_to_utm48n(row[0], row[1]) for row in coords_in], dtype=np.float64)
            coords_w = coords_in.copy()
        else:
            coords_u = coords_in.copy()
            lat_est = 20.9852 + (coords_u[:, 1] - 2320700.0) / 110574.0
            lon_est = 105.7972 + (coords_u[:, 0] - 499700.0) / (111320.0 * math.cos(math.radians(20.9852)))
            coords_w = np.column_stack([lat_est, lon_est])
    else:
        coords_u = STATION_COORDS_UTM[:N].copy()
        coords_w = STATION_COORDS_WGS84[:N].copy()

    N = coords_u.shape[0]
    if not is_multi_step:
        vals = vals[:N]
    else:
        vals = vals[:N, :]

    if station_ids is not None:
        s_ids = list(station_ids)[:N]
    else:
        s_ids = list(STATION_IDS[:N]) if N <= len(STATION_IDS) else [f"STATION_{i+1:02d}" for i in range(N)]

    # Bounding box
    span = max(buffer_m, 2600.0)
    x_min = float(np.min(coords_u[:, 0]) - span)
    x_max = float(np.max(coords_u[:, 0]) + span)
    y_min = float(np.min(coords_u[:, 1]) - span)
    y_max = float(np.max(coords_u[:, 1]) + span)
    lat_min = float(np.min(coords_w[:, 0]) - 0.02)
    lat_max = float(np.max(coords_w[:, 0]) + 0.02)
    lon_min = float(np.min(coords_w[:, 1]) - 0.02)
    lon_max = float(np.max(coords_w[:, 1]) + 0.02)

    x_axis = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    lat_axis = np.linspace(lat_min, lat_max, grid_size, dtype=np.float64)
    lon_axis = np.linspace(lon_min, lon_max, grid_size, dtype=np.float64)

    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)  # shape (G, G)

    w_vec = np.asarray(wind_vector if wind_vector is not None else [0.0, 0.0], dtype=np.float64).ravel()
    u_w = float(w_vec[0]) if w_vec.size >= 1 else 0.0
    v_w = float(w_vec[1]) if w_vec.size >= 2 else 0.0
    w_speed = math.sqrt(u_w * u_w + v_w * v_w)

    station_alphas = np.zeros((N, grid_size, grid_size), dtype=np.float64)
    eff_dists = np.zeros((N, grid_size, grid_size), dtype=np.float64)
    nearest_cells: List[Tuple[int, int]] = []

    for i in range(N):
        dx = mesh_x - coords_u[i, 0]
        dy = mesh_y - coords_u[i, 1]
        euclid_sq = dx * dx + dy * dy
        r_idx, c_idx = np.unravel_index(int(np.argmin(euclid_sq)), (grid_size, grid_size))
        nearest_cells.append((int(r_idx), int(c_idx)))

        if w_speed > 0.2:
            wx = u_w / w_speed
            wy = v_w / w_speed
            d_par = dx * wx + dy * wy
            d_perp = -dx * wy + dy * wx
            stretch_down = 1.0 + min(0.35, 0.08 * w_speed)
            compress_up = max(0.65, 1.0 - 0.10 * w_speed)
            scale_par = np.where(d_par >= 0.0, stretch_down, compress_up)
            d_eff = np.sqrt((d_par / scale_par) ** 2 + d_perp ** 2)
        else:
            d_eff = np.sqrt(euclid_sq)

        d_eff[r_idx, c_idx] = 0.0
        eff_dists[i] = d_eff
        alpha_i = np.exp(-(d_eff ** 2) / (2.0 * (radius_m ** 2)))
        alpha_i[r_idx, c_idx] = 1.0
        station_alphas[i] = alpha_i

    opacity_grid = 1.0 - np.prod(1.0 - station_alphas, axis=0)
    opacity_grid = np.clip(opacity_grid, 0.0, 1.0).astype(np.float32)

    if N == 1:
        mode = "single_station_plume"
        a0 = station_alphas[0]
        r0, c0 = nearest_cells[0]
        if not is_multi_step:
            c0_val = float(vals[0])
            c_bg = min(c0_val, 22.0)
            grid_pm25 = (a0 * c0_val + (1.0 - a0) * c_bg).astype(np.float32)
            grid_pm25[r0, c0] = float(c0_val)
        else:
            c0_series = vals[0, :]
            c_bg_series = np.minimum(c0_series, 22.0)
            grid_pm25 = (a0[None, :, :] * c0_series[:, None, None] + (1.0 - a0[None, :, :]) * c_bg_series[:, None, None]).astype(np.float32)
            grid_pm25[:, r0, c0] = c0_series.astype(np.float32)

    elif N == 2:
        mode = "dual_station_idw"
        w0 = 1.0 / ((eff_dists[0] + 10.0) ** 2) + 1.2 * station_alphas[0]
        w1 = 1.0 / ((eff_dists[1] + 10.0) ** 2) + 1.2 * station_alphas[1]
        w_sum = w0 + w1 + 1e-12
        W0 = w0 / w_sum
        W1 = w1 / w_sum

        if not is_multi_step:
            v0 = float(vals[0])
            v1 = float(vals[1])
            grid_pm25 = (W0 * v0 + W1 * v1).astype(np.float32)
            grid_pm25[nearest_cells[0]] = float(v0)
            grid_pm25[nearest_cells[1]] = float(v1)
        else:
            v0 = vals[0, :]
            v1 = vals[1, :]
            grid_pm25 = (W0[None, :, :] * v0[:, None, None] + W1[None, :, :] * v1[:, None, None]).astype(np.float32)
            grid_pm25[:, nearest_cells[0][0], nearest_cells[0][1]] = v0.astype(np.float32)
            grid_pm25[:, nearest_cells[1][0], nearest_cells[1][1]] = v1.astype(np.float32)

    else:
        mode = "multistation_ordinary_kriging"
        cache_key = (int(N), int(grid_size), round(float(coords_u[0, 0]), 1), round(float(coords_u[0, 1]), 1))
        if cache_key in _KRIGING_WEIGHTS_CACHE:
            weights = _KRIGING_WEIGHTS_CACHE[cache_key]
        else:
            c0_krig = float(nugget)
            a_krig = max(100.0, float(range_m))
            if sill is not None:
                c_krig = float(sill)
            else:
                var_v = float(np.var(vals))
                c_krig = max(10.0, var_v if var_v > 0.0 else 25.0)
    
            diff_st = coords_u[:, None, :] - coords_u[None, :, :]
            dist_st = np.sqrt(np.sum(diff_st ** 2, axis=-1))
            gamma_st = c0_krig + c_krig * (1.0 - np.exp(-dist_st / a_krig))
            np.fill_diagonal(gamma_st, 0.0)
    
            K = np.ones((N + 1, N + 1), dtype=np.float64)
            K[:N, :N] = gamma_st
            K[N, N] = 0.0
            K_reg = K + 1e-9 * np.eye(N + 1)
    
            K_inv = np.linalg.pinv(K_reg)
    
            P = grid_size * grid_size
            pts = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
            diff_pts = coords_u[:, None, :] - pts[None, :, :]
            dist_pts = np.sqrt(np.sum(diff_pts ** 2, axis=-1))
            gamma_pts = c0_krig + c_krig * (1.0 - np.exp(-dist_pts / a_krig))
            gamma_pts[dist_pts < 1e-6] = 0.0
    
            G = np.ones((N + 1, P), dtype=np.float64)
            G[:N, :] = gamma_pts
    
            Lambda = K_inv @ G
            weights = Lambda[:N, :]  # shape (N, P)
            _KRIGING_WEIGHTS_CACHE[cache_key] = weights

        if not is_multi_step:
            z_interp = vals @ weights
            grid_pm25 = z_interp.reshape((grid_size, grid_size)).astype(np.float32)
            for i, (ri, ci) in enumerate(nearest_cells):
                grid_pm25[ri, ci] = float(vals[i])
        else:
            z_interp = vals.T @ weights
            grid_pm25 = z_interp.reshape((M_steps, grid_size, grid_size)).astype(np.float32)
            for i, (ri, ci) in enumerate(nearest_cells):
                grid_pm25[:, ri, ci] = vals[i, :].astype(np.float32)

    grid_pm25 = np.maximum(0.0, grid_pm25)
    mean_val = float(np.mean(vals)) if vals.size > 0 else 35.0
    grid_pm25 = np.nan_to_num(grid_pm25, nan=mean_val, posinf=500.0, neginf=0.0).astype(np.float32)

    grid_aqi = _vectorized_vn_aqi_pm25(grid_pm25)

    markers = []
    if not is_multi_step:
        st_aqi = _vectorized_vn_aqi_pm25(vals)
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
                "aqi": float(st_aqi[i]),
            })
    else:
        st_aqi = _vectorized_vn_aqi_pm25(vals[:, -1])
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
                "pm25": float(vals[i, -1]),
                "aqi": float(st_aqi[i]),
            })

    exec_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "mode": mode,
        "num_active_stations": int(N),
        "grid_size": int(grid_size),
        "pm25_grid": grid_pm25,
        "aqi_grid": grid_aqi,
        "opacity_grid": opacity_grid,
        "grid_pm25": grid_pm25,  # Alias tương thích ngược
        "grid_aqi": grid_aqi,    # Alias tương thích ngược
        "markers": markers,
        "x_utm_axis": x_axis.astype(np.float32),
        "y_utm_axis": y_axis.astype(np.float32),
        "lat_axis": lat_axis.astype(np.float32),
        "lon_axis": lon_axis.astype(np.float32),
        "bounds": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "execution_time_ms": round(float(exec_ms), 3),
    }


def generate_web_heatmap(
    station_values: Union[np.ndarray, List[float]],
    coordinates_utm: Optional[np.ndarray] = None,
    coordinates_wgs84: Optional[np.ndarray] = None,
    station_ids: Optional[List[str]] = None,
    wind_vector: Optional[np.ndarray] = None,
    grid_size: int = 25,
    radius_m: float = 1000.0,
    buffer_m: float = 800.0,
) -> Dict[str, Any]:
    """Hàm bao bọc tương thích ngược, điều hướng về build_adaptive_spatial_heatmap."""
    coords = coordinates_utm if coordinates_utm is not None else coordinates_wgs84
    return build_adaptive_spatial_heatmap(
        station_values=station_values,
        coordinates=coords,
        wind_vector=wind_vector,
        grid_size=grid_size,
        radius_m=radius_m,
        buffer_m=buffer_m,
        station_ids=station_ids,
    )


class SpatialTemporalEngine:
    """
    Động cơ Tương quan Không gian - Thời gian Đa trạm (Milestone 2 & V5.0 Dynamic 1-to-N).
    Chạy định kỳ trên Raspberry Pi 5 để phân tích luồng phát tán ô nhiễm và vẽ bản đồ Web thích ứng.
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
        compute_heatmap: bool = True,
    ) -> Dict[str, Any]:
        """
        Phân tích toàn diện: Tương quan trễ -> Vector gió -> Định vị nguồn phát tán -> Bản đồ 2D thích ứng.
        Hỗ trợ hoàn hảo từ 1 trạm (N=1) đến N trạm.
        """
        hist = np.asarray(pm25_history, dtype=np.float64)
        if hist.ndim == 1:
            hist = hist[:, np.newaxis]

        R_mat, tau_mat = compute_cross_correlation_matrix(hist, max_lag=max_lag, dt_s=dt_s)
        active_n = min(R_mat.shape[0], len(self.coordinates)) if self.coordinates is not None else R_mat.shape[0]
        active_coords = np.asarray(self.coordinates, dtype=np.float64)[:active_n] if self.coordinates is not None else STATION_COORDS_UTM[:active_n]
        active_ids = list(self.station_ids)[:active_n] if self.station_ids is not None else [f"STATION_{i+1:02d}" for i in range(active_n)]

        wind_vec, wind_speed, bearing, r2 = estimate_wind_and_dispersion(
            tau_mat, coordinates=active_coords, r_matrix=R_mat
        )

        if hist.shape[1] >= active_n:
            current_concs = hist[-1, :active_n]
        else:
            current_concs = hist[:active_n, -1]

        src_x, src_y, q_rate, conf = localize_emission_source(
            current_concs, wind_vec, coordinates=active_coords
        )

        heatmap_2d = None
        if compute_heatmap:
            heatmap_2d = build_adaptive_spatial_heatmap(
                station_values=current_concs,
                coordinates=active_coords,
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
    ) -> Dict[str, Any]:
        """
        Dựng bản đồ nhiệt nồng độ bụi PM2.5 không gian 2D bằng Ordinary Kriging vector hóa NumPy.
        """
        t_start = time.perf_counter()

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
        points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

        c0 = float(nugget)
        if sill is not None:
            c = float(sill)
        else:
            var_z = float(np.var(z))
            c = max(10.0, var_z if var_z > 0.0 else 25.0)
        a = max(100.0, float(range_m))

        diff_stations = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_stations = np.sqrt(np.sum(diff_stations**2, axis=-1))

        gamma_stations = c0 + c * (1.0 - np.exp(-dist_stations / a))
        np.fill_diagonal(gamma_stations, 0.0)

        K = np.ones((N + 1, N + 1), dtype=np.float64)
        K[:N, :N] = gamma_stations
        K[N, N] = 0.0
        K_reg = K + 1e-9 * np.eye(N + 1)

        diff_grid = coords[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_grid = np.sqrt(np.sum(diff_grid**2, axis=-1))

        gamma_grid = c0 + c * (1.0 - np.exp(-dist_grid / a))
        gamma_grid[dist_grid < 1e-6] = 0.0

        G = np.ones((N + 1, M), dtype=np.float64)
        G[:N, :] = gamma_grid

        try:
            Lambda = np.linalg.solve(K_reg, G)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(K_reg) @ G

        weights = Lambda[:N, :]
        z_interp = z @ weights
        z_interp = np.maximum(0.0, z_interp)

        kriging_var = np.sum(Lambda * G, axis=0)
        kriging_var = np.maximum(0.0, kriging_var)

        heatmap = z_interp.reshape((grid_resolution, grid_resolution))
        variance_map = kriging_var.reshape((grid_resolution, grid_resolution))

        exec_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "grid_resolution": grid_resolution,
            "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
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
) -> Dict[str, Any]:
    """Hàm độc lập dựng bản đồ nhiệt nội suy không gian 2D bằng Ordinary Kriging."""
    engine = SpatialTemporalEngine(station_ids=station_ids, coordinates=coordinates)
    return engine.generate_pollution_heatmap(
        station_values=station_values,
        grid_resolution=grid_resolution,
        nugget=nugget,
        sill=sill,
        range_m=range_m,
        buffer_m=buffer_m,
    )
