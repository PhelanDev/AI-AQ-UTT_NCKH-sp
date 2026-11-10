"""Central AI Engine E2E Testing Suite Package.

Provides module availability discovery, reference mathematical oracles,
synthetic fixtures, and tolerance constants for all 4 test tiers.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from pi import config as central_config
    HAS_CONFIG = True
except (ImportError, ModuleNotFoundError):
    central_config = None
    HAS_CONFIG = False

try:
    from pi import simulator as central_simulator
    HAS_SIMULATOR = True
except (ImportError, ModuleNotFoundError):
    central_simulator = None
    HAS_SIMULATOR = False

try:
    from pi import data_pipeline as central_data_pipeline
    HAS_DATA_PIPELINE = True
except (ImportError, ModuleNotFoundError):
    central_data_pipeline = None
    HAS_DATA_PIPELINE = False

try:
    from pi import spatial_temporal_engine as central_spatial_temporal
    HAS_SPATIAL_TEMPORAL = True
except (ImportError, ModuleNotFoundError):
    central_spatial_temporal = None
    HAS_SPATIAL_TEMPORAL = False

try:
    from pi import forecasting_engine as central_forecasting
    HAS_FORECASTING = True
except (ImportError, ModuleNotFoundError):
    central_forecasting = None
    HAS_FORECASTING = False

try:
    from pi import anomaly_engine as central_anomaly
    HAS_ANOMALY = True
except (ImportError, ModuleNotFoundError):
    central_anomaly = None
    HAS_ANOMALY = False

try:
    from pi import central_engine as central_core
    HAS_CENTRAL_ENGINE = True
except (ImportError, ModuleNotFoundError):
    central_core = None
    HAS_CENTRAL_ENGINE = False

try:
    from pi import export_onnx as central_export_onnx
    HAS_EXPORT_ONNX = True
except (ImportError, ModuleNotFoundError):
    central_export_onnx = None
    HAS_EXPORT_ONNX = False

try:
    from pi import benchmark_pi5 as central_benchmark
    HAS_BENCHMARK = True
except (ImportError, ModuleNotFoundError):
    central_benchmark = None
    HAS_BENCHMARK = False

def oracle_magnus_tetens(temp: float, rh: float) -> float:
    """Calculates dew point (deg C) using standard Magnus-Tetens formula.

    Formula:
        alpha(T, RH) = (17.27 * T) / (237.7 + T) + ln(RH / 100.0)
        T_dp = (237.7 * alpha) / (17.27 - alpha)
    """
    rh_clamped = max(1e-4, min(100.0, rh))
    alpha = (17.27 * temp) / (237.7 + temp) + math.log(rh_clamped / 100.0)
    denom = 17.27 - alpha
    if abs(denom) < 1e-9:
        return temp
    return (237.7 * alpha) / denom

def oracle_humidity_gate(rh: float) -> float:
    """Calculates Sigmoid adaptive humidity gate centered at 55% RH.

    Formula:
        gate = 1.0 / (1.0 + exp(-(RH - 55.0) / 6.0))
    """
    z = -(rh - 55.0) / 6.0
    z_clamped = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(z_clamped))

def oracle_haenel_swelling(rh: float, a_h: float = 0.28, b_h: float = 0.92) -> float:
    """Calculates optical particle hygroscopic growth factor f(RH) by Hanel model.

    Formula:
        f(RH) = 1.0 if RH <= 60%
        f(RH) = 1.0 + a_h * ( (RH/100) / (1 - RH/100) )^b_h if RH > 60%
    """
    if rh <= 60.0:
        return 1.0
    rh_ratio = min(0.999, max(0.0, rh / 100.0))
    term = rh_ratio / (1.0 - rh_ratio)
    return 1.0 + a_h * (term ** b_h)

def oracle_diurnal_pblh(hour: float, h_min: float = 250.0, h_max: float = 1600.0) -> float:
    """Calculates diurnal Planetary Boundary Layer Height (PBLH) in meters.

    Formula:
        H_PBL(t) = H_min + (H_max - H_min) * max(0, sin(pi * (hour - 6) / 14))^1.6
    """
    hour_mod = hour % 24.0
    if 6.0 <= hour_mod <= 20.0:
        solar_phase = math.sin(math.pi * (hour_mod - 6.0) / 14.0)
        day_factor = max(0.0, solar_phase) ** 1.6
    else:
        day_factor = 0.0
    return h_min + (h_max - h_min) * day_factor

def oracle_bimodal_traffic(hour: float, is_weekend: bool = False) -> float:
    """Calculates bimodal traffic rush-hour emission factor (ug/s).

    Formula:
        Q(t) = Q_base + Q_morn * exp(-(hour - 8.0)^2 / (2 * 1.2^2))
                      + Q_eve * exp(-(hour - 18.0)^2 / (2 * 1.5^2))
    """
    hour_mod = hour % 24.0
    q_base = 15.0
    q_morn = 75.0 * math.exp(-((hour_mod - 8.0) ** 2) / (2.0 * (1.2 ** 2)))
    q_eve = 85.0 * math.exp(-((hour_mod - 18.0) ** 2) / (2.0 * (1.5 ** 2)))
    factor = 0.65 if is_weekend else 1.0
    return (q_base + q_morn + q_eve) * factor

def oracle_gaussian_puff(
    x: float, y: float,
    x_puff: float, y_puff: float,
    sigma: float, mass: float, pblh: float
) -> float:
    """Calculates single Gaussian puff concentration contribution at (x, y)."""
    sigma = max(1.0, sigma)
    pblh = max(50.0, pblh)
    dist_sq = (x - x_puff) ** 2 + (y - y_puff) ** 2
    denom = 2.0 * math.pi * (sigma ** 2) * pblh
    return (mass / denom) * math.exp(-dist_sq / (2.0 * (sigma ** 2)))

def oracle_slowness_wls(
    delta_coords: np.ndarray,
    tau_stars: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """Solves overdetermined system A * s = tau for Slowness Vector s = (s_x, s_y).

    Returns:
        wind_vector: (u, v) in m/s
        wind_speed: float in m/s
        wind_bearing: float in degrees [0, 360)
    """
    a_mat = np.asarray(delta_coords, dtype=np.float64)
    tau = np.asarray(tau_stars, dtype=np.float64)

    if weights is None:
        w_mat = np.eye(len(tau), dtype=np.float64)
    else:
        w_mat = np.diag(np.asarray(weights, dtype=np.float64) ** 2)

    ata = a_mat.T @ w_mat @ a_mat

    ata_reg = ata + 1e-8 * np.eye(2)
    at_tau = a_mat.T @ w_mat @ tau
    s_vec = np.linalg.solve(ata_reg, at_tau)

    s_sq = float(s_vec[0] ** 2 + s_vec[1] ** 2)
    if s_sq < 1e-12:
        return np.array([0.0, 0.0]), 0.0, 0.0

    u = s_vec[0] / s_sq
    v = s_vec[1] / s_sq
    speed = math.sqrt(u ** 2 + v ** 2)
    bearing = (math.atan2(u, v) * 180.0 / math.pi) % 360.0
    return np.array([u, v]), speed, bearing

def oracle_asymmetric_huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.5,
    lambda_peak: float = 0.4,
    lambda_trend: float = 0.2
) -> float:
    """Calculates reference Asymmetric Huber Loss with Peak and Trend penalties."""
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    huber = np.where(abs_diff <= delta, 0.5 * (diff ** 2), delta * (abs_diff - 0.5 * delta))
    base_loss = float(np.mean(huber))

    peak_mask = (y_true > 50.0).astype(np.float64)
    under_pred = np.maximum(0.0, y_true - y_pred)
    peak_penalty = float(np.mean(under_pred * peak_mask))

    if len(y_true) > 1:
        dy_true = np.diff(y_true, axis=0)
        dy_pred = np.diff(y_pred, axis=0)
        norm_true = np.linalg.norm(dy_true)
        norm_pred = np.linalg.norm(dy_pred)
        if norm_true > 1e-6 and norm_pred > 1e-6:
            cos_sim = float(np.sum(dy_true * dy_pred) / (norm_true * norm_pred))
            trend_penalty = 1.0 - cos_sim
        else:
            trend_penalty = 0.0
    else:
        trend_penalty = 0.0

    return base_loss + lambda_peak * peak_penalty + lambda_trend * trend_penalty

def oracle_mahalanobis_distance(
    residual: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """Calculates Mahalanobis distance with pseudo-inverse regularization."""
    res = np.asarray(residual, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov_matrix, dtype=np.float64)

    reg_cov = cov + 1e-6 * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(reg_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(reg_cov)
    dist_sq = float(res.T @ inv_cov @ res)
    return math.sqrt(max(0.0, dist_sq))
