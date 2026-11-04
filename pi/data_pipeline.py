"""
ai/central/data_pipeline.py
Multi-Station Spatial-Temporal Data Pipeline for Thanh Xuan Air Quality Network.
- Tier 1 TinyML ResMLP19k Edge Calibration (Magnus-Tetens, Dual Sensor Filter, Humidity Gate).
- Chronological Train / Val / Test Temporal Splitting (Strict Zero Data Leakage).
- Multi-station Sliding Window Feature Construction: X in R^{B x T_in x N_stations x F}.
- PyTorch Dataset & DataLoader Factory.
"""

from dataclasses import dataclass
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pi.config import (
    FEATURE_NAMES,
    NUM_FEATURES,
    NUM_STATIONS,
    STATION_IDS,
    STATIONS,
)
from pi.simulator import (
    AnomalyScenario,
    AnomalyType,
    AtmosphericSimulator,
)

def _load_tinyml_scaler() -> Tuple[np.ndarray, np.ndarray]:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(curr_dir)
    params_path = os.path.join(root_dir, "mlops", "models", "robust_resmlp_19k_params.json")
    if os.path.exists(params_path):
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            return (
                np.array(params["x_min"], dtype=np.float32),
                np.array(params["x_max"], dtype=np.float32),
            )
        except Exception:
            pass
    return (
        np.array([0.5, 0.0, -25.350000381469727, 5.199999809265137, -35.7599983215332], dtype=np.float32),
        np.array([1216.800048828125, 93.06999969482422, 44.45000076293945, 100.0, 84.0], dtype=np.float32),
    )


TINYML_X_MIN, TINYML_X_MAX = _load_tinyml_scaler()

def calc_dew_point_magnus_tetens(
    temp_c: Union[float, np.ndarray], rh_percent: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate dew point (°C) using the Magnus-Tetens approximation.

    alpha(T, RH) = (17.27 * T) / (237.7 + T) + ln(RH / 100)
    T_dew = (237.7 * alpha) / (17.27 - alpha)
    """
    is_scalar = np.isscalar(temp_c) and np.isscalar(rh_percent)
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_percent, dtype=np.float64)
    rh = np.clip(rh, 1.0, 100.0)

    a = 17.27
    b = 237.7
    alpha = ((a * t) / (b + t)) + np.log(rh / 100.0)
    denom = a - alpha
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    dew_point = (b * alpha) / denom

    return float(dew_point) if is_scalar else dew_point

def filter_dual_pm_sensor(
    pm1: Union[float, np.ndarray], pm2: Union[float, np.ndarray], threshold: float = 25.0
) -> Tuple[Union[float, np.ndarray], Union[bool, np.ndarray]]:
    """Dual PMS7003 cross-sensor difference filter matching tinyml_edge_engine.cpp.

    If |pm1 - pm2| > 25.0 ug/m3:
        Flag chamber condensation / mismatch anomaly and select min(pm1, pm2)
    Else:
        Return arithmetic average (pm1 + pm2) / 2
    """
    is_scalar = np.isscalar(pm1) and np.isscalar(pm2)
    p1 = np.asarray(pm1, dtype=np.float64)
    p2 = np.asarray(pm2, dtype=np.float64)
    p1, p2 = np.broadcast_arrays(p1, p2)

    diff = np.abs(p1 - p2)
    is_anomaly = diff > threshold
    pm_avg = np.where(is_anomaly, np.minimum(p1, p2), (p1 + p2) * 0.5)

    if is_scalar:
        return float(pm_avg), bool(is_anomaly)
    return pm_avg, is_anomaly

def calc_vn_aqi_pm25(
    pm25: Union[float, np.ndarray],
    standard: str = "EPA",
) -> Union[float, np.ndarray]:
    """Calculate Air Quality Index (AQI) from PM2.5 (ug/m3) according to EPA or QCVN breakpoints.

    Args:
        pm25: PM2.5 concentration in ug/m3 (scalar or numpy array).
        standard: Standard to use - "EPA" (US EPA, default) or "VN_QCVN" (Vietnam QCVN 05:2023).

    Returns:
        Calculated AQI value (scalar float or numpy array).
    """
    is_scalar = np.isscalar(pm25)
    p = np.asarray(pm25, dtype=np.float64)

    if standard.strip().upper() in ("VN_QCVN", "QCVN", "VN"):
        breakpoints = [
            (0.0, 25.0, 0.0, 50.0),
            (25.0, 50.0, 50.0, 100.0),
            (50.0, 80.0, 100.0, 150.0),
            (80.0, 150.0, 150.0, 200.0),
            (150.0, 250.0, 200.0, 300.0),
            (250.0, 500.0, 300.0, 500.0),
        ]
        top_c = 500.0
    else:
        breakpoints = [
            (0.0, 12.0, 0.0, 50.0),
            (12.0, 35.4, 50.0, 100.0),
            (35.4, 55.4, 100.0, 150.0),
            (55.4, 150.4, 150.0, 200.0),
            (150.4, 250.4, 200.0, 300.0),
            (250.4, 500.4, 300.0, 500.0),
        ]
        top_c = 500.4

    aqi = np.zeros_like(p)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = (p >= c_lo) & (p <= c_hi)
        aqi = np.where(
            mask,
            ((i_hi - i_lo) / (c_hi - c_lo)) * (p - c_lo) + i_lo,
            aqi,
        )

    aqi = np.where(p > top_c, 500.0, aqi)
    aqi = np.clip(aqi, 0.0, 500.0)

    return float(aqi) if is_scalar else aqi

class RobustResMLP19k(nn.Module):
    """PyTorch architecture of RobustResMLP19k matching Tier 1 TinyML edge engine.

    Structure: 5 -> 96 -> ResBlock(96->96) -> ResBlock(96->96) -> 1 (19,297 parameters)
    Output: Sigmoid(z) * 0.80 + 0.25 in [0.25, 1.05]
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 96):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_block1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_block2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.relu(self.fc_in(x))
        h1 = self.relu(self.fc_block1(h0) + h0)
        h2 = self.relu(self.fc_block2(h1) + h1)
        out = self.fc_out(h2)
        c_factor = self.sigmoid(out) * 0.80 + 0.25
        return c_factor

class TinyMLEdgeCalibrator:
    """Tier 1 TinyML calibration engine evaluating RobustResMLP19k and Adaptive Humidity Gating."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model = RobustResMLP19k(5, 96).to(self.device)

        resolved_path = model_path
        if resolved_path is None:

            curr_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(curr_dir)
            possible_paths = [
                os.path.join(root_dir, "mlops", "models", "robust_resmlp_19k.pt"),
                os.path.join(root_dir, "ai", "models", "robust_resmlp_19k.pt"),
                os.path.join(root_dir, "models", "robust_resmlp_19k.pt"),
                os.path.join(curr_dir, "models", "robust_resmlp_19k.pt"),
                os.path.join(curr_dir, "..", "mlops", "models", "robust_resmlp_19k.pt"),
            ]
            for p in possible_paths:
                p_norm = os.path.normpath(p)
                if os.path.exists(p_norm):
                    resolved_path = p_norm
                    break

        if resolved_path and os.path.exists(resolved_path):
            state_dict = torch.load(resolved_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.x_min = torch.from_numpy(TINYML_X_MIN).float().to(self.device)
        self.x_max = torch.from_numpy(TINYML_X_MAX).float().to(self.device)
        self.x_range = self.x_max - self.x_min

    def calibrate(
        self,
        pm1_raw: Union[float, np.ndarray],
        pm2_raw: Union[float, np.ndarray],
        temp_c: Union[float, np.ndarray],
        rh_percent: Union[float, np.ndarray],
    ) -> Tuple[
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        Union[bool, np.ndarray],
    ]:
        """Perform full edge calibration on dual PM sensor readings.

        Returns (pm25_calib, c_final, humidity_gate, dew_point, is_anomaly).
        """
        is_scalar = (
            np.isscalar(pm1_raw)
            and np.isscalar(pm2_raw)
            and np.isscalar(temp_c)
            and np.isscalar(rh_percent)
        )

        p1 = np.asarray(pm1_raw, dtype=np.float32)
        p2 = np.asarray(pm2_raw, dtype=np.float32)
        t = np.asarray(temp_c, dtype=np.float32)
        rh = np.asarray(rh_percent, dtype=np.float32)
        p1, p2, t, rh = np.broadcast_arrays(p1, p2, t, rh)

        pm_avg, is_anom = filter_dual_pm_sensor(p1, p2)
        diff_pm = np.abs(p1 - p2)

        dew_point = calc_dew_point_magnus_tetens(t, rh)

        orig_shape = pm_avg.shape
        flat_pm_avg = pm_avg.ravel()
        flat_diff = diff_pm.ravel()
        flat_t = t.ravel()
        flat_rh = rh.ravel()
        flat_dew = dew_point.ravel()

        features = np.stack(
            [flat_pm_avg, flat_diff, flat_t, flat_rh, flat_dew], axis=1
        )

        feat_tensor = torch.from_numpy(features).float().to(self.device)
        norm_feat = torch.clamp((feat_tensor - self.x_min) / self.x_range, 0.0, 1.0)

        with torch.no_grad():
            c_pred_tensor = self.model(norm_feat).squeeze(-1)
            c_pred = c_pred_tensor.cpu().numpy()

        c_pred = c_pred.reshape(orig_shape)

        gate = 1.0 / (1.0 + np.exp(-(rh - 55.0) / 6.0))
        c_final = (1.0 - gate) * 0.95 + gate * c_pred
        c_final = np.clip(c_final, 0.25, 1.05)

        pm25_calib = np.maximum(0.0, pm_avg * c_final)

        if is_scalar:
            return (
                float(pm25_calib),
                float(c_final),
                float(gate),
                float(dew_point),
                bool(is_anom),
            )

        return pm25_calib, c_final, gate, dew_point, is_anom

_GLOBAL_CALIBRATOR: Optional[TinyMLEdgeCalibrator] = None

def calibrate_pm25_edge(
    pm1_raw: Union[float, np.ndarray],
    pm2_raw: Union[float, np.ndarray],
    temp_c: Union[float, np.ndarray],
    rh_percent: Union[float, np.ndarray],
    calibrator: Optional[TinyMLEdgeCalibrator] = None,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
]:
    """Module-level calibration helper matching TinyML Edge Engine."""
    global _GLOBAL_CALIBRATOR
    if calibrator is None:
        if _GLOBAL_CALIBRATOR is None:
            _GLOBAL_CALIBRATOR = TinyMLEdgeCalibrator()
        calibrator = _GLOBAL_CALIBRATOR
    return calibrator.calibrate(pm1_raw, pm2_raw, temp_c, rh_percent)

class AirQualityDataPipeline:
    """High-level Data Pipeline generating multi-station simulation datasets,

    calibrating PM2.5, performing chronological splits, and structuring PyTorch tensors.
    """

    def __init__(self, calibrator: Optional[TinyMLEdgeCalibrator] = None):
        self.calibrator = calibrator or TinyMLEdgeCalibrator()

    def generate_dataset(
        self,
        duration_days: float = 7.0,
        sample_interval_s: float = 600.0,
        seed: int = 42,
        anomalies: Optional[List[AnomalyScenario]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run atmospheric simulation and apply edge calibration across all 8 stations."""
        sim = AtmosphericSimulator(dt_s=sample_interval_s, seed=seed)
        if anomalies:
            for anom in anomalies:
                sim.add_anomaly(anom)

        sim_data = sim.run_simulation(
            duration_days=duration_days, sample_interval_s=sample_interval_s
        )

        T = len(sim_data["time_s"])
        N = NUM_STATIONS

        pm25_calib = np.zeros((T, N), dtype=np.float32)
        dew_points = np.zeros((T, N), dtype=np.float32)
        c_factors = np.zeros((T, N), dtype=np.float32)

        temp_mat = np.tile(sim_data["temp_c"][:, np.newaxis], (1, N))
        rh_mat = np.tile(sim_data["rh_percent"][:, np.newaxis], (1, N))

        pm25_calib, c_factors, _, dew_points, _ = self.calibrator.calibrate(
            sim_data["pm25_raw_1"], sim_data["pm25_raw_2"], temp_mat, rh_mat
        )

        aqi = calc_vn_aqi_pm25(pm25_calib)

        time_hours = (sim_data["time_s"] / 3600.0) % 24.0
        hour_sin = np.sin(2.0 * np.pi * time_hours / 24.0).astype(np.float32)
        hour_cos = np.cos(2.0 * np.pi * time_hours / 24.0).astype(np.float32)

        hour_sin_mat = np.tile(hour_sin[:, np.newaxis], (1, N))
        hour_cos_mat = np.tile(hour_cos[:, np.newaxis], (1, N))
        wind_u_mat = np.tile(sim_data["wind_u"][:, np.newaxis], (1, N))
        wind_v_mat = np.tile(sim_data["wind_v"][:, np.newaxis], (1, N))

        features = np.stack(
            [
                pm25_calib,
                sim_data["pm10_raw"],
                temp_mat,
                rh_mat,
                dew_points,
                sim_data["co2"],
                sim_data["voc"],
                wind_u_mat,
                wind_v_mat,
                hour_sin_mat,
                hour_cos_mat,
            ],
            axis=-1,
        )

        return {
            "timestamps": sim_data["time_s"],
            "features": features.astype(np.float32),
            "pm25_calib": pm25_calib.astype(np.float32),
            "pm25_true": sim_data["pm25_true"].astype(np.float32),
            "aqi": aqi.astype(np.float32),
            "anomaly_flags": sim_data["anomaly_flags"],
            "wind_u": sim_data["wind_u"],
            "wind_v": sim_data["wind_v"],
            "temp_c": sim_data["temp_c"],
            "rh_percent": sim_data["rh_percent"],
            "pblh": sim_data["pblh"],
        }

    def chronological_split(
        self,
        dataset: Dict[str, np.ndarray],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Strict chronological split (70% train, 15% val, 15% test).

        Zero temporal data leakage guaranteed: max(train_t) < min(val_t) <= max(val_t) < min(test_t).
        """
        assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5)

        total_steps = len(dataset["timestamps"])
        train_end = int(round(total_steps * train_ratio))
        val_end = int(round(total_steps * (train_ratio + val_ratio)))

        train_dict = {}
        val_dict = {}
        test_dict = {}

        for k, v in dataset.items():
            train_dict[k] = v[:train_end]
            val_dict[k] = v[train_end:val_end]
            test_dict[k] = v[val_end:]

        assert (
            train_dict["timestamps"][-1] < val_dict["timestamps"][0]
        ), "Temporal leakage detected between train and val!"
        assert (
            val_dict["timestamps"][-1] < test_dict["timestamps"][0]
        ), "Temporal leakage detected between val and test!"

        return train_dict, val_dict, test_dict

    def fit_transform_scaler(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        test_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Standardize features strictly using statistics computed from training set only."""

        mean = np.mean(train_features, axis=(0, 1), keepdims=True)
        std = np.std(train_features, axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)

        train_norm = (train_features - mean) / std
        val_norm = (val_features - mean) / std
        test_norm = (test_features - mean) / std

        scaler_meta = {
            "mean": mean.squeeze().tolist(),
            "std": std.squeeze().tolist(),
            "feature_names": FEATURE_NAMES,
        }

        return train_norm, val_norm, test_norm, scaler_meta

class SpatialTemporalAirQualityDataset(Dataset):
    """Sliding-window PyTorch Dataset for multi-station spatial-temporal neural networks.

    Output per index:
    - x: Tensor (T_in, N_stations, F)
    - y: Tensor (T_out, N_stations, 2) -> (PM2.5, AQI) targets
    - timestamp: float
    - anomaly: int
    """

    def __init__(
        self,
        features: np.ndarray,
        pm25_targets: np.ndarray,
        aqi_targets: np.ndarray,
        timestamps: np.ndarray,
        anomaly_flags: np.ndarray,
        seq_len_in: int = 24,
        seq_len_out: int = 24,
    ):
        self.features = torch.from_numpy(features).float()
        self.pm25_targets = torch.from_numpy(pm25_targets).float()
        self.aqi_targets = torch.from_numpy(aqi_targets).float()
        self.timestamps = timestamps
        self.anomaly_flags = anomaly_flags

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        self.total_len = len(features)
        self.valid_samples = max(0, self.total_len - (seq_len_in + seq_len_out) + 1)

    def __len__(self) -> int:
        return self.valid_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_start = idx
        in_end = idx + self.seq_len_in
        out_start = in_end
        out_end = in_end + self.seq_len_out

        x = self.features[in_start:in_end]

        target_pm25 = self.pm25_targets[out_start:out_end]
        target_aqi = self.aqi_targets[out_start:out_end]
        y = torch.stack([target_pm25, target_aqi], dim=-1)

        return {
            "x": x,
            "y": y,
            "timestamp": torch.tensor(self.timestamps[in_end - 1], dtype=torch.float64),
            "anomaly_flag": torch.tensor(self.anomaly_flags[in_end - 1], dtype=torch.int32),
        }

def create_dataloaders(
    dataset_dict: Dict[str, np.ndarray],
    batch_size: int = 16,
    seq_len_in: int = 24,
    seq_len_out: int = 24,
    normalize: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Generate Train, Validation, and Test PyTorch DataLoaders with zero temporal leakage."""
    pipeline = AirQualityDataPipeline()
    train_dict, val_dict, test_dict = pipeline.chronological_split(
        dataset_dict, train_ratio, val_ratio, test_ratio
    )

    if normalize:
        train_feat, val_feat, test_feat, scaler_meta = pipeline.fit_transform_scaler(
            train_dict["features"], val_dict["features"], test_dict["features"]
        )
    else:
        train_feat = train_dict["features"]
        val_feat = val_dict["features"]
        test_feat = test_dict["features"]
        scaler_meta = {}

    train_ds = SpatialTemporalAirQualityDataset(
        features=train_feat,
        pm25_targets=train_dict["pm25_calib"],
        aqi_targets=train_dict["aqi"],
        timestamps=train_dict["timestamps"],
        anomaly_flags=train_dict["anomaly_flags"],
        seq_len_in=seq_len_in,
        seq_len_out=seq_len_out,
    )

    val_ds = SpatialTemporalAirQualityDataset(
        features=val_feat,
        pm25_targets=val_dict["pm25_calib"],
        aqi_targets=val_dict["aqi"],
        timestamps=val_dict["timestamps"],
        anomaly_flags=val_dict["anomaly_flags"],
        seq_len_in=seq_len_in,
        seq_len_out=seq_len_out,
    )

    test_ds = SpatialTemporalAirQualityDataset(
        features=test_feat,
        pm25_targets=test_dict["pm25_calib"],
        aqi_targets=test_dict["aqi"],
        timestamps=test_dict["timestamps"],
        anomaly_flags=test_dict["anomaly_flags"],
        seq_len_in=seq_len_in,
        seq_len_out=seq_len_out,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    metadata = {
        "num_stations": NUM_STATIONS,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "seq_len_in": seq_len_in,
        "seq_len_out": seq_len_out,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "scaler": scaler_meta,
    }

    return train_loader, val_loader, test_loader, metadata
