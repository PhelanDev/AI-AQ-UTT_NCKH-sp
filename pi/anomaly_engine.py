"""
DEEP AUTOENCODER ANOMALY DETECTION ENGINE
Mạng Nơ-ron Deep Autoencoder Phát hiện Dị thường Đa trạm & Phân loại Dấu vết Vật lý
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5 (8GB RAM)
Thuộc Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MultiStationDeepAutoencoder(nn.Module):
    """
    Mạng Nơ-ron Deep Autoencoder Đa trạm với nút thắt cổ chai đối xứng (Symmetric Bottleneck).
    Nén đa biến không gian - môi trường từ 8 trạm (D_in) xuống không gian ẩn z (16 chiều).
    Tổng số tham số siêu nhỏ gọn (< 15,000 params), chạy thời gian thực trên Raspberry Pi 5.
    """

    def __init__(self, in_features: int = 48, latent_dim: int = 16):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, in_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất biểu diễn ẩn trong không gian latent (16D)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Tái tạo dữ liệu từ vector tiềm ẩn."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            reconstructed: Tensor tái tạo cùng kích thước với x.
            latent: Tensor đặc trưng ẩn (B, 16).
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

class AdaptiveAnomalyDetector:
    """
    Bộ phát hiện dị thường thích ứng kết hợp khoảng cách Mahalanobis / MSE tái tạo
    và ngưỡng động trượt 3-Sigma: Theta(t) = mu_EMA + 3 * sigma_EMA.
    """

    def __init__(self, beta_ema: float = 0.98, n_sigma: float = 3.0):
        self.beta_ema = beta_ema
        self.n_sigma = n_sigma
        self.mu_ema = 0.0
        self.var_ema = 1.0
        self.initialized = False
        self.cov_inv: Optional[np.ndarray] = None

    def fit_baseline(self, normal_errors: np.ndarray):
        """Khởi tạo giá trị thống kê nền từ tập dữ liệu sạch."""
        errs = np.asarray(normal_errors, dtype=np.float64)
        if errs.ndim == 2:

            cov = np.cov(errs, rowvar=False)
            cov_reg = cov + 1e-5 * np.eye(cov.shape[0])
            try:
                self.cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                self.cov_inv = np.eye(cov.shape[0])
            scalar_scores = np.mean(errs ** 2, axis=1)
        else:
            scalar_scores = errs

        self.mu_ema = float(np.mean(scalar_scores))
        self.var_ema = float(np.var(scalar_scores)) + 1e-6
        self.initialized = True

    def compute_score(self, error_vector: np.ndarray) -> float:
        """Tính điểm bất thường (Reconstruction score)."""
        err = np.asarray(error_vector, dtype=np.float64).ravel()
        if self.cov_inv is not None and len(err) == self.cov_inv.shape[0]:

            d_sq = float(err @ self.cov_inv @ err)
            return float(math.sqrt(max(0.0, d_sq)))

        return float(np.mean(err ** 2))

    def update_and_detect(self, score: float) -> Tuple[bool, float, float]:
        """
        Cập nhật baseline EMA và phát hiện bất thường:
        Returns:
            is_anomaly: True nếu score > threshold.
            threshold: Giá trị ngưỡng động hiện tại Theta(t).
            score: Điểm dị thường hiện tại.
        """
        if not self.initialized:
            self.mu_ema = score
            self.var_ema = 1.0
            self.initialized = True

        sigma = math.sqrt(max(1e-8, self.var_ema))
        threshold = self.mu_ema + self.n_sigma * sigma
        is_anomaly = (score > threshold)

        if not is_anomaly:
            diff = score - self.mu_ema
            self.mu_ema = self.beta_ema * self.mu_ema + (1.0 - self.beta_ema) * score
            self.var_ema = self.beta_ema * self.var_ema + (1.0 - self.beta_ema) * (diff ** 2)

        return is_anomaly, threshold, score

class PhysicalSignatureClassifier:
    """
    Bộ phân loại dấu vết vật lý 3 lớp dị thường (kèm lớp 0: Clean/Normal):
    - Lớp 0: Sạch / Hoạt động bình thường (Clean Normal)
    - Lớp 1: Khói cháy / Đốt rác / Sự cố công nghiệp (Fire / Combustion)
             (Đa trạm >= 2 trạm xuôi gió, tăng đồng thời PM2.5, PM10, CO2, VOC)
    - Lớp 2: Trôi cảm biến / Bám bẩn buồng quang học (Sensor Drift / Degradation)
             (Cô lập 1 trạm duy nhất, kênh khí CO2/VOC bình thường, kéo dài > 1 ngày)
    - Lớp 3: Đột biến vi khí hậu cục bộ tức thời (Hyper-Local Micro Burst / Spike)
             (1 trạm đột biến trong thời gian ngắn < 15 phút, tự triệt tiêu nhanh)
    """

    LABEL_CLEAN = 0
    LABEL_FIRE = 1
    LABEL_DRIFT = 2
    LABEL_LOCAL_SPIKE = 3

    CLASS_NAMES = ["Clean", "Fire/Combustion", "Sensor Drift", "Local Spike"]

    @staticmethod
    def classify(
        stations_affected: int,
        gases_elevated: List[str],
        duration_minutes: float = 5.0,
        is_anomaly: bool = True,
    ) -> Tuple[int, str, List[float]]:
        """
        Phân loại dấu vết vật lý và tính xác suất tin cậy.
        Returns:
            class_id: Mã nhãn (0, 1, 2, 3)
            class_name: Tên loại dị thường
            probabilities: Mảng xác suất tin cậy 4 lớp [P0, P1, P2, P3] với tổng bằng 1.0.
        """
        if not is_anomaly or stations_affected == 0:
            return 0, "Clean", [0.94, 0.02, 0.02, 0.02]

        elevated_set = set(g.lower() for g in gases_elevated)
        has_gas = bool(elevated_set.intersection({"co2", "voc"}))
        has_pm = bool(elevated_set.intersection({"pm25", "pm10"}))

        if stations_affected >= 2 and (has_gas and has_pm):
            p1 = 0.88
            rem = (1.0 - p1) / 3.0
            return 1, "Fire/Combustion", [rem, p1, rem, rem]

        duration_days = duration_minutes / (60.0 * 24.0)
        if stations_affected == 1 and has_pm and (not has_gas) and duration_days >= 1.0:
            p2 = 0.85
            rem = (1.0 - p2) / 3.0
            return 2, "Sensor Drift", [rem, rem, p2, rem]

        if stations_affected == 1 and duration_minutes <= 15.0:
            p3 = 0.86
            rem = (1.0 - p3) / 3.0
            return 3, "Local Spike", [rem, rem, rem, p3]

        return 3, "Local Spike", [0.05, 0.25, 0.10, 0.60]

class AnomalyEngine:
    """
    Động cơ Phát hiện Dị thường Toàn diện trên Raspberry Pi 5.
    Tích hợp Deep Autoencoder, Adaptive Threshold và Physical Signature Classifier.
    """

    def __init__(self, in_features: int = 48, latent_dim: int = 16, weights_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model = MultiStationDeepAutoencoder(in_features=in_features, latent_dim=latent_dim).to(self.device)
        self.detector = AdaptiveAnomalyDetector(beta_ema=0.98, n_sigma=3.0)
        self.classifier = PhysicalSignatureClassifier()

        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def inspect(
        self,
        features_vector: np.ndarray,
        stations_affected: int = 0,
        gases_elevated: Optional[List[str]] = None,
        duration_minutes: float = 5.0,
    ) -> Dict[str, Union[bool, int, str, float, List[float]]]:
        """
        Kiểm tra và gắn nhãn dị thường cho một khung đo dữ liệu đa trạm.
        """
        x_np = np.asarray(features_vector, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, device=self.device)
            x_rec, z = self.model(x_tensor)
            err_vec = (x_tensor - x_rec).cpu().numpy().ravel()

        score = self.detector.compute_score(err_vec)
        is_anom, thresh, _ = self.detector.update_and_detect(score)

        gases = gases_elevated if gases_elevated is not None else []
        cid, cname, probs = self.classifier.classify(
            stations_affected=stations_affected,
            gases_elevated=gases,
            duration_minutes=duration_minutes,
            is_anomaly=is_anom,
        )

        return {
            "is_anomaly": is_anom,
            "reconstruction_score": float(score),
            "dynamic_threshold": float(thresh),
            "class_id": int(cid),
            "class_name": cname,
            "probabilities": [float(p) for p in probs],
            "latent_vector": z.cpu().numpy().ravel().tolist(),
        }
