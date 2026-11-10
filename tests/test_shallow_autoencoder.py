import pytest
import numpy as np
import torch
from pi.anomaly_engine import MultiStationDeepAutoencoder, AnomalyEngine, AdaptiveAnomalyDetector

def test_shallow_autoencoder_architecture():
    model = MultiStationDeepAutoencoder(in_features=48, latent_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    # 48*32 + 32 + 32*16 + 16 + 16*32 + 32 + 32*48 + 48 = 4208
    assert n_params <= 5000, f"Expected < 5000 params, got {n_params}"

    x = torch.randn(2, 48)
    rec, z = model(x)
    assert rec.shape == (2, 48)
    assert z.shape == (2, 16)

def test_adaptive_anomaly_detector_mse():
    detector = AdaptiveAnomalyDetector(beta_ema=0.98, n_sigma=3.0)
    baseline_errs = np.random.normal(0.0, 0.5, size=(100, 48))
    detector.fit_baseline(baseline_errs)
    assert detector.initialized

    # Lỗi nhỏ -> Không bất thường
    small_err = np.full(48, 0.1, dtype=np.float32)
    score = detector.compute_score(small_err)
    is_anom, thresh, _ = detector.update_and_detect(score)
    assert not is_anom

    # Lỗi đột biến lớn -> Bất thường
    large_err = np.full(48, 5.0, dtype=np.float32)
    score_large = detector.compute_score(large_err)
    is_anom_large, thresh_large, _ = detector.update_and_detect(score_large)
    assert is_anom_large
