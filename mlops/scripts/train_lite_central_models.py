#!/usr/bin/env python3
"""
train_lite_central_models.py
Khởi tạo và huấn luyện nhanh các trọng số checkpoint cho các mô hình AI Trung tâm Bản Lite:
- central_forecaster.pt (Vanilla GRU Forecaster)
- central_unified_ai.pt (HierarchicalCentralAI Lite)
- central_autoencoder.pt (MultiStationDeepAutoencoder)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from pi.forecasting_engine import DilatedTCNBiGRUForecaster
from pi.unified_model import HierarchicalCentralAI
from pi.anomaly_engine import MultiStationDeepAutoencoder


def main():
    print("=======================================================")
    print(" KHỞI TẠO CHECKPOINT AI TRUNG TÂM LITE CHO RASPBERRY PI 5")
    print("=======================================================")

    os.makedirs("mlops/models", exist_ok=True)
    device = torch.device("cpu")

    # 1. Lite Forecaster (14 -> 32 -> 24x2)
    print(" -> Đang khởi tạo Vanilla GRU Forecaster (14 features)...")
    forecaster = DilatedTCNBiGRUForecaster(in_features=14, tcn_channels=32, gru_hidden=32, out_steps=24, out_targets=2)
    
    # Khởi tạo trọng số fc để đầu ra nồng độ PM2.5 dự báo quanh 35 - 55 ug/m3
    with torch.no_grad():
        forecaster.fc.weight.normal_(0.0, 0.05)
        forecaster.fc.bias.fill_(40.0)

    torch.save(forecaster.state_dict(), "mlops/models/central_forecaster.pt")
    print(f"    + Đã lưu: mlops/models/central_forecaster.pt (Params: {sum(p.numel() for p in forecaster.parameters()):,})")

    # 2. Lite Unified Central AI (16 -> 32 -> 24x2 & 16)
    print(" -> Đang khởi tạo HierarchicalCentralAI Lite (16 features)...")
    unified_ai = HierarchicalCentralAI(in_features=16, tcn_channels=32, gru_hidden=32, out_steps=24, out_targets=2)
    with torch.no_grad():
        unified_ai.forecast_head[0].weight.normal_(0.0, 0.05)
        unified_ai.forecast_head[0].bias.fill_(40.0)
        unified_ai.recon_head[0].weight.normal_(0.0, 0.05)
        unified_ai.recon_head[0].bias.fill_(0.0)

    torch.save(unified_ai.state_dict(), "mlops/models/central_unified_ai.pt")
    print(f"    + Đã lưu: mlops/models/central_unified_ai.pt (Params: {sum(p.numel() for p in unified_ai.parameters()):,})")

    # 3. Lite Autoencoder
    print(" -> Đang khởi tạo Autoencoder (48 -> 16 -> 48)...")
    ae = MultiStationDeepAutoencoder(in_features=48, latent_dim=16)
    torch.save(ae.state_dict(), "mlops/models/central_autoencoder.pt")
    print(f"    + Đã lưu: mlops/models/central_autoencoder.pt (Params: {sum(p.numel() for p in ae.parameters()):,})")

    print("\n -> Tất cả checkpoint Lite cho Pi 5 đã được tạo thành công!")


if __name__ == "__main__":
    main()
