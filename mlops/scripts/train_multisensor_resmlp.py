"""
Huấn Luyện Mô Hình Multi-Sensor ResMLP 19.5k Tham Số với Thuật Toán Dẫn Hướng Vật Lý
(Physics-Informed Gated ResMLP với Ràng Buộc Đơn Điệu & Neo Khói Thực Nghiệm)
Dành cho Vi Điều Khiển ESP32-S3 (Bù nồm ẩm RH > 75% & Phân biệt khói ô nhiễm)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import csv
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Thiết lập seed tái lập
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MultiSensorResMLP(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 96):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.15)

        self.block1 = nn.Linear(hidden_dim, hidden_dim)
        self.act_b1 = nn.ReLU()

        self.block2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_b2 = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8)
        h0 = self.drop1(self.act1(self.fc_in(x)))
        # Res block 1
        h1 = self.act_b1(self.block1(h0)) + h0
        # Res block 2
        h2 = self.act_b2(self.block2(h1)) + h1
        # Bounded physics multiplier C in [0.25, 1.05]
        raw_out = self.fc_out(h2)
        c = 0.25 + 0.80 * self.sigmoid(raw_out)
        return c

def load_dataset(csv_path: str = "mlops/data/multisensor_training_set.csv"):
    X, Y, PM_raws = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pm_avg = float(r["pm_avg"])
            diff_pm = float(r["diff_pm"])
            r_pm = float(r["r_pm"])
            t = float(r["temp"])
            rh = float(r["rh"])
            dp = float(r["dew_point"])
            co2 = float(r["co2"])
            gas = float(r["gas_proxy"])
            ref_pm = float(r["ref_pm25"])

            # Tinh target multiplier C = ref_pm / pm_avg
            c_target = min(1.05, max(0.25, ref_pm / max(pm_avg, 1.0)))

            X.append([pm_avg, diff_pm, r_pm, t, rh, dp, co2, gas])
            Y.append([c_target])
            PM_raws.append(pm_avg)

    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)
    PM_raws_np = np.array(PM_raws, dtype=np.float32)
    return X_np, Y_np, PM_raws_np

def predict_calibrated_pm25(
    model: MultiSensorResMLP,
    pm_avg: float,
    diff_pm: float,
    r_pm: float,
    temp: float,
    rh: float,
    dew_point: float,
    co2: float,
    gas_proxy: float,
    scaler_path: str = "mlops/models/multisensor_scaler.json",
):
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    x_min = np.array(scaler["x_min"], dtype=np.float32)
    x_max = np.array(scaler["x_max"], dtype=np.float32)

    raw_feat = np.array([pm_avg, diff_pm, r_pm, temp, rh, dew_point, co2, gas_proxy], dtype=np.float32)
    norm_feat = (raw_feat - x_min) / np.maximum(x_max - x_min, 1e-6)

    tensor_in = torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        c_pred = float(model(tensor_in).item())

    calibrated_pm25 = round(pm_avg * c_pred, 2)
    return round(c_pred, 3), calibrated_pm25

def train_and_evaluate(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-3,
    save_pt: str = "mlops/models/multisensor_resmlp.pt",
    save_params: str = "mlops/models/multisensor_resmlp_params.json",
):
    print("===============================================================================")
    print("    HUẤN LUYỆN MULTI-SENSOR RESMLP (PIR-MLP VỚI THUẬT TOÁN DẪN HƯỚNG VẬT LÝ)")
    print("===============================================================================")
    X_raw, Y_raw, PM_raws = load_dataset()
    n_samples = len(X_raw)
    print(f"Tổng số mẫu nạp: {n_samples:,}")

    # Min-max normalization
    x_min = np.min(X_raw, axis=0)
    x_max = np.max(X_raw, axis=0)
    X_norm = (X_raw - x_min) / np.maximum(x_max - x_min, 1e-6)

    # Train / Test split 80/20
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(0.80 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, Y_train = X_norm[train_idx], Y_raw[train_idx]
    X_test, Y_test = X_norm[test_idx], Y_raw[test_idx]
    PM_test = PM_raws[test_idx]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Kiến trúc Physics-Informed ResMLP: 8 -> 96 -> 96 -> 96 -> 1 ({total_params:,} tham số)")

    criterion_base = nn.HuberLoss(delta=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n=== TIẾN TRÌNH HUẤN LUYỆN (PHYSICS-CONSTRAINED LOSS) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            # Data Jittering data augmentation (+- 2%)
            noise = torch.randn_like(bx) * 0.02
            bx_noisy = torch.clamp(bx + noise, 0.0, 1.0)

            optimizer.zero_grad()
            out = model(bx_noisy)

            # 1. Base Huber Loss
            loss_base = criterion_base(out, by)

            # 2. Physics Constraint 1: Monotonicity in clean high humidity (RH > 0.70 & low gas)
            # RH là index 4, CO2 index 6, Gas index 7
            clean_humid_mask = (bx_noisy[:, 4] > 0.70) & (bx_noisy[:, 6] < 0.40) & (bx_noisy[:, 7] < 0.35)
            if clean_humid_mask.any():
                # Phạt nếu hệ số C không giảm để bù phồng nước (C > 0.75)
                loss_humidity_over = torch.mean(torch.relu(out[clean_humid_mask] - 0.75))
            else:
                loss_humidity_over = torch.tensor(0.0)

            # 3. Physics Constraint 2: Smoke consistency (CO2 > 0.50 or Gas > 0.40)
            smoke_mask = (bx_noisy[:, 6] > 0.50) | (bx_noisy[:, 7] > 0.40)
            if smoke_mask.any():
                # Phạt nếu C bị gọt quá thấp khi có khói ô nhiễm thật (C < 0.80)
                loss_smoke_under = torch.mean(torch.relu(0.80 - out[smoke_mask]))
            else:
                loss_smoke_under = torch.tensor(0.0)

            # Tổng hợp Physics-Informed Loss
            total_loss = loss_base + 0.15 * loss_humidity_over + 0.15 * loss_smoke_under
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in test_loader:
                    out = model(bx)
                    val_loss += criterion_base(out, by).item()
            val_loss /= len(test_loader)
            print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Đánh giá độ chính xác trên tập kiểm thử độc lập
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test)).numpy().flatten()
        test_targets = Y_test.flatten()

    calib_pm_pred = PM_test * test_preds
    calib_pm_ref = PM_test * test_targets

    # Metrics
    mae_all = np.mean(np.abs(calib_pm_pred - calib_pm_ref))
    rmse_all = np.sqrt(np.mean((calib_pm_pred - calib_pm_ref) ** 2))
    ss_res = np.sum((calib_pm_pred - calib_pm_ref) ** 2)
    ss_tot = np.sum((calib_pm_ref - np.mean(calib_pm_ref)) ** 2)
    r2_all = 1.0 - (ss_res / max(ss_tot, 1e-6))

    # Đánh giá riêng phân khúc nồm ẩm (RH > 75%)
    rh_unnorm = X_raw[test_idx, 4]
    humid_mask = rh_unnorm > 75.0
    if np.sum(humid_mask) > 0:
        pred_humid = calib_pm_pred[humid_mask]
        ref_humid = calib_pm_ref[humid_mask]
        mae_humid = np.mean(np.abs(pred_humid - ref_humid))
        rmse_humid = np.sqrt(np.mean((pred_humid - ref_humid) ** 2))
        r2_humid = 1.0 - (np.sum((pred_humid - ref_humid) ** 2) / max(np.sum((ref_humid - np.mean(ref_humid)) ** 2), 1e-6))
    else:
        mae_humid, rmse_humid, r2_humid = mae_all, rmse_all, r2_all

    print(f"\n=== KẾT QUẢ NGHIỆM THU ĐỘC LẬP (TEST SET: {len(test_idx):,} MẪU) ===")
    print(f"Toàn bộ dải khí hậu : R² = {r2_all:.4f} | MAE = {mae_all:.2f} µg/m³ | RMSE = {rmse_all:.2f} µg/m³")
    print(f"Phân khúc nồm ẩm >75%: R² = {r2_humid:.4f} | MAE = {mae_humid:.2f} µg/m³ | RMSE = {rmse_humid:.2f} µg/m³")

    # Lưu mô hình PyTorch
    os.makedirs(os.path.dirname(save_pt), exist_ok=True)
    torch.save(model.state_dict(), save_pt)
    print(f"Đã lưu trọng số PyTorch vào {save_pt}")

    # Lưu siêu tham số
    params = {
        "architecture": "MultiSensorResMLP",
        "in_features": 8,
        "hidden_dim": 96,
        "num_params": total_params,
        "metrics_all": {"r2": float(r2_all), "mae": float(mae_all), "rmse": float(rmse_all)},
        "metrics_humid": {"r2": float(r2_humid), "mae": float(mae_humid), "rmse": float(rmse_humid)},
        "x_min": x_min.tolist(),
        "x_max": x_max.tolist(),
    }
    with open(save_params, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Đã lưu siêu tham số vào {save_params}")

    return model

if __name__ == "__main__":
    train_and_evaluate()
