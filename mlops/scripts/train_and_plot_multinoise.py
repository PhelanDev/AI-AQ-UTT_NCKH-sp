#!/usr/bin/env python3
"""
train_and_plot_multinoise.py
Thực thi huấn luyện toàn diện mô hình Multi-Sensor ResMLP trên môi trường đa nhiễu
và tự động kết xuất sơ đồ - biểu đồ khoa học 300 DPI đánh giá nghiệm thu.
"""

import os
import sys
import csv
import json
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

torch.manual_seed(100)
np.random.seed(100)

from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    compute_dew_point,
    generate_dry_air_boundary_anchors,
    load_dataset_from_csv,
)


def run_training_with_logging(
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 3e-3,
    train_csv: str = "mlops/data/multisensor_training_set.csv",
    test_csv: str = "mlops/data/multisensor_test_set.csv",
    scaler_path: str = "mlops/models/multisensor_scaler.json",
    save_pt: str = "mlops/models/multisensor_resmlp.pt",
    save_params: str = "mlops/models/multisensor_resmlp_params.json",
):
    print("=" * 88)
    print(" 1. BẮT ĐẦU HUẤN LUYỆN MULTI-SENSOR RESMLP (200 EPOCHS VỚI MULTI-NOISE)")
    print("=" * 88)

    X_train_raw, PM_train, Ref_train, RH_train = load_dataset_from_csv(train_csv)
    X_test_raw, PM_test, Ref_test, RH_test = load_dataset_from_csv(test_csv)
    print(f" -> Nạp dữ liệu: {len(X_train_raw):,} Train samples | {len(X_test_raw):,} Out-of-Time Test samples")

    X_anc, PM_anc, Ref_anc, RH_anc = generate_dry_air_boundary_anchors(n_samples=600, seed=100)
    X_train_raw = np.concatenate([X_train_raw, X_anc], axis=0)
    PM_train = np.concatenate([PM_train, PM_anc], axis=0)
    Ref_train = np.concatenate([Ref_train, Ref_anc], axis=0)
    RH_train = np.concatenate([RH_train, RH_anc], axis=0)
    CO2_train = X_train_raw[:, 6].copy()

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    x_min = np.array(scaler["x_min"], dtype=np.float32)
    x_max = np.array(scaler["x_max"], dtype=np.float32)
    denom = np.maximum(x_max - x_min, 1e-6)

    X_train = np.clip((X_train_raw - x_min) / denom, 0.0, 1.0)
    X_test = np.clip((X_test_raw - x_min) / denom, 0.0, 1.0)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(PM_train, dtype=torch.float32),
        torch.tensor(Ref_train, dtype=torch.float32),
        torch.tensor(RH_train, dtype=torch.float32),
        torch.tensor(CO2_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(PM_test, dtype=torch.float32),
        torch.tensor(Ref_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    criterion = nn.HuberLoss(delta=4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_mae": [],
        "test_rmse": [],
        "test_r2": [],
    }

    ss_tot = float(np.sum((Ref_test - np.mean(Ref_test)) ** 2))

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, bpm, bref, brh, bco2 in train_loader:
            noise = torch.randn_like(bx) * 0.005
            bx_noisy = torch.clamp(bx + noise, 0.0, 1.0)

            c_factor = model(bx_noisy).squeeze(-1)
            pred_pm = bpm * c_factor
            loss = criterion(pred_pm, bref)

            dry_mask = (brh < 55.0) & (bco2 < 750.0)
            if torch.any(dry_mask):
                dry_penalty = torch.mean((c_factor[dry_mask] - 1.0) ** 2)
                loss = loss + 0.8 * dry_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Đánh giá sau mỗi epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bpm, bref in test_loader:
                c_factor = model(bx).squeeze(-1)
                pred_pm = bpm * c_factor
                val_loss += criterion(pred_pm, bref).item()
            val_loss /= len(test_loader)

            c_test = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()
            calib_pm = PM_test * c_test
            mae = float(np.mean(np.abs(calib_pm - Ref_test)))
            rmse = float(np.sqrt(np.mean((calib_pm - Ref_test) ** 2)))
            ss_res = float(np.sum((calib_pm - Ref_test) ** 2))
            r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(val_loss)
        history["test_mae"].append(mae)
        history["test_rmse"].append(rmse)
        history["test_r2"].append(r2)

        if ep % 20 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch [{ep:3d}/{epochs:3d}] | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f} | Test MAE: {mae:.2f} µg/m³ | R²: {r2:.4f}")

    os.makedirs(os.path.dirname(save_pt), exist_ok=True)
    torch.save(model.state_dict(), save_pt)
    print(f" -> Đã lưu trọng số PyTorch vào: {save_pt}")

    # Tính toán chỉ số tổng kết và lưu multisensor_resmlp_params.json
    model.eval()
    with torch.no_grad():
        c_final = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()
    calib_final = PM_test * c_final

    mae_raw = float(np.mean(np.abs(PM_test - Ref_test)))
    rmse_raw = float(np.sqrt(np.mean((PM_test - Ref_test) ** 2)))
    r2_raw = float(1.0 - np.sum((PM_test - Ref_test) ** 2) / max(ss_tot, 1e-6))

    mae_cal = float(np.mean(np.abs(calib_final - Ref_test)))
    rmse_cal = float(np.sqrt(np.mean((calib_final - Ref_test) ** 2)))
    r2_cal = float(1.0 - np.sum((calib_final - Ref_test) ** 2) / max(ss_tot, 1e-6))

    mask_wet = RH_test > 75.0
    if np.sum(mask_wet) > 0:
        raw_h = PM_test[mask_wet]
        calib_h = calib_final[mask_wet]
        ref_h = Ref_test[mask_wet]
        ss_tot_h = max(float(np.sum((ref_h - np.mean(ref_h)) ** 2)), 1e-6)
        mae_h_raw = float(np.mean(np.abs(raw_h - ref_h)))
        rmse_h_raw = float(np.sqrt(np.mean((raw_h - ref_h) ** 2)))
        r2_h_raw = float(1.0 - np.sum((raw_h - ref_h) ** 2) / ss_tot_h)
        mae_h_calib = float(np.mean(np.abs(calib_h - ref_h)))
        rmse_h_calib = float(np.sqrt(np.mean((calib_h - ref_h) ** 2)))
        r2_h_calib = float(1.0 - np.sum((calib_h - ref_h) ** 2) / ss_tot_h)
    else:
        mae_h_raw, rmse_h_raw, r2_h_raw = mae_raw, rmse_raw, r2_raw
        mae_h_calib, rmse_h_calib, r2_h_calib = mae_cal, rmse_cal, r2_cal

    params = {
        "architecture": "MultiSensorResMLP_MultiNoise",
        "in_features": 8,
        "hidden_dim": 96,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "validation_strategy": "Chronological_Split_48h_Buffer_Out_Of_Time",
        "noise_layers": [
            "Hysteresis_Kohler_Efflorescence_Deliquescence",
            "Aerosol_Composition_Drift",
            "Dual_Channel_Optical_Mismatch_Spikes",
            "Thermal_Self_Heating_Board_Microclimate",
            "Online_Batch_Data_Jittering"
        ],
        "metrics_overall_test": {"r2": r2_cal, "mae": mae_cal, "rmse": rmse_cal},
        "metrics_wet_season_test": {"r2": r2_h_calib, "mae": mae_h_calib, "rmse": rmse_h_calib},
        "metrics_raw_baseline": {"r2": r2_raw, "mae": mae_raw, "rmse": rmse_raw},
        "x_min": x_min.tolist(),
        "x_max": x_max.tolist(),
    }
    with open(save_params, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f" -> Đã lưu cấu hình siêu tham số vào: {save_params}")

    return model, history, X_test, PM_test, Ref_test, RH_test, x_min, x_max


def plot_comprehensive_evaluation(
    model, history, X_test, PM_test, Ref_test, RH_test, x_min, x_max,
    save_path: str = "docs/figures/fig_multinoise_training_evaluation.png"
):
    print("\n" + "=" * 88)
    print(" 2. KẾT XUẤT BIỂU ĐỒ & SƠ ĐỒ ĐÁNH GIÁ NGHIỆM THU ĐA NHIỄU (300 DPI)")
    print("=" * 88)

    denom = np.maximum(x_max - x_min, 1e-6)

    # Đánh giá suy luận trên tập test
    model.eval()
    with torch.no_grad():
        c_test = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()
    calib_pm = PM_test * c_test

    # Chỉ số toàn thể
    mae_raw = np.mean(np.abs(PM_test - Ref_test))
    rmse_raw = np.sqrt(np.mean((PM_test - Ref_test) ** 2))
    r2_raw = 1.0 - np.sum((PM_test - Ref_test) ** 2) / np.sum((Ref_test - np.mean(Ref_test)) ** 2)

    mae_cal = np.mean(np.abs(calib_pm - Ref_test))
    rmse_cal = np.sqrt(np.mean((calib_pm - Ref_test) ** 2))
    r2_cal = 1.0 - np.sum((calib_pm - Ref_test) ** 2) / np.sum((Ref_test - np.mean(Ref_test)) ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)

    # -------------------------------------------------------------
    # (a) LEARNING CURVES & ERROR METRICS
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    epochs = history["epoch"]
    l1 = ax1.plot(epochs, history["train_loss"], color="#1E40AF", lw=2.2, label="Train Huber Loss")
    l2 = ax1.plot(epochs, history["test_loss"], color="#DC2626", lw=2.0, ls="--", label="Test Huber Loss (Unseen)")

    l3 = ax1_twin.plot(epochs, history["test_mae"], color="#059669", lw=1.8, ls="-.", label="Test MAE (µg/m³)")
    l4 = ax1_twin.plot(epochs, history["test_rmse"], color="#D97706", lw=1.8, ls=":", label="Test RMSE (µg/m³)")

    ax1.set_xlabel("Epochs", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Huber Loss (δ = 4.0)", fontsize=10.5, fontweight="bold", color="#1E40AF")
    ax1_twin.set_ylabel("Error Metric (µg/m³)", fontsize=10.5, fontweight="bold", color="#059669")

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.92)
    ax1.set_title(f"(a) Quá Trình Huấn Luyện & Kiểm Thử Kháng Nhiễu ({len(history['epoch'])} Epochs)", fontsize=11.5, fontweight="bold", color="#0F172A", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # (b) SCATTER PLOT: RAW SENSOR VS REFERENCE
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    mask_wet = RH_test > 75.0

    ax2.scatter(Ref_test[~mask_wet], PM_test[~mask_wet], c="#64748B", alpha=0.35, s=12, label="Khô / Bình thường (RH ≤ 75%)")
    ax2.scatter(Ref_test[mask_wet], PM_test[mask_wet], c="#EF4444", alpha=0.55, s=16, label="Nồm ẩm cao (RH > 75% phồng sương)")

    max_val = min(160.0, max(np.max(Ref_test), np.max(PM_test)))
    ax2.plot([0, max_val], [0, max_val], color="#0F172A", lw=1.5, ls="--", label="Đường lý tưởng 1:1 (y = x)")

    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.set_xlabel("Chuẩn Tham Chiếu BAM-1020 FEM (µg/m³)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Cảm Biến Thô PMS7003 (µg/m³)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Tương Quan Cảm Biến Thô (Chưa Bù)", fontsize=11.5, fontweight="bold", color="#0F172A", pad=8)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax2.text(0.52, 0.10, f"R² = {r2_raw:.4f}\nMAE = {mae_raw:.2f} µg/m³\nRMSE = {rmse_raw:.2f} µg/m³",
             transform=ax2.transAxes, fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor="#DC2626", alpha=0.9))
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # (c) SCATTER PLOT: CALIBRATED VS REFERENCE
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.scatter(Ref_test[~mask_wet], calib_pm[~mask_wet], c="#64748B", alpha=0.35, s=12, label="Khô / Bình thường (RH ≤ 75%)")
    ax3.scatter(Ref_test[mask_wet], calib_pm[mask_wet], c="#10B981", alpha=0.55, s=16, label="Đã bù nồm ẩm (RH > 75%)")
    ax3.plot([0, max_val], [0, max_val], color="#0F172A", lw=1.5, ls="--", label="Đường lý tưởng 1:1 (y = x)")

    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.set_xlabel("Chuẩn Tham Chiếu BAM-1020 FEM (µg/m³)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Multi-Sensor ResMLP Đã Bù (µg/m³)", fontsize=10.5, fontweight="bold")
    ax3.set_title("(c) Tương Quan Sau Hiệu Chuẩn Đa Nhiễu (Đề Tài)", fontsize=11.5, fontweight="bold", color="#0F172A", pad=8)
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    delta_r2 = r2_cal - r2_raw
    pct_mae = (mae_cal - mae_raw) / max(mae_raw, 1e-6) * 100.0
    pct_rmse = (rmse_cal - rmse_raw) / max(rmse_raw, 1e-6) * 100.0
    ax3.text(0.52, 0.10, f"R² = {r2_cal:.4f} ({delta_r2:+.3f})\nMAE = {mae_cal:.2f} µg/m³ ({pct_mae:.1f}%)\nRMSE = {rmse_cal:.2f} µg/m³ ({pct_rmse:.1f}%)",
             transform=ax3.transAxes, fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.9))
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # (d) HUMIDITY RESPONSE & SMOKE DISCRIMINATION CURVE
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    rh_sweep = np.linspace(25.0, 98.0, 100)

    # 1. Quét đường sương mù nồm ẩm sạch (CO2=420, Gas=100)
    t_val = 22.0
    c_clean_curve = []
    for rh_val in rh_sweep:
        dp_val = compute_dew_point(np.array([t_val]), np.array([rh_val]))[0]
        feat = np.array([100.0, 2.0, 0.7, t_val, rh_val, dp_val, 420.0, 100.0], dtype=np.float32)
        norm = np.clip((feat - x_min) / denom, 0.0, 1.0)
        with torch.no_grad():
            c = float(model(torch.tensor(norm).unsqueeze(0)).squeeze(-1).item())
        c_clean_curve.append(c)

    # 2. Quét đường khói đốt / xe cộ thực tế (CO2=1400, Gas=380)
    c_smoke_curve = []
    for rh_val in rh_sweep:
        dp_val = compute_dew_point(np.array([t_val]), np.array([rh_val]))[0]
        feat = np.array([100.0, 2.0, 0.7, t_val, rh_val, dp_val, 1400.0, 380.0], dtype=np.float32)
        norm = np.clip((feat - x_min) / denom, 0.0, 1.0)
        with torch.no_grad():
            c = float(model(torch.tensor(norm).unsqueeze(0)).squeeze(-1).item())
        c_smoke_curve.append(c)

    ax4.plot(rh_sweep, c_clean_curve, color="#0284C7", lw=2.4, label="Sương mù nồm ẩm sạch (CO₂=420, Gas=100)")
    ax4.plot(rh_sweep, c_smoke_curve, color="#EA580C", lw=2.4, ls="--", label="Khói xe / Đốt rác thực (CO₂=1400, Gas=380)")

    ax4.axhline(1.0, color="#64748B", ls=":", lw=1.2, label="Không can thiệp (C = 1.0)")
    ax4.axvline(75.0, color="#EF4444", ls=":", lw=1.2, label="Ngưỡng bắt đầu nồm ẩm (RH = 75%)")

    ax4.set_xlabel("Độ Ẩm Tương Đối RH (%)", fontsize=10.5, fontweight="bold")
    ax4.set_ylabel("Hệ Số Hiệu Chuẩn Bounded Multiplier C", fontsize=10.5, fontweight="bold")
    ax4.set_ylim(0.2, 1.1)
    ax4.set_title("(d) Đường Đáp Ứng Thích Ứng & Phân Biệt Khói Thực", fontsize=11.5, fontweight="bold", color="#0F172A", pad=8)
    ax4.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Lưu file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" -> Đã lưu biểu đồ khoa học 300 DPI vào: {save_path}")

    # Đồng bộ sang brain artifact nếu có
    alt_paths = [
        r"C:\Users\LEGION\.gemini\antigravity\brain\bc457dc3-55b0-4bf5-9a8f-81d9bdbf6728\fig_multinoise_training_evaluation.png"
    ]
    for p in alt_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(save_path, p)
            print(f" -> Đã đồng bộ sang: {p}")
        except Exception as e:
            print(f" -> Lỗi sao chép sang {p}: {e}")

    plt.close(fig)


if __name__ == "__main__":
    model, history, X_test, PM_test, Ref_test, RH_test, x_min, x_max = run_training_with_logging()
    plot_comprehensive_evaluation(model, history, X_test, PM_test, Ref_test, RH_test, x_min, x_max)
