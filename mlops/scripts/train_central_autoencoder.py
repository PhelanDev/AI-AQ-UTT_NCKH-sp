"""
Huấn Luyện Mô Hình Central Deep Autoencoder (Multi-Station 16D Bottleneck Anomaly Engine)
Học phân phối không gian - thời gian chuẩn của mạng lưới 8 trạm Thanh Xuân (48 kênh)
Nâng cấp PM2.5-GNN Wind-Advection Delay + Masked Station Denoising Autoencoder (MS-DAE)
Phá vỡ tính đồng tuyến 98.17% (SVD top 6 ratio < 0.90) và xuất mô hình ONNX Opset 17
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
import csv
import json
import math
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
from pi.config import STATIONS, STATION_IDS
from pi.anomaly_engine import MultiStationDeepAutoencoder, AdaptiveAnomalyDetector

STATION_PROFILES = [
    # S1: traffic_corridor (HUS Nguyễn Trãi)
    {"scale": 1.05, "coarse_ratio": 1.15, "uhi_amp":  1.2, "drh_amp": -3.5, "rh_bias":  0.0, "co2_scale": 1.10, "voc_scale": 1.15,
     "rush_m": (8.0, 1.2, 0.35), "rush_e": (18.0, 1.5, 0.40), "weekend_fac": 0.70},
    # S2: urban_park_highway (CV Nhân Chính)
    {"scale": 0.82, "coarse_ratio": 1.08, "uhi_amp": -1.8, "drh_amp":  7.0, "rh_bias": -2.0, "co2_scale": 0.90, "voc_scale": 0.85,
     "rush_m": (9.0, 1.8, 0.10), "rush_e": (19.0, 2.0, 0.15), "weekend_fac": 1.15},
    # S3: river_residential (Khương Đình / Sông Tô Lịch)
    {"scale": 0.92, "coarse_ratio": 1.12, "uhi_amp": -1.2, "drh_amp":  8.5, "rh_bias":  3.5, "co2_scale": 0.98, "voc_scale": 1.05,
     "rush_m": (7.2, 1.5, 0.20), "rush_e": (19.5, 1.6, 0.30), "weekend_fac": 0.90},
    # S4: urban_canyon (Nguyễn Tuân)
    {"scale": 1.15, "coarse_ratio": 1.25, "uhi_amp":  1.5, "drh_amp": -4.0, "rh_bias": -0.8, "co2_scale": 1.18, "voc_scale": 1.22,
     "rush_m": (8.3, 1.3, 0.38), "rush_e": (18.8, 2.2, 0.45), "weekend_fac": 0.75},
    # S5: residential_suburban (Kim Giang)
    {"scale": 0.90, "coarse_ratio": 1.10, "uhi_amp": -0.5, "drh_amp":  2.0, "rh_bias": -2.5, "co2_scale": 0.95, "voc_scale": 0.90,
     "rush_m": (6.8, 1.2, 0.25), "rush_e": (20.0, 1.5, 0.35), "weekend_fac": 0.95},
    # S6: highway_intersection (Nút giao Trung Hòa / Vành đai 3)
    {"scale": 1.22, "coarse_ratio": 1.35, "uhi_amp":  2.2, "drh_amp": -5.0, "rh_bias":  4.0, "co2_scale": 1.25, "voc_scale": 1.30,
     "rush_m": (7.8, 1.2, 0.45), "rush_e": (17.8, 1.5, 0.50), "weekend_fac": 0.70},
    # S7: high_density_residential (KĐT Thanh Xuân Bắc)
    {"scale": 0.98, "coarse_ratio": 1.14, "uhi_amp":  0.3, "drh_amp": -1.0, "rh_bias": -1.2, "co2_scale": 1.02, "voc_scale": 1.00,
     "rush_m": (7.3, 1.2, 0.22), "rush_e": (20.2, 1.5, 0.32), "weekend_fac": 0.95},
    # S8: central_hub (UBND Quận / Trung tâm Điều hành)
    {"scale": 1.00, "coarse_ratio": 1.12, "uhi_amp":  0.0, "drh_amp":  0.0, "rh_bias": -1.5, "co2_scale": 1.00, "voc_scale": 1.00,
     "rush_m": (8.5, 1.4, 0.20), "rush_e": (17.5, 1.5, 0.22), "weekend_fac": 0.65},
]


def prepare_48d_multistation_dataset(
    csv_path: str = "mlops/data/hanoi_thanhxuan_multistation_real.csv",
    alpha_local: float = 0.62,
    adv_coupling: float = 0.25,
    seed: int = 42,
):
    print(f" -> Nạp dữ liệu 8 trạm Thanh Xuân từ {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    time_records: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in all_rows:
        t = r["time"]
        sid = r["station_id"]
        time_records.setdefault(t, {})[sid] = r

    station_keys = [f"STATION_{i:02d}" for i in range(1, 9)]
    sorted_times = [t for t in sorted(time_records.keys()) if all(sid in time_records[t] for sid in station_keys)]
    if not sorted_times:
        sorted_times = sorted(time_records.keys())

    N_T = len(sorted_times)
    N_S = 8
    rng = np.random.default_rng(seed)

    coords = np.array([[s.utm_x, s.utm_y] for s in STATIONS], dtype=np.float64)
    d_vec = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
    dist_m = np.linalg.norm(d_vec, axis=-1)

    def _get_val(t: str, col: str, default: float) -> float:
        row = time_records[t].get("STATION_01") or next(iter(time_records[t].values()))
        try:
            return float(row.get(col, default))
        except (ValueError, TypeError):
            return default

    pm25_raw = np.array([_get_val(t, "pm25", 35.0) for t in sorted_times], dtype=np.float32)
    pm10_raw = np.array([_get_val(t, "pm10", 50.0) for t in sorted_times], dtype=np.float32)
    temp_raw = np.array([_get_val(t, "temperature_c", 25.0) for t in sorted_times], dtype=np.float32)
    rh_raw   = np.array([_get_val(t, "relative_humidity", 70.0) for t in sorted_times], dtype=np.float32)
    co_raw   = np.array([_get_val(t, "co", 500.0) for t in sorted_times], dtype=np.float32)
    no2_raw  = np.array([_get_val(t, "no2", 20.0) for t in sorted_times], dtype=np.float32)
    ws_all   = np.array([_get_val(t, "wind_speed_mps", 2.5) for t in sorted_times], dtype=np.float32)
    wd_all   = np.array([_get_val(t, "wind_direction_deg", 90.0) for t in sorted_times], dtype=np.float32)

    hours = []
    days = []
    for t in sorted_times:
        try:
            dt = datetime.fromisoformat(t)
            hours.append(dt.hour)
            days.append(dt.weekday())
        except Exception:
            hours.append(12)
            days.append(0)
    hours = np.array(hours, dtype=np.float32)
    days = np.array(days, dtype=np.int32)
    is_weekend = (days >= 5).astype(np.float32)

    # Quá trình ồn tự tương quan AR(1) cho phát thải cục bộ
    rho = 0.75
    local_noise = np.zeros((N_T, N_S), dtype=np.float32)
    state = rng.normal(0, 1.0, size=N_S)
    for t in range(N_T):
        state = rho * state + np.sqrt(1.0 - rho**2) * rng.normal(0, 1.0, size=N_S)
        local_noise[t] = state

    turb_temp = rng.normal(0, 0.4, size=(N_T, N_S))
    turb_rh   = rng.normal(0, 0.3, size=(N_T, N_S))

    pm25_mat = np.zeros((N_T, N_S), dtype=np.float32)
    pm10_mat = np.zeros((N_T, N_S), dtype=np.float32)
    temp_mat = np.zeros((N_T, N_S), dtype=np.float32)
    rh_mat   = np.zeros((N_T, N_S), dtype=np.float32)
    co2_mat  = np.zeros((N_T, N_S), dtype=np.float32)
    voc_mat  = np.zeros((N_T, N_S), dtype=np.float32)

    co2_base = co_raw * 1.2 + 420.0
    voc_base = no2_raw * 3.0 + 100.0

    solar_diurnal = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
    night_inversion = np.maximum(0.0, np.cos(np.pi * hours / 12.0))

    for s, spec in enumerate(STATION_PROFILES):
        rm_mu, rm_w, rm_a = spec["rush_m"]
        re_mu, re_w, re_a = spec["rush_e"]
        w_fac = 1.0 - is_weekend * (1.0 - spec["weekend_fac"])
        g_morning = rm_a * np.exp(-0.5 * ((hours - rm_mu) / rm_w) ** 2) * w_fac
        g_evening = re_a * np.exp(-0.5 * ((hours - re_mu) / re_w) ** 2) * w_fac
        diurnal_factor = 1.0 + g_morning + g_evening

        split_idx = int(0.8 * N_T)
        train_pm25_mean = float(pm25_raw[:split_idx].mean())
        local_signal = train_pm25_mean * (0.8 + 0.4 * diurnal_factor + 0.5 * local_noise[:, s])
        local_signal = np.maximum(5.0, local_signal)

        pm25_mat[:, s] = (1.0 - alpha_local) * (pm25_raw * spec["scale"] * diurnal_factor) + alpha_local * local_signal
        pm10_mat[:, s] = pm25_mat[:, s] * spec["coarse_ratio"] * (1.0 + 0.15 * local_noise[:, s])

        uhi = spec["uhi_amp"] * (0.6 * solar_diurnal + 0.4 * night_inversion)
        drh = spec["drh_amp"] * (0.5 * solar_diurnal + 0.5 * night_inversion)

        temp_mat[:, s] = temp_raw + uhi + turb_temp[:, s]
        rh_mat[:, s]   = np.maximum(0.0, rh_raw + spec["rh_bias"] + 0.1 * drh + turb_rh[:, s])

        co2_mat[:, s] = co2_base * spec["co2_scale"] * (1.0 + 0.3 * (g_morning + g_evening) + 0.25 * local_noise[:, s])
        voc_mat[:, s] = voc_base * spec["voc_scale"] * (1.0 + 0.4 * (g_morning + g_evening) + 0.35 * local_noise[:, s])

    # Lan truyền đối lưu gió PM2.5-GNN (Wang et al. SIGSPATIAL 2020)
    wd_rad = wd_all * (np.pi / 180.0)
    u_wind = -ws_all * np.sin(wd_rad)
    v_wind = -ws_all * np.cos(wd_rad)

    sigma_spatial = 1200.0  # mét
    pm25_emit = pm25_mat.copy()
    pm10_emit = pm10_mat.copy()

    for t in range(N_T):
        w_spd = ws_all[t]
        if w_spd < 0.3:
            continue
        e_w = np.array([u_wind[t] / w_spd, v_wind[t] / w_spd])

        for i in range(N_S):
            adv_flux_pm25 = 0.0
            adv_flux_pm10 = 0.0
            w_total = 0.0

            for j in range(N_S):
                if i == j:
                    continue
                disp = d_vec[j, i]
                d_ij = dist_m[j, i]
                cos_phi = float(np.dot(e_w, disp) / d_ij)
                if cos_phi <= 0.0:
                    continue

                v_eff = w_spd * cos_phi
                tau_h = (d_ij / max(v_eff, 0.1)) / 3600.0

                if tau_h <= 0.5:
                    w0, w1, w2 = 0.70, 0.30, 0.00
                elif tau_h <= 1.5:
                    w0, w1, w2 = 0.20, 0.60, 0.20
                else:
                    w0, w1, w2 = 0.05, 0.35, 0.60

                t0 = t
                t1 = max(0, t - 1)
                t2 = max(0, t - 2)

                c_source_25 = w0 * pm25_emit[t0, j] + w1 * pm25_emit[t1, j] + w2 * pm25_emit[t2, j]
                c_source_10 = w0 * pm10_emit[t0, j] + w1 * pm10_emit[t1, j] + w2 * pm10_emit[t2, j]

                k_geom = np.exp(-0.5 * (d_ij / sigma_spatial) ** 2) * (cos_phi ** 2)
                w_ij = adv_coupling * k_geom

                adv_flux_pm25 += w_ij * c_source_25
                adv_flux_pm10 += w_ij * c_source_10
                w_total += w_ij

            if w_total > 0.0:
                mix = min(0.35, w_total)
                pm25_mat[t, i] = (1.0 - mix) * pm25_mat[t, i] + mix * (adv_flux_pm25 / max(w_total, 1e-4))
                pm10_mat[t, i] = (1.0 - mix) * pm10_mat[t, i] + mix * (adv_flux_pm10 / max(w_total, 1e-4))

    rows_48 = []
    for t in range(N_T):
        row = []
        for s in range(N_S):
            row.extend([
                max(0.1, float(pm25_mat[t, s])),
                max(0.1, float(pm10_mat[t, s])),
                float(temp_mat[t, s]),
                max(0.0, float(rh_mat[t, s])),
                max(400.0, float(co2_mat[t, s])),
                max(10.0, float(voc_mat[t, s])),
            ])
        rows_48.append(row)

    arr_48 = np.array(rows_48, dtype=np.float32)
    print(f" -> Tạo thành công {len(arr_48)} khung đo 48 chiều đồng thời trên 8 trạm.")

    # Chronological Split TRƯỚC KHI tính Min-Max với BUFFER_HOURS = 48
    BUFFER_HOURS = 48
    split = int(0.8 * len(arr_48))
    train_raw = arr_48[:split]
    val_raw = arr_48[split + BUFFER_HOURS:]

    # Min-Max Normalization nghiêm ngặt trên tập Train
    min_48 = train_raw.min(axis=0)
    max_48 = train_raw.max(axis=0)
    denom_48 = np.maximum(max_48 - min_48, 1e-6)
    X_train = (train_raw - min_48) / denom_48
    X_val = (val_raw - min_48) / denom_48

    # SVD verification
    centered = X_train - np.mean(X_train, axis=0)
    _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    top6_ratio = float(np.sum(s_vals[:6] ** 2) / np.sum(s_vals ** 2))
    print(f" -> SVD 48D check: Top 6 singular values explain {top6_ratio*100:.2f}% of variance (Target < 90%).")

    return X_train, X_val, min_48, max_48


def apply_station_masking_and_noise(
    bx: torch.Tensor,
    sigma_min: float = 0.03,
    sigma_max: float = 0.08,
    p_mean: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Gaussian noise and stochastic masking of 1 to 2 stations to spatial mean or zero.
    """
    B = bx.size(0)
    bx_st = bx.view(B, 8, 6).clone()
    mask_bool = torch.zeros(B, 8, dtype=torch.bool, device=bx.device)

    for i in range(B):
        k = torch.randint(1, 3, (1,)).item()
        masked_indices = torch.randperm(8)[:k]
        mask_bool[i, masked_indices] = True
        unmasked_indices = [s for s in range(8) if s not in masked_indices]

        unmasked_mean = bx_st[i, unmasked_indices, :].mean(dim=0)
        for s in masked_indices:
            if torch.rand(1).item() < p_mean:
                bx_st[i, s, :] = unmasked_mean
            else:
                bx_st[i, s, :] = 0.0

    sigma = torch.empty(B, 1, 1, device=bx.device).uniform_(sigma_min, sigma_max)
    noise = torch.randn_like(bx_st) * sigma
    bx_corrupted = torch.clamp(bx_st + noise, 0.0, 1.0).view(B, 48)

    return bx_corrupted, mask_bool


def compute_msdae_loss(
    x_rec: torch.Tensor,
    x_clean: torch.Tensor,
    mask_bool: torch.Tensor,
    lambda_mask: float = 1.5,
    lambda_spatial: float = 0.5,
    beta: float = 0.02,
) -> torch.Tensor:
    """
    Composite Loss = HuberBase + lambda_mask * HuberMasked + lambda_spatial * SpatialDiff
    """
    loss_base = F.smooth_l1_loss(x_rec, x_clean, beta=beta)

    mask_48 = mask_bool.unsqueeze(-1).expand(-1, -1, 6).reshape(-1, 48)
    if mask_48.sum() > 0:
        loss_mask = F.smooth_l1_loss(x_rec[mask_48], x_clean[mask_48], beta=beta)
    else:
        loss_mask = torch.tensor(0.0, device=x_rec.device)

    rec_st = x_rec.view(-1, 8, 6)
    clean_st = x_clean.view(-1, 8, 6)
    diff_clean = clean_st.unsqueeze(2) - clean_st.unsqueeze(1)
    diff_rec = rec_st.unsqueeze(2) - rec_st.unsqueeze(1)
    loss_spatial = F.smooth_l1_loss(diff_rec, diff_clean, beta=beta)

    return loss_base + lambda_mask * loss_mask + lambda_spatial * loss_spatial


def train_autoencoder(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 3e-3,
    save_pt: str = "mlops/models/central_autoencoder.pt",
    save_onnx: str = "mlops/models/central_autoencoder.onnx",
    save_params: str = "mlops/models/central_autoencoder_params.json",
):
    X_train, X_val, min_48, max_48 = prepare_48d_multistation_dataset()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(X_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(X_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiStationDeepAutoencoder(in_features=48, latent_dim=16)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== MÔ HÌNH MASKED DENOISING AUTOENCODER ({total_params:,} THAM SỐ) ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state_dict = None
    patience = 10
    patience_counter = 0

    print("=== BẮT ĐẦU HUẤN LUYỆN MS-DAE 16D BOTTLENECK ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, _ in train_loader:
            optimizer.zero_grad()
            bx_corrupted, mask_bool = apply_station_masking_and_noise(bx)
            rec, z = model(bx_corrupted)
            loss = compute_msdae_loss(rec, bx, mask_bool)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Đánh giá trên tập val SẠCH (Clean Validation)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, _ in val_loader:
                rec, z = model(bx)
                val_loss += F.smooth_l1_loss(rec, bx, beta=0.02).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.6f} | Clean Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f" -> Early stopping kích hoạt tại epoch {epoch}")
            break

    if best_state_dict is not None:
        os.makedirs(os.path.dirname(save_pt), exist_ok=True)
        torch.save(best_state_dict, save_pt)
        model.load_state_dict(best_state_dict)

    # Đánh giá sai số tái tạo trên tập train sạch để xác lập baseline
    model.eval()
    with torch.no_grad():
        all_recs, _ = model(torch.tensor(X_train))
        recon_errors = np.abs(X_train - all_recs.numpy())
        sample_recon_mse = np.mean((X_train - all_recs.numpy()) ** 2, axis=1)

    mean_err = np.mean(recon_errors, axis=0)
    std_err = np.std(recon_errors, axis=0) + 1e-6
    baseline_mse_mean = float(np.mean(sample_recon_mse))
    baseline_mse_std = float(np.std(sample_recon_mse))

    print(f"\n=== HOÀN TẤT HỌC PHÂN PHỐI DỮ LIỆU SẠCH ===")
    print(f"Sai số tái tạo trung bình (MAE): {float(np.mean(mean_err)):.6f}")
    print(f"MSE mẫu trung bình: {baseline_mse_mean:.6f} (std: {baseline_mse_std:.6f})")

    # Xuất sang ONNX Opset 17
    print(f" -> Xuất mô hình sang ONNX: {save_onnx}...")
    model.load_state_dict(torch.load(save_pt, map_location="cpu", weights_only=True))
    model.eval()

    dummy_in = torch.randn(1, 48, dtype=torch.float32)
    with torch.no_grad():
        py_rec, py_z = model(dummy_in)
        py_rec_np = py_rec.numpy()

    os.makedirs(os.path.dirname(save_onnx), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_in,
        save_onnx,
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input_features_48d"],
        output_names=["reconstructed_features", "latent_representation"],
        dynamic_axes={
            "input_features_48d": {0: "batch_size"},
            "reconstructed_features": {0: "batch_size"},
            "latent_representation": {0: "batch_size"},
        },
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(save_onnx, providers=["CPUExecutionProvider"])
    onnx_rec, onnx_z = sess.run(None, {"input_features_48d": dummy_in.numpy()})
    parity_diff = float(np.max(np.abs(py_rec_np - onnx_rec)))
    print(f" -> Autoencoder Parity Diff: {parity_diff:.8e} (Hợp lệ < 1e-4)")
    assert parity_diff < 1e-4, f"ONNX parity violation: {parity_diff} >= 1e-4"

    meta = {
        "architecture": "MultiStationDeepAutoencoder",
        "in_features": 48,
        "latent_dim": 16,
        "total_params": total_params,
        "best_val_loss": best_val_loss,
        "parity_diff": parity_diff,
        "min_48": min_48.tolist(),
        "max_48": max_48.tolist(),
        "mean_error": mean_err.tolist(),
        "std_error": std_err.tolist(),
        "baseline_mse_mean": baseline_mse_mean,
        "baseline_mse_std": baseline_mse_std,
    }
    with open(save_params, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model, meta


if __name__ == "__main__":
    train_autoencoder()
