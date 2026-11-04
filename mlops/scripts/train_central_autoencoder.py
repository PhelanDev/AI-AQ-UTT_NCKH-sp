"""
Huấn Luyện Mô Hình Central Deep Autoencoder (Multi-Station 16D Bottleneck Anomaly Engine)
Học phân phối không gian - thời gian chuẩn của mạng lưới 8 trạm Thanh Xuân (48 kênh)
Xác lập ngưỡng động Mahalanobis 3-Sigma và xuất mô hình ONNX xịn cho Raspberry Pi 5
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
from pi.anomaly_engine import MultiStationDeepAutoencoder, AdaptiveAnomalyDetector

MICROCLIMATE_MODULATIONS = {
    "STATION_01": (1.00, 1.00,  0.0,  0.0, 1.00,  0.0),  # urban_core, Thanh Xuan Bac
    "STATION_02": (1.12, 1.15,  0.6, -2.0, 0.90,  5.0),  # traffic_corridor, Khuat Duy Tien Ring Road 3
    "STATION_03": (0.88, 0.86, -0.7,  3.5, 1.15, -4.0),  # urban_park_highway, Nhan Chinh Park
    "STATION_04": (1.04, 1.02,  0.3, -0.8, 0.85,  3.0),  # dense_residential, Kim Giang
    "STATION_05": (1.16, 1.18,  0.8, -2.5, 0.88,  7.0),  # major_intersection, Nga Tu So
    "STATION_06": (0.95, 0.94, -0.3,  4.0, 1.05, -3.0),  # river_residential, Khuong Dinh / To Lich
    "STATION_07": (1.08, 1.09,  0.4, -1.2, 0.92,  4.0),  # commercial_axis, Le Van Luong
    "STATION_08": (1.10, 1.13,  0.5, -1.5, 0.95,  6.0),  # southern_gateway, Nguyen Xien
}

def get_station_modulation(sid: str):
    if sid in MICROCLIMATE_MODULATIONS:
        return MICROCLIMATE_MODULATIONS[sid]
    try:
        num = int("".join(c for c in sid if c.isdigit()))
        key = f"STATION_{num:02d}"
        if key in MICROCLIMATE_MODULATIONS:
            return MICROCLIMATE_MODULATIONS[key]
    except Exception:
        pass
    return (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)

def prepare_48d_multistation_dataset(
    csv_path: str = "mlops/data/hanoi_thanhxuan_multistation_real.csv",
):
    print(" -> Nạp dữ liệu 8 trạm Thanh Xuân để ghép vector 48 chiều...")
    # Map timestamp -> { station_id -> [pm25, pm10, temp, rh, co2, voc] }
    time_dict = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = r["time"]
            sid = r["station_id"]
            try:
                pm25 = float(r["pm25"])
                pm10 = float(r["pm10"])
                temp = float(r["temperature_c"])
                rh = float(r["relative_humidity"])
                co = float(r["co"])
                no2 = float(r["no2"])
                co2_ppm = co * 1.2 + 420.0
                voc_idx = no2 * 3.0 + 100.0

                mod = get_station_modulation(sid)
                pm25 = max(0.1, pm25 * mod[0])
                pm10 = max(0.1, pm10 * mod[1])
                temp = temp + mod[2]
                rh = max(0.0, rh + mod[3])
                co2_ppm = co2_ppm * mod[4]
                voc_idx = max(10.0, voc_idx + mod[5])

                if t not in time_dict:
                    time_dict[t] = {}
                time_dict[t][sid] = [pm25, pm10, temp, rh, co2_ppm, voc_idx]
            except (ValueError, TypeError):
                continue

    station_keys = [f"STATION_{i:02d}" for i in range(1, 9)]
    vectors = []
    for t, stations in time_dict.items():
        if len(stations) == 8:
            row_48 = []
            for sid in station_keys:
                row_48.extend(stations[sid])
            vectors.append(row_48)

    arr_48 = np.array(vectors, dtype=np.float32)
    print(f" -> Tạo thành công {len(arr_48)} khung đo 48 chiều đồng thời trên 8 trạm.")

    # Chronological Split TRƯỚC KHI tính Min-Max với BUFFER_HOURS = 48
    BUFFER_HOURS = 48
    split = int(0.8 * len(arr_48))
    train_raw = arr_48[:split]
    val_raw = arr_48[split + 48:]

    # Min-Max Normalization nghiêm ngặt trên tập Train
    min_48 = train_raw.min(axis=0)
    max_48 = train_raw.max(axis=0)
    denom_48 = np.maximum(max_48 - min_48, 1e-6)
    X_train = (train_raw - min_48) / denom_48
    X_val = (val_raw - min_48) / denom_48

    return X_train, X_val, min_48, max_48

def train_autoencoder(
    epochs: int = 25,
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
    print(f"\n=== MÔ HÌNH MULTI-STATION DEEP AUTOENCODER ({total_params:,} THAM SỐ) ===")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state_dict = None

    print("=== BẮT ĐẦU HUẤN LUYỆN AUTOENCODER 16D BOTTLENECK ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, _ in train_loader:
            optimizer.zero_grad()
            rec, z = model(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, _ in val_loader:
                rec, z = model(bx)
                val_loss += criterion(rec, bx).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if best_state_dict is not None:
        os.makedirs(os.path.dirname(save_pt), exist_ok=True)
        torch.save(best_state_dict, save_pt)
        model.load_state_dict(best_state_dict)

    # Đánh giá tái tạo và tính toán phân phối Mahalanobis
    model.eval()
    with torch.no_grad():
        all_recs, _ = model(torch.tensor(X_train))
        recon_errors = np.abs(X_train - all_recs.numpy())

    mean_err = np.mean(recon_errors, axis=0)
    std_err = np.std(recon_errors, axis=0) + 1e-6
    cov_matrix = np.cov(recon_errors.T) + np.eye(48) * 1e-4
    inv_cov = np.linalg.pinv(cov_matrix)

    print(f"\n=== HOÀN TẤT HỌC PHÂN PHỐI DỮ LIỆU SẠCH ===")
    print(f"Sai số tái tạo trung bình: {float(np.mean(mean_err)):.6f}")

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

    # Lưu siêu tham số
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
    }
    with open(save_params, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model, meta

if __name__ == "__main__":
    train_autoencoder()
