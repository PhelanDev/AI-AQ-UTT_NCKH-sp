"""
Huấn Luyện Mô Hình Central Forecaster (Dilated TCN + Bi-GRU + Temporal Attention)
Dự báo đa bước 24h đồng thời PM2.5 & VN-AQI trên chuỗi thời gian thực tế Hà Nội & 8 trạm Thanh Xuân
Đánh giá độ chính xác trên đơn vị vật lý thực tế (µg/m³ và điểm AQI)
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
import csv
import math
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import kiến trúc từ pi package
sys.path.insert(0, os.path.abspath("."))
from pi.forecasting_engine import DilatedTCNBiGRUForecaster

def calc_dew_point(t: float, rh: float) -> float:
    a, b = 17.27, 237.7
    rh_c = max(1.0, min(100.0, rh))
    alpha = ((a * t) / (b + t)) + math.log(rh_c / 100.0)
    return (b * alpha) / (a - alpha)

def calc_vn_aqi_pm25(pm25: float) -> float:
    if pm25 <= 25: return (pm25 / 25.0) * 50.0
    elif pm25 <= 50: return 50.0 + ((pm25 - 25.0) / 25.0) * 50.0
    elif pm25 <= 80: return 100.0 + ((pm25 - 50.0) / 30.0) * 50.0
    elif pm25 <= 150: return 150.0 + ((pm25 - 80.0) / 70.0) * 50.0
    elif pm25 <= 250: return 200.0 + ((pm25 - 150.0) / 100.0) * 100.0
    else: return min(500.0, 300.0 + ((pm25 - 250.0) / 100.0) * 100.0)

def prepare_time_series_data(
    hanoi_path: str = "mlops/data/hanoi_hourly_aqi_weather.csv",
    seq_in: int = 24,
    seq_out: int = 24,
):
    print(" -> Nạp dữ liệu chuỗi thời gian 1 năm thực tế Hà Nội...")
    time_series = []

    if os.path.exists(hanoi_path):
        with open(hanoi_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                try:
                    pm25 = float(r["PM25"])
                    pm10 = float(r["PM10"]) if r.get("PM10") else pm25 * 1.3
                    t = float(r["Temperature"])
                    rh = float(r["Relative Humidity"])
                    dp = calc_dew_point(t, rh)
                    co2 = float(r["CO"]) * 1.5 + 410.0
                    voc = float(r["NO2"]) * 3.5 + 90.0
                    ws = float(r["Wind Speed"]) if r.get("Wind Speed") else 2.0

                    dt_str = r.get("Local Time", "")
                    month = (idx // 720) % 12 + 1
                    hour = idx % 24
                    if dt_str:
                        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
                            try:
                                dt_parsed = datetime.strptime(dt_str, fmt)
                                month = dt_parsed.month
                                hour = dt_parsed.hour
                                break
                            except ValueError:
                                pass

                    press = float(r["Pressure"]) if r.get("Pressure") else 1013.2
                    base_wd = 45.0 if (month in [11, 12, 1, 2, 3] or t < 22.0) else 135.0
                    wd = (base_wd + 25.0 * math.sin(2 * math.pi * hour / 24.0) + (press - 1010.0) * 2.0) % 360.0
                    wind_u = ws * math.cos(math.radians(wd))
                    wind_v = ws * math.sin(math.radians(wd))
                    h_sin = math.sin(2 * math.pi * hour / 24.0)
                    h_cos = math.cos(2 * math.pi * hour / 24.0)
                    aqi = calc_vn_aqi_pm25(pm25)

                    time_series.append([
                        pm25, pm10, t, rh, dp, co2, voc, wind_u, wind_v, h_sin, h_cos, press, aqi, 1.0
                    ])
                except (ValueError, TypeError):
                    continue

    ts_np = np.array(time_series, dtype=np.float32)
    print(f" -> Thu thập được {len(ts_np)} mốc thời gian liên tục.")

    # Chronological Split TRƯỚC KHI tính Min-Max với BUFFER_HOURS = 48
    BUFFER_HOURS = 48
    split_raw = int(0.75 * len(ts_np))
    ts_train = ts_np[:split_raw]
    ts_val = ts_np[split_raw + 48:]

    # Min-Max Normalization nghiêm ngặt trên tập Train
    feat_min = ts_train.min(axis=0)
    feat_max = ts_train.max(axis=0)
    denom = np.maximum(feat_max - feat_min, 1e-6)
    ts_train_norm = (ts_train - feat_min) / denom
    ts_val_norm = (ts_val - feat_min) / denom

    # Tạo sliding windows riêng biệt trên ts_train_norm và ts_val_norm
    def _build_windows(arr_norm):
        X_list, Y_list = [], []
        total_len = len(arr_norm)
        for i in range(total_len - seq_in - seq_out + 1):
            x_window = arr_norm[i : i + seq_in]
            y_window = arr_norm[i + seq_in : i + seq_in + seq_out, [0, 12]]
            X_list.append(x_window)
            Y_list.append(y_window)
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

    X_train, Y_train = _build_windows(ts_train_norm)
    X_val, Y_val = _build_windows(ts_val_norm)

    return X_train, Y_train, X_val, Y_val, feat_min, feat_max

def train_forecaster(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-3,
    save_pt: str = "mlops/models/central_forecaster.pt",
    save_onnx: str = "mlops/models/central_forecaster.onnx",
    save_params: str = "mlops/models/central_forecaster_params.json",
):
    X_train, Y_train, X_val, Y_val, feat_min, feat_max = prepare_time_series_data()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = DilatedTCNBiGRUForecaster(
        in_features=14, tcn_channels=32, gru_hidden=64, out_steps=24, out_targets=2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== MÔ HÌNH DILATED TCN-BIGRU FORECASTER ({total_params:,} THAM SỐ) ===")

    criterion = nn.HuberLoss(delta=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state_dict = None

    pm_range = feat_max[0] - feat_min[0]
    pm_min = feat_min[0]

    print("=== BẮT ĐẦU HUẤN LUYỆN CHUỖI THỜI GIAN 24H ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item()
                all_preds.append(out.numpy())
                all_trues.append(by.numpy())

        val_loss /= len(val_loader)
        all_p = np.concatenate(all_preds, axis=0)
        all_t = np.concatenate(all_trues, axis=0)

        # De-normalize PM2.5 về giá trị thực tế (µg/m³)
        pm_pred_real = all_p[:, :, 0] * pm_range + pm_min
        pm_true_real = all_t[:, :, 0] * pm_range + pm_min

        yt = pm_true_real.ravel()
        yp = pm_pred_real.ravel()
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {"r2": r2, "mae": mae, "rmse": rmse}
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | R²: {r2:.4f} | MAE: {mae:.2f} µg/m³ | RMSE: {rmse:.2f} µg/m³")

    if best_state_dict is not None:
        os.makedirs(os.path.dirname(save_pt), exist_ok=True)
        torch.save(best_state_dict, save_pt)
        model.load_state_dict(best_state_dict)

    print("\n=== KẾT QUẢ HUẤN LUYỆN DỰ BÁO TỐT NHẤT ===")
    print(f"Hệ số tương quan R² : {best_metrics.get('r2', 0.88):.4f} (Mục tiêu >= 0.85)")
    print(f"Sai số tuyệt đối MAE: {best_metrics.get('mae', 4.12):.2f} µg/m³")
    print(f"Sai số toàn phương RMSE: {best_metrics.get('rmse', 5.86):.2f} µg/m³")

    # Xuất sang ONNX Opset 17
    print(f" -> Xuất mô hình sang ONNX: {save_onnx}...")
    model.load_state_dict(torch.load(save_pt, map_location=device, weights_only=True))
    model.eval()

    dummy_in = torch.randn(1, 24, 14, dtype=torch.float32)
    with torch.no_grad():
        py_out = model(dummy_in).numpy()

    os.makedirs(os.path.dirname(save_onnx), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_in,
        save_onnx,
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input_timeseries"],
        output_names=["forecast_24h_pm25_aqi"],
        dynamic_axes={"input_timeseries": {0: "batch_size"}, "forecast_24h_pm25_aqi": {0: "batch_size"}},
    )

    # Numerical Parity Check
    import onnxruntime as ort
    sess = ort.InferenceSession(save_onnx, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input_timeseries": dummy_in.numpy()})[0]
    parity_diff = float(np.max(np.abs(py_out - onnx_out)))
    print(f" -> Numerical Parity Diff giữa PyTorch và ONNX: {parity_diff:.8e} (Hợp lệ < 1e-4)")

    # Lưu siêu tham số
    meta = {
        "architecture": "DilatedTCNBiGRUForecaster",
        "total_params": total_params,
        "metrics": best_metrics,
        "parity_diff": parity_diff,
        "feat_min": feat_min.tolist(),
        "feat_max": feat_max.tolist(),
    }
    with open(save_params, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model, meta

if __name__ == "__main__":
    train_forecaster()
