#!/usr/bin/env python3
"""
train_and_quantize_standard_mlp_lite.py
Huấn luyện mạng Standard Multi-Layer Perceptron (8D -> 32 -> 16 -> 1) cho bản AQ_Lite
- Kiến trúc phẳng kinh điển, không Residual Connections, không over-engineering.
- Quy mô: ~833 tham số, lượng tử hóa Full INT8 (Flash < 1 KB, Heap: 0 Byte).
- Tự động lượng tử hóa và xuất file C/C++ Header: esp/include/model_data.h
"""

import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Đảm bảo UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT_DIR)

torch.manual_seed(42)
np.random.seed(42)


class StandardMLPLite(nn.Module):
    """Mạng Nơ-ron Truyền thẳng 2 lớp ẩn (Vanilla FNN / Standard MLP)."""
    def __init__(self, in_dim: int = 8, h1_dim: int = 32, h2_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(h2_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        raw = self.fc_out(h2)
        # Giới hạn vật lý hệ số C trong [0.25, 1.05]
        c = 0.25 + 0.80 * self.sigmoid(raw)
        return c


def quantize_matrix(weight_tensor):
    w = weight_tensor.detach().cpu().numpy()
    max_val = np.max(np.abs(w))
    scale = float(max_val / 127.0) if max_val > 0 else 1.0
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_int8, scale


def load_dataset(csv_path: str, max_rows: int = 60000):
    print(f" -> Đang nạp dữ liệu từ {csv_path}...")
    X, y_c, y_pm = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            pm_avg = float(r["pm_avg"])
            ref = float(r["ref_pm25"])
            if pm_avg <= 0.5:
                continue
            c_target = np.clip(ref / pm_avg, 0.25, 1.05)
            X.append([
                pm_avg, float(r["diff_pm"]), float(r["r_pm"]),
                float(r["temp"]), float(r["rh"]), float(r["dew_point"]),
                float(r["co2"]), float(r["gas_proxy"])
            ])
            y_c.append([c_target])
            y_pm.append([ref])
    return np.array(X, dtype=np.float32), np.array(y_c, dtype=np.float32), np.array(y_pm, dtype=np.float32)


def main():
    train_csv = "mlops/data/multisensor_training_set.csv"
    test_csv = "mlops/data/multisensor_test_set.csv"

    X_train, y_c_train, y_pm_train = load_dataset(train_csv, max_rows=50000)
    X_test, y_c_test, y_pm_test = load_dataset(test_csv, max_rows=10000)

    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    denom = np.maximum(x_max - x_min, 1e-6)

    X_train_norm = np.clip((X_train - x_min) / denom, 0.0, 1.0)
    X_test_norm = np.clip((X_test - x_min) / denom, 0.0, 1.0)

    t_X_train = torch.from_numpy(X_train_norm)
    t_y_train = torch.from_numpy(y_c_train)
    t_pm_train = torch.from_numpy(X_train[:, 0:1])
    t_ref_train = torch.from_numpy(y_pm_train)

    t_X_test = torch.from_numpy(X_test_norm)
    t_ref_test = torch.from_numpy(y_pm_test)
    t_pm_test = torch.from_numpy(X_test[:, 0:1])

    model = StandardMLPLite(in_dim=8, h1_dim=32, h2_dim=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN STANDARD MLP LITE (TOTAL PARAMS: {total_params:,})")
    print(f"=======================================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    batch_size = 256
    n_batches = len(X_train_norm) // batch_size
    epochs = 12

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_train_norm))
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            bx = t_X_train[idx]
            b_pm = t_pm_train[idx]
            b_ref = t_ref_train[idx]

            optimizer.zero_grad()
            c_pred = model(bx)
            pm_pred = b_pm * c_pred
            # Tối ưu hóa trực tiếp nồng độ PM2.5 bù ẩm
            loss = loss_fn(pm_pred, b_ref)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            c_test = model(t_X_test)
            pred_pm_test = (t_pm_test * c_test).numpy().flatten()
            true_pm_test = t_ref_test.numpy().flatten()
            rmse = np.sqrt(np.mean((pred_pm_test - true_pm_test) ** 2))
            ss_tot = np.sum((true_pm_test - np.mean(true_pm_test)) ** 2)
            ss_res = np.sum((true_pm_test - pred_pm_test) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            print(f" [Epoch {ep+1:02d}/{epochs:02d}] Train Loss: {epoch_loss/n_batches:.4f} | Test RMSE: {rmse:.2f} ug/m3 | Test R2: {r2:.4f}")

    # Lượng tử hóa sang INT8 và xuất C Header
    print("\n -> Tiến hành Lượng tử hóa Full INT8 cho ESP32-S3...")
    model.eval()
    sd = model.state_dict()

    w1_int8, scale1 = quantize_matrix(sd["fc1.weight"])
    b1 = sd["fc1.bias"].numpy()

    w2_int8, scale2 = quantize_matrix(sd["fc2.weight"])
    b2 = sd["fc2.bias"].numpy()

    w_out_int8, scale_out = quantize_matrix(sd["fc_out.weight"])
    b_out = sd["fc_out.bias"].numpy()

    # Lưu model PyTorch
    os.makedirs("mlops/models", exist_ok=True)
    torch.save(sd, "mlops/models/standard_mlp_lite.pt")
    with open("mlops/models/standard_mlp_lite_params.json", "w", encoding="utf-8") as f:
        json.dump({"x_min": x_min.tolist(), "x_max": x_max.tolist()}, f, indent=2)

    # Xuất file esp/include/model_data.h
    header_path = "esp/include/model_data.h"
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * model_data.h - Standard MLP Lite (8D -> 32 -> 16 -> 1)\n")
        f.write(" * Mạng Nơ-ron Chuẩn Sinh Viên cho Bản Phân Phối AQ_Lite\n")
        f.write(f" * Tổng số tham số: {total_params} | Kích thước INT8: {w1_int8.size + w2_int8.size + w_out_int8.size} Bytes\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_MULTISENSOR_H\n")
        f.write("#define MODEL_DATA_MULTISENSOR_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define TINYML_INPUT_DIM 8\n")
        f.write("#define TINYML_HIDDEN1_DIM 32\n")
        f.write("#define TINYML_HIDDEN2_DIM 16\n\n")

        # X_MIN & X_MAX
        f.write("// Ngưỡng Min/Max chuẩn hóa dữ liệu đầu vào\n")
        f.write("static const float TINYML_X_MIN[TINYML_INPUT_DIM] = {\n    ")
        f.write(", ".join(f"{x:.6f}f" for x in x_min) + "\n};\n\n")
        f.write("static const float TINYML_X_MAX[TINYML_INPUT_DIM] = {\n    ")
        f.write(", ".join(f"{x:.6f}f" for x in x_max) + "\n};\n\n")

        # Layer 1: fc1 (8 -> 32)
        f.write(f"// Tầng 1: fc1 (8 -> 32)\n")
        f.write(f"static const float FC1_SCALE = {scale1:.8f}f;\n")
        f.write("static const float FC1_BIAS[TINYML_HIDDEN1_DIM] = {\n")
        for i in range(0, 32, 8):
            f.write("    " + ", ".join(f"{b:.6f}f" for b in b1[i:i+8]) + ",\n")
        f.write("};\n")
        f.write("static const int8_t FC1_WEIGHT[TINYML_HIDDEN1_DIM][TINYML_INPUT_DIM] = {\n")
        for row in w1_int8:
            f.write("    { " + ", ".join(f"{int(v):4d}" for v in row) + " },\n")
        f.write("};\n\n")

        # Layer 2: fc2 (32 -> 16)
        f.write(f"// Tầng 2: fc2 (32 -> 16)\n")
        f.write(f"static const float FC2_SCALE = {scale2:.8f}f;\n")
        f.write("static const float FC2_BIAS[TINYML_HIDDEN2_DIM] = {\n")
        for i in range(0, 16, 8):
            f.write("    " + ", ".join(f"{b:.6f}f" for b in b2[i:i+8]) + ",\n")
        f.write("};\n")
        f.write("static const int8_t FC2_WEIGHT[TINYML_HIDDEN2_DIM][TINYML_HIDDEN1_DIM] = {\n")
        for row in w2_int8:
            f.write("    { " + ", ".join(f"{int(v):4d}" for v in row) + " },\n")
        f.write("};\n\n")

        # Layer Output: fc_out (16 -> 1)
        f.write(f"// Tầng 3: fc_out (16 -> 1)\n")
        f.write(f"static const float FC_OUT_SCALE = {scale_out:.8f}f;\n")
        f.write(f"static const float FC_OUT_BIAS[1] = {{ {b_out[0]:.6f}f }};\n")
        f.write("static const int8_t FC_OUT_WEIGHT[1][TINYML_HIDDEN2_DIM] = {\n")
        f.write("    { " + ", ".join(f"{int(v):4d}" for v in w_out_int8[0]) + " }\n")
        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_MULTISENSOR_H\n")

    print(f" -> Đã xuất thành công model C++ nhúng vào {header_path}!")
    print(f" -> Kích thước bảng trọng số INT8: {w1_int8.size + w2_int8.size + w_out_int8.size} Bytes (< 1 KB)!")


if __name__ == "__main__":
    main()
