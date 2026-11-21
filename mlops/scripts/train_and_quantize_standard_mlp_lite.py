#!/usr/bin/env python3
"""
train_and_quantize_standard_mlp_lite.py
Huấn luyện mạng Standard Multi-Layer Perceptron (8D -> 32 -> 16 -> 1) cho bản AQ_Lite
- Kiến trúc phẳng kinh điển, không Residual Connections, không over-engineering.
    - Quy mô: 833 tham số, trọng số INT8 và suy luận tĩnh không cấp phát heap.
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


def firmware_sensor_values(pm_avg: float, diff_pm: float, r_pm: float):
    """Reconstruct the dual-PM filter and PM ratio used by the firmware."""
    pm_used = pm_avg
    if diff_pm > 25.0:
        pm_used = max(0.0, pm_avg - diff_pm * 0.5)

    pm10 = pm_avg / max(r_pm, 0.1)
    r_used = float(np.clip(pm_used / max(pm10, 0.1), 0.1, 1.0))
    return pm_used, r_used


def load_dataset(csv_path: str, max_rows: int | None = None):
    print(f" -> Đang nạp dữ liệu từ {csv_path}...")
    X, y_c, y_pm = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            pm_avg = float(r["pm_avg"])
            diff_pm = float(r["diff_pm"])
            r_pm = float(r["r_pm"])
            ref = float(r["ref_pm25"])
            if pm_avg <= 0.5:
                continue
            pm_used, r_used = firmware_sensor_values(pm_avg, diff_pm, r_pm)
            if pm_used <= 0.5:
                continue
            c_target = np.clip(ref / pm_used, 0.25, 1.05)
            X.append([
                pm_used, diff_pm, r_used,
                float(r["temp"]), float(r["rh"]), float(r["dew_point"]),
                float(r["co2"]), float(r["gas_proxy"])
            ])
            y_c.append([c_target])
            y_pm.append([ref])
    return np.array(X, dtype=np.float32), np.array(y_c, dtype=np.float32), np.array(y_pm, dtype=np.float32)


def gate_weight_torch(rh: torch.Tensor, co2: torch.Tensor) -> torch.Tensor:
    sig_rh = torch.sigmoid((rh - 68.0) / 7.0)
    sig_co2 = torch.sigmoid((co2 - 800.0) / 240.0)
    return sig_rh * (1.0 - sig_co2)


def quantized_parameters(model: StandardMLPLite):
    state = model.state_dict()
    w1, scale1 = quantize_matrix(state["fc1.weight"])
    w2, scale2 = quantize_matrix(state["fc2.weight"])
    w_out, scale_out = quantize_matrix(state["fc_out.weight"])
    return {
        "w1": w1,
        "scale1": scale1,
        "b1": state["fc1.bias"].detach().cpu().numpy(),
        "w2": w2,
        "scale2": scale2,
        "b2": state["fc2.bias"].detach().cpu().numpy(),
        "w_out": w_out,
        "scale_out": scale_out,
        "b_out": state["fc_out.bias"].detach().cpu().numpy(),
    }


def predict_firmware_path(model: StandardMLPLite, X_norm, X_raw):
    params = quantized_parameters(model)
    w1 = params["w1"].astype(np.float32)
    w2 = params["w2"].astype(np.float32)
    w_out = params["w_out"].astype(np.float32)

    h0 = np.maximum((X_norm @ w1.T) * params["scale1"] + params["b1"], 0.0)
    h1 = np.maximum((h0 @ w2.T) * params["scale2"] + params["b2"], 0.0)
    raw_out = (h1 @ w_out.T) * params["scale_out"] + params["b_out"]
    c_factor = 0.25 + 0.80 / (1.0 + np.exp(-np.clip(raw_out[:, 0], -80.0, 80.0)))

    rh = X_raw[:, 4]
    co2 = X_raw[:, 6]
    gate = (1.0 / (1.0 + np.exp(-np.clip((rh - 68.0) / 7.0, -80.0, 80.0))))
    gate *= 1.0 - 1.0 / (1.0 + np.exp(-np.clip((co2 - 800.0) / 240.0, -80.0, 80.0)))
    c_effective = 1.0 - gate + gate * c_factor
    return X_raw[:, 0] * c_effective


def regression_metrics(y_true, y_pred):
    errors = y_pred - y_true
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "r2": float(1.0 - np.sum(errors ** 2) / max(ss_tot, 1e-12)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def validation_checkpoint_score(simulation_metrics, epa_metrics):
    """Score checkpoints on both deployment domains using their R2 values."""
    return float(
        0.75 * simulation_metrics["r2"]
        + 0.25 * epa_metrics["r2"]
    )


def c_float(value):
    """Format a Python/NumPy float as a valid C++ float literal."""
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "f"


def train_epoch(
    model,
    X_tensor,
    pm_tensor,
    rh_tensor,
    co2_tensor,
    y_tensor,
    c_target_tensor,
    optimizer,
    rng,
    sample_indices,
    batch_size=256,
):
    """Train one epoch through the same gated path that the firmware executes."""
    model.train()
    permutation = rng.permutation(sample_indices)
    for start in range(0, len(permutation), batch_size):
        idx = permutation[start : start + batch_size]
        c_pred = model(X_tensor[idx]).squeeze(-1)
        gate = gate_weight_torch(rh_tensor[idx], co2_tensor[idx]).squeeze(-1)
        pm_pred = pm_tensor[idx, 0] * (1.0 - gate + gate * c_pred)

        # Direct C supervision keeps gradients alive when the firmware gate is
        # nearly closed; the PM term preserves the end-to-end firmware target.
        c_error = nn.functional.smooth_l1_loss(
            c_pred, c_target_tensor[idx, 0], reduction="none"
        )
        pm_error = nn.functional.smooth_l1_loss(
            pm_pred, y_tensor[idx, 0], reduction="none"
        )
        c_loss = torch.mean((0.10 + gate) * c_error)
        pm_loss = torch.mean(pm_error / 10.0)
        loss = c_loss + 0.30 * pm_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def fit_epochs(model, X_norm, X_raw, y_ref, c_target, epochs, seed):
    X_tensor = torch.from_numpy(X_norm)
    pm_tensor = torch.from_numpy(X_raw[:, 0:1])
    rh_tensor = torch.from_numpy(X_raw[:, 4:5])
    co2_tensor = torch.from_numpy(X_raw[:, 6:7])
    y_tensor = torch.from_numpy(y_ref)
    c_target_tensor = torch.from_numpy(c_target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    sample_indices = np.arange(len(X_tensor), dtype=np.int64)
    for _ in range(epochs):
        train_epoch(
            model,
            X_tensor,
            pm_tensor,
            rh_tensor,
            co2_tensor,
            y_tensor,
            c_target_tensor,
            optimizer,
            rng,
            sample_indices,
        )


def main():
    training_csv_path = "mlops/data/multisensor_training_set.csv"
    test_csv = "mlops/data/multisensor_test_set.csv"

    X_train, y_c_train, y_pm_train = load_dataset(training_csv_path)
    X_test, _, y_pm_test = load_dataset(test_csv)
    if len(X_train) < 2 or len(X_test) == 0:
        raise ValueError("Training and test datasets must contain usable samples")

    # The generated files append the chronological EPA slices at the end.
    # Keep validation stratified so the large Simulation slice cannot hide an
    # EPA regression and vice versa.
    epa_train_count = min(15000, len(X_train) - 1)
    epa_test_count = min(3000, len(X_test))
    simulation_train_count = len(X_train) - epa_train_count
    simulation_test_count = len(X_test) - epa_test_count

    split_rng = np.random.default_rng(42)
    simulation_indices = np.arange(simulation_train_count, dtype=np.int64)
    epa_indices = np.arange(simulation_train_count, len(X_train), dtype=np.int64)
    split_rng.shuffle(simulation_indices)
    split_rng.shuffle(epa_indices)
    simulation_validation_count = max(1, int(round(len(simulation_indices) * 0.10)))
    epa_validation_count = max(1, int(round(len(epa_indices) * 0.10)))
    validation_simulation_indices = simulation_indices[:simulation_validation_count]
    validation_epa_indices = epa_indices[:epa_validation_count]
    validation_indices = np.concatenate(
        [validation_simulation_indices, validation_epa_indices]
    )
    fit_indices = np.concatenate(
        [simulation_indices[simulation_validation_count:], epa_indices[epa_validation_count:]]
    )

    validation_min = np.min(X_train[fit_indices], axis=0)
    validation_max = np.max(X_train[fit_indices], axis=0)
    validation_denom = np.maximum(validation_max - validation_min, 1e-6)
    X_fit_norm = np.clip((X_train[fit_indices] - validation_min) / validation_denom, 0.0, 1.0)
    X_validation_norm = np.clip((X_train[validation_indices] - validation_min) / validation_denom, 0.0, 1.0)

    torch.manual_seed(42)
    model = StandardMLPLite(in_dim=8, h1_dim=32, h2_dim=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN STANDARD MLP LITE (TOTAL PARAMS: {total_params:,})")
    print(f"=======================================================")

    max_epochs = 60
    patience = 10
    best_validation_score = -float("inf")
    best_validation_simulation_metrics = {}
    best_validation_epa_metrics = {}
    best_epoch = 1
    stale_epochs = 0
    X_fit_tensor = torch.from_numpy(X_fit_norm)
    pm_fit_tensor = torch.from_numpy(X_train[fit_indices, 0:1])
    rh_fit_tensor = torch.from_numpy(X_train[fit_indices, 4:5])
    co2_fit_tensor = torch.from_numpy(X_train[fit_indices, 6:7])
    y_fit_tensor = torch.from_numpy(y_pm_train[fit_indices])
    c_fit_tensor = torch.from_numpy(y_c_train[fit_indices])
    validation_optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    validation_rng = np.random.default_rng(1042)
    fit_local_indices = np.arange(len(fit_indices), dtype=np.int64)
    for epoch in range(1, max_epochs + 1):
        train_epoch(
            model, X_fit_tensor, pm_fit_tensor, rh_fit_tensor, co2_fit_tensor,
            y_fit_tensor, c_fit_tensor, validation_optimizer, validation_rng,
            fit_local_indices,
        )
        validation_simulation_predictions = predict_firmware_path(
            model,
            X_validation_norm[:len(validation_simulation_indices)],
            X_train[validation_simulation_indices],
        )
        validation_epa_predictions = predict_firmware_path(
            model,
            X_validation_norm[len(validation_simulation_indices):],
            X_train[validation_epa_indices],
        )
        validation_simulation_metrics = regression_metrics(
            y_pm_train[validation_simulation_indices, 0],
            validation_simulation_predictions,
        )
        validation_epa_metrics = regression_metrics(
            y_pm_train[validation_epa_indices, 0],
            validation_epa_predictions,
        )
        # Select the checkpoint on the deployment metric rather than MAE. The
        # test contract requires preserving both the large Simulation domain
        # and the chronological EPA domain in the selection signal.
        validation_score = validation_checkpoint_score(
            validation_simulation_metrics,
            validation_epa_metrics,
        )
        print(
            f" [Epoch {epoch:02d}/{max_epochs}] Validation INT8 | "
            f"Simulation RMSE: {validation_simulation_metrics['rmse']:.2f} "
            f"R2: {validation_simulation_metrics['r2']:.4f} | "
            f"EPA RMSE: {validation_epa_metrics['rmse']:.2f} "
            f"R2: {validation_epa_metrics['r2']:.4f} | "
            f"Score: {validation_score:.4f}"
        )
        if validation_score > best_validation_score + 1e-4:
            best_validation_score = validation_score
            best_validation_simulation_metrics = validation_simulation_metrics.copy()
            best_validation_epa_metrics = validation_epa_metrics.copy()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    print(f" -> Chọn epoch {best_epoch} theo validation; huấn luyện lại trên toàn bộ {len(X_train):,} mẫu.")
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    denom = np.maximum(x_max - x_min, 1e-6)
    X_train_norm = np.clip((X_train - x_min) / denom, 0.0, 1.0)
    X_test_norm = np.clip((X_test - x_min) / denom, 0.0, 1.0)

    torch.manual_seed(42)
    model = StandardMLPLite(in_dim=8, h1_dim=32, h2_dim=16)
    fit_epochs(model, X_train_norm, X_train, y_pm_train, y_c_train, best_epoch, 4242)
    test_predictions = predict_firmware_path(model, X_test_norm, X_test)
    test_metrics = regression_metrics(y_pm_test[:, 0], test_predictions)
    simulation_test_metrics = regression_metrics(
        y_pm_test[:simulation_test_count, 0], test_predictions[:simulation_test_count]
    )
    epa_test_metrics = regression_metrics(
        y_pm_test[-epa_test_count:, 0], test_predictions[-epa_test_count:]
    )
    print(
        f" [Final firmware-path test, INT8 weights] N={len(X_test):,} | "
        f"RMSE: {test_metrics['rmse']:.2f} ug/m3 | "
        f"MAE: {test_metrics['mae']:.2f} | R2: {test_metrics['r2']:.4f}"
    )
    print(
        f" [Simulation test] N={simulation_test_count:,} | "
        f"RMSE: {simulation_test_metrics['rmse']:.2f} | "
        f"MAE: {simulation_test_metrics['mae']:.2f} | "
        f"R2: {simulation_test_metrics['r2']:.4f}"
    )
    print(
        f" [EPA test] N={epa_test_count:,} | "
        f"RMSE: {epa_test_metrics['rmse']:.2f} | "
        f"MAE: {epa_test_metrics['mae']:.2f} | "
        f"R2: {epa_test_metrics['r2']:.4f}"
    )

    # Lượng tử hóa sang INT8 và xuất C Header
    print("\n -> Tiến hành Lượng tử hóa Full INT8 cho ESP32-S3...")
    model.eval()
    sd = model.state_dict()

    params = quantized_parameters(model)
    w1_int8, scale1, b1 = params["w1"], params["scale1"], params["b1"]
    w2_int8, scale2, b2 = params["w2"], params["scale2"], params["b2"]
    w_out_int8, scale_out, b_out = params["w_out"], params["scale_out"], params["b_out"]

    # Lưu model PyTorch & thông số thực tế
    os.makedirs("mlops/models", exist_ok=True)
    torch.save(sd, "mlops/models/standard_mlp_lite.pt")
    weight_flash_bytes = int(w1_int8.size + w2_int8.size + w_out_int8.size)
    bias_flash_bytes = int((b1.size + b2.size + b_out.size) * np.dtype(np.float32).itemsize)
    normalization_flash_bytes = int((x_min.size + x_max.size) * np.dtype(np.float32).itemsize)
    scale_flash_bytes = 3 * np.dtype(np.float32).itemsize
    model_constants_bytes = (
        weight_flash_bytes + bias_flash_bytes + normalization_flash_bytes + scale_flash_bytes
    )
    with open("mlops/models/standard_mlp_lite_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_params": total_params,
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(validation_indices)),
            "simulation_train_samples": int(simulation_train_count),
            "epa_train_samples": int(epa_train_count),
            "simulation_test_samples": int(simulation_test_count),
            "epa_test_samples": int(epa_test_count),
            "best_validation_epoch": int(best_epoch),
            "best_validation_score": round(float(best_validation_score), 5),
            "validation_score_metric": "weighted_r2 (0.75 simulation, 0.25 EPA)",
            "test_samples": int(len(X_test)),
            "weight_quantization": "symmetric per-layer INT8; float32 activations and biases",
            "training_objective": "gate-weighted direct C SmoothL1 + firmware-path PM SmoothL1",
            "flash_footprint_bytes": weight_flash_bytes,
            "weight_flash_bytes": weight_flash_bytes,
            "bias_flash_bytes": bias_flash_bytes,
            "normalization_flash_bytes": normalization_flash_bytes,
            "scale_flash_bytes": scale_flash_bytes,
            "model_constants_bytes": model_constants_bytes,
            "test_r2": round(test_metrics["r2"], 5),
            "test_mae_ugm3": round(test_metrics["mae"], 5),
            "test_rmse_ugm3": round(test_metrics["rmse"], 5),
            "metrics_overall_test": {
                key: round(float(value), 5) for key, value in test_metrics.items()
            },
            "metrics_simulation_test": {
                key: round(float(value), 5)
                for key, value in simulation_test_metrics.items()
            },
            "metrics_epa_test": {
                key: round(float(value), 5)
                for key, value in epa_test_metrics.items()
            },
            "metrics_validation_simulation": {
                key: round(float(value), 5)
                for key, value in best_validation_simulation_metrics.items()
            },
            "metrics_validation_epa": {
                key: round(float(value), 5)
                for key, value in best_validation_epa_metrics.items()
            },
            "x_min": x_min.tolist(),
            "x_max": x_max.tolist()
        }, f, indent=2)

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
        f.write("static const float TINYML_FEATURE_MIN[TINYML_INPUT_DIM] = {\n    ")
        f.write(", ".join(c_float(x) for x in x_min) + "\n};\n\n")
        f.write("static const float TINYML_X_MAX[TINYML_INPUT_DIM] = {\n    ")
        f.write(", ".join(c_float(x) for x in x_max) + "\n};\n\n")

        # Layer 1: fc1 (8 -> 32)
        f.write(f"// Tầng 1: fc1 (8 -> 32)\n")
        f.write(f"static const float FC1_SCALE = {c_float(scale1)};\n")
        f.write("static const float FC1_BIAS[TINYML_HIDDEN1_DIM] = {\n")
        for i in range(0, 32, 8):
            f.write("    " + ", ".join(c_float(b) for b in b1[i:i+8]) + ",\n")
        f.write("};\n")
        f.write("static const int8_t FC1_WEIGHT[TINYML_HIDDEN1_DIM][TINYML_INPUT_DIM] = {\n")
        for row in w1_int8:
            f.write("    { " + ", ".join(f"{int(v):4d}" for v in row) + " },\n")
        f.write("};\n\n")

        # Layer 2: fc2 (32 -> 16)
        f.write(f"// Tầng 2: fc2 (32 -> 16)\n")
        f.write(f"static const float FC2_SCALE = {c_float(scale2)};\n")
        f.write("static const float FC2_BIAS[TINYML_HIDDEN2_DIM] = {\n")
        for i in range(0, 16, 8):
            f.write("    " + ", ".join(c_float(b) for b in b2[i:i+8]) + ",\n")
        f.write("};\n")
        f.write("static const int8_t FC2_WEIGHT[TINYML_HIDDEN2_DIM][TINYML_HIDDEN1_DIM] = {\n")
        for row in w2_int8:
            f.write("    { " + ", ".join(f"{int(v):4d}" for v in row) + " },\n")
        f.write("};\n\n")

        # Layer Output: fc_out (16 -> 1)
        f.write(f"// Tầng 3: fc_out (16 -> 1)\n")
        f.write(f"static const float FC_OUT_SCALE = {c_float(scale_out)};\n")
        f.write(f"static const float FC_OUT_BIAS[1] = {{ {c_float(b_out[0])} }};\n")
        f.write("static const int8_t FC_OUT_WEIGHT[1][TINYML_HIDDEN2_DIM] = {\n")
        f.write("    { " + ", ".join(f"{int(v):4d}" for v in w_out_int8[0]) + " }\n")
        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_MULTISENSOR_H\n")

    print(f" -> Đã xuất thành công model C++ nhúng vào {header_path}!")
    print(
        f" -> Weights: {weight_flash_bytes} bytes INT8; "
        f"all model constants: {model_constants_bytes} bytes."
    )


if __name__ == "__main__":
    main()
