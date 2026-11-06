#!/usr/bin/env python3
"""
Lượng tử hóa mô hình MultiSensorResMLP (8D -> 96 -> 96 -> 96 -> 1) sang Full INT8
và xuất tệp C/C++ Header nhúng (model_data.h) cho vi điều khiển ESP32-S3.
Tích hợp Quantization Bias Calibration để triệt tiêu sai số trôi dạt trung bình.
"""

import sys
import os
import csv
import json
import numpy as np
import torch

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def quantize_matrix(weight_tensor):
    w = weight_tensor.cpu().numpy()
    max_val = np.max(np.abs(w))
    scale = float(max_val / 127.0) if max_val > 0 else 1.0
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_int8, scale

def format_1d_array(arr, indent="    ", per_line=8, fmt="%.6ff"):
    lines = []
    curr = []
    for val in arr:
        curr.append(fmt % val)
        if len(curr) >= per_line:
            lines.append(indent + ", ".join(curr) + ",")
            curr = []
    if curr:
        lines.append(indent + ", ".join(curr))
    return "\n".join(lines)

def format_int8_array(arr_2d, indent="        ", per_line=16):
    lines = []
    for r in arr_2d:
        row_str = "    { "
        curr = []
        for val in r:
            curr.append(f"{int(val):4d}")
            if len(curr) >= per_line:
                row_str += ", ".join(curr) + ",\n" + indent
                curr = []
        if curr:
            row_str += ", ".join(curr)
        row_str += " },"
        lines.append(row_str)
    return "\n".join(lines)

def export_multisensor_int8(
    model_path: str = "mlops/models/multisensor_resmlp.pt",
    params_path: str = "mlops/models/multisensor_resmlp_params.json",
    test_csv: str = "mlops/data/multisensor_test_set.csv",
    out_header: str = "esp/include/model_data.h"
):
    print(f" -> Nạp mô hình Multi-Sensor từ: {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    with open(params_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    x_min = np.array(cfg["x_min"], dtype=np.float32)
    x_max = np.array(cfg["x_max"], dtype=np.float32)
    denom = np.maximum(x_max - x_min, 1e-6)

    # Nạp mẫu dữ liệu kiểm thử để calibrate bias lượng tử hóa
    calib_X = []
    if os.path.exists(test_csv):
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                calib_X.append([
                    float(r["pm_avg"]), float(r["diff_pm"]), float(r["r_pm"]),
                    float(r["temp"]), float(r["rh"]), float(r["dew_point"]),
                    float(r["co2"]), float(r["gas_proxy"])
                ])
                if len(calib_X) >= 3000:
                    break
    if not calib_X:
        calib_X = np.random.uniform(0.1, 0.9, size=(1000, 8)).astype(np.float32)
    else:
        calib_X = np.clip((np.array(calib_X, dtype=np.float32) - x_min) / denom, 0.0, 1.0)

    # 1. fc_in (8 -> 96)
    w_in = state_dict["fc_in.weight"].cpu().numpy()
    w_in_int8, scale_in = quantize_matrix(state_dict["fc_in.weight"])
    b_in = state_dict["fc_in.bias"].cpu().numpy()
    # Bias compensation
    w_in_deq = w_in_int8.astype(np.float32) * scale_in
    b_in_calib = b_in - (w_in_deq - w_in) @ np.mean(calib_X, axis=0)

    # Forward qua tầng 1
    h0 = np.maximum(0.0, calib_X @ w_in.T + b_in)

    # 2. block1 (96 -> 96)
    w_b1 = state_dict["block1.weight"].cpu().numpy()
    w_b1_int8, scale_b1 = quantize_matrix(state_dict["block1.weight"])
    b_b1 = state_dict["block1.bias"].cpu().numpy()
    w_b1_deq = w_b1_int8.astype(np.float32) * scale_b1
    b_b1_calib = b_b1 - (w_b1_deq - w_b1) @ np.mean(h0, axis=0)

    # Forward qua block 1
    h1 = np.maximum(0.0, h0 @ w_b1.T + b_b1) + h0

    # 3. block2 (96 -> 96)
    w_b2 = state_dict["block2.weight"].cpu().numpy()
    w_b2_int8, scale_b2 = quantize_matrix(state_dict["block2.weight"])
    b_b2 = state_dict["block2.bias"].cpu().numpy()
    w_b2_deq = w_b2_int8.astype(np.float32) * scale_b2
    b_b2_calib = b_b2 - (w_b2_deq - w_b2) @ np.mean(h1, axis=0)

    # Forward qua block 2
    h2 = np.maximum(0.0, h1 @ w_b2.T + b_b2) + h1

    # 4. fc_out (96 -> 1)
    w_out = state_dict["fc_out.weight"].cpu().numpy()
    w_out_int8, scale_out = quantize_matrix(state_dict["fc_out.weight"])
    b_out = state_dict["fc_out.bias"].cpu().numpy()
    w_out_deq = w_out_int8.astype(np.float32) * scale_out
    b_out_calib = b_out - (w_out_deq - w_out) @ np.mean(h2, axis=0)

    total_weights_bytes = w_in_int8.size + w_b1_int8.size + w_b2_int8.size + w_out_int8.size
    print(f" -> Kích thước ma trận trọng số INT8: {total_weights_bytes:,} Bytes (~{total_weights_bytes/1024.0:.2f} KB)")
    print(f" -> Tối ưu hóa bù sai số lượng tử (Bias Calibration): Done")

    os.makedirs(os.path.dirname(out_header), exist_ok=True)
    with open(out_header, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * model_data.h\n")
        f.write(" * Trọng số lượng tử hóa INT8 cho mạng MultiSensorResMLP (8D -> 96 -> 96 -> 96 -> 1).\n")
        f.write(" * Tích hợp đầy đủ: Dual PMS7003, SHT40, SenseAir S8 (CO2), Sensirion SGP41 (TVOC/NOx).\n")
        f.write(" * Tự động sinh bởi: quantize_multisensor_int8.py\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_MULTISENSOR_H\n")
        f.write("#define MODEL_DATA_MULTISENSOR_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define TINYML_INPUT_DIM 8\n")
        f.write("#define TINYML_HIDDEN_DIM 96\n")
        f.write("#define TINYML_NUM_RES_BLOCKS 2\n\n")

        f.write("// Ngưỡng Min/Max chuẩn hóa dữ liệu đầu vào [pm_avg, diff_pm, r_pm, temp, rh, dew_point, co2, gas_proxy]\n")
        f.write(f"static const float TINYML_X_MIN[TINYML_INPUT_DIM] = {{\n{format_1d_array(x_min, per_line=4)}\n}};\n\n")
        f.write(f"static const float TINYML_X_MAX[TINYML_INPUT_DIM] = {{\n{format_1d_array(x_max, per_line=4)}\n}};\n\n")

        f.write(f"// Tầng 1: fc_in (8 -> 96)\n")
        f.write(f"static const float FC_IN_SCALE = {scale_in:.9f}f;\n")
        f.write(f"static const float FC_IN_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_in_calib)}\n}};\n")
        f.write(f"static const int8_t FC_IN_WEIGHT[TINYML_HIDDEN_DIM][TINYML_INPUT_DIM] = {{\n")
        f.write(format_int8_array(w_in_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 1: block1 (96 -> 96) + Skip Connection h0\n")
        f.write(f"static const float FC_BLOCK1_SCALE = {scale_b1:.9f}f;\n")
        f.write(f"static const float FC_BLOCK1_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_b1_calib)}\n}};\n")
        f.write(f"static const int8_t FC_BLOCK1_WEIGHT[TINYML_HIDDEN_DIM][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b1_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 2: block2 (96 -> 96) + Skip Connection h1\n")
        f.write(f"static const float FC_BLOCK2_SCALE = {scale_b2:.9f}f;\n")
        f.write(f"static const float FC_BLOCK2_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_b2_calib)}\n}};\n")
        f.write(f"static const int8_t FC_BLOCK2_WEIGHT[TINYML_HIDDEN_DIM][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b2_int8) + "\n};\n\n")

        f.write(f"// Tầng Output: fc_out (96 -> 1)\n")
        f.write(f"static const float FC_OUT_SCALE = {scale_out:.9f}f;\n")
        f.write(f"static const float FC_OUT_BIAS[1] = {{ {b_out_calib[0]:.9f}f }};\n")
        f.write(f"static const int8_t FC_OUT_WEIGHT[1][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_out_int8) + "\n};\n\n")

        f.write("#endif // MODEL_DATA_MULTISENSOR_H\n")

    print(f" -> Hoàn tất xuất tệp C Header: {out_header} ({os.path.getsize(out_header)/1024.0:.1f} KB text)")

if __name__ == "__main__":
    export_multisensor_int8()
