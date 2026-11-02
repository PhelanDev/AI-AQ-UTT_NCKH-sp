#!/usr/bin/env python3
"""
Lượng tử hóa mô hình MultiSensorResMLP (8D -> 96 -> 96 -> 96 -> 1) sang Full INT8
và xuất tệp C/C++ Header nhúng (model_data_multisensor.h & model_data.h) cho vi điều khiển ESP32-S3.
"""

import sys
import os
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
    out_header: str = "esp/include/model_data.h"
):
    print(f" -> Nạp mô hình Multi-Sensor từ: {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    with open(params_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    x_min = cfg["x_min"]
    x_max = cfg["x_max"]

    w_in_int8, scale_in = quantize_matrix(state_dict["fc_in.weight"])
    b_in = state_dict["fc_in.bias"].cpu().numpy()

    w_b1_int8, scale_b1 = quantize_matrix(state_dict["block1.weight"])
    b_b1 = state_dict["block1.bias"].cpu().numpy()

    w_b2_int8, scale_b2 = quantize_matrix(state_dict["block2.weight"])
    b_b2 = state_dict["block2.bias"].cpu().numpy()

    w_out_int8, scale_out = quantize_matrix(state_dict["fc_out.weight"])
    b_out = state_dict["fc_out.bias"].cpu().numpy()

    total_weights_bytes = w_in_int8.size + w_b1_int8.size + w_b2_int8.size + w_out_int8.size
    print(f" -> Kích thước ma trận trọng số INT8: {total_weights_bytes:,} Bytes (~{total_weights_bytes/1024.0:.2f} KB)")

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
        f.write(f"static const float FC_IN_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_in)}\n}};\n")
        f.write(f"static const int8_t FC_IN_WEIGHT[TINYML_HIDDEN_DIM][TINYML_INPUT_DIM] = {{\n")
        f.write(format_int8_array(w_in_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 1: block1 (96 -> 96) + Skip Connection h0\n")
        f.write(f"static const float FC_BLOCK1_SCALE = {scale_b1:.9f}f;\n")
        f.write(f"static const float FC_BLOCK1_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_b1)}\n}};\n")
        f.write(f"static const int8_t FC_BLOCK1_WEIGHT[TINYML_HIDDEN_DIM][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b1_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 2: block2 (96 -> 96) + Skip Connection h1\n")
        f.write(f"static const float FC_BLOCK2_SCALE = {scale_b2:.9f}f;\n")
        f.write(f"static const float FC_BLOCK2_BIAS[TINYML_HIDDEN_DIM] = {{\n{format_1d_array(b_b2)}\n}};\n")
        f.write(f"static const int8_t FC_BLOCK2_WEIGHT[TINYML_HIDDEN_DIM][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b2_int8) + "\n};\n\n")

        f.write(f"// Tầng Output: fc_out (96 -> 1)\n")
        f.write(f"static const float FC_OUT_SCALE = {scale_out:.9f}f;\n")
        f.write(f"static const float FC_OUT_BIAS[1] = {{ {b_out[0]:.9f}f }};\n")
        f.write(f"static const int8_t FC_OUT_WEIGHT[1][TINYML_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_out_int8) + "\n};\n\n")

        f.write("#endif // MODEL_DATA_MULTISENSOR_H\n")

    print(f" -> Hoàn tất xuất tệp C Header: {out_header} ({os.path.getsize(out_header)/1024.0:.1f} KB text)")

if __name__ == "__main__":
    export_multisensor_int8()
