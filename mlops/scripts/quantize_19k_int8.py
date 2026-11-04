#!/usr/bin/env python3
"""
quantize_19k_int8.py
Lượng tử hóa mô hình Robust Deep Residual TinyML 19.297 tham số sang INT8
và xuất tệp C/C++ Header nhúng (model_data.h) cho vi điều khiển ESP32-S3 (Xtensa LX7).
- Dung lượng bảng trọng số: 19.008 Bytes (~18.56 KB)
- Toàn bộ mô hình nằm gọn trong bộ nhớ đệm nhanh L1 SRAM của ESP32-S3.
- Tiêu thụ Heap RAM: 0 Bytes (Thuần bộ nhớ tĩnh và ngăn xếp Stack < 1 KB).
- Thích ứng miền dữ liệu thực tế Quận Thanh Xuân, Hà Nội.
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

def format_1d_array(arr, indent="    ", per_line=10, fmt="%.6ff"):
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

def export_c_header_19k(model_path: str, params_json: str, output_header: str):
    print(f" -> Đang nạp mô hình 19k từ: {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")

    with open(params_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    x_min = cfg["x_min"]
    x_max = cfg["x_max"]

    w_in_int8, scale_in = quantize_matrix(state_dict["fc_in.weight"])
    b_in = state_dict["fc_in.bias"].cpu().numpy()

    w_b1_int8, scale_b1 = quantize_matrix(state_dict["fc_block1.weight"])
    b_b1 = state_dict["fc_block1.bias"].cpu().numpy()

    w_b2_int8, scale_b2 = quantize_matrix(state_dict["fc_block2.weight"])
    b_b2 = state_dict["fc_block2.bias"].cpu().numpy()

    w_out_int8, scale_out = quantize_matrix(state_dict["fc_out.weight"])
    b_out = state_dict["fc_out.bias"].cpu().numpy()

    total_weights_bytes = w_in_int8.size + w_b1_int8.size + w_b2_int8.size + w_out_int8.size
    print(f" -> Kích thước ma trận trọng số INT8: {total_weights_bytes:,} Bytes (~{total_weights_bytes/1024.0:.2f} KB)")

    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    print(f" -> Đang xuất file Header C++ nhúng vào: {output_header}...")
    with open(output_header, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * model_data_19k.h\n")
        f.write(" * Trọng số lượng tử hóa INT8 cho mạng Robust Residual TinyML 19.297 tham số.\n")
        f.write(" * Tối ưu hóa triệt tiêu học vẹt và đã Domain Adaptation trên dữ liệu thực tế Thanh Xuân, Hà Nội.\n")
        f.write(" * Tự động sinh bởi: quantize_19k_int8.py\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_19K_H\n")
        f.write("#define MODEL_DATA_19K_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define TINYML_19K_TOTAL_PARAMS 19297\n")
        f.write("#define TINYML_19K_INPUT_DIM 5\n")
        f.write("#define TINYML_19K_HIDDEN_DIM 96\n")
        f.write("#define TINYML_19K_NUM_RES_BLOCKS 2\n\n")

        f.write("// Ngưỡng Min/Max chuẩn hóa dữ liệu đầu vào [pm_avg, diff_pm, temp, rh, dew_point]\n")
        f.write(f"static const float TINYML_19K_X_MIN[TINYML_19K_INPUT_DIM] = {{\n{format_1d_array(x_min, per_line=5)}\n}};\n\n")
        f.write(f"static const float TINYML_19K_X_MAX[TINYML_19K_INPUT_DIM] = {{\n{format_1d_array(x_max, per_line=5)}\n}};\n\n")

        f.write(f"// Tầng 1: fc_in (5 -> 96)\n")
        f.write(f"static const float FC19K_IN_SCALE = {scale_in:.9f}f;\n")
        f.write(f"static const float FC19K_IN_BIAS[TINYML_19K_HIDDEN_DIM] = {{\n{format_1d_array(b_in)}\n}};\n")
        f.write(f"static const int8_t FC19K_IN_WEIGHT[TINYML_19K_HIDDEN_DIM][TINYML_19K_INPUT_DIM] = {{\n")
        f.write(format_int8_array(w_in_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 1: fc_block1 (96 -> 96) + Skip Connection h0\n")
        f.write(f"static const float FC19K_BLOCK1_SCALE = {scale_b1:.9f}f;\n")
        f.write(f"static const float FC19K_BLOCK1_BIAS[TINYML_19K_HIDDEN_DIM] = {{\n{format_1d_array(b_b1)}\n}};\n")
        f.write(f"static const int8_t FC19K_BLOCK1_WEIGHT[TINYML_19K_HIDDEN_DIM][TINYML_19K_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b1_int8) + "\n};\n\n")

        f.write(f"// Khối Residual 2: fc_block2 (96 -> 96) + Skip Connection h1\n")
        f.write(f"static const float FC19K_BLOCK2_SCALE = {scale_b2:.9f}f;\n")
        f.write(f"static const float FC19K_BLOCK2_BIAS[TINYML_19K_HIDDEN_DIM] = {{\n{format_1d_array(b_b2)}\n}};\n")
        f.write(f"static const int8_t FC19K_BLOCK2_WEIGHT[TINYML_19K_HIDDEN_DIM][TINYML_19K_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_b2_int8) + "\n};\n\n")

        f.write(f"// Tầng Output: fc_out (96 -> 1)\n")
        f.write(f"static const float FC19K_OUT_SCALE = {scale_out:.9f}f;\n")
        f.write(f"static const float FC19K_OUT_BIAS[1] = {{ {b_out[0]:.9f}f }};\n")
        f.write(f"static const int8_t FC19K_OUT_WEIGHT[1][TINYML_19K_HIDDEN_DIM] = {{\n")
        f.write(format_int8_array(w_out_int8) + "\n};\n\n")

        f.write("#endif // MODEL_DATA_19K_H\n")

    print(f" -> Hoàn tất xuất tệp C Header: {output_header} ({os.path.getsize(output_header)/1024.0:.1f} KB text)")

def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "models", "robust_resmlp_19k.pt")
    params_path = os.path.join(base_dir, "..", "models", "robust_resmlp_19k_params.json")
    out_path = os.path.join(base_dir, "..", "..", "esp", "include", "model_data_19k.h")
    export_c_header_19k(model_path, params_path, out_path)

if __name__ == "__main__":
    main()
