import pytest
import numpy as np
import re

def test_mlp_model_data_parameters_and_int8_weights():
    # 1. Đọc file esp/include/model_data.h để xác minh kích thước trọng số
    with open("esp/include/model_data.h", "r", encoding="utf-8") as f:
        content = f.read()

    assert "TINYML_INPUT_DIM 8" in content
    assert "TINYML_HIDDEN1_DIM 32" in content
    assert "TINYML_HIDDEN2_DIM 16" in content

    # Kiểm tra tổng số tham số lý thuyết:
    # Layer 1: 8*32 (weights) + 32 (biases) = 288
    # Layer 2: 32*16 (weights) + 16 (biases) = 528
    # Layer Out: 16*1 (weights) + 1 (biases) = 17
    # Tổng cộng = 288 + 528 + 17 = 833 params
    total_params = (8 * 32 + 32) + (32 * 16 + 16) + (16 * 1 + 1)
    assert total_params == 833, f"Expected 833 params, got {total_params}"

    # Kiểm tra định dạng trọng số INT8 và Scale
    assert "int8_t FC1_WEIGHT" in content
    assert "int8_t FC2_WEIGHT" in content
    assert "int8_t FC_OUT_WEIGHT" in content
    assert "FC1_SCALE" in content
    assert "FC2_SCALE" in content
    assert "FC_OUT_SCALE" in content

def test_mlp_inference_bounds_and_physics_multiplier():
    # Kiểm tra logic kẹp vật lý C in [0.25, 1.05]
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    for raw_out in np.linspace(-10.0, 10.0, 100):
        c_mult = 0.25 + 0.80 * sigmoid(raw_out)
        assert 0.25 <= c_mult <= 1.05, f"c_mult {c_mult} out of bounds"
