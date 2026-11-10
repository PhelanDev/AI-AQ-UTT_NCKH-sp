"""
Test Suite kiểm tra Mô hình AI Phân Tầng Thống Nhất (Hierarchical Unified Central AI)
Gộp 2 mô hình (Dự báo 24h + Phát hiện Dị thường) thành 1 mô hình duy nhất trên Raspberry Pi 5.
"""

import os
import pytest
import numpy as np
import torch

def test_unified_model_architecture_and_export():
    """Kiểm tra mô hình gộp phân tầng có cả 2 đầu ra (Forecast & Recon) và xuất ONNX chuẩn xác."""
    pt_path = "mlops/models/central_unified_ai.pt"
    onnx_path = "mlops/models/central_unified_ai.onnx"

    assert os.path.exists(pt_path), f"File {pt_path} chua duoc huan luyen!"
    assert os.path.exists(onnx_path), f"File {onnx_path} chua duoc xuat!"

    import sys
    sys.path.insert(0, os.path.abspath("."))
    from pi.unified_model import HierarchicalCentralAI
    import onnxruntime as ort

    model = HierarchicalCentralAI(in_features=16, tcn_channels=32, gru_hidden=32, out_steps=24, out_targets=2)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_in = np.random.randn(1, 24, 16).astype(np.float32)

    with torch.no_grad():
        py_forecast, py_recon = model(torch.tensor(dummy_in))
        py_forecast_np = py_forecast.numpy()
        py_recon_np = py_recon.numpy()

    # Kiểm tra kích thước đầu ra 2 tầng
    assert py_forecast_np.shape == (1, 24, 2), f"Forecast shape sai: {py_forecast_np.shape}"
    assert py_recon_np.shape == (1, 16), f"Recon shape sai: {py_recon_np.shape}"

    # Kiểm tra ONNX parity
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_forecast, onnx_recon = sess.run(None, {"input_timeseries": dummy_in})

    diff_forecast = float(np.max(np.abs(py_forecast_np - onnx_forecast)))
    diff_recon = float(np.max(np.abs(py_recon_np - onnx_recon)))

    assert diff_forecast < 1e-4, f"Forecast Parity diff qua lon: {diff_forecast}"
    assert diff_recon < 1e-4, f"Recon Parity diff qua lon: {diff_recon}"
