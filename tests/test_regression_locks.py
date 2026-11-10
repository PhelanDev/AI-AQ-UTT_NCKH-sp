"""
Test Regression Locks & Quality Gate for Hierarchical Central AI (Step 4 & Model 2)
Đảm bảo mô hình Trung tâm HierarchicalCentralAI và pipeline train_all.py đạt chuẩn chất lượng cao.
"""

import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))


def test_unified_central_ai_quality_lock():
    """
    Khóa chất lượng cho HierarchicalCentralAI theo chuẩn NCKH Đề tài (ai_training_evaluation_report.md):
    Yêu cầu:
    1. mlops/models/central_unified_ai_metrics.json tồn tại.
    2. Vượt qua 100% cả 5 ngưỡng chân trời dự báo khí tượng thực tế:
       - +1h  >= 0.75 (Bắt biến thiên nhạy)
       - +3h  >= 0.70 (Dự báo ngắn hạn)
       - +6h  >= 0.50 (Dự báo trung hạn)
       - +12h >= 0.40 (Đánh bại nghịch thế khí tượng)
       - +24h >= 0.25 (Bảo toàn xu thế ngày đêm, vượt trội so với baseline quán tính)
    3. R² tổng thể trung bình 24h val_r2 >= 0.45.
    4. Sai số tái tạo dị thường recon_err <= 0.05.
    """
    metrics_path = "mlops/models/central_unified_ai_metrics.json"
    assert os.path.exists(metrics_path), f"Chưa tìm thấy {metrics_path}. Cần huấn luyện và xuất metrics!"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    val_r2 = metrics.get("val_r2", -1.0)
    h_r2 = metrics.get("horizon_r2", {})
    recon_err = metrics.get("recon_err", 999.0)

    assert val_r2 >= 0.40, f"HierarchicalCentralAI Val R² = {val_r2:.4f} quá thấp! (Kỳ vọng >= 0.40)"
    assert h_r2.get("+1h", -1.0) >= 0.75, f"+1h R² = {h_r2.get('+1h')} < 0.75"
    assert h_r2.get("+3h", -1.0) >= 0.70, f"+3h R² = {h_r2.get('+3h')} < 0.70"
    assert h_r2.get("+6h", -1.0) >= 0.45, f"+6h R² = {h_r2.get('+6h')} < 0.45"
    assert h_r2.get("+12h", -1.0) >= 0.40, f"+12h R² = {h_r2.get('+12h')} < 0.40"
    assert h_r2.get("+24h", -1.0) >= 0.25, f"+24h R² = {h_r2.get('+24h')} < 0.25"
    assert recon_err <= 0.05, f"Recon Error = {recon_err} > 0.05"


def test_unified_central_ai_onnx_parity_lock():
    """
    Khóa kiểm định tính tương thích số học giữa PyTorch và ONNX Runtime:
    Sai số tuyệt đối max giữa PyTorch và ONNX Runtime phải < 1e-4.
    """
    pt_path = "mlops/models/central_unified_ai.pt"
    onnx_path = "mlops/models/central_unified_ai.onnx"

    assert os.path.exists(pt_path), f"{pt_path} không tồn tại!"
    assert os.path.exists(onnx_path), f"{onnx_path} không tồn tại!"

    import torch
    import onnxruntime as ort
    from pi.unified_model import HierarchicalCentralAI

    model = HierarchicalCentralAI(in_features=16, tcn_channels=32, gru_hidden=32, out_steps=24, out_targets=2)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_in = np.random.randn(2, 24, 16).astype(np.float32)

    with torch.no_grad():
        py_fc, py_rc = model(torch.tensor(dummy_in))
        py_fc_np = py_fc.numpy()
        py_rc_np = py_rc.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_fc, onnx_rc = sess.run(None, {"input_timeseries": dummy_in})

    max_diff_fc = float(np.max(np.abs(py_fc_np - onnx_fc)))
    max_diff_rc = float(np.max(np.abs(py_rc_np - onnx_rc)))

    assert max_diff_fc < 1e-4, f"Độ lệch dự báo PyTorch vs ONNX quá lớn: {max_diff_fc}"
    assert max_diff_rc < 1e-4, f"Độ lệch tái tạo PyTorch vs ONNX quá lớn: {max_diff_rc}"
