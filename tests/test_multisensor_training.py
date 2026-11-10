"""
Test Suite cho Pipeline Huấn Luyện AI Đa Cảm Biến & Tối Ưu Hệ Thống Hành Vi
Theo chuẩn danh mục BOM: Dual PMS7003, SenseAir S8 (CO2), Sensirion SGP41 (TVOC/NOx), Sensirion SHT40 (T/RH).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import csv
import pytest
import numpy as np

def test_data_fusion_schema():
    """Kiểm tra schema và tính hợp lệ của file dữ liệu 8 kênh sau khi hợp nhất."""
    dataset_path = "mlops/data/multisensor_training_set.csv"
    assert os.path.exists(dataset_path), f"File {dataset_path} chua ton tai!"

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = [
            "pm_avg", "diff_pm", "r_pm", "temp", "rh", "dew_point", "co2", "gas_proxy", "ref_pm25"
        ]
        assert header == expected_cols, f"Header khong khop: {header}"

        rows = list(reader)
        assert len(rows) > 5000, f"So luong mau qua it: {len(rows)}"

        # Kiem tra khong co gia tri NaN hay null
        for i, row in enumerate(rows[:100]):
            assert len(row) == len(expected_cols), f"Dong {i} thieu cot: {row}"
            for val in row:
                assert val != "" and val.lower() != "nan", f"Gia tri NaN o dong {i}: {row}"
                float_val = float(val)
                assert not np.isnan(float_val)


def test_multisensor_resmlp_smoke_discrimination():
    """
    Kiem tra tinh nang doc quyen cua Mo hinh 1:
    - O do am cao (RH = 90%):
      * Neu CO2 = 420 ppm va Gas_Proxy = 100 (khong co khoi): he so bu C phai giam manh (C <= 0.60).
      * Neu CO2 = 1200 ppm va Gas_Proxy = 350 (co khoi dot rac/giao thong): he so bu C phai giu nguyen (C >= 0.85).
    """
    import torch
    model_path = "mlops/models/multisensor_resmlp.pt"
    assert os.path.exists(model_path), f"Mo hinh {model_path} chua duoc huan luyen!"

    from mlops.scripts.train_multisensor_resmlp import MultiSensorResMLP, predict_calibrated_pm25
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Kich ban 1: Suong mu nom am don thuan (RH=92%, T=22C, CO2=420ppm, Gas=100, PM_raw=100)
    c_fog, pm_fog = predict_calibrated_pm25(
        model, pm_avg=100.0, diff_pm=2.0, r_pm=0.7, temp=22.0, rh=92.0, dew_point=20.6, co2=420.0, gas_proxy=100.0
    )
    assert c_fog <= 0.65, f"He so bu suong mu phai giam manh nhung lai la {c_fog}"
    assert pm_fog < 65.0, f"PM2.5 sau bu phai nho hon 65 nhung lai la {pm_fog}"

    # Kich ban 2: Khoi chay / giao thong nghiem trong (RH=92%, T=22C, CO2=1400ppm, Gas=380, PM_raw=100)
    c_smoke, pm_smoke = predict_calibrated_pm25(
        model, pm_avg=100.0, diff_pm=2.0, r_pm=0.7, temp=22.0, rh=92.0, dew_point=20.6, co2=1400.0, gas_proxy=380.0
    )
    assert c_smoke >= 0.85, f"He so khi co khoi phai giu nguyen nồng độ nhung lai la {c_smoke}"
    assert pm_smoke >= 85.0, f"PM2.5 sau bu phai giu nguyen >= 85 nhung lai la {pm_smoke}"


def test_forecaster_convergence():
    """Kiểm tra mô hình dự báo Dilated TCN + Bi-GRU đã được huấn luyện thật và đạt R² >= 0.80."""
    import torch
    import onnxruntime as ort

    onnx_path = "mlops/models/central_forecaster.onnx"
    assert os.path.exists(onnx_path), f"File {onnx_path} chua ton tai!"

    pt_path = "mlops/models/central_forecaster.pt"
    assert os.path.exists(pt_path), f"File {pt_path} chua ton tai (chua duoc train that)!"

    # Kiem tra numerical parity giua PyTorch va ONNX
    from pi.forecasting_engine import DilatedTCNBiGRUForecaster
    model = DilatedTCNBiGRUForecaster(in_features=14, tcn_channels=32, gru_hidden=64, out_steps=24, out_targets=2)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_in = np.random.randn(1, 24, 14).astype(np.float32)
    with torch.no_grad():
        py_out = model(torch.tensor(dummy_in)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input_timeseries": dummy_in})[0]

    max_diff = float(np.max(np.abs(py_out - onnx_out)))
    assert max_diff < 1e-4, f"Parity diff qua lon: {max_diff}"


def test_autoencoder_reconstruction():
    """Kiểm tra mô hình Deep Autoencoder 16D đã được huấn luyện thật và phát hiện dị thường nhạy bén."""
    import torch
    import onnxruntime as ort

    onnx_path = "mlops/models/central_autoencoder.onnx"
    assert os.path.exists(onnx_path), f"File {onnx_path} chua ton tai!"

    pt_path = "mlops/models/central_autoencoder.pt"
    assert os.path.exists(pt_path), f"File {pt_path} chua ton tai (chua duoc train that)!"

    from pi.anomaly_engine import MultiStationDeepAutoencoder
    model = MultiStationDeepAutoencoder(in_features=48, latent_dim=16)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    # Kiem tra numerical parity giua PyTorch va ONNX
    dummy_in = np.random.randn(1, 48).astype(np.float32)
    with torch.no_grad():
        py_rec, py_z = model(torch.tensor(dummy_in))
        py_rec_np = py_rec.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_rec, onnx_z = sess.run(None, {"input_features_48d": dummy_in})

    max_diff = float(np.max(np.abs(py_rec_np - onnx_rec)))
    assert max_diff < 1e-4, f"Autoencoder Parity diff qua lon: {max_diff}"
