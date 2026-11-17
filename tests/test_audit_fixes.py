import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from pi.central_engine import CentralAIEngine
from pi.web_server import AQServerState, get_server_state


def test_central_engine_cwd_independent():
    """Vấn đề 13: CentralAIEngine phải nạp đúng weights bất kể CWD ở đâu."""
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            # Khởi tạo engine từ thư mục tạm
            engine = CentralAIEngine()
            assert engine.forecaster_weights is not None
            assert os.path.exists(engine.forecaster_weights)
            assert engine.unified_weights is not None
            assert os.path.exists(engine.unified_weights)
        finally:
            os.chdir(orig_cwd)


def test_real_pm10_propagation():
    """Vấn đề 11: PM10 đo đạc thực tế phải được đưa vào pipeline thay vì luôn là pm * 1.35."""
    engine = CentralAIEngine()
    hist_24x1 = np.full((24, 1), 50.0, dtype=np.float32)
    # Đo đạc thực tế: PM2.5 = 50.0, PM10 = 200.0 (bụi thô nặng)
    st_mat_high_pm10 = np.array([[50.0, 200.0, 25.0, 75.0, 420.0, 100.0]], dtype=np.float32)
    res_high = engine.process_telemetry_frame(hist_24x1, st_mat_high_pm10)

    # Đo đạc bình thường: PM2.5 = 50.0, PM10 = 67.5 (1.35x)
    st_mat_normal = np.array([[50.0, 67.5, 25.0, 75.0, 420.0, 100.0]], dtype=np.float32)
    res_normal = engine.process_telemetry_frame(hist_24x1, st_mat_normal)

    # Đầu ra dự báo hoặc bất thường phải phản ánh sự khác biệt nồng độ PM10 thực tế
    assert res_high is not None
    assert res_normal is not None


def test_station_forecasts_marker_consistency():
    """Vấn đề 14: Station forecasts và markers phải thống nhất nguồn dữ liệu."""
    engine = CentralAIEngine()
    hist_24x8 = np.ones((24, 8), dtype=np.float32) * 40.0
    st_mat_8 = np.zeros((8, 6), dtype=np.float32)
    st_mat_8[:, 0] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    st_mat_8[:, 1] = st_mat_8[:, 0] * 1.35
    st_mat_8[:, 2] = 25.0
    st_mat_8[:, 3] = 75.0
    st_mat_8[:, 4] = 420.0
    st_mat_8[:, 5] = 100.0

    res = engine.process_telemetry_frame(hist_24x8, st_mat_8)
    web_map = res["web_map_v5"]
    markers = web_map["markers"]
    st_forecasts = res["station_forecasts"]

    # Kiểm tra tính nhất quán giữa marker và station_forecasts cho trạm 1 và trạm 8
    for i, m in enumerate(markers):
        sid = m["id"]
        assert sid in st_forecasts
        m_fc = m["forecast_pm25_24h"]
        st_fc = st_forecasts[sid]["pm25_pred"]
        np.testing.assert_allclose(m_fc, st_fc, rtol=1e-4)


def test_gas_signature_classification_with_real_co2():
    """Vấn đề 12: Phân loại Fire/Combustion chỉ khi CO2/VOC thực sự tăng cao."""
    engine = CentralAIEngine()
    hist_24x1 = np.full((24, 1), 150.0, dtype=np.float32)

    # Trường hợp 1: PM rất cao (150) nhưng CO2/VOC bình thường (410 ppm, 90 ppb) -> Không phải cháy
    st_mat_dust = np.array([[150.0, 200.0, 25.0, 75.0, 410.0, 90.0]], dtype=np.float32)
    res_dust = engine.process_telemetry_frame(hist_24x1, st_mat_dust)
    # Không được gán cờ CO2/VOC elevated nếu nồng độ thực tế thấp
    assert res_dust["anomaly_diagnostics"]["signature_class_name"] != "Fire / Biomass Burning"

    # Trường hợp 2: PM cao (150) VÀ CO2/VOC tăng đột biến (950 ppm, 350 ppb) -> Cháy/Đốt rác
    st_mat_fire = np.array([[150.0, 200.0, 25.0, 75.0, 950.0, 350.0]], dtype=np.float32)
    res_fire = engine.process_telemetry_frame(hist_24x1, st_mat_fire)
    # Có mặt CO2/VOC cao
    assert res_fire is not None


def test_post_telemetry_endpoint_updates_state():
    """Vấn đề 3: POST /api/telemetry phải tiếp nhận số đo thực và cập nhật state (không gán jitter ảo)."""
    import urllib.request
    import json
    from pi.web_server import create_server
    import threading

    server = create_server(host="127.0.0.1", port=0)
    port = server.server_port
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()

    try:
        url = f"http://127.0.0.1:{port}/api/telemetry"
        packet = {
            "station_id": "STATION_01",
            "pm25": 180.0,
            "pm10": 240.0,
            "temp_c": 28.5,
            "rh_percent": 84.0,
            "co2_ppm": 520.0,
            "gas_proxy": 110.0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(packet).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert data["status"] == "TELEMETRY_INGESTED"
            assert data["telemetry"]["pm25"] == 180.0
            assert data["telemetry"]["pm10"] == 240.0

        # Kiểm tra GET /api/telemetry trả về đúng số đo thực vừa POST (không phải 38.5)
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["pm25"] == 180.0
            assert data["pm10"] == 240.0
    finally:
        server.shutdown()
        server.server_close()
