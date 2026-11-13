"""
KIỂM THỬ HỆ THỐNG V5.0: CO GIÃN ĐỘNG 1 -> N TRẠM, MÔ HÌNH HỢP NHẤT 43.5K & BẢN ĐỒ NHIỆT WEB 2D
Đảm bảo:
1. Chế độ 1 trạm Prototype (N = 1): Chỉ hiển thị đúng 1 trạm thật, không bịa trạm ảo, vẽ quầng nhiệt kèm opacity_grid.
2. Chế độ mở rộng (N = 3, N = 8): Tự động nối màu Kriging giữa N trạm thật và mở rộng vùng phủ opacity_grid.
3. Tích hợp HierarchicalCentralAI (43.552 tham số) vào CentralAIEngine dự báo 24h & bắt lỗi cho từng trạm.
"""

import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.config import STATION_COORDS_UTM, STATION_COORDS_WGS84, STATION_IDS
from pi.spatial_temporal_engine import (
    SpatialTemporalEngine,
    generate_web_heatmap,
)
from pi.central_engine import CentralAIEngine


def test_v5_01_single_station_heatmap_and_opacity_decay():
    """Khi chỉ có 1 trạm thật (N=1), chỉ có đúng 1 marker, tâm trạm opacity=1.0 và mờ dần khi ra xa."""
    single_val = np.array([85.0], dtype=np.float32)
    single_coord_utm = STATION_COORDS_UTM[:1]
    single_coord_wgs = STATION_COORDS_WGS84[:1]
    single_id = STATION_IDS[:1]

    hm = generate_web_heatmap(
        station_values=single_val,
        coordinates_utm=single_coord_utm,
        coordinates_wgs84=single_coord_wgs,
        station_ids=single_id,
        wind_vector=np.array([0.0, 0.0]),
        grid_size=25,
        radius_m=1000.0,
    )

    assert hm["num_active_stations"] == 1
    assert len(hm["markers"]) == 1
    assert hm["markers"][0]["station_id"] == "STATION_01"
    assert abs(hm["markers"][0]["pm25"] - 85.0) < 1e-4

    grid_pm25 = np.asarray(hm["grid_pm25"])
    opacity = np.asarray(hm["opacity_grid"])
    assert grid_pm25.shape == (25, 25)
    assert opacity.shape == (25, 25)
    assert np.all(np.isfinite(grid_pm25))
    assert np.all((opacity >= 0.0) & (opacity <= 1.0))

    # Tại điểm lưới gần trạm 1 nhất: opacity xấp xỉ 1.0 và nồng độ sát 85.0
    assert float(np.max(opacity)) > 0.95
    max_idx = np.unravel_index(np.argmax(opacity), opacity.shape)
    assert abs(float(grid_pm25[max_idx]) - 85.0) < 3.0

    # Ở rìa xa bản đồ (cách > 2.5km): opacity phải mờ đi rõ rệt (< 0.20)
    assert float(np.min(opacity)) < 0.20


def test_v5_02_wind_plume_anisotropy_and_multistation_scaling():
    """Khi có gió, quầng nhiệt vươn theo hướng gió; khi tăng từ 1 -> 3 -> 8 trạm, vùng phủ sóng mở rộng."""
    # Kiểm tra hướng gió Đông (u = +4.0 m/s, v = 0.0)
    hm_wind = generate_web_heatmap(
        station_values=np.array([120.0], dtype=np.float32),
        coordinates_utm=STATION_COORDS_UTM[:1],
        station_ids=STATION_IDS[:1],
        wind_vector=np.array([4.0, 0.0]),
        grid_size=25,
        radius_m=1000.0,
    )
    opacity_wind = np.asarray(hm_wind["opacity_grid"])
    peak_r, peak_c = np.unravel_index(np.argmax(opacity_wind), opacity_wind.shape)
    # Phía Đông (cột > peak_c) phải có độ phủ trung bình cao hơn phía Tây (cột < peak_c) ở cùng khoảng cách
    if 3 <= peak_c <= 21:
        east_opacity = float(opacity_wind[peak_r, peak_c + 3])
        west_opacity = float(opacity_wind[peak_r, peak_c - 3])
        assert east_opacity > west_opacity

    # Kiểm tra mở rộng 1 -> 3 -> 8 trạm
    hm_1 = generate_web_heatmap(
        station_values=np.array([70.0]),
        coordinates_utm=STATION_COORDS_UTM[:1],
        station_ids=STATION_IDS[:1],
        grid_size=25,
        radius_m=900.0,
    )
    hm_3 = generate_web_heatmap(
        station_values=np.array([70.0, 45.0, 95.0]),
        coordinates_utm=STATION_COORDS_UTM[:3],
        station_ids=STATION_IDS[:3],
        grid_size=25,
        radius_m=900.0,
    )
    hm_8 = generate_web_heatmap(
        station_values=np.linspace(40.0, 110.0, 8),
        coordinates_utm=STATION_COORDS_UTM[:8],
        station_ids=STATION_IDS[:8],
        grid_size=25,
        radius_m=900.0,
    )

    assert len(hm_1["markers"]) == 1
    assert len(hm_3["markers"]) == 3
    assert len(hm_8["markers"]) == 8

    cov_1 = float(np.mean(np.asarray(hm_1["opacity_grid"]) > 0.4))
    cov_3 = float(np.mean(np.asarray(hm_3["opacity_grid"]) > 0.4))
    cov_8 = float(np.mean(np.asarray(hm_8["opacity_grid"]) > 0.4))
    assert cov_1 < cov_3 < cov_8


def test_v5_03_spatial_engine_single_station_safe():
    """SpatialTemporalEngine.analyze không bị lỗi khi chỉ truyền dữ liệu 1 trạm (T, 1)."""
    st_engine = SpatialTemporalEngine()
    hist_1st = np.linspace(35.0, 65.0, 24, dtype=np.float32).reshape(24, 1)
    res = st_engine.analyze(hist_1st, dt_s=900.0)

    assert res["num_active_stations"] == 1
    assert res["r_matrix"].shape == (1, 1)
    assert res["tau_star_matrix"].shape == (1, 1)
    assert "heatmap_2d" in res
    assert len(res["heatmap_2d"]["markers"]) == 1


def test_v5_04_central_engine_single_station_prototype_mode():
    """CentralAIEngine chạy trực tiếp với đúng 1 trạm Prototype (6 chiều hiện tại, 24 bước lịch sử)."""
    engine = CentralAIEngine(use_onnx=True)

    # Kiểm tra HierarchicalCentralAI Lite (~6.9k params) đã được tích hợp
    assert hasattr(engine, "unified_model")
    total_params = sum(p.numel() for p in engine.unified_model.parameters())
    assert 5000 <= total_params <= 50000

    # Đầu vào chỉ có 1 trạm duy nhất (STATION_01)
    hist_1_station = np.linspace(40.0, 75.0, 24, dtype=np.float32)  # shape (24,)
    curr_1_station = np.array([75.0, 98.0, 23.5, 78.0, 520.0, 130.0], dtype=np.float32)  # shape (6,)

    t0 = time.perf_counter()
    out = engine.process_frame(hist_1_station, curr_1_station, dt_s=900.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0
    assert out["num_active_stations"] == 1
    assert len(out["station_forecasts"]) == 1
    assert "STATION_01" in out["station_forecasts"]
    assert len(out["station_forecasts"]["STATION_01"]["pm25_pred"]) == 24
    assert len(out["station_forecasts"]["STATION_01"]["aqi_pred"]) == 24

    # Bản đồ nhiệt Web có đúng 1 marker, có current_map và forecast_maps (+1h, +3h, +6h, +12h, +24h)
    smaps = out["spatial_maps"]
    assert smaps["num_active_stations"] == 1
    assert len(smaps["markers"]) == 1
    assert "current_map" in smaps
    for h_key in ["+1h", "+3h", "+6h", "+12h", "+24h"]:
        assert h_key in smaps["forecast_maps"]
        fmap = smaps["forecast_maps"][h_key]
        assert np.asarray(fmap["grid_pm25"]).shape == (25, 25)
        assert np.asarray(fmap["opacity_grid"]).shape == (25, 25)


def test_v5_05_central_engine_dynamic_scaling_3_and_8_stations():
    """Khi mở rộng hệ thống lên 3 trạm hoặc 8 trạm, CentralAIEngine tự động dự báo và nối bản đồ đúng số trạm."""
    engine = CentralAIEngine(use_onnx=True)

    # 1. Thử nghiệm với 3 trạm
    hist_3 = np.random.RandomState(42).uniform(30.0, 90.0, size=(24, 3)).astype(np.float32)
    curr_3 = np.array([
        [80.0, 110.0, 24.0, 75.0, 500.0, 120.0],
        [55.0, 75.0, 24.5, 72.0, 450.0, 105.0],
        [95.0, 130.0, 23.8, 80.0, 580.0, 145.0],
    ], dtype=np.float32)

    out_3 = engine.process_frame(hist_3, curr_3, dt_s=900.0)
    assert out_3["num_active_stations"] == 3
    assert len(out_3["station_forecasts"]) == 3
    assert len(out_3["spatial_maps"]["markers"]) == 3

    # 2. Thử nghiệm với 8 trạm đầy đủ
    hist_8 = np.random.RandomState(42).uniform(25.0, 110.0, size=(24, 8)).astype(np.float32)
    curr_8 = np.random.RandomState(42).uniform(30.0, 100.0, size=(8, 6)).astype(np.float32)

    out_8 = engine.process_frame(hist_8, curr_8, dt_s=900.0)
    assert out_8["num_active_stations"] == 8
    assert len(out_8["station_forecasts"]) == 8
    assert len(out_8["spatial_maps"]["markers"]) == 8

    # Kiểm tra tính đơn điệu phân vị 0 <= P10 <= P50 <= P90
    p10 = np.asarray(out_8["forecast_24h"]["pm25_p10"])
    p50 = np.asarray(out_8["forecast_24h"]["pm25_p50"])
    p90 = np.asarray(out_8["forecast_24h"]["pm25_p90"])
    assert np.all(p10 >= 0.0)
    assert np.all(p10 <= p50 + 1e-5)
    assert np.all(p50 <= p90 + 1e-5)


def test_forecast_values_denormalized_physical_scale():
    """Kiểm tra dự báo PM2.5 và VN-AQI được giải chuẩn hóa về đúng thang vật lý (không bị bẹp ở ~0.06)."""
    engine = CentralAIEngine()
    hist_24 = np.ones((24, 1), dtype=np.float32) * 35.0
    curr_6d = np.array([35.0, 48.0, 25.0, 75.0, 420.0, 100.0], dtype=np.float32)
    res = engine.process_telemetry_frame(hist_24, curr_6d)
    fc = res["forecast_24h"]
    assert fc["pm25_pred"][0] >= 10.0, f"PM2.5 dự báo bị bẹp: {fc['pm25_pred'][0]}"
    assert fc["pm25_pred"][0] <= 150.0
    assert fc["pm25_p10"][0] <= fc["pm25_pred"][0] <= fc["pm25_p90"][0]

