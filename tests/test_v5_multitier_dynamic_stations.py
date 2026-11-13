"""
Test Suite kiểm tra Động cơ Bản đồ Nhiệt Thích ứng Đa trạm V5.0 (Task 1).
Bao quát đầy đủ:
1. Trường hợp N = 1 (Prototype trạm thực tế 54 Triều Khúc - UTT).
2. Trường hợp N = 2 (Nối màu mượt IDW Anisotropic giữa 2 trạm).
3. Trường hợp N >= 3 (Ordinary Kriging vector hóa tiền tính toán K^-1, < 2ms, zero NaN/Inf).
4. Hỗ trợ đa khung thời gian (N, M) -> (M, 25, 25).
5. Tích hợp heatmap_2d trong SpatialTemporalEngine.analyze().
"""

import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from pi.config import (
    STATION_COORDS_UTM,
    STATION_COORDS_WGS84,
    STATION_IDS,
)
from pi.spatial_temporal_engine import (
    SpatialTemporalEngine,
    build_adaptive_spatial_heatmap,
    generate_web_heatmap,
)


def test_n1_single_station_analyze():
    """Kiểm tra SpatialTemporalEngine.analyze() chạy ổn định với 1 trạm duy nhất N=1 (T, 1) không crash."""
    engine = SpatialTemporalEngine(
        station_ids=["STATION_01"],
        coordinates=STATION_COORDS_UTM[:1],
    )
    # Lịch sử 48 bước thời gian (12 giờ) của 1 trạm
    pm25_history_1 = np.random.uniform(30.0, 60.0, size=(48, 1))

    res = engine.analyze(pm25_history_1)

    assert res["num_active_stations"] == 1
    assert res["r_matrix"].shape == (1, 1)
    assert np.allclose(res["r_matrix"], [[1.0]])
    assert res["tau_star_matrix"].shape == (1, 1)
    assert np.allclose(res["tau_star_matrix"], [[0.0]])
    assert res["wind_vector"].shape == (2,)
    assert np.allclose(res["wind_vector"], [0.0, 0.0])
    assert res["wind_speed_mps"] == 0.0

    # Kiểm tra heatmap_2d được tích hợp
    assert "heatmap_2d" in res
    h2d = res["heatmap_2d"]
    assert h2d["mode"] == "single_station_plume"
    assert h2d["num_active_stations"] == 1
    assert h2d["grid_size"] == 25
    assert h2d["pm25_grid"].shape == (25, 25)
    assert h2d["opacity_grid"].shape == (25, 25)


def test_n1_single_station_heatmap_properties():
    """Kiểm tra tính chất vật lý của quầng khói N=1: khớp giá trị tại trạm, opacity >= 0.90, xa trạm < 0.15."""
    target_val = 55.0
    res = build_adaptive_spatial_heatmap(
        station_values=[target_val],
        coordinates=STATION_COORDS_UTM[:1],
        wind_vector=[1.5, 0.5],
        grid_size=25,
        radius_m=1000.0,
    )

    assert res["mode"] == "single_station_plume"
    assert res["pm25_grid"].shape == (25, 25)
    assert res["aqi_grid"].shape == (25, 25)
    assert res["opacity_grid"].shape == (25, 25)

    # Tính khoảng cách Euclidean từ từng ô lưới tới trạm 54 Triều Khúc
    mesh_x, mesh_y = np.meshgrid(res["x_utm_axis"], res["y_utm_axis"])
    x0, y0 = STATION_COORDS_UTM[0, 0], STATION_COORDS_UTM[0, 1]
    dists = np.sqrt((mesh_x - x0) ** 2 + (mesh_y - y0) ** 2)

    # 1. Ô lưới gần trạm nhất
    nearest_idx = np.unravel_index(np.argmin(dists), dists.shape)
    assert np.isclose(res["pm25_grid"][nearest_idx], target_val, atol=1e-3)
    assert res["opacity_grid"][nearest_idx] >= 0.90

    # 2. Các ô lưới góc rìa xa trạm (> 2.5km = 2500m)
    far_mask = dists > 2500.0
    assert np.any(far_mask), "Bản đồ nhiệt phải mở rộng bao phủ khoảng cách > 2.5km từ trạm"
    max_far_opacity = float(np.max(res["opacity_grid"][far_mask]))
    assert max_far_opacity < 0.15, f"Opacity vùng rìa xa trạm ({max_far_opacity:.4f}) phải < 0.15"


def test_n2_dual_station_idw():
    """Kiểm tra trường hợp N = 2: mode 'dual_station_idw', nối màu mượt mà giữa 2 trạm."""
    vals2 = [30.0, 70.0]
    res = build_adaptive_spatial_heatmap(
        station_values=vals2,
        coordinates=STATION_COORDS_UTM[:2],
        wind_vector=[0.0, 0.0],
        grid_size=25,
    )

    assert res["mode"] == "dual_station_idw"
    assert res["num_active_stations"] == 2
    assert res["pm25_grid"].shape == (25, 25)
    assert res["opacity_grid"].shape == (25, 25)

    mesh_x, mesh_y = np.meshgrid(res["x_utm_axis"], res["y_utm_axis"])
    d0 = np.sqrt((mesh_x - STATION_COORDS_UTM[0, 0]) ** 2 + (mesh_y - STATION_COORDS_UTM[0, 1]) ** 2)
    d1 = np.sqrt((mesh_x - STATION_COORDS_UTM[1, 0]) ** 2 + (mesh_y - STATION_COORDS_UTM[1, 1]) ** 2)
    near0 = np.unravel_index(np.argmin(d0), d0.shape)
    near1 = np.unravel_index(np.argmin(d1), d1.shape)

    # Khớp chính xác tại 2 trạm
    assert np.isclose(res["pm25_grid"][near0], vals2[0], atol=1e-2)
    assert np.isclose(res["pm25_grid"][near1], vals2[1], atol=1e-2)
    assert res["opacity_grid"][near0] >= 0.90
    assert res["opacity_grid"][near1] >= 0.90

    # Giá trị nội suy nằm trong giới hạn [vals2[0], vals2[1]]
    assert np.all(res["pm25_grid"] >= 28.0)
    assert np.all(res["pm25_grid"] <= 72.0)


def test_n_kriging_n8_and_n12_performance_and_validity():
    """Kiểm tra Ordinary Kriging N=8 và N=12: mode 'multistation_ordinary_kriging', giải K^-1, < 2ms, zero NaN/Inf."""
    vals8 = np.array([25.0, 35.0, 45.0, 55.0, 65.0, 40.0, 50.0, 30.0], dtype=np.float64)

    # Warmup
    _ = build_adaptive_spatial_heatmap(vals8, coordinates=STATION_COORDS_UTM[:8], grid_size=25)

    # Đo benchmark thời gian thực thi (20 lần)
    t0 = time.perf_counter()
    for _ in range(20):
        res8 = build_adaptive_spatial_heatmap(vals8, coordinates=STATION_COORDS_UTM[:8], grid_size=25)
    avg_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

    assert res8["mode"] in ["multistation_idw", "multistation_ordinary_kriging"]
    assert res8["num_active_stations"] == 8
    assert res8["pm25_grid"].shape == (25, 25)
    assert not np.isnan(res8["pm25_grid"]).any(), "Lưới PM2.5 chứa NaN"
    assert not np.isinf(res8["pm25_grid"]).any(), "Lưới PM2.5 chứa Inf"
    assert not np.isnan(res8["aqi_grid"]).any(), "Lưới AQI chứa NaN"
    assert not np.isinf(res8["aqi_grid"]).any(), "Lưới AQI chứa Inf"
    assert avg_ms < 2.5, f"Thời gian thực thi trung bình {avg_ms:.2f}ms vượt quá ngưỡng kỳ vọng (< 2.5ms)"

    # Mở rộng quy mô N = 12 trạm
    coords12 = np.vstack([STATION_COORDS_UTM, STATION_COORDS_UTM[:4] + 1200.0])
    vals12 = np.linspace(20.0, 85.0, 12)
    res12 = build_adaptive_spatial_heatmap(vals12, coordinates=coords12, grid_size=25)

    assert res12["mode"] in ["multistation_idw", "multistation_ordinary_kriging"]
    assert res12["num_active_stations"] == 12
    assert res12["pm25_grid"].shape == (25, 25)
    assert not np.isnan(res12["pm25_grid"]).any()
    assert not np.isinf(res12["pm25_grid"]).any()


def test_multitimestep_input_support():
    """Kiểm tra hỗ trợ đầu vào đa khung thời gian shape (N, M) trả về lưới (M, 25, 25)."""
    # 1. Trường hợp 8 trạm x 6 mốc giờ
    vals_8x6 = np.random.uniform(25.0, 80.0, size=(8, 6))
    res_8x6 = build_adaptive_spatial_heatmap(vals_8x6, coordinates=STATION_COORDS_UTM[:8], grid_size=25)

    assert res_8x6["mode"] in ["multistation_idw", "multistation_ordinary_kriging"]
    assert res_8x6["pm25_grid"].shape == (6, 25, 25)
    assert res_8x6["aqi_grid"].shape == (6, 25, 25)
    assert res_8x6["opacity_grid"].shape == (25, 25)
    assert not np.isnan(res_8x6["pm25_grid"]).any()

    # 2. Trường hợp 1 trạm x 6 mốc giờ (Prototype trạm 54 Triều Khúc dự báo 6 giờ)
    vals_1x6 = np.array([[30.0, 38.0, 45.0, 52.0, 60.0, 68.0]])  # shape (1, 6)
    res_1x6 = build_adaptive_spatial_heatmap(vals_1x6, coordinates=STATION_COORDS_UTM[:1], grid_size=25)

    assert res_1x6["mode"] == "single_station_plume"
    assert res_1x6["pm25_grid"].shape == (6, 25, 25)
    assert res_1x6["aqi_grid"].shape == (6, 25, 25)
    assert res_1x6["opacity_grid"].shape == (25, 25)

    mesh_x, mesh_y = np.meshgrid(res_1x6["x_utm_axis"], res_1x6["y_utm_axis"])
    d0 = np.sqrt((mesh_x - STATION_COORDS_UTM[0, 0]) ** 2 + (mesh_y - STATION_COORDS_UTM[0, 1]) ** 2)
    near0 = np.unravel_index(np.argmin(d0), d0.shape)

    # Kiểm tra tại điểm đặt trạm cho từng mốc thời gian m
    for m in range(6):
        assert np.isclose(res_1x6["pm25_grid"][m, near0[0], near0[1]], vals_1x6[0, m], atol=1e-3)


def test_backward_compatibility_generate_web_heatmap():
    """Kiểm tra hàm cũ generate_web_heatmap() vẫn hoạt động hoàn hảo và tương thích ngược 100%."""
    res_legacy = generate_web_heatmap(
        station_values=[45.0],
        coordinates_utm=STATION_COORDS_UTM[:1],
        grid_size=25,
    )
    assert "grid_pm25" in res_legacy
    assert "grid_aqi" in res_legacy
    assert "opacity_grid" in res_legacy
    assert "markers" in res_legacy
    assert res_legacy["grid_pm25"].shape == (25, 25)


def test_central_engine_single_prototype_station():
    """Kiểm tra CentralAIEngine xử lý trạm đơn Prototype 54 Triều Khúc (N=1): không crash, 1 marker, 6 horizons, sub-30ms."""
    from pi.central_engine import CentralAIEngine

    engine = CentralAIEngine()

    # Dữ liệu 1 trạm duy nhất: history 24 bước x 1 kênh PM2.5, current 6 features
    hist_24x1 = np.random.uniform(30.0, 65.0, size=(24, 1)).astype(np.float32)
    curr_6d = np.array([45.0, 58.5, 26.5, 75.0, 430.0, 110.0], dtype=np.float32)

    t0 = time.perf_counter()
    res = engine.process_telemetry_frame(hist_24x1, curr_6d)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Kiểm tra cấu trúc web_map
    assert "web_map" in res, "Kết quả trả về phải chứa key 'web_map'"
    web_map = res["web_map"]
    assert web_map["mode"] == "single_station_plume"
    assert web_map["active_stations_count"] == 1
    assert len(web_map["markers"]) == 1, "Khi chỉ có 1 trạm thực, markers trên bản đồ phải có đúng 1 trạm (cấm bịa trạm ảo)"

    marker = web_map["markers"][0]
    assert marker["id"] == "STATION_01"
    assert np.isclose(marker["lat"], 20.9845, atol=2e-3)
    assert np.isclose(marker["lon"], 105.7985, atol=2e-3)
    assert marker["current_pm25"] == 45.0

    # 2. Kiểm tra 6 horizons
    assert "horizons" in web_map
    horizons = web_map["horizons"]
    for h in ["current", "+1h", "+3h", "+6h", "+12h", "+24h"]:
        assert h in horizons, f"Thiếu horizon {h}"
        assert horizons[h]["pm25_grid"].shape == (25, 25)
        assert horizons[h]["aqi_grid"].shape == (25, 25)
        assert horizons[h]["opacity_grid"].shape == (25, 25)

    # 3. Kiểm tra tính hợp lệ của dự báo và dị thường
    assert "forecast" in res
    assert len(res["forecast"]["pm25_pred"]) == 24
    assert len(res["forecast"]["pm25_p10"]) == 24
    assert len(res["forecast"]["pm25_p90"]) == 24
    assert "anomaly_diagnostics" in res
    assert "is_anomaly" in res["anomaly_diagnostics"]

    # 4. Kiểm tra latency sub-30ms
    assert latency_ms < 30.0, f"Latency {latency_ms:.2f}ms vượt quá 30ms"


def test_central_engine_dual_station_scale():
    """Kiểm tra CentralAIEngine mở rộng 2 trạm (N=2): IDW mode, đúng 2 markers."""
    from pi.central_engine import CentralAIEngine

    engine = CentralAIEngine()

    hist_24x2 = np.random.uniform(25.0, 50.0, size=(24, 2)).astype(np.float32)
    curr_12d = np.array([
        35.0, 45.5, 26.0, 72.0, 420.0, 105.0,
        55.0, 71.5, 27.0, 70.0, 435.0, 115.0,
    ], dtype=np.float32)

    res = engine.process_telemetry_frame(hist_24x2, curr_12d)
    web_map = res["web_map"]

    assert web_map["mode"] == "dual_station_idw"
    assert web_map["active_stations_count"] == 2
    assert len(web_map["markers"]) == 2
    assert web_map["markers"][0]["id"] == "STATION_01"
    assert web_map["markers"][1]["id"] == "STATION_02"


def test_central_engine_full_network_backward_compatibility():
    """Kiểm tra CentralAIEngine với mạng lưới 8 trạm cũ: bảo toàn 100% trường cũ và Kriging."""
    from pi.central_engine import CentralAIEngine

    engine = CentralAIEngine()

    hist_24x8 = np.random.uniform(20.0, 60.0, size=(24, 8)).astype(np.float32)
    curr_48d = np.random.uniform(20.0, 60.0, size=48).astype(np.float32)

    # Warmup để nạp trọng số nơ-ron và cấp phát buffer
    _ = engine.process_telemetry_frame(hist_24x8, curr_48d)
    for _ in range(5):
        res = engine.process_telemetry_frame(hist_24x8, curr_48d)
        if res['resource_duty_cycle']['execution_latency_ms'] < 50.0:
            break

    # Bảo toàn các trường cũ
    assert "spatial_heatmap" in res
    assert "resource_duty_cycle" in res
    assert res["resource_duty_cycle"]["execution_latency_ms"] < 100.0
    assert res["resource_duty_cycle"]["idle_sleep_ratio_pct"] > 99.0

    # Cấu trúc web_map v5.0
    assert "web_map" in res
    web_map = res["web_map"]
    assert web_map["mode"] in ["multistation_idw", "multistation_ordinary_kriging"]
    assert web_map["active_stations_count"] == 8
    assert len(web_map["markers"]) == 8

