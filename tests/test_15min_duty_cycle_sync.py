"""
tests/test_15min_duty_cycle_sync.py
Kiểm chứng Đồng bộ Hóa Chu Kỳ 15 Phút (900s), Bản Đồ Nhiệt Kriging Không Gian 2D
và Tối Ưu Hóa Tài Nguyên Tiết Kiệm Năng Lượng (> 99.9% Idle) Trên Raspberry Pi 5.
"""

import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.config import (
    TELEMETRY_INTERVAL_S,
    ESP_WARMUP_SECONDS,
    ESP_ACTIVE_PROCESS_SECONDS,
    ESP_DEEP_SLEEP_SECONDS,
    STEPS_PER_HOUR,
    SUBHOURLY_RING_BUFFER_LEN,
    WLS_MAX_LAG_STEPS,
    STATION_IDS,
    STATION_COORDS_UTM,
)
from pi.spatial_temporal_engine import (
    SpatialTemporalEngine,
    generate_pollution_heatmap,
    compute_cross_correlation_matrix,
)
from pi.central_engine import CentralAIEngine


class Test15MinDutyCycleConstants:
    """Kiểm tra các hằng số chu kỳ 15 phút đồng bộ ESP32-S3 & Raspberry Pi 5."""

    def test_duty_cycle_constants_values(self):
        """Xác thực giá trị các hằng số đặc tả kỹ thuật theo chuẩn 15 phút."""
        assert TELEMETRY_INTERVAL_S == 900.0, "TELEMETRY_INTERVAL_S phai la 900.0 giay (15 phut)"
        assert ESP_WARMUP_SECONDS == 30.0, "ESP_WARMUP_SECONDS phai la 30.0 giay lam am quat/laser"
        assert ESP_ACTIVE_PROCESS_SECONDS == 2.0, "ESP_ACTIVE_PROCESS_SECONDS phai la 2.0 giay doc cam bien"
        assert ESP_DEEP_SLEEP_SECONDS == 868.0, "ESP_DEEP_SLEEP_SECONDS phai la 868.0 giay ngu sau"
        assert STEPS_PER_HOUR == 4, "STEPS_PER_HOUR phai la 4 buoc / gio"
        assert SUBHOURLY_RING_BUFFER_LEN == 96, "SUBHOURLY_RING_BUFFER_LEN phai la 96 buoc (24 gio)"
        assert WLS_MAX_LAG_STEPS == 12, "WLS_MAX_LAG_STEPS phai la 12 buoc (3 gio tre toi da)"

    def test_cycle_time_conservation_and_duty_ratio(self):
        """Bảo toàn tổng thời gian chu kỳ và tỷ lệ tiêu thụ năng lượng cực thấp."""
        total_cycle = ESP_WARMUP_SECONDS + ESP_ACTIVE_PROCESS_SECONDS + ESP_DEEP_SLEEP_SECONDS
        assert total_cycle == TELEMETRY_INTERVAL_S, (
            f"Tong thoi gian cac pha ({total_cycle}s) phai bang chu ky ({TELEMETRY_INTERVAL_S}s)"
        )

        active_time = ESP_WARMUP_SECONDS + ESP_ACTIVE_PROCESS_SECONDS
        active_ratio = active_time / TELEMETRY_INTERVAL_S
        # 32s / 900s = 3.55% active, 96.44% sleep trên node ESP32-S3
        assert active_ratio < 0.04, f"Active ratio {active_ratio:.4f} phai < 4% de keo dai pin trạm"


class TestOrdinaryKrigingHeatmap:
    """Kiểm chứng giải thuật Ordinary Kriging vector hóa NumPy tạo bản đồ nhiệt 2D."""

    @pytest.fixture
    def spatial_engine(self):
        return SpatialTemporalEngine()

    def test_heatmap_output_shape_and_keys(self, spatial_engine):
        """Kiểm tra cấu trúc từ điển và kích thước lưới (20, 20)."""
        station_pm25 = np.array([32.5, 41.2, 28.0, 55.4, 38.0, 62.1, 44.5, 36.8])
        res = spatial_engine.generate_pollution_heatmap(station_pm25, grid_resolution=20)

        # Kiểm tra đầy đủ các khóa
        expected_keys = [
            "grid_resolution",
            "bounds",
            "grid_x",
            "grid_y",
            "mesh_x",
            "mesh_y",
            "heatmap_pm25",
            "kriging_variance",
            "variance_map",
            "min_pm25",
            "max_pm25",
            "mean_pm25",
            "execution_time_ms",
        ]
        for k in expected_keys:
            assert k in res, f"Thieu khoa '{k}' trong ket qua Kriging heatmap"

        assert res["grid_resolution"] == 20
        assert res["heatmap_pm25"].shape == (20, 20), "Lưới nồng độ PM2.5 phai co shape (20, 20)"
        assert res["kriging_variance"].shape == (20, 20), "Lưới phương sai phai co shape (20, 20)"
        assert res["grid_x"].shape == (20,)
        assert res["grid_y"].shape == (20,)

    def test_heatmap_physical_positivity_and_variance(self, spatial_engine):
        """Tất cả giá trị nồng độ bụi nội suy và phương sai Kriging phải không âm."""
        station_pm25 = np.array([12.0, 18.5, 14.2, 25.0, 16.0, 30.5, 22.0, 15.0])
        res = spatial_engine.generate_pollution_heatmap(station_pm25, grid_resolution=20)

        assert np.all(res["heatmap_pm25"] >= 0.0), "Nồng độ bụi PM2.5 nội suy không được âm"
        assert res["min_pm25"] >= 0.0, "min_pm25 phai >= 0"
        assert np.all(res["kriging_variance"] >= 0.0), "Phương sai Kriging phai >= 0"

    def test_kriging_constant_field_preservation(self, spatial_engine):
        """Kiểm tra tính chất bảo toàn trường đồng nhất (Unbiasedness): nếu mọi trạm = C thì lưới = C."""
        const_val = 48.0
        station_pm25 = np.full(len(STATION_IDS), const_val)
        res = spatial_engine.generate_pollution_heatmap(station_pm25, grid_resolution=20)

        heatmap = res["heatmap_pm25"]
        # Do trọng số Ordinary Kriging tổng bằng 1 (sum lambda_i = 1), trường đồng nhất phải bằng đúng const_val
        np.testing.assert_allclose(heatmap, const_val, rtol=1e-3, atol=1e-2)

    def test_kriging_execution_speed_benchmark(self, spatial_engine):
        """Kiểm tra thời gian thực thi vector hóa siêu tốc (< 30ms theo brief, < 50ms giới hạn test)."""
        station_pm25 = np.random.uniform(20.0, 80.0, len(STATION_IDS))

        # Khởi động (warmup)
        _ = spatial_engine.generate_pollution_heatmap(station_pm25, grid_resolution=20)

        # Chạy benchmark 20 lần
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = spatial_engine.generate_pollution_heatmap(station_pm25, grid_resolution=20)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        assert avg_lat < 30.0, f"Thoi gian trung binh {avg_lat:.2f}ms vuot qua 30ms"
        assert p95_lat < 50.0, f"P95 latency {p95_lat:.2f}ms vuot qua 50ms"
        assert out["execution_time_ms"] < 50.0

    def test_heatmap_input_polymorphism(self, spatial_engine):
        """Hỗ trợ linh hoạt nhiều định dạng đầu vào: dict, 1D array, 48D array, 2D matrix."""
        # 1. Dict
        dict_input = {sid: float(10 + i * 5) for i, sid in enumerate(STATION_IDS)}
        res_dict = spatial_engine.generate_pollution_heatmap(dict_input, grid_resolution=15)
        assert res_dict["heatmap_pm25"].shape == (15, 15)

        # 2. 48-dim array
        arr_48 = np.ones(48, dtype=np.float32) * 35.0
        res_48 = spatial_engine.generate_pollution_heatmap(arr_48, grid_resolution=15)
        assert res_48["heatmap_pm25"].shape == (15, 15)

        # 3. Module-level standalone function
        res_standalone = generate_pollution_heatmap(dict_input, grid_resolution=15)
        assert res_standalone["heatmap_pm25"].shape == (15, 15)


class TestCentralAIEngine15MinDutyCycle:
    """Kiểm tra tích hợp chu kỳ 15 phút và tỷ lệ nhàn rỗi trên CentralAIEngine."""

    @pytest.fixture
    def central_engine(self):
        return CentralAIEngine(forecaster_weights=None, autoencoder_weights=None)

    def test_process_telemetry_frame_default_15m_duty_cycle(self, central_engine):
        """Xử lý khung dữ liệu với chu kỳ mặc định 15 phút (dt_s = 900s)."""
        history_24x8 = np.random.uniform(25.0, 55.0, (24, 8)).astype(np.float32)
        current_48d = np.random.uniform(25.0, 55.0, 48).astype(np.float32)

        res = central_engine.process_telemetry_frame(history_24x8, current_48d)

        # 1. Kiểm tra spatial_heatmap hiện diện và hợp lệ
        assert "spatial_heatmap" in res, "process_telemetry_frame phai tra ve 'spatial_heatmap'"
        heatmap_info = res["spatial_heatmap"]
        assert heatmap_info["heatmap_pm25"].shape == (20, 20)
        assert heatmap_info["min_pm25"] >= 0.0

        # 2. Kiểm tra resource_duty_cycle
        assert "resource_duty_cycle" in res, "process_telemetry_frame phai tra ve 'resource_duty_cycle'"
        duty = res["resource_duty_cycle"]

        assert duty["cycle_interval_s"] == 900.0, "cycle_interval_s phai la 900.0s"
        assert duty["execution_latency_ms"] < 50.0, (
            f"Latency {duty['execution_latency_ms']}ms vuot nguong 50ms"
        )
        assert duty["idle_sleep_time_s"] >= 899.0, (
            f"idle_sleep_time_s {duty['idle_sleep_time_s']}s phai gan bang 900s"
        )
        assert duty["idle_sleep_ratio_pct"] > 99.9, (
            f"idle_sleep_ratio_pct {duty['idle_sleep_ratio_pct']}% phai > 99.9%"
        )

    def test_run_periodic_15m_cycle_method(self, central_engine):
        """Kiểm tra phương thức run_periodic_15m_cycle thực thi nhanh chóng và dọn dẹp tensor."""
        history_24x8 = np.random.uniform(20.0, 60.0, (24, 8)).astype(np.float32)
        current_48d = np.random.uniform(20.0, 60.0, 48).astype(np.float32)

        t_start = time.perf_counter()
        res = central_engine.run_periodic_15m_cycle(history_24x8, current_48d)
        elapsed_total_ms = (time.perf_counter() - t_start) * 1000.0

        assert elapsed_total_ms < 50.0, f"Tong thoi gian run_periodic_15m_cycle {elapsed_total_ms:.2f}ms vuot 50ms"
        assert "spatial_heatmap" in res
        assert "resource_duty_cycle" in res
        assert res["resource_duty_cycle"]["idle_sleep_ratio_pct"] > 99.9

    def test_backward_compatibility_with_30s_cycle(self, central_engine):
        """Đảm bảo tương thích ngược 100% khi caller truyền chu kỳ cũ dt_s = 30.0s."""
        history_24x8 = np.random.uniform(20.0, 40.0, (24, 8)).astype(np.float32)
        current_48d = np.random.uniform(20.0, 40.0, 48).astype(np.float32)

        res_30s = central_engine.process_telemetry_frame(history_24x8, current_48d, dt_s=30.0)

        assert res_30s["resource_duty_cycle"]["cycle_interval_s"] == 30.0
        assert res_30s["resource_duty_cycle"]["idle_sleep_time_s"] < 30.1
        # Với dt_s = 30s và latency ~10ms: idle ratio ~ 99.96%
        assert res_30s["resource_duty_cycle"]["idle_sleep_ratio_pct"] > 99.0
        assert "spatial_heatmap" in res_30s
