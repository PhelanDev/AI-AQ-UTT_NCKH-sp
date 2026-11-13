import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("."))


@pytest.fixture(scope="class")
def test_server(request):
    """Khởi động Web Server trên cổng ngẫu nhiên trong luồng riêng biệt."""
    from pi.web_server import run_server_in_thread, stop_server

    server, port = run_server_in_thread(port=0)  # Port 0: OS tự cấp port rảnh
    base_url = f"http://127.0.0.1:{port}"
    time.sleep(0.1)  # Chờ server sẵn sàng

    yield base_url

    stop_server(server)


class TestWebServerEndpoints:
    """Kiểm thử TDD toàn diện cho Máy chủ Web & REST API V5.0."""

    def test_get_root_dashboard_html(self, test_server):
        """Kiểm tra đường dẫn gốc '/' trả về trang HTML Dashboard đầy đủ."""
        url = f"{test_server}/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "text/html" in content_type
            html_body = resp.read().decode("utf-8")
            assert "54 Triều Khúc" in html_body
            assert "VN-AQI" in html_body
            assert "leaflet" in html_body.lower()

    def test_api_status(self, test_server):
        """Kiểm tra API trạng thái hệ thống: /api/status."""
        url = f"{test_server}/api/status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "OK"
            assert data["version"] == "5.0"
            assert "UTT" in data.get("prototype_station", "")
            assert "uptime_seconds" in data

    def test_api_telemetry(self, test_server):
        """Kiểm tra API thông số cảm biến thời gian thực: /api/telemetry."""
        url = f"{test_server}/api/telemetry"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "pm25" in data
            assert "vn_aqi" in data
            assert "temp_c" in data
            assert "rh_percent" in data
            assert "co2_ppm" in data
            assert "gas_proxy" in data
            assert "station_id" in data
            assert data["station_id"] == "STATION_01"

    def test_api_map_v5(self, test_server):
        """Kiểm tra API bản đồ nhiệt đa chân thời gian V5.0: /api/map."""
        url = f"{test_server}/api/map"
        t0 = time.perf_counter()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))

            assert "web_map" in data
            wm = data["web_map"]
            assert wm["mode"] == "single_station_plume"
            assert wm["active_stations_count"] == 1
            assert len(wm["markers"]) == 1

            marker = wm["markers"][0]
            assert marker["id"] == "STATION_01"
            assert np.isclose(marker["lat"], 20.9845, atol=2e-3)
            assert np.isclose(marker["lon"], 105.7985, atol=2e-3)

            assert "bounds" in wm
            assert "horizons" in wm
            horizons = wm["horizons"]
            for h in ["current", "+1h", "+3h", "+6h", "+12h", "+24h"]:
                assert h in horizons, f"Thiếu mốc thời gian {h}"
                assert "pm25_grid" in horizons[h]
                assert "aqi_grid" in horizons[h]
                assert "opacity_grid" in horizons[h]
                # Kiểm tra ma trận lưới 25x25
                assert len(horizons[h]["pm25_grid"]) == 25
                assert len(horizons[h]["pm25_grid"][0]) == 25

            # Phản hồi dưới 35ms
            assert elapsed_ms < 35.0, f"Độ trễ API /api/map {elapsed_ms:.2f}ms vượt 35ms"

    def test_api_forecast(self, test_server):
        """Kiểm tra API dự báo chuỗi thời gian 24 giờ: /api/forecast."""
        url = f"{test_server}/api/forecast"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "pm25_pred" in data
            assert len(data["pm25_pred"]) == 24
            assert "pm25_p10" in data
            assert "pm25_p90" in data
            assert len(data["pm25_p10"]) == 24
            assert len(data["pm25_p90"]) == 24
            assert "aqi_pred" in data

    def test_api_trigger_cycle_post(self, test_server):
        """Kiểm tra API kích hoạt tính toán chu kỳ mới: POST /api/trigger_cycle."""
        url = f"{test_server}/api/trigger_cycle"
        post_data = json.dumps({"reason": "live_nckh_demo"}).encode("utf-8")
        req = urllib.request.Request(url, data=post_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert "cycle_result" in data
            assert "execution_latency_ms" in data["cycle_result"]
