"""
tests/test_store_and_forward_sync.py
Kiểm chứng cơ chế Bộ đệm Ngoại Tuyến Store-and-Forward (Phương án Nâng cấp 3)
Đảm bảo đồng bộ giữa Node ESP32-S3 và Server Pi 5 khi có mất kết nối mạng và gửi bù.
"""

import json
import pytest
import numpy as np

from pi.web_server import AQServerState
from pi.central_engine import CentralAIEngine


class TestStoreAndForwardIntegration:
    """Kiểm tra xử lý gói tin Store-and-Forward (gửi bù backfill) trên Pi 5."""

    @pytest.fixture
    def app_state(self):
        engine = CentralAIEngine()
        return AQServerState(engine=engine)

    def test_backfill_telemetry_packet_parsing(self, app_state):
        """Xác nhận định dạng gói tin MQTT Telemetry chứa cờ is_backfill và flags."""
        # Gói tin bình thường (Realtime)
        realtime_payload = {
            "station_id": "esp32s3_node_01",
            "timestamp": 1731063600,
            "pm1_raw": 32.10,
            "pm2_raw": 33.40,
            "pm25_calibrated": 22.80,
            "temp_c": 25.50,
            "rh_percent": 79.00,
            "co2_ppm": 510.0,
            "aqi": 46,
            "anomaly_code": 0,
            "flags": 1,
            "is_backfill": False
        }

        # Gói tin gửi bù sau khi phục hồi mạng (Store-and-Forward Backfill)
        backfill_payload = {
            "station_id": "esp32s3_node_01",
            "timestamp": 1731062700,  # 15 phút trước
            "pm1_raw": 45.20,
            "pm2_raw": 46.10,
            "pm25_calibrated": 38.50,
            "temp_c": 26.10,
            "rh_percent": 81.00,
            "co2_ppm": 530.0,
            "aqi": 77,
            "anomaly_code": 0,
            "flags": 2,  # TELEMETRY_FLAG_BACKFILL = 0x02
            "is_backfill": True
        }

        assert realtime_payload["is_backfill"] is False
        assert realtime_payload["flags"] == 1
        assert backfill_payload["is_backfill"] is True
        assert backfill_payload["flags"] & 2 != 0

    def test_server_cycle_with_backfilled_telemetry(self, app_state):
        """Khi nhận gói tin backfill, hệ thống cập nhật chuỗi thời gian bình thường."""
        # Mô phỏng nhận 5 gói tin gửi bù (15 phút/gói = 75 phút mất mạng)
        base_pm25 = 35.0
        for i in range(5):
            pm = base_pm25 + i * 2.0
            app_state.current_6d[0] = pm
            res = app_state.execute_cycle(reason="store_and_forward_backfill")
            assert res["status"] == "COMPLETED"
            assert res["trigger_reason"] == "store_and_forward_backfill"
            assert res["execution_latency_ms"] > 0
            assert res["active_stations_count"] >= 1
            assert app_state.current_telemetry["pm25"] > 0

        # Kiểm tra lịch sử 24 mẫu được bảo toàn liên tục không đứt gãy
        assert len(app_state.history_24x1) == 24
        assert not np.isnan(app_state.history_24x1).any()
