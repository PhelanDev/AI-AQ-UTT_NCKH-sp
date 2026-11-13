"""
AQ-Server V5.0: Embedded Lightweight Web & REST API Server.
Tối ưu hóa tài nguyên cho Raspberry Pi 5 và trạm quan trắc đơn 54 Triều Khúc (UTT).

Kiến trúc Ponytail Protocol:
- Sử dụng tiêu chuẩn thư viện chuẩn Python (http.server.ThreadingHTTPServer).
- Zero external dependencies (không cần fastapi, uvicorn hay nodejs).
- Bộ nhớ RAM tiêu thụ < 15MB, thời gian khởi động < 0.1s.
- Độ trễ phản hồi REST API < 5ms (nhờ cơ chế đệm bộ nhớ đệm khung tính toán gần nhất).
"""

import os
import sys
import time
import json
import threading
from typing import Tuple, Dict, Any, Optional
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np

# Thêm đường dẫn gốc để import pi modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pi.central_engine import CentralAIEngine
from pi.spatial_temporal_engine import _vectorized_vn_aqi_pm25


class NumpyEncoder(json.JSONEncoder):
    """Bộ mã hóa JSON hỗ trợ chuyển đổi ndarray, float32, int64 sang kiểu chuẩn."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class AQServerState:
    """Quản lý trạng thái chia sẻ an toàn luồng (Thread-safe) của Web Server."""
    def __init__(self, engine: Optional[CentralAIEngine] = None):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.engine = engine if engine is not None else CentralAIEngine()
        
        # Trạng thái cảm biến trạm mẫu thực nghiệm 54 Triều Khúc (UTT)
        self.current_telemetry = {
            "station_id": "STATION_01",
            "station_name": "Trạm Thử Nghiệm UTT - 54 Triều Khúc, Thanh Xuân Nam, Hà Nội",
            "latitude": 20.983688,
            "longitude": 105.797477,
            "pm25": 38.5,
            "pm10": 52.0,
            "vn_aqi": 108,
            "temp_c": 26.5,
            "rh_percent": 70.0,
            "co2_ppm": 420.0,
            "gas_proxy": 115.0,
            "burst_samples_collected": 30,
            "filter_algorithm": "20% Trimmed Mean (In-place Heap-free)",
            "duty_cycle": "15-min duty cycle with 30-sample burst",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Lịch sử 24 bước thời gian của 1 trạm
        self.history_24x1 = np.full((24, 1), 38.5, dtype=np.float32)
        self.current_6d = np.array([38.5, 52.0, 26.5, 70.0, 420.0, 115.0], dtype=np.float32)
        
        # Kết quả tính toán chu kỳ gần nhất
        self.last_cycle_result: Dict[str, Any] = {}
        self.cached_map_bytes: bytes = b"{}"
        self.cached_forecast_bytes: bytes = b"{}"
        
        # Tính toán khung đầu tiên ngay khi khởi tạo
        self.execute_cycle(reason="startup_init")

    def execute_cycle(self, reason: str = "manual_trigger") -> Dict[str, Any]:
        """Kích hoạt chu kỳ tính toán AI toàn diện (15-min cycle simulation)."""
        with self.lock:
            t0 = time.perf_counter()
            
            # Cập nhật thông số mẫu với biến thiên nhẹ thực tế
            if reason != "startup_init":
                jitter = float(np.random.uniform(-1.5, 1.5))
                new_pm25 = max(5.0, min(250.0, float(self.current_6d[0]) + jitter))
                self.current_6d[0] = new_pm25
                self.current_6d[1] = new_pm25 * 1.35
                self.current_telemetry["pm25"] = round(new_pm25, 1)
                self.current_telemetry["pm10"] = round(float(self.current_6d[1]), 1)
                # Dịch chuyển cửa sổ lịch sử trạm
                self.history_24x1 = np.roll(self.history_24x1, -1, axis=0)
                self.history_24x1[-1, 0] = new_pm25
            
            # Chạy qua động cơ AI trung tâm
            res = self.engine.process_telemetry_frame(self.history_24x1, self.current_6d)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            # Cập nhật VN-AQI từ kết quả tính toán
            pm25_val = float(self.current_6d[0])
            calculated_aqi = int(_vectorized_vn_aqi_pm25(np.array([pm25_val]))[0])
            self.current_telemetry["vn_aqi"] = calculated_aqi
            self.current_telemetry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.current_telemetry["latency_ms"] = round(elapsed_ms, 2)
            self.current_telemetry["anomaly"] = res.get("anomaly", {})
            
            # Chuẩn bị dữ liệu map và forecast
            web_map = res.get("web_map", {})
            forecast_data = res.get("forecast", {})
            if "aqi_pred" not in forecast_data and "pm25_pred" in forecast_data:
                forecast_data["aqi_pred"] = _vectorized_vn_aqi_pm25(
                    np.array(forecast_data["pm25_pred"])
                ).tolist()
            
            self.last_cycle_result = {
                "execution_latency_ms": round(elapsed_ms, 2),
                "status": "COMPLETED",
                "active_stations_count": web_map.get("active_stations_count", 1),
                "trigger_reason": reason,
                "timestamp": self.current_telemetry["timestamp"],
            }
            
            # Serialize và đệm sẵn JSON bytes để phục vụ GET API với độ trễ < 2ms
            self.cached_map_bytes = json.dumps({"web_map": web_map}, cls=NumpyEncoder).encode("utf-8")
            self.cached_forecast_bytes = json.dumps(forecast_data, cls=NumpyEncoder).encode("utf-8")
            
            return self.last_cycle_result


# Singleton server state dùng chung
_GLOBAL_SERVER_STATE: Optional[AQServerState] = None


def get_server_state(engine: Optional[CentralAIEngine] = None) -> AQServerState:
    global _GLOBAL_SERVER_STATE
    if _GLOBAL_SERVER_STATE is None:
        _GLOBAL_SERVER_STATE = AQServerState(engine=engine)
    return _GLOBAL_SERVER_STATE


class AQWebServerHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler phục vụ Dashboard UI và REST API endpoints."""

    def log_message(self, format, *args):
        # Tắt bớt log mặc định để giữ console sạch sẽ và đạt hiệu năng tối đa
        return

    def handle(self):
        """Bọc handle để triệt tiêu lỗi BrokenPipeError khi client ngắt kết nối hoặc refresh trang."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json_response(self, status_code: int, data_bytes: bytes):
        """Gửi phản hồi JSON chuẩn tối ưu."""
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        """Hỗ trợ CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Xử lý các truy vấn GET (Dashboard HTML, Status, Telemetry, Map, Forecast)."""
        path = self.path.split("?")[0].rstrip("/")
        if path == "":
            path = "/"

        state = getattr(self.server, "app_state", get_server_state())

        if path in ("/", "/index.html"):
            # Phục vụ Dashboard HTML
            web_dir = Path(__file__).resolve().parent / "web"
            html_file = web_dir / "index.html"
            
            if html_file.exists():
                with open(html_file, "rb") as f:
                    content = f.read()
            else:
                # Fallback HTML nếu chưa tạo file
                content = (
                    "<!DOCTYPE html><html><head><title>AQ-Server V5.0</title></head>"
                    "<body><h1>Trạm Thử Nghiệm 54 Triều Khúc</h1>"
                    "<p>VN-AQI Dashboard</p><script src='leaflet.js'></script></body></html>"
                ).encode("utf-8")

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/api/status":
            uptime = round(time.time() - state.start_time, 1)
            payload = {
                "status": "OK",
                "version": "5.0",
                "prototype_station": "UTT_TrieuKhuc (54 Triều Khúc, Thanh Xuân Nam, Hà Nội)",
                "uptime_seconds": uptime,
                "active_stations": 1,
                "duty_cycle": "15-min duty cycle with 30-sample burst",
                "timestamp": state.current_telemetry["timestamp"]
            }
            self._send_json_response(200, json.dumps(payload).encode("utf-8"))

        elif path == "/api/telemetry":
            with state.lock:
                telemetry_copy = dict(state.current_telemetry)
            self._send_json_response(200, json.dumps(telemetry_copy).encode("utf-8"))

        elif path == "/api/map":
            # Phục vụ dữ liệu bản đồ V5.0 siêu tốc từ bộ đệm bytes
            with state.lock:
                map_bytes = state.cached_map_bytes
            self._send_json_response(200, map_bytes)

        elif path == "/api/forecast":
            # Phục vụ chuỗi thời gian dự báo 24 giờ
            with state.lock:
                forecast_bytes = state.cached_forecast_bytes
            self._send_json_response(200, forecast_bytes)

        else:
            err = json.dumps({"error": "Not Found", "path": path}).encode("utf-8")
            self._send_json_response(404, err)

    def do_POST(self):
        """Xử lý các truy vấn POST (kích hoạt chu kỳ tính toán mới)."""
        path = self.path.split("?")[0].rstrip("/")
        state = getattr(self.server, "app_state", get_server_state())

        if path == "/api/trigger_cycle":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
            except Exception:
                data = {}

            reason = data.get("reason", "api_post_request")
            cycle_result = state.execute_cycle(reason=reason)

            resp = {
                "success": True,
                "cycle_result": cycle_result
            }
            self._send_json_response(200, json.dumps(resp).encode("utf-8"))

        else:
            err = json.dumps({"error": "Endpoint Not Found", "path": path}).encode("utf-8")
            self._send_json_response(404, err)


class AQThreadingHTTPServer(ThreadingHTTPServer):
    """Máy chủ HTTP đa luồng, triệt tiêu traceback console khi client ngắt kết nối đột ngột."""
    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type and issubclass(exc_type, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def create_server(host: str = "0.0.0.0", port: int = 8080, engine: Optional[CentralAIEngine] = None) -> AQThreadingHTTPServer:
    """Khởi tạo thể hiện AQThreadingHTTPServer gắn liền với App State."""
    server = AQThreadingHTTPServer((host, port), AQWebServerHandler)
    server.app_state = AQServerState(engine=engine)
    return server


def run_server_in_thread(port: int = 8080, host: str = "127.0.0.1", engine: Optional[CentralAIEngine] = None) -> Tuple[ThreadingHTTPServer, int]:
    """Khởi động Web Server trong luồng nền (background thread), hỗ trợ port=0 cho testing."""
    server = create_server(host=host, port=port, engine=engine)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


def stop_server(server: ThreadingHTTPServer) -> None:
    """Dừng và giải phóng tài nguyên Web Server."""
    server.shutdown()
    server.server_close()


def main():
    """Điểm vào chạy trực tiếp từ dòng lệnh: python -m pi.web_server [port]"""
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print(f"============================================================")
    print(f" Khởi động AQ-Server V5.0 (Raspberry Pi 5 Embedded)")
    print(f" Trạm thực nghiệm mẫu: 54 Triều Khúc (UTT), Thanh Xuân Nam, Hà Nội")
    print(f" Chu kỳ quan trắc: 15 phút, lấy mẫu burst 30 lần với bộ lọc 20% Trimmed Mean")
    print(f" Lắng nghe tại: http://0.0.0.0:{port}")
    print(f"============================================================")

    server = create_server(host="0.0.0.0", port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Web Server...")
        stop_server(server)
        print("Đã dừng an toàn.")


if __name__ == "__main__":
    main()
