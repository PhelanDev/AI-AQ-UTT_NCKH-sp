# Web Dashboard & Interactive Spatial Map V5.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng máy chủ Web Server nhúng siêu nhẹ và Giao diện Web Dashboard tương tác hoàn chỉnh cho hệ thống quan trắc chất lượng không khí, trực quan hóa trạm Prototype tại 54 Triều Khúc, quầng nhiệt thích ứng V5.0 (6 mốc thời gian) trên bản đồ tương tác và biểu đồ chuỗi thời gian 24 giờ.

**Architecture:** Kiến trúc Client-Server siêu nhẹ tuân thủ kỷ luật Ponytail (Stdlib-first). Backend sử dụng Python `http.server.ThreadingHTTPServer` kết nối trực tiếp với `CentralAIEngine` để cung cấp các REST API (`/api/status`, `/api/telemetry`, `/api/map`, `/api/forecast`, `/api/trigger_cycle`). Frontend là Single-Page Application (SPA) hiện đại (HTML5 + Canvas Heatmap Renderer + Leaflet.js 1.9.4 + Chart.js 4.4.1), không yêu cầu cài đặt `node_modules` hay build-step cồng kềnh, chạy mượt mà ngay trên Raspberry Pi 5.

**Tech Stack:** Python 3.12 (Standard Library `http.server`, `json`, `socketserver`), HTML5, CSS3 (Modern Glassmorphism/Dark Mode), JavaScript (ES6+), Leaflet.js (OpenStreetMap), Chart.js (24h Uncertainty Band), Pytest.

**Spec:** Bản đặc tả kỹ thuật V5.0 tại [`docs/reports/ai_training_evaluation_report.md`](file:///e:/AQ/docs/reports/ai_training_evaluation_report.md#L325-L440) và [`README.md`](file:///e:/AQ/README.md).

## Global Constraints

- **Zero Heavy Web Dependencies**: Không cài thêm framework nặng (như React/Vue/Node.js monorepo build) để bảo đảm Raspberry Pi 5 khởi động tức thì trong $< 1\text{s}$ và tiêu thụ RAM $< 15\text{ MB}$.
- **Zero-Heap/Zero-Drift Ingestion**: Dữ liệu nạp trực tiếp từ `CentralAIEngine.process_telemetry_frame`.
- **Honest Coordinate Pin**: Ghim bản đồ đặt chính xác tại 54 Triều Khúc, Thanh Xuân Nam, Hà Nội (`lat: 20.9852, lon: 105.7972`), không sinh trạm ảo khi chỉ có 1 trạm.
- **Multi-Horizon Heatmap Sync**: Lớp phủ bản đồ nhiệt phải render mượt mà 6 khung thời gian (`current`, `+1h`, `+3h`, `+6h`, `+12h`, `+24h`) theo thanh trượt thời gian của người dùng.
- **Sub-25ms Response Latency**: Toàn bộ API endpoint phải phản hồi dưới 25ms.

## Review Focus

1. **Map Overlay Coordinate Bounds Alignment**: Tọa độ góc của Canvas Heatmap trên Leaflet phải khớp 100% với `web_map["bounds"]` xuất từ `CentralAIEngine`.
2. **Opacity Grid Blending**: Quầng màu nhiệt xung quanh trạm UTT phải đậm ở tâm và chuyển trong suốt mượt mà ở rìa xa theo `opacity_grid`, không che khuất đường phố OpenStreetMap.
3. **Empty/Offline Handling**: Khi chưa có dữ liệu cảm biến hoặc lỗi mạng, dashboard hiển thị trạng thái chờ mượt mà, không bị crash JavaScript console.
4. **Interactive Timeline Scrubbing**: Kéo thanh trượt thời gian 6 mốc phải cập nhật canvas tức thì dưới 16ms (60 FPS).
5. **Full Test Suite Integrity**: Toàn bộ 358 bài kiểm thử cũ và các bài kiểm thử mới của web server phải đạt 100% PASS.

---

## Danh Mục Tệp (File Structure)

```
E:\AQ\
├── pi/
│   ├── web_server.py           # 🔵 [NEW] Embedded Threading HTTP Server & REST API Dispatcher
│   └── web/
│       ├── index.html          # 🌐 [NEW] Interactive Dashboard SPA (Single-File Architecture)
│       └── css/                # Optional static styles or embedded in index.html
│
├── tests/
│   └── test_web_server.py      # 🧪 [NEW] TDD Test suite cho Web Server & REST Endpoints
│
└── README.md                   # 📚 [UPDATE] Hướng dẫn khởi chạy Web Dashboard
```

---

## Tasks Thực Hiện

### Task 1: Xây dựng Test Suite TDD cho Web Server (`tests/test_web_server.py`) (RED)
- [ ] Viết test `test_web_server_endpoints`:
  - `GET /` trả về mã 200, Content-Type `text/html`, chứa các thẻ dashboard.
  - `GET /api/status` trả về mã 200, JSON chứa `status == "OK"`, uptime, model info.
  - `GET /api/telemetry` trả về mã 200, JSON chứa thông số trạm UTT (PM2.5, VN-AQI, Temp, RH, CO2, VOC).
  - `GET /api/map` trả về mã 200, JSON chứa `web_map` V5.0 (1 marker tại 54 Triều Khúc, bounds, 6 horizons).
  - `GET /api/forecast` trả về mã 200, JSON chứa 24 giờ dự báo kèm $P_{10}, P_{50}, P_{90}$.
  - `POST /api/trigger_cycle` kích hoạt tính toán chu kỳ mới tức thì.
  - Đo độ trễ phản hồi API $< 25\text{ ms}$.
- [ ] Chạy pytest xác nhận bài test FAIL (RED) do chưa có file `pi/web_server.py`.

### Task 2: Cài đặt Máy chủ Web Siêu Nhẹ (`pi/web_server.py`) (GREEN)
- [ ] Xây dựng class `AQWebServerHandler(http.server.BaseHTTPRequestHandler)`:
  - Phân luồng REST API router: `/api/status`, `/api/telemetry`, `/api/map`, `/api/forecast`, `/api/trigger_cycle`.
  - Phục vụ tệp tĩnh `index.html` cho đường dẫn gốc `/`.
  - Hỗ trợ CORS headers và `application/json`.
- [ ] Tích hợp singleton `CentralAIEngine` để xử lý chu kỳ dữ liệu tự động hoặc theo yêu cầu.
- [ ] Hỗ trợ chạy trực tiếp từ dòng lệnh: `python -m pi.web_server --port 8080 --host 0.0.0.0`.
- [ ] Chạy `pytest tests/test_web_server.py` xác nhận 100% PASS (GREEN).

### Task 3: Thiết kế Giao Diện Web Dashboard Tương Tác (`pi/web/index.html`)
- [ ] Xây dựng layout Single-Page Application (SPA) hiện đại (Clean Dark-Mode / Glassmorphism):
  - **Header**: Tên đề tài NCKH, Logo, Trạng thái hệ thống (Online/Offline), Đồng hồ chu kỳ 15 phút.
  - **Khối Bản Đồ Không Gian (Leaflet.js)**:
    - Bản đồ OpenStreetMap nền tối (CartoDB DarkMatter hoặc OSM tiêu chuẩn).
    - Marker ghim tại **Trường ĐH Công nghệ Giao thông Vận tải (54 Triều Khúc)** với pulse animation.
    - Custom Canvas Layer: Vẽ ma trận nhiệt $25 \times 25$ với gradient màu EPA/QCVN và độ mờ `opacity_grid`.
    - **Thanh điều khiển Horizon Slider**: 6 nút bấm chọn nhanh (`Hiện tại`, `+1h`, `+3h`, `+6h`, `+12h`, `+24h`).
  - **Khối Thẻ Chỉ Số (Metric Cards)**:
    - Thẻ VN-AQI lớn với cấp độ chất lượng không khí Việt Nam.
    - Các thẻ: Bụi mịn PM2.5, PM10, Nhiệt độ, Độ ẩm nồm, SenseAir S8 CO2, SGP41 Gas VOC, Điểm sương.
  - **Khối Biểu Đồ Chuỗi Thời Gian (Chart.js)**:
    - Đồ thị đường PM2.5 trong 24 giờ tới.
    - Dải tin cậy $P_{10} - P_{90}$ dạng vùng bóng mờ bán trong suốt (Shaded Confidence Band).
    - Đường ngưỡng quy chuẩn QCVN 05:2023 ($50\ \mu\text{g/m}^3$).
  - **Khối Điều Khiển NCKH Demo**:
    - Nút "Chạy Chu Kỳ Mới (Trigger Cycle)" cập nhật tức thì phục vụ thuyết minh hội đồng.

### Task 4: Kiểm Thử Toàn Diện & Tích Hợp Hệ Thống
- [ ] Chạy toàn bộ test suite `pytest -q` xác nhận 100% tests PASS (358 tests cũ + tests web mới).
- [ ] Cập nhật [`README.md`](file:///e:/AQ/README.md) hướng dẫn khởi chạy web server.
- [ ] Commit toàn bộ mã nguồn lên git branch `feat/v5-prototype-heatmap`.
