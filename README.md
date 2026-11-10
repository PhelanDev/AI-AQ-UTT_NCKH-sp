# HỆ THỐNG QUAN TRẮC VÀ DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ (AIoT PRODUCTION ARCHITECTURE)
> **Trường Đại học Công nghệ Giao thông Vận tải (UTT)** - Số 54 Triều Khúc, Thanh Xuân, Hà Nội  
> **Kiến trúc**: Edge TinyML trên vi điều khiển ESP32-S3 & Central Gateway AI trên Raspberry Pi 5  
> **Triết lý thiết kế**: *Pragmatic Engineering • Zero Over-Engineering • Deterministic Execution*

---

## 📌 1. TỔNG QUAN HỆ THỐNG

Dự án phát triển một hệ thống quan trắc và dự báo nồng độ ô nhiễm bụi mịn $PM_{2.5}$ vi khí hậu thời gian thực, phục vụ khuôn viên giảng đường và các nút giao thông lân cận trường ĐH Công nghệ GTVT (UTT).

Hệ thống được thiết kế theo mô hình **AIoT 2 tầng phân cấp**:
1. **Tầng Trạm Đo Biên (Edge TinyML - ESP32-S3)**: 
   * Thu thập dữ liệu từ cảm biến bụi **Plantower PMS7003** (UART) và cảm biến nhiệt/ẩm **Sensirion SHT40** (I2C).
   * Hiển thị trạng thái màu cảnh báo theo chuẩn VN-AQI trên màn hình **LCD ST7789 IPS 2.0"** qua giao tiếp SPI tốc độ cao (20MHz).
   * Thực thi mô hình **Standard MLP lượng tử hóa đối xứng INT8** (833 tham số) với chính sách **0-Byte Heap RAM**, khử hiện tượng phồng hạt do nồm ẩm miền Bắc ($RH > 90\%$).
   * Vận hành chu kỳ tiết kiệm điện FSM 15 phút (Warmup 30s $\to$ Sample 10s $\to$ Infer $0.3\mu\text{s}$ $\to$ MQTT/LCD 5s $\to$ Deep Sleep 14m15s với dòng nghỉ $\approx 10\mu\text{A}$).
2. **Tầng Trạm Thu Trung Tâm (Gateway AI - Raspberry Pi 5)**:
   * Thu nhận luồng dữ liệu trạm đo qua MQTT.
   * Lọc dữ liệu ngoại lai bằng thuật toán thống kê trượt **3-Sigma ($Z > 3$)** kết hợp ranh giới kỹ thuật quốc gia **QCVN 05:2023/BTNMT**.
   * Dự báo xu hướng nồng độ bụi 24 giờ tiếp theo bằng mạng hồi quy **Vanilla GRU (32 hidden units)**.
   * Nội suy bản đồ nhiệt ô nhiễm toàn trường bằng thuật toán khoảng cách nghịch đảo **IDW ($p=2.0$)**.
   * Cung cấp Web Dashboard trực quan hóa qua FastAPI và Leaflet.

---

## ⚖️ 2. PHÂN TÍCH ĐÁNH ĐỔI KỸ THUẬT: FULL ACADEMIC VS. PRODUCTION LITE

Trong quá trình R&D, hệ thống đã được thử nghiệm và đo kiểm độc lập qua 2 phiên bản kiến trúc. Dựa trên số liệu thực nghiệm đo đạc thực tế, nhóm kỹ sư đã lựa chọn **Bản Lite** làm kiến trúc triển khai thực tế trên nhánh `main`:

| Tiêu chí kỹ thuật | Bản Nghiên Cứu (`research/academic-full`) | Bản Thực Chiến (`main`) | Ý nghĩa kỹ thuật & Lựa chọn triển khai |
| :--- | :--- | :--- | :--- |
| **Mô hình Edge (ESP32-S3)** | **ResMLP** (Residual Skip-Connection, LayerNorm) | **Standard MLP (2 tầng ẩn)** (Dense 32 $\to$ Dense 16, ReLU) | Bản Lite dùng cấu trúc nén kinh điển, loại bỏ rủi ro overfit trên vector đặc trưng 8 chiều. |
| **Số tham số Edge** | **19.297 tham số** | **833 tham số** | **Bản Lite nhẹ hơn 23 lần**, giảm triệt để gánh nặng tính toán trên MCU. |
| **Bảng trọng số (`model_data.h`)** | ~19.2 KB Flash | **784 Bytes** (< 1 KB) | Bảng trọng số Lite nằm gọn trong L1 Cache của CPU ESP32. |
| **Độ trễ suy luận Edge** | $10.60\ \mu\text{s}$ / lần (94.340 inferences/s) | **$0.30\ \mu\text{s}$ / lần** (**> 3.330.000 inferences/s**) | **Bản Lite nhanh hơn 35 lần**. |
| **Độ chính xác Edge ($R^2$, RMSE)** | $R^2 \approx 0.88$, $\text{RMSE} \approx 8.9\ \mu\text{g/m}^3$ | **$R^2 = 0.8182$, $\text{RMSE} = 11.23\ \mu\text{g/m}^3$** | Sai số chênh lệch chỉ $\sim 2.3\ \mu\text{g/m}^3$, hoàn toàn chìm trong dải sai số nội tại ($\pm 10\ \mu\text{g/m}^3$) của cảm biến quang PMS7003. |
| **Mô hình Gateway (Pi 5)** | **Dilated TCN + Bi-GRU + Dual Heads** (~43k params) | **Vanilla GRU 1 tầng** (**6.192 params**) | Giảm 7 lần số lượng tham số, tránh bẫy overfitting chuỗi thời gian ngắn. |
| **Thời gian tính toán Gateway** | ~45 ms / frame (CPU load 12-15%) | **1.6 ms / frame** (CPU load **< 1%**) | **Bản Lite nhanh hơn 28 lần**, Gateway có thể mở rộng lên hàng trăm trạm đo. |
| **Nội suy bản đồ nhiệt** | Ordinary Kriging (Bán phương sai Semivariogram) | **IDW ($p=2.0$, Nghịch đảo bình phương)** | Kriging dễ bị lỗi suy biến ma trận (Singular Matrix) khi số trạm đo ít ($N < 4$). IDW chạy ổn định $100\%$ không bao giờ crash. |
| **Lọc dữ liệu dị thường** | 16D Deep Autoencoder + Mahalanobis Distance | **3-Sigma ($Z > 3$) + Ngưỡng QCVN 05:2023** | 3-Sigma mượt mà, bám sát quy chuẩn môi trường nhà nước, loại trừ báo động giả (False Alarms). |

> 📌 **Lưu trữ mã nguồn**:
> * Nhánh **`main`**: Bản triển khai thực tế tối ưu hóa (Production / Applied Edition).
> * Nhánh **`research/academic-full`**: Bản nghiên cứu chuyên sâu phục vụ công bố bài báo khoa học (Academic Master Edition).

---

## 📁 3. CẤU TRÚC MÃ NGUỒN

```
E:\AQ
├── esp/                         # Firmware C++ cho ESP32-S3
│   ├── include/
│   │   ├── config.h             # Cấu hình GPIO phần cứng, SPI LCD, I2C, Octal PSRAM lock
│   │   ├── model_data.h         # Bảng trọng số INT8 Standard MLP Lite (784 Bytes)
│   │   └── tinyml_edge_engine.h # Giao diện suy luận 0-Heap
│   ├── src/
│   │   ├── sensor_drivers.cpp   # Drivers đọc PMS7003 UART, Sensirion SHT40 I2C
│   │   ├── st7789_lcd_driver.cpp# Driver màn hình màu ST7789 SPI 20MHz
│   │   ├── power_scheduler.cpp  # FSM chu kỳ 15 phút và RTC Deep Sleep
│   │   └── tinyml_edge_engine.cpp # Động cơ nhân ma trận số nguyên INT8
│   └── tests/                   # Bộ test C++ Native (17/17 tests PASS)
├── pi/                          # Trung tâm Gateway trên Raspberry Pi 5
│   ├── central_engine.py        # Pipeline thu nhận, xử lý và lưu trữ dữ liệu trạm
│   ├── forecasting_engine.py    # Mô hình Vanilla GRU 24 giờ
│   ├── anomaly_engine.py        # Bộ phát hiện ngoại lai 3-Sigma + QCVN 05:2023
│   ├── spatial_temporal_engine.py # Thuật toán nội suy không gian IDW (p=2.0)
│   └── web_server.py            # API FastAPI và Web Dashboard Leaflet
├── mlops/                       # Huấn luyện và lượng tử hóa mô hình
│   ├── models/                  # Checkpoints PyTorch (.pt)
│   └── scripts/
│       ├── train_and_quantize_standard_mlp_lite.py # Huấn luyện và xuất header C
│       └── train_lite_central_models.py           # Huấn luyện mô hình GRU trung tâm
├── docs/                        # Tài liệu kỹ thuật & Cẩm nang bảo vệ
│   └── SO_SANH_BAN_FULL_VA_LITE.md # Đối chiếu chi tiết & Bộ câu hỏi phản biện
└── tests/                       # Test suite Python toàn diện (46/46 tests PASS)
```

---

## 🚀 4. HƯỚNG DẪN KHỞI CHẠY VÀ KIỂM THỬ

### 4.1. Kiểm thử Firmware C++
```powershell
Set-Location E:\AQ\esp\tests
# Biên dịch và chạy bộ test FSM 15 phút, LCD ST7789, SHT40 (17/17 tests):
g++ -std=c++17 -I../include test_firmware_15min_cycle.cpp ../src/sensor_drivers.cpp ../src/st7789_lcd_driver.cpp ../src/power_scheduler.cpp ../src/tinyml_edge_engine.cpp ../src/display_mqtt_hal.cpp -o test_firmware_15min_cycle.exe
.\test_firmware_15min_cycle.exe

# Biên dịch và chạy benchmark độ trễ TinyML 833 tham số:
g++ -std=c++17 -I../include test_main.cpp ../src/tinyml_edge_engine.cpp -o test_edge_engine.exe
.\test_edge_engine.exe
```

### 4.2. Kiểm thử Python Test Suite
```powershell
Set-Location E:\AQ
pytest tests/ -v
```

### 4.3. Khởi chạy Web Server Gateway
```powershell
Set-Location E:\AQ
python -m uvicorn pi.web_server:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập giao diện Web thời gian thực tại: `http://localhost:8000`

---

## 📜 5. BẢN QUYỀN VÀ TRÍCH DẪN
Dự án Nghiên cứu Khoa học Trường Đại học Công nghệ Giao thông Vận tải (UTT).  
Tác giả: PhelanDev.
