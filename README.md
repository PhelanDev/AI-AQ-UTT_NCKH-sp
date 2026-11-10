# HỆ THỐNG QUAN TRẮC VÀ DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ (PHIÊN BẢN BÁO CÁO ĐỒ ÁN / LITE)

> **Dự án**: Nghiên cứu Khoa học & Báo cáo Đồ án Tốt nghiệp  
> **Địa bàn thử nghiệm**: Trường Đại học Công nghệ Giao thông Vận tải (UTT) - Số 54 Triều Khúc, Thanh Xuân, Hà Nội  
> **Kiến trúc**: Edge TinyML trên vi điều khiển ESP32-S3 kết hợp AI Trung tâm trên Raspberry Pi 5

---

## 📌 1. TỔNG QUAN HỆ THỐNG

Phiên bản **AQ_Lite** được thiết kế tinh giản, tối ưu hóa các mô hình toán học và học máy nhằm đảm bảo:
1. **Tính trực quan và chuẩn mực**: Sử dụng các thuật toán chuẩn mực trong giáo trình khoa học máy tính và kỹ thuật viễn thông (Standard MLP, GRU, IDW, 3-Sigma), giúp sinh viên dễ dàng chứng minh công thức và bảo vệ trước Hội đồng chấm thi.
2. **Hiệu năng thực thi cao trên phần cứng hạn chế**: Bảng trọng số AI Edge chỉ chiếm **784 Bytes** (< 1 KB), tuyệt đối **0-Byte cấp phát động (Heap RAM)** trên ESP32-S3, tốc độ suy luận dưới **1 microgiây**.
3. **Tiết kiệm năng lượng triệt để**: Chu kỳ FSM 15 phút với chế độ Deep Sleep, kéo dài tuổi thọ nguồn pin/ắc quy ngoài trời.

```
[ Trạm Đo ESP32-S3 ]
  ├── Cảm biến SHT40 (I2C) & PMS7003 (UART)
  ├── Edge TinyML: Standard MLP (833 tham số, Full INT8, 0B Heap)
  ├── Màn hình LCD ST7789 SPI (Hiển thị AQI & Năng lượng)
  └── Truyền thông: MQTT qua Wi-Fi / ESP-NOW
            │ (Chu kỳ 15 phút)
            ▼
[ Gateway Raspberry Pi 5 ]
  ├── Gateway Engine: Thu nhận dữ liệu & Lọc ngoại lai 3-Sigma / QCVN 05:2023
  ├── Central AI Forecaster: Vanilla GRU 1 tầng (32 units, ~6.1k tham số)
  ├── Spatial Engine: Nội suy ô nhiễm không gian IDW (p=2.0) & Gaussian Plume
  └── Web Dashboard: FastAPI + Leaflet Map thời gian thực
```

---

## ⚙️ 2. THÔNG SỐ KỸ THUẬT CỐT LÕI

| Thành phần | Đặc tả kỹ thuật Phiên bản Lite | Đơn vị / Ý nghĩa |
| :--- | :--- | :--- |
| **Edge Vi điều khiển** | ESP32-S3 WROOM-1 (Xtensa Dual-Core 240MHz) | SRAM 512KB, ROM 384KB |
| **Edge TinyML Model** | Standard MLP (Input 8 -> Dense 32 ReLU -> Dense 16 ReLU -> Dense 1 Linear) | 833 tham số |
| **Lượng tử hóa Edge** | INT8 đối xứng (Symmetric INT8 Quantization) | Bảng trọng số 784 Bytes (< 1 KB) |
| **Cấp phát RAM Edge** | **0 Byte Dynamic Heap** (Toàn bộ mảng tĩnh) | An toàn chống rò rỉ bộ nhớ |
| **Thời gian suy luận Edge** | ~0.30 microgiây / lần | > 3.300.000 lần/giây |
| **Độ chính xác hiệu chỉnh Edge** | $R^2 = 0.8182$, $\text{RMSE} = 11.23\ \mu\text{g/m}^3$ | Hiệu chỉnh độ ẩm, nhiệt độ cho PMS7003 |
| **Chu kỳ hoạt động FSM** | 15 phút (Warmup 30s -> Sample 10s -> Infer 0.3µs -> Send 5s -> Sleep 14m15s) | Dòng nghỉ Deep Sleep ~10µA |
| **Gateway AI Model** | Vanilla GRU 1 tầng (32 hidden units, 1 fc) | 6.192 tham số |
| **Nhiệm vụ Gateway AI** | Dự báo nồng độ $PM_{2.5}$ cho 24 giờ tiếp theo | Khởi chạy < 2ms / frame |
| **Phát hiện dị thường** | Shallow Autoencoder MSE (4.208 tham số) + Ngưỡng động 3-Sigma + QCVN 05:2023 | Học máy không giám sát chuẩn mực |
| **Nội suy bản đồ ô nhiễm** | IDW (Inverse Distance Weighting, số mũ khoảng cách $p = 2.0$) | Bản đồ nhiệt phân giải cao |

---

## 📁 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```
E:\AQ_Lite
├── esp/                         # Firmware và TinyML cho ESP32-S3
│   ├── include/
│   │   ├── config.h             # Cấu hình phần cứng, GPIO LCD ST7789, Sensirion SHT40
│   │   ├── model_data.h         # Bảng trọng số INT8 của StandardMLPLite (784 Bytes)
│   │   └── tinyml_edge_engine.h # Khai báo engine suy luận 0-Heap
│   ├── src/
│   │   ├── main.cpp             # Mã nguồn chính ESP-IDF / Arduino FreeRTOS FSM
│   │   ├── tinyml_edge_engine.cpp # Triển khai toán học nhân ma trận INT8
│   │   └── display_mqtt_hal.cpp # Driver ST7789 SPI LCD và MQTT Ring Buffer
│   └── tests/                   # Bộ kiểm thử C++ Native (17/17 tests PASS)
├── pi/                          # Hệ thống Gateway và Web Server trên Raspberry Pi 5
│   ├── central_engine.py        # Điều phối nạp dữ liệu, tiền xử lý và suy luận
│   ├── forecasting_engine.py    # Mô hình Vanilla GRU 24 giờ
│   ├── spatial_temporal_engine.py # Thuật toán nội suy không gian IDW (p=2.0)
│   └── web_server.py            # API FastAPI và Web Dashboard Leaflet
├── mlops/                       # Huấn luyện và lượng tử hóa mô hình
│   ├── models/                  # Checkpoints PyTorch (.pt)
│   └── scripts/
│       ├── train_and_quantize_standard_mlp_lite.py # Train & Export model_data.h
│       └── train_lite_central_models.py           # Huấn luyện mô hình GRU trung tâm
├── docs/                        # Tài liệu kỹ thuật & Hướng dẫn bảo vệ
│   └── SO_SANH_BAN_FULL_VA_LITE.md # Bảng đối chiếu Full vs Lite & Bộ câu hỏi Hội đồng
└── tests/                       # Test suite toàn diện (pytest)
```

---

## 🚀 4. HƯỚNG DẪN KHỞI CHẠY VÀ KIỂM THỬ

### 4.1. Chạy Bộ Test Firmware C++ (Trên PC / Môi trường mô phỏng)
Yêu cầu đã cài đặt GCC (MinGW / MSYS2 UCRT64):
```powershell
Set-Location E:\AQ_Lite\esp\tests
# Biên dịch và chạy test chu kỳ FSM 15 phút:
g++ -std=c++17 -I../include test_firmware_15min_cycle.cpp ../src/display_mqtt_hal.cpp -o test_firmware_15min_cycle.exe
.\test_firmware_15min_cycle.exe

# Biên dịch và chạy test kiểm tra độ trễ và độ chính xác TinyML:
g++ -std=c++17 -I../include test_edge_engine.cpp ../src/tinyml_edge_engine.cpp -o test_edge_engine.exe
.\test_edge_engine.exe
```

### 4.2. Huấn Luyện Lại Mô Hình (Nếu Thay Đổi Dataset)
```powershell
Set-Location E:\AQ_Lite
# Huấn luyện và xuất file C header model_data.h cho ESP32-S3:
python mlops/scripts/train_and_quantize_standard_mlp_lite.py

# Huấn luyện mô hình Vanilla GRU cho Gateway:
python mlops/scripts/train_lite_central_models.py
```

### 4.3. Chạy Toàn Bộ Test Suite Python
```powershell
Set-Location E:\AQ_Lite
pytest tests/ -v
```

### 4.4. Khởi Chạy Web Dashboard Gateway
```powershell
Set-Location E:\AQ_Lite
python -m uvicorn pi.web_server:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập giao diện Web thời gian thực tại: `http://localhost:8000`

---

## 🎓 5. TÀI LIỆU HỘI ĐỒNG BẢO VỆ
Xem tài liệu phân tích ưu nhược điểm và bí quyết bảo vệ đồ án tại:  
👉 [Báo Cáo Đối Chiếu Kỹ Thuật & Kịch Bản Bảo Vệ Đồ Án](file:///E:/AQ_Lite/docs/reports/bang_so_sanh_mo_hinh_ban_xin_vs_ban_nop.md)
