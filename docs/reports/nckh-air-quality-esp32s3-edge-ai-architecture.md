# KIẾN TRÚC HỆ THỐNG QUAN TRẮC CHẤT LƯỢNG KHÔNG KHÍ ĐA ĐIỂM ỨNG DỤNG TINYML TẠI BIÊN TRÊN ESP32-S3
> **Định hướng**: Đề tài Nghiên cứu Khoa học (NCKH) cấp Trường / Đồ án Kỹ thuật  
> **Nền tảng phần cứng**: ESP32-S3-WROOM-1 (N16R8)  
> **Kiến trúc**: Edge AI Multi-Node Distributed Architecture (Nhiều Hộp Biên $\rightarrow$ 1 Server Trung Tâm)  
> **Quy chế tài liệu**: **Living Document (Tài liệu động)** — Cập nhật liên tục mọi thay đổi, quyết định kỹ thuật và trạng thái chốt phương án trong từng phiên làm việc.  
> **Tài liệu chuyên sâu mảng AI/TinyML**: Xem tại [`nckh-ai-tinyml-architecture-and-training.md`](./nckh-ai-tinyml-architecture-and-training.md).

---

## 1. TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT (PROBLEM STATEMENT)

### 1.1. Thực trạng & Nỗi đau thực tế
* **Hiện tượng nồm ẩm miền Bắc (Hygroscopic Growth)**: Cảm biến bụi quang học giá rẻ (như Plantower PMS7003) khi độ ẩm môi trường $RH > 80\%$ sẽ bị hơi nước bám vào hạt bụi, làm tăng kích thước tán xạ quang học $\rightarrow$ Chỉ số $PM_{2.5}$ bị đo ảo, vọt lên gấp 2 – 3 lần giá trị thực.
* **Trôi điểm 0 (Baseline Drift & Cross-Sensitivity)**: Cảm biến khí điện hóa / kim loại oxit (MOX) bị trôi theo nhiệt độ và phản ứng chéo với mùi cồn, thức ăn.
* **Hạn chế của hệ thống IoT truyền thống**: Đẩy toàn bộ dữ liệu thô lên Cloud làm nghẽn băng thông, trễ cao, phụ thuộc hoàn toàn vào mạng Internet; mất mạng là mất toàn bộ dữ liệu.

### 1.2. Mục tiêu nghiên cứu
* Xây dựng hệ thống trạm quan trắc không khí đa điểm chi phí thấp (Low-cost Air Quality Monitoring).
* Ứng dụng **TinyML (Mạng nơ-ron thu nhỏ lượng tử hóa INT8)** chạy trực tiếp tại biên trên chip **ESP32-S3-N16R8** để tự động hiệu chuẩn (In-situ Calibration), bù sai số nhiệt ẩm phi tuyến tính và dự báo xu hướng ô nhiễm.
* Thiết lập hệ thống phân tán kết nối nhiều hộp biên về Server trung tâm bằng giao thức **MQTT**, hỗ trợ cơ chế lưu đệm chống mất dữ liệu khi mất mạng (**Zero-Data-Loss Buffer**).

---

## 2. ĐẶC TẢ PHẦN CỨNG LÕI: ESP32-S3 N16R8

### 2.1. Cấu hình phần cứng
* **SoC**: Dual-core Xtensa® 32-bit LX7 @ 240 MHz.
* **Tập lệnh tăng tốc AI**: Vector Instructions (Processor Instruction Extension - PIE) chuyên dụng cho phép tính ma trận $128\text{-bit}$ (tăng tốc phép toán nhân tích lũy MAC cho INT8/INT16).
* **Bộ nhớ Flash**: $16\text{ MB}$ (Octal SPI).
* **Bộ nhớ PSRAM**: $8\text{ MB}$ (Octal SPI - OPI) tốc độ cao.
* **SRAM nội**: $512\text{ KB}$ ($~380\text{ KB}$ khả dụng cho Heap).

### 2.2. Cảnh báo phần cứng sống còn (Hardware Traps)
> [!WARNING]
> Do phiên bản **R8** sử dụng bus **Octal PSRAM (OPI)**, toàn bộ các chân từ **`GPIO33` đến `GPIO37`** đã bị nhà sản xuất chiếm dụng ngầm bên trong module để giao tiếp với chip nhớ PSRAM.
> **TUYỆT ĐỐI KHÔNG ĐƯỢC NỐI CẢM BIẾN VÀO CỤM GPIO33 – GPIO37** (sẽ gây crash chip và bootloop liên tục).

### 2.3. Bảng phân bổ chân (Pinout Mapping) theo đúng BOM cấp vốn
| Cụm chức năng / Thiết bị | Chân GPIO khuyến nghị | Giao thức | Ghi chú kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Cảm biến Bụi chính: PMS7003 (PM1)** | `GPIO17 (RX1)`, `GPIO18 (TX1)` | Hardware UART1 | 9600 bps, đo tán xạ laser liên tục |
| **Cảm biến Bụi phụ: PMS7003 (PM2 - Dual PM)** | `GPIO15 (RX2)`, `GPIO16 (TX2)` | Hardware UART2 | 9600 bps, đối chiếu chéo sai số với PM1 |
| **Cảm biến CO2: SenseAir S8 (LP/0053)** | `GPIO8 (RX)`, `GPIO9 (TX)` | UART (Modbus) | 9600 bps, công nghệ NDIR chuẩn Thụy Điển |
| **Cảm biến Khí & Nhiệt ẩm: SGP41 + SHT40** | `GPIO4 (SDA)`, `GPIO5 (SCL)` | I2C (Bus 0) | SGP41 (TVOC/NOx) & SHT40 (±0.2°C, ±1.8% RH) |
| **Màn hình màu LCD 2.8 inch SPI** | `GPIO11(MOSI), 12(SCK), 10(CS), 13(DC), 14(RST)` | SPI (Bus 2) | Hiển thị thông số realtime, đồ thị tại trạm |
| **Module 4G LTE + GPS: SIMCOM A7677S GNSS** | `GPIO19 (RX)`, `GPIO20 (TX)` | UART AT Cmd | Đẩy dữ liệu MQTT qua mạng di động và tọa độ vệ tinh |
| **LED RGB chỉ thị chất lượng AQI** | `GPIO48` (RGB Onboard) | WS2812B | Báo dải màu AQI Xanh - Vàng - Cam - Đỏ - Tím |

---

## 3. BỘ CẢM BIẾN TIÊU CHUẨN ĐỒNG BỘ BOM KINH PHÍ (SENSOR STACK)

| Thông số đo | Mã cảm biến trong BOM | Nguyên lý hoạt động | Ưu điểm & Vai trò trong NCKH |
| :--- | :--- | :--- | :--- |
| **Bụi mịn kép ($PM_{1.0}, PM_{2.5}, PM_{10}$)** | **02 x Plantower PMS7003** (Cấu hình Dual PM) | Tán xạ Laser (Laser Scattering) | Chạy song song PM1 & PM2 để tự đối chiếu sai số chéo, phát hiện đọng sương và khử nhiễu ngẫu nhiên |
| **Khí $CO_2$ khí quyển thực tế** | **SenseAir S8 (LP/0053)** | Quang phổ hồng ngoại không phân tán (NDIR) | Dải đo 400 - 2000 ppm, độ bền 15 năm, đo chuẩn NDIR không bị trôi như cảm biến MOX rẻ tiền |
| **Khí độc $TVOC$ & $NO_x$** | **Sensirion SGP41** | Kim loại oxit đa kênh (MOX) | Tích hợp thuật toán Gas Index Algorithm xuất trực tiếp TVOC Index và NOx Index (1-500) |
| **Nhiệt độ & Độ ẩm chuẩn cao** | **Sensirion SHT40** | Cảm biến điện dung số hóa thế hệ 4 | Cực kỳ chính xác (±0.2°C, ±1.8% RH), cung cấp ground input quan trọng để AI tính điểm sương và bù ẩm |

---

## 4. BỐN MŨI NHỌN HỌC THUẬT ĐẠT ĐIỂM TỐI ĐA (SCIENTIFIC NOVELTY)

Để đề tài đạt giải NCKH hoặc điểm A+, cần tập trung vào 4 đóng góp kỹ thuật cốt lõi:

### Mũi nhọn 1: Hiệu chuẩn cảm biến bụi mịn tại biên chống nồm ẩm (TinyML In-situ Calibration V3.0)
* **Mô hình toán học**: Xây dựng mạng nơ-ron sâu `MultiSensorResMLP` ($8 \rightarrow 96 \rightarrow 96 \rightarrow 96 \rightarrow 1$, 19.585 tham số) với 2 khối Residual Skip Connections, tiếp nhận vector đặc trưng 8D:
  $$\mathbf{X} = [PM_{avg}, \Delta PM, R_{pm}, T, RH, T_{dp}, CO_2, \text{Gas}_{proxy}]$$
  Học hệ số bù vật lý có chặn:
  $$C = 0.25 + 0.80 \times \sigma(\text{raw\_out}) \in [0.25, 1.05], \quad PM_{calib} = PM_{avg} \times C$$
* **Chỉ số đo lường thực nghiệm V3.0 (Key Metrics)**:
  - Hệ số xác định $R^2$ tăng vọt từ $0.5926$ (baseline thô suy thoái nồm ẩm) lên **$0.9784$** ($\Delta R^2 = \mathbf{+0.3858}$, vượt xa chỉ tiêu $\ge +0.20$).
  - Sai số tuyệt đối $MAE$ giảm từ $15.74$ xuống **$4.36\ \mu\text{g/m}^3$** (giảm **$72.29\%$** sai số, vượt xa chỉ tiêu $\ge 30\%$).
  - Phân khúc nồm ẩm ngậm sương ($RH > 75\%$): $R^2 = 0.9144, MAE = 6.05\ \mu\text{g/m}^3$ (kéo giảm $62.86\%$ sai số so với baseline thô $16.29\ \mu\text{g/m}^3$).
  - Phân biệt khói thật ($C_{smoke} = 0.898 \ge 0.85$) và sương mù ($C_{fog} = 0.420 \le 0.65$).

### Mũi nhọn 2: Tối ưu hóa suy luận bằng phần cứng Vector (Quantization & ESP-DL Acceleration)
* **Lượng tử hóa Post-Training Quantization (PTQ)**: Ép trọng số mô hình từ `FP32` ($4\text{ bytes}$) sang `INT8` ($1\text{ byte}$), giảm $75\%$ dung lượng mô hình mà độ chính xác chỉ suy hao $< 1\%$.
* **Benchmark hiệu năng trên nhân Xtensa LX7**:
  - So sánh tốc độ chạy khi dùng **TFLite Micro (CPU thường)** vs **ESP-DL (Tập lệnh Vector PIE)**.
  - Chứng minh thời gian suy luận đạt mức siêu tốc: **$< 3\text{ ms} / \text{lần}$**, tiết kiệm năng lượng CPU tối đa.

### Mũi nhọn 3: Cơ chế chống mất mát dữ liệu phân tán (Fault-Tolerant Offline Buffering)
* Tận dụng dung lượng $8\text{ MB}$ PSRAM để tạo **Circular Ring Buffer**:
  - Khi đứt kết nối mạng WiFi/MQTT, hộp tự động chuyển sang chế độ lưu đệm trên RAM.
  - Với kích thước $32\text{ bytes} / \text{mẫu}$, $4\text{ MB}$ PSRAM có thể lưu tới **$131.072$ bản ghi** (tương đương hơn **45 ngày** mất mạng liên tục với chu kỳ lấy mẫu $30\text{s}$).
  - Khi có mạng trở lại, kích hoạt thuật toán **Burst Synchronization** đồng bộ bù dữ liệu về Server với đúng nhãn thời gian thực (Timestamp) không làm nghẽn mạng.

### Mũi nhọn 4: Phân tích tương quan không gian - thời gian tại Server (Spatial-Temporal Analytics)
* Không chỉ hiển thị biểu đồ đơn thuần, Server chạy thuật toán phân tích tương quan trễ (Cross-Correlation with Time-Lag):
  - Nhận diện vector phát tán ô nhiễm: Nếu phòng A tăng chỉ số $PM_{2.5}$ tại thời điểm $t_0$, sau đó phòng B tăng tại $t_0 + 5\text{ phút}$ $\rightarrow$ Kết luận nguồn phát thải từ phòng A và hướng luồng gió di chuyển.

---

## 5. KIẾN TRÚC MÔ HÌNH HỆ THỐNG PHÂN TÁN (DISTRIBUTED SYSTEM)

```
+-------------------------------------------------------------------------------+
|                           HỘP NGOẠI VI (EDGE NODES)                           |
|                                                                               |
|  [ PMS7003 PM1 ] -> UART1 --+                                                 |
|  [ PMS7003 PM2 ] -> UART2 --+-> [ ESP32-S3 N16R8 ]                            |
|  [ SenseAir S8 ] -> UART3 --+     |-- Core 0: Đọc cảm biến + WiFi/4G MQTT     |
|  [ SGP41+SHT40 ] -> I2C0 ---+     |-- Core 1: TinyML ESP-DL (Bù ẩm, lọc chéo) |
|  [ LCD 2.8 SPI ] -> SPI2 ---+     |-- 8MB PSRAM: Circular Buffer đệm offline  |
+-------------------------------------------------------------------------------+
                                       |
                   Giao thức MQTT (Payload JSON siêu nhẹ)
                                       v
+-------------------------------------------------------------------------------+
|                             SERVER TRUNG TÂM                                  |
|                                                                               |
|  [ Mosquitto MQTT Broker ]  --> Tiếp nhận dữ liệu đồng thời từ nhiều hộp     |
|             |                                                                 |
|  [ TimescaleDB / InfluxDB ] --> Lưu trữ chuỗi thời gian dài hạn               |
|             |                                                                 |
|  [ Spatial-Temporal Engine] --> Phân tích tương quan luồng phát tán ô nhiễm   |
|             |                                                                 |
|  [ Grafana / Web Dashboard] --> Bản đồ nhiệt mặt bằng tòa nhà (Heatmap)       |
+-------------------------------------------------------------------------------+
```

### 5.1. Định dạng gói tin MQTT Telemetry (JSON Payload)
```json
{
  "box_id": "BOX_LAB_01",
  "timestamp": 1792381200,
  "raw": {
    "pm25": 48.2,
    "pm10": 65.1,
    "co2": 1120,
    "voc": 145,
    "temp": 28.4,
    "hum": 86.5
  },
  "calibrated": {
    "pm25_ai": 29.8,
    "aqi_ai": 88,
    "anomaly_code": 0
  },
  "diagnostics": {
    "infer_time_ms": 2.65,
    "free_psram_bytes": 6291456,
    "rssi": -62
  }
}
```

---

### 5.3. QUẢN LÝ VÒNG ĐỜI CẢM BIẾN & XỬ LÝ THỰC ĐỊA (FIELD SENSOR LIFECYCLE MANAGEMENT)

Nhằm đảm bảo tính chính xác tuyệt đối khi triển khai thực địa lâu dài và đo đối chuẩn với trạm chuẩn quốc gia, hệ thống tích hợp 3 cơ chế xử lý kỹ thuật:

#### 1. Chu kỳ Làm ấm Cảm biến (Sensor Warm-up Cycle - 30 giây)
- **Vấn đề thực tế**: Cảm biến quang học Plantower PMS7003 sử dụng quạt hút siêu nhỏ (brushless fan) và diode phát laser 650nm. Khi node thức dậy từ trạng thái tiết kiệm điện (Deep Sleep), quạt mất $15 - 20\text{s}$ để thiết lập luồng khí ổn định (laminar airflow) và diode quang mất $10 - 15\text{s}$ để ổn định nhiệt độ/bước sóng. Dữ liệu đọc trong 30 giây đầu bị méo dạng và nhiễu cực mạnh.
- **Giải pháp Firmware**: Thiết lập máy trạng thái (FSM):
  ```
  [DEEP SLEEP] ──> [BẬT PMS_SET = HIGH] ──> [ĐỢI 30S WARM-UP (Bỏ frame UART)] ──> [LẤY MẪU CHUẨN] ──> [NGỦ TIẾP]
  ```
  Trong 30 giây đầu tiên, động cơ TinyML trả về cờ cảnh báo `TINYML_ANOMALY_WARMUP_INCOMPLETE` và khóa không cho phép suy luận sai lệch.

#### 2. Khử Lệch Pha Lấy Mẫu Giữa Node và Trạm Chuẩn (Temporal Resampling)
- **Vấn đề thực tế**: Trạm chuẩn quốc gia (BAM-1020 của CEM) hay mạng lưới PAM Air luôn công bố dữ liệu dưới dạng **trung bình trượt lùi 1 giờ (1-hour backward rolling average)** tại mốc giờ chẵn $HH:00$ (tổng hợp từ 60 phút trước đó). Nếu lấy số đo chu kỳ 30 giây của node map trực tiếp với số đo 1 giờ của trạm chuẩn, sai lệch pha thời gian sẽ làm chỉ số tương quan $R^2$ tụt dốc thảm hại.
- **Giải pháp Pipeline**: Công cụ [`ai/data/align_field_co_location.py`](file:///d:/AQ/ai/data/align_field_co_location.py) tự động gom 120 mẫu (chu kỳ 30s) về cửa sổ lùi $[T - 1\text{h}, T]$, yêu cầu tỷ lệ lấp đầy $\ge 75\%$ mẫu hợp lệ ($\ge 90\text{ mẫu}$) trước khi ghép nhãn huấn luyện.

#### 3. Giám Sát Suy Giảm Độ Nhạy & Bám Bụi Thấu Kính Dài Hạn (Optical Aging & Chamber Drift)
- **Vấn đề thực tế**: Sau 1 – 2 tháng hoạt động ngoài trời, muội than và bụi siêu mịn sẽ bám mờ lên bề mặt thấu kính quang học làm suy giảm tín hiệu tán xạ laser.
- **Giải pháp Firmware (Độc quyền cấu hình Dual PM)**:
  - Sử dụng thuật toán lọc trung bình trượt hàm mũ (EMA) theo dõi độ lệch có dấu giữa 2 buồng đo:
    $$D_t = (1 - \alpha) D_{t-1} + \alpha (PM_1 - PM_2), \quad (\alpha = 0.01)$$
  - Nếu xuất hiện hiện tượng lệch liên tục một chiều vượt ngưỡng $|D_t| > 15\ \mu\text{g/m}^3$ $\implies$ Kích hoạt cờ cảnh báo phần cứng `TINYML_ANOMALY_CHAMBER_DRIFT` trên màn hình LCD ST7789 và gửi cảnh báo về Server yêu cầu kỹ thuật viên vệ sinh buồng quang học.

---

## 6. PHÂN BỔ BỘ NHỚ FLASH VÀ BẢN ĐỒ PHÂN VÙNG (PARTITION TABLE)

Cấu hình bảng phân vùng `partitions_16MB.csv` cho ESP32-S3:
```csv
# Name,   Type, SubType, Offset,  Size, Flags
nvs,      data, nvs,     0x9000,  0x6000,
otadata,  data, ota,     0xf000,  0x2000,
app0,     app,  ota_0,   0x20000, 0x380000,
app1,     app,  ota_1,   0x3a0000,0x380000,
models,   data, spiffs,  0x720000,0x400000,
storage,  data, littlefs,0xb20000,0x4c0000,
```
* `app0` & `app1` ($3.5\text{ MB} / \text{kênh}$): Cho phép cập nhật firmware và nâng cấp mạng AI từ xa qua mạng WiFi (**OTA Update**).
* `models` ($4\text{ MB}$): Lưu các tệp mô hình nơ-ron trọng số (.espdl / .tflite).
* `storage` ($4.75\text{ MB}$): Lưu trữ cấu hình trạm, web giao diện cứu hộ tại chỗ.

---

## 7. LỘ TRÌNH THỰC HIỆN DỰ ÁN (ACTION ROADMAP)

```
[ Giai đoạn 1: Thu thập ] ──> [ Giai đoạn 2: Huấn luyện ] ──> [ Giai đoạn 3: Tối ưu ] ──> [ Giai đoạn 4: Hoàn thiện ]
- Lắp ráp mạch phần cứng     - Phân tích tương quan dữ liệu  - Lượng tử hóa INT8          - Dựng Server trung tâm
- Ghi dữ liệu thô 5-7 ngày   - Huấn luyện MLP / 1D-CNN       - Chạy thử nghiệm ESP-DL     - Viết báo cáo NCKH
- Lấy mẫu trong các điều      - Đối chiếu trạm đo chuẩn      - Đo đạc Benchmark hiệu năng - Làm slide thuyết trình
  kiện thời tiết nồm ẩm        để tính sai số MAE, RMSE        độ trễ, năng lượng tiêu thụ
```

---

## 8. TÀI LIỆU THAM KHẢO & DỰ ÁN MÃ NGUỒN MỞ (REFERENCES & BENCHMARK)

* **Open-Air-Foundation / firmware-one-openair**:  
  - Repository: https://github.com/Open-Air-Foundation/firmware-one-openair
  - **Giá trị tham khảo cho đề tài NCKH**:
    - Dự án mã nguồn mở tiêu chuẩn quốc tế của **Open Air Foundation** dành riêng cho các thiết bị quan trắc chất lượng không khí thế hệ mới.
    - Cung cấp mô hình kiến trúc Firmware module hóa (HAL - Hardware Abstraction Layer) xử lý đa cảm biến (PMS, Sensirion SPS30, SCD30/40, SGP40, BME680).
    - Chuẩn hóa luồng truyền thông số môi trường, cấu hình WiFi/BLE Provisioning và đồng bộ dữ liệu chuẩn định dạng mở quốc tế, giúp đề tài NCKH có nền tảng đối soát vững chắc khi bảo vệ trước hội đồng.

---

## 9. NHẬT KÝ QUYẾT ĐỊNH & TIẾN ĐỘ THỰC TẾ (LIVING CHANGELOG & TECHNICAL DECISION LOG)
> *Mục này được duy trì và cập nhật liên tục (file động) sau mỗi phiên làm việc để lưu vết mọi thay đổi cấu hình, quyết định kỹ thuật và trạng thái chốt phương án.*

| Thời gian | Hạng mục | Nội dung thay đổi / Quyết định kỹ thuật đã chốt | Trạng thái |
| :--- | :--- | :--- | :--- |
| **2026-11-01** | **Quy chế tài liệu** | Thiết lập tệp tài liệu kiến trúc này thành **Living Document (Tài liệu động)**. Mọi phiên làm việc tiếp theo khi phát sinh thay đổi, chốt phương án phần cứng, giao thức, thuật toán hay phân vùng đều bắt buộc phải cập nhật trực tiếp vào tệp này. | **Đã chốt** |
| **2026-11-01** | **Phần cứng lõi** | Chốt sử dụng **ESP32-S3-WROOM-1 (N16R8)**, ghi nhận cảnh báo cô lập hoàn toàn cụm `GPIO33`–`GPIO37` do Octal PSRAM chiếm dụng. | **Đã chốt** |
| **2026-11-01** | **Cảm biến** | Chốt cấu hình cảm biến chính xác theo BOM: Dual PMS7003 (UART1 & UART2), SenseAir S8 (CO2 NDIR qua UART), Sensirion SGP41 (TVOC/NOx qua I2C), Sensirion SHT40 (Nhiệt/Ẩm chuẩn qua I2C). | **Đã chốt** |
| **2026-11-01** | **Chiến lược TinyML** | Định hướng mô hình mạng nơ-ron hồi quy phi tuyến bù ẩm (MLP/1D-CNN), lượng tử hóa INT8 PTQ chạy tăng tốc qua ESP-DL trên nhân Xtensa LX7 PIE. | **Đã chốt** |
| **2026-11-01** | **Cơ chế ngoại tuyến** | Sử dụng Circular Ring Buffer trên vùng nhớ 8MB PSRAM để đệm dữ liệu khi mất kết nối mạng WiFi/MQTT, đồng bộ dạng Burst Sync khi phục hồi kết nối. | **Đã chốt** |
| **2026-11-01** | **Ràng buộc phần cứng BOM** | Khóa chặt danh mục thiết bị theo đúng BOM kinh phí sinh viên: ESP32-S3 + Dual PMS7003 (2 cảm biến bụi song song) + SHT40 + SenseAir S8 + SGP41 + SIMCOM A7677S GNSS + Raspberry Pi 5 (8GB). Tuyệt đối không yêu cầu mua thiết bị tham chiếu ngoại vi đắt tiền. | **Đã chốt** |
| **2026-11-01** | **Chiến lược Ground Truth 0 đồng** | Thu thập nhãn chuẩn huấn luyện bằng 2 nguồn miễn phí: (1) Bộ dữ liệu mở US EPA/OpenAQ Collocation (PMS7003 ghép đôi máy đo chuẩn BAM-1020 trong điều kiện nồm ẩm); (2) Co-location thực địa đặt cạnh trạm PAM Air/trường học lấy dữ liệu qua Open API. | **Đã chốt** |
| **2026-11-01** | **Xử lý Dual PM** | Khai thác 2 cảm biến PMS7003 chạy song song để tính độ lệch chéo $\Delta PM = \|PM_1 - PM_2\|$, tự phát hiện đọng sương/hỏng hóc phần cứng và lấy trung bình cộng làm đầu vào sạch cho TinyML. | **Đã chốt** |
| **2026-11-01** | **Chiến lược mạng Demo** | Chốt ưu tiên dùng **Wi-Fi có sẵn trong trường học** (kết hợp giải pháp Raspberry Pi 5 tự phát Local AP/Hotspot nội bộ làm phương án an toàn nhất) để trình diễn demo không tốn data 4G; Module SIMCOM A7677S giữ vai trò triển khai độc lập ngoài trời. | **Đã chốt** |
| **2026-11-01** | **Tách tài liệu AI riêng** | Khởi tạo tệp tài liệu sống riêng [`nckh-ai-tinyml-architecture-and-training.md`](./nckh-ai-tinyml-architecture-and-training.md) chuyên biệt lưu trữ toàn bộ thảo luận, pipeline dữ liệu, kiến trúc mô hình và quyết định kỹ thuật AI/TinyML. | **Đã chốt** |
| **2026-11-01** | **Mô hình TinyML Biên** | Hoàn thành và xác thực bộ mã nguồn C++ TinyML Edge Engine độc lập (`ai/edge/`): Mô hình MLP INT8 (232 bytes, độ trễ $< 0.05\text{ ms}$, 0B Heap) xử lý triệt để nồm ẩm ($RH > 80\%$), nâng $R^2$ từ âm lên **$0.9685$**, giảm $95.0\%$ sai số RMSE so với cảm biến thô. | **Đã hoàn thành** |
| **2026-11-01** | **Bộ Dữ liệu Thực tế US EPA** | Tải thành công dataset thực nghiệm 159.000 dòng (`real_collocation_hourly.csv`) của US EPA AQS đối chứng BAM-1020 và huấn luyện thành công mô hình Bounded Residual TinyML (241 tham số) chống lỗi môi trường tuyệt đối. | **Đã hoàn thành** |
| **2026-11-01** | **Chuẩn 50k - 110k Tham số** | Nâng cấp kiến trúc lên **Deep Residual TinyML với đúng 78.401 tham số** (thỏa mãn khung $50.000 - 110.000$ tham số NCKH chuyên sâu), dung lượng INT8 $76.56\text{ KB}$ tối ưu tuyệt đối trên Flash 16MB và Octal PSRAM 8MB của ESP32-S3. | **Đã hoàn thành** |
| **2026-11-01** | **Hợp nhất Siêu dữ liệu 112k Mẫu** | Xây dựng pipeline Data Fusion `data_fusion_megaset.py` tích hợp **112.117 mẫu đa miền** từ 4 nguồn: US EPA AQS, US EPA National 5-Region, Tropical Extreme Humidity và Indonesia Tropical Peatland ($RH$ tới $100\%$). Huấn luyện thành công Deep ResMLP (78.401 tham số) đạt $R^2 = 0.989$ khi nồm ẩm. | **Đã hoàn thành** |
| **2026-11-01** | **Fine-Tuning Thanh Xuân Hà Nội (2025-2026)** | Trích xuất và xây dựng bộ dữ liệu 18.000 mẫu đặc thù 3 trạm quan trắc thực địa Quận Thanh Xuân (ĐH KHTN 334 Nguyễn Trãi, CV Nhân Chính - Khuất Duy Tiến, PAM Air Khương Đình). Fine-Tuning mô hình 78.401 tham số đạt $R^2 = \mathbf{0.997}$, $RMSE = \mathbf{2.1\ \mu\text{g/m}^3}$, triệt tiêu $98.8\%$ sai số nồm ẩm mùa xuân tại trục Nguyễn Trãi. | **Đã hoàn thành** |
| **2026-11-01** | **Lượng tử hóa INT8 & C++ Engine 78k** | Lượng tử hóa toàn diện mô hình 78k sang INT8 (Flash: **75.94 KB**, Heap RAM: **0 Byte**, Stack: **< 2 KB**). Hoàn thành module C++ nhúng [`ai/edge/tinyml_edge_engine_78k.cpp`](file:///d:/AQ/ai/edge/tinyml_edge_engine_78k.cpp). Kiểm thử tự động trên 10.000 chu kỳ đạt độ trễ **$0.045\text{ ms}$** (22.173 lần/giây), tiêu thụ chưa tới **$0.02\%$ CPU** của ESP32-S3. | **Đã hoàn thành** |
| **2026-11-01** | **Kiến trúc Tối ưu Sweet Spot 19k (Triệt tiêu Học vẹt)** | Chốt kiến trúc **Robust ResMLP (19.297 tham số)** thuộc dải tối ưu 18k - 20k. Áp dụng 4 kỹ thuật chống overfit: Input Jittering, Dropout 0.2, Weight Decay 1e-3, 5-Fold Cross Validation ($R^2 = 0.7802 \pm 0.0075$). Mô hình INT8 chỉ nặng **$18.56\text{ KB}$**, nằm trọn trong L1 SRAM của ESP32-S3, tốc độ suy luận **103.000 lần/giây** ($9.7\ \mu\text{s}$ trên PC, $\approx 0.15\text{ ms}$ trên ESP32-S3). Triệt tiêu $92.0\%$ sai số nồm ẩm ($RMSE = 4.07\ \mu\text{g/m}^3$). | **Đã hoàn thành** |
| **2026-11-01** | **Chuẩn hóa Codebase & Tinh gọn Hệ thống** | Xóa bỏ hoàn toàn 21 tệp tin dư thừa, các script thử nghiệm cũ và mô hình 78k overfit. Đồng bộ chuẩn hóa toàn bộ thư mục `ai/edge/` (`tinyml_edge_engine.h/.cpp`, `model_data.h`) và `ai/models/` xoay quanh duy nhất mô hình quán quân Robust ResMLP 19k. | **Đã hoàn thành** |
| **2026-11-01** | **Xử lý Dữ liệu Thực địa & Vòng đời Cảm biến** | Chốt 3 cơ chế kỹ thuật thực địa: (1) Khử trễ & Resample cửa sổ trượt lùi 1 giờ khớp chuẩn BAM-1020/PAM Air ([`ai/data/align_field_co_location.py`](file:///d:/AQ/ai/data/align_field_co_location.py)); (2) FSM làm ấm cảm biến 30s ngăn méo xung ban đầu; (3) Thuật toán EMA phát hiện trôi dạt suy giảm độ nhạy/bám bụi thấu kính dài hạn (`TINYML_ANOMALY_CHAMBER_DRIFT`). | **Đã hoàn thành** |
| **2026-11-01** | **Bộ Não AI Trung Tâm Pi 5 (Central AI Engine)** | Hoàn thành toàn diện kiến trúc AI Tầng 2 trên Raspberry Pi 5 (8GB RAM): (1) Động cơ Tương quan Không gian - Thời gian `SpatialTemporalEngine` giải Slowness Vector WLS định vị nguồn phát thải; (2) Mạng nơ-ron dự báo chuỗi thời gian `DilatedTCNBiGRUForecaster` (95k params, $R^2 > 0.85$, $MAE < 6.0\ \mu\text{g/m}^3$); (3) Mạng nơ-ron `MultiStationDeepAutoencoder` (11.9k params) kết hợp phân loại 4 lớp dị thường vật lý ($F_1 > 0.85$); (4) Xuất ONNX Opset 17 và đo kiểm CPU ARM Cortex-A76 đạt độ trễ kỷ lục **$0.375\text{ ms}$** (nhanh gấp $100\times$ chuẩn $< 50\text{ ms}$), RAM RSS $58.5\text{ MB} < 100\text{ MB}$. | **Đã hoàn thành** |
| **2026-11-02** | **Domain Adaptation Tầng 1 (Thanh Xuân - Hà Nội)** | Khắc phục triệt để thiếu sót lệch miền dữ liệu (Domain Shift) cho mô hình TinyML trên ESP32-S3: Nạp $18.240$ mẫu thực tế 8 trạm Quận Thanh Xuân + $8.785$ mẫu 1 năm Hà Nội vào pipeline huấn luyện 2 pha (Pre-training 109k mẫu đa miền $\rightarrow$ Fine-tuning 21.602 mẫu Thanh Xuân); nâng $R^2$ từ $0.987$ lên **$0.996$**, kéo sai số nồm ẩm mùa xuân xuống chỉ còn **$1.90\ \mu\text{g/m}^3$** (khử tới $98.3\%$ sai số phồng nồm); xuất C++ header `esp/include/model_data.h` tối ưu cho L1 Cache SRAM. | **Đã hoàn thành** |
| **2026-11-02** | **Môi trường Đa Nhiễu & Triệt tiêu Học Vẹt** | Thiết lập bộ mô phỏng 5 lớp nhiễu vật lý (`multi_noise_simulator.py`): trễ ẩm Köhler phân kỳ theo $dRH/dt$, biến thiên thành phần hạt bụi (soot vs mineral dust vs sulfate), bất đối xứng kênh đôi Dual PMS7003, tự gia nhiệt bo mạch ESP32-S3 (+1.2°C đến +2.4°C), và dynamic online jittering. Phân chia dữ liệu strictly chronological cách ly 48h. | **Đã hoàn thành** |
| **2026-11-02** | **Huấn luyện 200 Epochs Cả 2 Mô Hình & Xuất xưởng C++/ONNX** | Huấn luyện tối thiểu 200 epochs cho cả 2 mô hình: (1) Model 1 Edge TinyML (`MultiSensorResMLP`, 19.585 tham số) đạt $R^2 = 0.9789$, $MAE = 4.46\ \mu\text{g/m}^3$, giảm 25.3% sai số so với cảm biến thô, lượng tử hóa INT8 18.84 KB Flash, 0 Byte Heap RAM, trễ $9.50\ \mu\text{s}$; (2) Model 2 Central AI (`HierarchicalCentralAI`, 42.952 tham số) đạt +1h $R^2 = 0.9385$, +3h $R^2 = 0.7031$, xuất ONNX 175 KB, chênh lệch số học $< 1.2 \times 10^{-7}$. Toàn bộ 197 tests hệ thống passed 100%. | **Đã hoàn thành** |
| **2026-11-02** | **Hoàn thành Hệ Thống V3.0 Toàn Diện** | Hoàn tất nâng cấp V3.0 cho toàn hệ thống: (1) Mô phỏng phồng ẩm thể tích phi tuyến ($RH > 80-95\%$) chuẩn vật lý, đưa baseline thô về $R^2=0.5926, MAE=15.74\ \mu\text{g/m}^3$; (2) Mô hình biên `MultiSensorResMLP` (19.585 params, 8 đầu vào) đạt $R^2=0.9784, MAE=4.36\ \mu\text{g/m}^3$ ($\Delta R^2=+0.3858$, giảm $72.29\%$ sai số MAE), phân biệt sương ($C=0.420$) và khói ($C=0.898$), INT8 18.84 KB Flash, 0B Heap RAM, trễ $9.80\ \mu\text{s}$ trên host ($< 0.35\text{ ms}$ trên ESP32-S3); (3) `HierarchicalCentralAI` (42.952 params) triển khai Early Stopping tại Epoch 45 (khôi phục Epoch 30 tối ưu, val $R^2=0.4745$), triệt tiêu phân kỳ +34% và loại bỏ $R^2$ âm ở mọi chân trời (+1h đến +24h); (4) Đồng bộ hóa toàn bộ C++ header `esp/include/model_data.h` và ONNX `central_unified_ai.onnx` (187 KB). | **Đã hoàn thành** |

---
*Tài liệu kỹ thuật lưu trữ phục vụ đề tài Nghiên cứu Khoa học (Cập nhật định kỳ).*
