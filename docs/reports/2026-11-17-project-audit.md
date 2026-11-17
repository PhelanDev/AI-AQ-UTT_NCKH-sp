# Rà soát dự án AQ — 17/11/2026

**Kết luận: chưa sẵn sàng triển khai quan trắc thực tế.** Mã hiện tại có các phần chạy được và bộ kiểm thử đều pass, nhưng luồng đo → truyền → nhận → suy luận còn thiếu tích hợp và có những chỗ dùng dữ liệu giả trong đường triển khai.

Phạm vi: toàn bộ mã hiện tại trên `main`, commit `9b27fc4d9fd5cb4823977d89d5c7039b60371f5f`; không giới hạn ở commit cuối. P1 là lỗi cần sửa trước khi vận hành; P2 là lỗi chức năng, độ tin cậy hoặc triển khai cần xử lý tiếp. Đây là báo cáo phát hiện, chưa sửa mã chạy hay checkpoint.

## Những vấn đề cần ưu tiên

### 1. P1 — Script được hướng dẫn để huấn luyện ghi đè mô hình bằng trọng số chưa học

Vị trí: `mlops/scripts/train_lite_central_models.py:40`, `:51`, `:55`; README hướng dẫn chạy script ở phần 5.3.

Script không nạp dataset, không tối ưu loss, chỉ tạo mạng ngẫu nhiên, đặt bias đầu ra bằng 40 và lưu đè cả ba checkpoint đang được gateway sử dụng. Đầu ra mạng là giá trị chuẩn hóa, nên 40 không có nghĩa là 40 µg/m³. Script cũng không cập nhật ONNX và scaler tương ứng.

**Đã tái hiện:** chạy script trong thư mục tạm, nạp checkpoint unified vừa sinh vào engine với PM2.5 đầu vào 38,5 µg/m³; dự báo khoảng **23.750–23.910 µg/m³**. Các checkpoint sẵn có trong repo không bị thay đổi. Test hiện tại chỉ kiểm artifact đã commit, không kiểm hậu quả của lệnh trong README.

Hướng sửa: bỏ lệnh khởi tạo này khỏi quy trình huấn luyện; gọi trainer thực, đánh giá trên dữ liệu giữ riêng và xuất checkpoint/scaler/ONNX cùng phiên bản. Không cho checkpoint chưa huấn luyện ghi đè artifact triển khai.

### 2. P1 — ESP32 báo gửi MQTT thành công nhưng không truyền gói tin

Vị trí: `esp/src/power_scheduler.cpp:142`, `esp/src/main_firmware.cpp:243`.

`telemetry_sent` chỉ phụ thuộc việc format JSON thành công. Không có lệnh publish, kết nối Wi-Fi/MQTT hay transport thực trong đường này. `mqtt_online` luôn là `true`. Gói backfill còn bị lấy khỏi queue và đếm là đã gửi ngay sau format JSON.

Test firmware chỉ kiểm JSON/cờ kết quả nên vẫn pass. Cần transport thực, trạng thái kết nối thực và chỉ xoá gói sau khi xác nhận gửi theo cơ chế tin cậy đã chọn.

### 3. P1 — Gateway phục vụ số mô phỏng, chưa nhận telemetry từ trạm

Vị trí: `pi/web_server.py:53`, `:89`, `:248`.

Server khởi tạo PM2.5=38,5; mỗi lần cập nhật lại cộng nhiễu ngẫu nhiên và thay PM10 bằng `PM2.5 × 1,35`. Không có subscriber MQTT hay endpoint tiếp nhận gói đo; POST chỉ hỗ trợ `/api/trigger_cycle`. Không có scheduler chạy suy luận mỗi 900 giây trong server; trình duyệt chỉ polling GET.

**Đã tái hiện:** POST gói PM2.5=180 vào `/api/telemetry` trả **404**, số phục vụ vẫn là **38,5**. Ngay cả khi đưa PM=80 vào state để giả lập backfill, `execute_cycle` vẫn sửa thành **81** khi cố định jitter=1. Test store-and-forward hiện sửa trực tiếp state, không đi qua transport/parser.

Hướng sửa: tách chế độ demo; thêm ingestion theo station ID/timestamp, lưu lịch sử, xử lý backfill đúng thứ tự và kích hoạt suy luận bằng dữ liệu nhận thực tế. Nút cập nhật giao diện không được tự tạo số đo mới.

### 4. P1 — Cảm biến lỗi vẫn tạo ra số đo có vẻ hợp lệ

Vị trí: `esp/src/main_firmware.cpp:153`.

PMS1 lỗi được thay bằng PM2.5=38,5/PM10=52; PMS2 lỗi dùng PMS1+0,4; SHT40 lỗi dùng 28,5°C/84%. Hàm vẫn trả `true`. CO2/VOC trong nhánh phần cứng luôn bằng 520/110. Vì vậy mất cảm biến có thể bị hiển thị như trạm vẫn đo bình thường.

Test parser kiểm checksum nhưng không kiểm chính sách xử lý lỗi trong acquisition thật. Cần trả lỗi/validity cho từng kênh, bỏ mẫu không hợp lệ và báo tình trạng cảm biến; chỉ sinh số giả trong chế độ mô phỏng rõ ràng.

### 5. P1 — Warmup 30 giây chỉ là một giá trị gán

Vị trí: `esp/src/main_firmware.cpp:235`, `:242`.

Firmware lấy burst ngay sau khởi tạo bus, rồi gán `warmup_elapsed=30.2f` để vượt guard. Không có bước chờ đủ 30 giây trước lấy mẫu. Thời gian đọc UART cũng biến động nhưng sleep luôn là 868 giây, nên phép cộng hằng số không bảo đảm chu kỳ thực 900 giây.

Test chỉ truyền elapsed giả định vào scheduler, không đo thời gian acquisition/setup. Cần đo từ lúc bật cảm biến, hoàn tất warmup trước burst và tính thời gian ngủ từ thời gian thực đã trôi qua. Chưa đo trên board thật.

### 6. P1 — Dữ liệu cách nhau 15 phút bị đưa vào mô hình như dữ liệu theo giờ

Vị trí: `pi/web_server.py:72`, `pi/central_engine.py:882`; phía huấn luyện: `mlops/scripts/train_unified_central_ai.py:190`, `mlops/scripts/train_central_forecaster.py:37`.

Gateway giữ 24 mẫu và dịch một mẫu mỗi chu kỳ. Với chu kỳ 900 giây, khoảng giữa mẫu đầu và cuối là **5 giờ 45 phút**. Engine lại mã hóa giờ bằng `base_hour - 23 + t`, và mô hình được huấn luyện từ chuỗi hourly. Không có resampling từ 96 mẫu/24 giờ sang 24 mốc theo giờ trước suy luận. `dt_s` chưa điều khiển các đặc trưng thời gian này.

**Đã tái hiện:** hook đầu vào mạng cho thấy 24 mẫu, `dt_s=900`, nhưng giờ sin/cos trải **23 giờ**. Test chu kỳ hiện chỉ kiểm hằng số và định dạng kết quả.

Hướng sửa: giữ timestamp cho từng mẫu, gom/resample hourly để khớp mô hình hoặc huấn luyện lại theo bước 15 phút; thống nhất cả history, đặc trưng thời gian và nhãn chân trời dự báo.

## Những lỗi chức năng và độ tin cậy còn lại

| # | Mức | Vị trí | Phát hiện, bằng chứng và hướng sửa |
|---|---|---|---|
| 7 | P2 | `mlops/scripts/train_and_quantize_standard_mlp_lite.py:132`; `esp/src/tinyml_edge_engine.cpp:163` | Train/eval dùng `pm_avg × c`, firmware thêm gate RH/CO2. Trên **8.688** mẫu test, công thức đánh giá có **R²=0,81853; RMSE=11,2166**, đường C++ đầy đủ có **R²=0,69952; RMSE=14,4334**. INT8 coefficient chưa qua gate có R²≈0,81771, nên phần lệch lớn đến từ gate. Cần tối ưu/đánh giá đúng đường triển khai và kiểm định artifact C++ sau xuất. |
| 8 | P2 | `mlops/scripts/train_all_lite.py:59`; `mlops/scripts/train_and_quantize_standard_mlp_lite.py:168` | Report lấy mặc định **R²=0,957; MAE=3,12** vì params chỉ có `x_min/x_max`. `mlops/models/training_summary_report.json:12` cũng chứa các số này, trong khi đo checkpoint hiện tại cho **R²=0,81853; MAE=5,86554** trước gate. Cần ghi metric thật kèm nhận dạng artifact; thiếu metric phải báo lỗi thay vì điền số tốt mặc định. |
| 9 | P2 | `esp/src/display_mqtt_hal.cpp:146` | Queue offline/head/tail/count nằm trong SRAM thường, mất sau deep sleep (`esp/src/power_scheduler.cpp:186`). Test giữ nguyên process nên không mô phỏng mất RAM khi ngủ. Cần lưu bền phù hợp và kiểm qua sleep/reset. Queue hiện 8.192 byte, cần tính dung lượng nếu dùng RTC memory. |
| 10 | P2 | `esp/src/st7789_lcd_driver.cpp:192` | LCD chỉ vẽ hình chữ nhật; không render chữ/số. Probe nhánh Arduino bằng SPI shim: thay PM2.5 **21,8→300**, nhiệt độ **26→40**, RH **70→95**, CO2 **500→2000**, status **HEALTHY→ANOMALY**, giữ AQI=45, vẫn cho **cùng chính xác 236.388 SPI byte**. Test chỉ xem mock state. Cần thêm render glyph và kiểm nội dung thực gửi LCD. |
| 11 | P2 | `pi/central_engine.py:857` | Nhánh trạm động bỏ PM10 đo được và dùng `pm_seq × 1,35`. Hook cho thấy đổi PM10 **52→500** tạo đầu vào mạng giống hệt nhau, kể cả vector tái tạo phát hiện bất thường. Cần đưa PM10 thực vào thời điểm hiện tại và giữ lịch sử đa kênh khi có. |
| 12 | P2 | `pi/central_engine.py:918`, `:923` | Chỉ cần PM>120 là tự khai CO2/VOC tăng, dù khí đo bình thường. Khi xác nhận có anomaly và hai trạm PM180/CO2=420/VOC=100, orchestrator phân loại **Fire/Combustion**. Duration luôn 10 phút còn khiến điều kiện Sensor Drift yêu cầu ≥1 ngày không bao giờ đạt; gọi classifier trực tiếp với 1.440 phút thì ra Sensor Drift. Cần dùng trạng thái khí thực và theo dõi thời lượng sự kiện theo trạm. |
| 13 | P2 | `pi/central_engine.py:89`, `:105` | Checkpoint tương đối với CWD; nếu chạy service từ thư mục khác thì âm thầm giữ trọng số ngẫu nhiên. Cùng input, ở repo dự báo **30,29–42,99**, ở thư mục tạm với seed cố định dự báo **0,30–156,66**. Scaler lại có fallback theo repo root, nên trạng thái trông như đã nạp cấu hình. Cần resolve đường mặc định từ repo/package và báo lỗi khi thiếu artifact bắt buộc. |
| 14 | P2 | `pi/central_engine.py:644` | Nhánh 8 trạm trả `station_forecasts` giống nhau cho mọi trạm, nhưng marker lại nhân theo tỷ lệ nồng độ trạm. Probe STATION_01 có PM2.5=10 và STATION_08 có PM2.5=80: API forecast theo station cùng **41,09**, marker tương ứng **9,13** và **73,04**. Cần một nguồn forecast thống nhất và test contract giữa hai trường. |
| 15 | P2 | `pi/web_server.py:187`, `:251` | Đối số mặc định của `getattr(..., get_server_state())` được tính ngay cả khi `app_state` tồn tại. GET đầu tiên tạo thêm một engine/state toàn cục ngoài engine/state của server. Probe xác nhận singleton từ `None` thành object sau GET dù server đã có state. Cần kiểm attribute trước khi gọi fallback, tránh nạp mô hình thừa và phụ thuộc CWD ngoài ý muốn. |
| 16 | P2 | `README.md:165` | Lệnh uvicorn trỏ tới `pi.web_server:app`, nhưng module dùng `ThreadingHTTPServer` và không có `app`; requirements cũng không cài uvicorn. Đã kiểm `hasattr(web_server, 'app') == False`. Cần cập nhật hướng dẫn chạy bằng `python -m pi.web_server 8000`, hoặc triển khai ASGI thực rồi đồng bộ dependency. |

## Bằng chứng kiểm thử và giới hạn

- `python -m pytest tests/ -q --tb=short`: **384 passed, 1 warning, 36,25 giây**. `pytest.ini` bỏ qua `tests/test_docx_report_verification.py`; kết quả không bao gồm test này.
- Biên dịch/chạy `esp/tests/test_firmware_15min_cycle.cpp`: **18/18 passed**. Đây là kiểm thử native, không phải chạy trên ESP32 thật.
- Các probe dùng model/checkpoint/CSV/header thật; script tạo checkpoint chưa huấn luyện được chạy trong thư mục tạm. Không sửa artifact triển khai.
- Đã dựng GitNexus với `--pdg --index-only` cho đúng HEAD: **136 file, 8.398 node, 27.171 edge, 103 luồng**. Đã đối chiếu `context`, `impact`, luồng `new_pm25` và taint với source. Graph báo thiếu 17 call site của `process_telemetry_frame` do không xác định được kiểu receiver; không coi danh sách caller là đầy đủ.
- Chưa kiểm tra phần cứng, broker thật, độ chính xác sensor thực địa hay trình duyệt thật. Probe LCD dùng SPI shim ở nhánh Arduino. Các kết luận triển khai phần cứng dựa trên source và probe native tương ứng.
- Đã kiểm hướng dẫn `modern-web-guidance` về sanitizer và web security. Chưa xác nhận lỗ hổng XSS từ các sink được rà soát và không thay đổi giao diện.

## Thứ tự xử lý đề xuất

1. Bảo vệ checkpoint và sửa lệnh huấn luyện/khởi chạy để người dùng không làm hỏng mô hình khi làm theo README.
2. Hoàn thiện đường truyền telemetry, ingestion theo timestamp và trạng thái lỗi sensor; tách dữ liệu demo khỏi chế độ quan trắc.
3. Sửa warmup, lịch sử 15 phút/theo giờ và lưu bền offline qua deep sleep.
4. Đồng nhất train/runtime và số liệu báo cáo; bổ sung kiểm thử tích hợp tại các điểm nối đang bị mock hoặc bỏ qua.
5. Hoàn thiện LCD, các trường forecast theo trạm và phân loại bất thường dựa trên dữ liệu thật.
