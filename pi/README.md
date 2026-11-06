# Raspberry Pi 5 Central AI Gateway & Spatial-Temporal Engine

Thư mục này chứa toàn bộ hệ thống gateway trung tâm xử lý dữ liệu từ **Trạm Cảm Biến Prototype Demo (ESP32-S3)**, phát hiện dị thường và dự báo chất lượng không khí (kèm khả năng mở rộng phân tích không - thời gian đa trạm) vận hành trên máy tính biên **Raspberry Pi 5 (8GB RAM)**.

---

## 🎯 1. Phần Cứng Mục Tiêu
- **Bộ xử lý trung tâm**: Broadcom BCM2712 Quad-Core ARM Cortex-A76 @ 2.4 GHz.
- **Bộ nhớ RAM**: 8GB LPDDR4X (băng thông bộ nhớ $17\text{ GB/s}$).
- **Hệ điều hành**: Raspberry Pi OS 64-bit (Debian Bookworm) / Ubuntu Server 24.04 LTS.

---

## 📂 2. Cấu Trúc Module & Trách Nhiệm
| Tệp Tin | Module / Class Chính | Chức Năng Kỹ Thuật |
|---|---|---|
| [`config.py`](file:///e:/AQ/pi/config.py) | `STATIONS`, `StationInfo`, `PROTOTYPE_STATION_ID` | Cấu hình trạm quan trắc Prototype (Thanh Xuân, Hà Nội), tọa độ UTM Zone 48N, danh mục tọa độ mở rộng mô phỏng và hằng số khí quyển. |
| [`simulator.py`](file:///e:/AQ/pi/simulator.py) | `AtmosphericSimulator`, `WindFieldOU`, `GaussianPuff` | Mô phỏng vật lý khí quyển vi khí hậu đô thị: trường gió ngẫu nhiên Ornstein-Uhlenbeck, lớp biên xáo trộn ngày đêm (PBLH), lan truyền khói Gaussian và các kịch bản kiểm thử dị thường. |
| [`data_pipeline.py`](file:///e:/AQ/pi/data_pipeline.py) | `AirQualityDataPipeline`, `TinyMLEdgeCalibrator` | Tiền xử lý dữ liệu, chuẩn hóa Min-Max/StandardScaler, tạo Sliding Window Dataset và nạp mô hình TinyML đối chuẩn. |
| [`spatial_temporal_engine.py`](file:///e:/AQ/pi/spatial_temporal_engine.py) | `SpatialTemporalEngine` | Tính ma trận tương quan chéo thời gian trễ, ước lượng vectơ vận tốc nghịch đảo dòng khí (Slowness Vector $\mathbf{s}$) bằng WLS và nội suy trường nồng độ bụi bằng Ordinary Kriging. |
| [`forecasting_engine.py`](file:///e:/AQ/pi/forecasting_engine.py) | `DilatedTCNBiGRUForecaster`, `AsymmetricHuberLoss` | Mạng nơ-ron không - thời gian kết hợp Dilated TCN + Bi-GRU, dự báo nồng độ bụi $PM_{2.5}$ trước $1 - 6\text{ giờ}$ với độ tin cậy $R^2 > 0.85$. |
| [`anomaly_engine.py`](file:///e:/AQ/pi/anomaly_engine.py) | `MultiStationDeepAutoencoder`, `AdaptiveAnomalyDetector` | Mạng Autoencoder học sâu kết hợp khoảng cách Mahalanobis thích ứng để phát hiện dị thường ô nhiễm đột xuất (cháy rơm rạ, kẹt xe) vs lỗi cảm biến. |
| [`central_engine.py`](file:///d:/AQ/pi/central_engine.py) | `CentralAIEngine` | Bộ điều phối toàn diện: liên kết nhận dữ liệu từ trạm prototype, chạy suy luận chuỗi AI trung tâm và đưa ra khuyến nghị điều hành. |
| [`benchmark_pi5.py`](file:///e:/AQ/pi/benchmark_pi5.py) | `run_full_pi5_benchmark` | Công cụ đo kiểm hiệu năng thực tế trên Raspberry Pi 5 (độ trễ, thông lượng, mức chiếm dụng RAM và CPU). |

---

## 🚀 3. Hướng Dẫn Sử Dụng & Đo Kiểm Hiệu Năng

### Sử dụng như một Python Package:
```python
import pi
from pi.config import STATIONS, PROTOTYPE_STATION_ID
from pi.central_engine import CentralAIEngine

# Khởi tạo động cơ trung tâm
engine = CentralAIEngine()
print(f"Trạm Prototype Demo hoạt động: {PROTOTYPE_STATION_ID}")
```

### Chạy Benchmark Hiệu Năng:
```bash
python -m pi.benchmark_pi5
```

### Kết quả Đo đạc trên Raspberry Pi 5 (8GB):
- **Độ trễ suy luận chuỗi AI (Pipeline Latency)**: $\approx 2.9\text{ ms}$ (chu kỳ thu nhận tin $60\text{ giây}$).
- **Mức chiếm dụng CPU**: $< 2.5\%$ tổng công suất 4 nhân Cortex-A76.
- **Mức chiếm dụng bộ nhớ RAM**: $\approx 267\text{ MB}$ (chỉ chiếm $3.3\%$ dung lượng 8GB của Pi 5).
