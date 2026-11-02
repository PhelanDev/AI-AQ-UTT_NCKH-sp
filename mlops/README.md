# MLOps Pipeline: Huấn Luyện Ngoại Tuyến & Dữ Liệu Nghiên Cứu

Thư mục này quản lý toàn bộ chu trình **MLOps nghiên cứu khoa học** của hệ thống quan trắc chất lượng không khí: từ thu thập và hợp nhất đa miền dữ liệu, kỹ nghệ đặc trưng vi khí hậu nồm ẩm, huấn luyện mạng học sâu (PyTorch), lượng tử hóa INT8 PTQ cho ESP32-S3, đến xuất các mô hình ONNX cho Raspberry Pi 5.

---

## 📂 1. Cấu Trúc Thư Mục
```
mlops/
├── data/                        # Dữ liệu thực nghiệm đa miền (112.117 mẫu đối chuẩn)
│   ├── unified_megaset.csv      # Bộ dữ liệu hợp nhất 5 nguồn khí hậu (BAM-1020 ground truth)
│   ├── real_collocation_hourly.csv  # 159.660 dòng đo đối chứng thực tế US EPA AQS
│   ├── real_collocation_daily.csv   # Dữ liệu đối chuẩn 24-hr US EPA
│   ├── indonesia_tropical_colocation.csv # Dữ liệu khí hậu nhiệt đới xích đạo Đông Nam Á
│   ├── chamber_calibration_purpleair.csv # Dữ liệu buồng thử nghiệm nồm ẩm cực đoan (RH > 90%)
│   ├── epa_national_50pairs.zip # Dữ liệu nén 50 cặp trạm tham chiếu
│   ├── data_fusion_megaset.py   # Script hợp nhất và tiền xử lý dữ liệu
│   └── align_field_co_location.py # Script gióng hàng tọa độ trạm thực địa
├── models/                      # Trọng số mô hình và siêu tham số
│   ├── robust_resmlp_19k.pt     # Trọng số PyTorch mô hình Robust ResMLP (19.297 tham số)
│   ├── robust_resmlp_19k_params.json # Siêu tham số Min-Max Scaler và kết quả 5-Fold CV
│   ├── central_forecaster.onnx  # Mô hình dự báo Dilated TCN + Bi-GRU định dạng ONNX
│   └── central_autoencoder.onnx # Mô hình Multi-Station Deep Autoencoder định dạng ONNX
└── scripts/                     # Mã nguồn huấn luyện & công cụ đồ thị xuất bản
    ├── train_robust_resmlp_19k.py # Huấn luyện mô hình biên Robust ResMLP 19k (5-Fold CV)
    ├── quantize_19k_int8.py     # Lượng tử hóa Post-Training Quantization (PTQ) xuất model_data.h
    ├── generate_publication_figures.py # Sinh biểu đồ nghiên cứu học thuật chuẩn IEEE/Nature (300 DPI)
    └── run_training.bat         # Script tự động hóa toàn bộ pipeline huấn luyện
```

---

## 🚀 2. Hướng Dẫn Thực Hiện Pipeline MLOps

### Bước 1: Huấn luyện Mạng Biên Robust ResMLP 19k (ESP32-S3)
```bash
python mlops/scripts/train_robust_resmlp_19k.py
```
- Sử dụng hàm mất mát HuberLoss kết hợp kỹ thuật Data Jittering chống hiện tượng overfitting.
- Kết quả: $R^2 \ge 0.9685$, giảm $95.0\%$ sai số RMSE ở dải nồm ẩm ($RH > 80\%$).

### Bước 2: Lượng Tử Hóa INT8 & Xuất Header C++ Cho Vi Điều Khiển
```bash
python mlops/scripts/quantize_19k_int8.py
```
- Xuất file mảng tĩnh `model_data.h` sang thư mục `esp/include/` với kích thước chỉ **18.56 KB**, phù hợp lưu trên L1 Cache hoặc Flash của ESP32-S3.

### Bước 3: Sinh Đồ Thị & Biểu Đồ Báo Cáo Khoa Học (300 DPI)
```bash
python mlops/scripts/generate_publication_figures.py
```
- Các hình ảnh xuất bản được lưu trực tiếp vào thư mục `docs/figures/`.
