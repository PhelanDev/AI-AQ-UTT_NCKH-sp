# Tài Liệu & Báo Cáo Nghiên Cứu Khoa Học (NCKH)

Thư mục này quản lý toàn bộ báo cáo khoa học, bản vẽ kỹ thuật phần cứng, biểu đồ trực quan hóa và bộ công cụ tự động tạo lập tài liệu nghiệm thu.

---

## 📂 1. Cấu Trúc Thư Mục
```
docs/
├── reports/                     # Các tài liệu tổng hợp và báo cáo khoa học chính thức
│   ├── Bao_Cao_Ban_Dau_NCKH_Edge_AI.docx # Báo cáo khoa học ban đầu (Đầy đủ biểu đồ 300 DPI)
│   ├── Bao_Cao_Huong_Di_Quy_Trinh_Edge_AI.docx
│   ├── Bao_Cao_Tong_Hop_Du_An_Va_GPIO_AirGradient_OpenAir.docx
│   ├── Tai_Lieu_Ky_Thuat_Huong_Di_Quy_Trinh_Edge_AI_Hop_Quan_Trac.docx
│   ├── Tong_Quan_Du_An_AirGradient_OpenAir_Pi5_AI.docx
│   ├── nckh-air-quality-esp32s3-edge-ai-architecture.md # Living Architecture (ESP32-S3)
│   └── nckh-ai-tinyml-architecture-and-training.md      # Living AI & Training Specs
├── hardware/                    # Bản thiết kế phần cứng và danh mục linh kiện
│   └── AirGradient_OpenAir_Final_CustomBOM.xlsx # Bảng BOM chi tiết từng linh kiện, chi phí, footprint
├── figures/                     # Hình vẽ kiến trúc & biểu đồ mô hình hóa (chuẩn xuất bản 300 DPI)
│   ├── fig1_robust_resmlp_architecture.png
│   ├── fig2_residual_block_skip_connection.png
│   ├── fig3_adaptive_humidity_gating.png
│   └── fig4_bounded_output_activation.png
└── generators/                  # Bộ script Python tự động kết xuất văn bản Word (.docx)
    ├── build_clean_accessible_report.py
    ├── build_academic_report_docx.py
    ├── build_human_scientific_report.py
    ├── build_practical_edge_ai_doc.py
    ├── generate_bulletproof_edge_ai_doc.py
    └── generate_focused_edge_ai_doc.py
```

---

## 🛠️ 2. Hướng Dẫn Tái Tạo Báo Cáo Word
Để sinh lại văn bản báo cáo khoa học hoàn chỉnh nhất từ mã nguồn:
```bash
python docs/generators/build_clean_accessible_report.py
```
File xuất ra sẽ được cập nhật trực tiếp tại `docs/reports/Bao_Cao_Ban_Dau_NCKH_Edge_AI.docx`.
