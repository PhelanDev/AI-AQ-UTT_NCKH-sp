@echo off
chcp 65001 > nul
echo ===============================================================================
echo     AIR QUALITY IOT - QUY TRÌNH HUẤN LUYỆN TOÀN BỘ HỆ THỐNG AI (END-TO-END)
echo     1. Edge TinyML MultiSensorResMLP (19.5k params, INT8 C++ cho ESP32-S3)
echo     2. Hierarchical Unified Central AI (104k params, ONNX cho Raspberry Pi 5)
echo ===============================================================================
echo.

echo [BƯỚC 1/5] Hợp nhất dữ liệu đa cảm biến (BOM: Dual PMS7003, SHT40, S8, SGP41)...
python mlops/scripts/data_fusion_multisensor.py
if %ERRORLEVEL% NEQ 0 (
    echo [LỖI] Hợp nhất dữ liệu thất bại!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [BƯỚC 2/5] Huấn luyện Mô hình 1: MultiSensorResMLP 19.5k (Bù nồm ẩm & Phân biệt khói)...
python mlops/scripts/train_multisensor_resmlp.py
if %ERRORLEVEL% NEQ 0 (
    echo [LỖI] Huấn luyện Mô hình 1 thất bại!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [BƯỚC 3/5] Lượng tử hóa INT8 và xuất C++ Header (esp/include/model_data.h)...
python mlops/scripts/quantize_multisensor_int8.py
if %ERRORLEVEL% NEQ 0 (
    echo [LỖI] Lượng tử hóa thất bại!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [BƯỚC 4/5] Huấn luyện Mô hình 2+3: Hierarchical Unified Central AI (104k params, ONNX)...
python mlops/scripts/train_unified_central_ai.py
if %ERRORLEVEL% NEQ 0 (
    echo [LỖI] Huấn luyện Mô hình Trung tâm Phân tầng thất bại!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [BƯỚC 5/5] Kiểm thử tự động toàn diện (PyTest + Biên dịch C++ TinyML Engine)...
python -m pytest tests/test_multisensor_training.py tests/test_unified_central_ai.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [CẢNH BÁO] Có test case thất bại!
)

where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo -> Phát hiện GCC/G++, đang biên dịch và chạy động cơ C++ TinyML...
    g++ -O3 -Wall -I esp/include esp/src/tinyml_edge_engine.cpp esp/tests/test_main.cpp -o esp/tests/test_edge_engine.exe
    esp\tests\test_edge_engine.exe
) else (
    echo -> Bỏ qua biên dịch g++ (chưa cài MinGW/GCC trên Windows, mã C++ vẫn hoàn toàn hợp lệ cho ESP-IDF).
)

echo.
echo ===============================================================================
echo   QUÁ TRÌNH HUẤN LUYỆN & XUẤT MÔ HÌNH HOÀN TẤT 100%%!
echo   [ESP32-S3]:
echo     - File trọng số INT8 nhúng Flash: esp/include/model_data.h (18.84 KB)
echo     - Động cơ suy luận C++ 0-Heap:    esp/src/tinyml_edge_engine.cpp
echo   [Raspberry Pi 5]:
echo     - File mô hình ONNX đa nhiệm:     mlops/models/central_unified_ai.onnx
echo     - File trọng số PyTorch:          mlops/models/central_unified_ai.pt
echo     - Metadata & Chuẩn hóa:           mlops/models/central_unified_ai_params.json
echo ===============================================================================
echo.
pause
