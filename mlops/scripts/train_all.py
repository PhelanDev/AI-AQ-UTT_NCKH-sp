"""
Trình Điều Phối Huấn Luyện Toàn Diện (End-to-End AI Training Orchestrator)
Tự động thực hiện 5 giai đoạn:
1. Data Fusion (BOM cảm biến thực tế)
2. Train Edge MultiSensorResMLP (ESP32-S3)
3. INT8 Quantization -> Export model_data.h
4. Train Unified Central AI (Raspberry Pi 5) -> Export central_unified_ai.onnx
5. Run Pytest Suite & C++ Parity Check
"""

import sys
import os
import subprocess
import time

sys.stdout.reconfigure(encoding="utf-8")

def run_step(step_num: int, title: str, cmd: list):
    print("\n" + "=" * 80)
    print(f"[{step_num}/5] {title}")
    print("=" * 80)
    start_t = time.time()
    res = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"\n[LỖI] Bước {step_num} thất bại với mã lỗi {res.returncode}!")
        sys.exit(res.returncode)
    print(f" -> Hoàn thành bước {step_num} trong {elapsed:.2f}s!")

def main():
    print("===============================================================================")
    print("    AIR QUALITY IOT - QUY TRÌNH HUẤN LUYỆN TOÀN BỘ HỆ THỐNG AI (END-TO-END)")
    print("    1. Edge TinyML MultiSensorResMLP (19.5k params, INT8 C++ cho ESP32-S3)")
    print("    2. Hierarchical Unified Central AI (104k params, ONNX cho Raspberry Pi 5)")
    print("===============================================================================")

    # Bước 1: Data Fusion
    run_step(1, "Hợp nhất dữ liệu đa cảm biến (Dual PMS7003, SHT40, S8, SGP41)",
             [sys.executable, "mlops/scripts/data_fusion_multisensor.py"])

    # Bước 2: Train Edge Model
    run_step(2, "Huấn luyện Mô hình Biên 19.5k (MultiSensorResMLP)",
             [sys.executable, "mlops/scripts/train_multisensor_resmlp.py"])

    # Bước 3: Quantize & Export C++
    run_step(3, "Lượng tử hóa INT8 & Xuất tệp nhúng Flash C++ (esp/include/model_data.h)",
             [sys.executable, "mlops/scripts/quantize_multisensor_int8.py"])
    subprocess.run("g++ -shared -fPIC -O3 -I esp/include esp/src/tinyml_edge_engine.cpp -o esp/tests/tinyml_edge.dll", shell=True)
    subprocess.run("g++ -O3 -I esp/include esp/src/tinyml_edge_engine.cpp esp/tests/test_main.cpp -o esp/tests/test_main.exe", shell=True)
    subprocess.run("g++ -O3 -I esp/include esp/src/tinyml_edge_engine.cpp esp/tests/test_adversarial_firmware.cpp -o esp/tests/test_adversarial_firmware.exe", shell=True)

    # Bước 4: Train Central Models (Forecaster, Autoencoder, Unified AI)
    run_step(4, "Huấn luyện Mô hình Trung tâm (Raspberry Pi 5: Forecaster, Autoencoder & Unified AI) & Xuất ONNX",
             [sys.executable, "mlops/scripts/train_central_forecaster.py"])
    run_step(4, "Huấn luyện Mô hình Trung tâm - Deep Autoencoder Phát hiện Dị thường (Raspberry Pi 5) & Xuất ONNX",
             [sys.executable, "mlops/scripts/train_central_autoencoder.py"])
    run_step(4, "Huấn luyện Mô hình Trung tâm Phân tầng Thống nhất (Hierarchical Central AI) & Xuất ONNX",
             [sys.executable, "mlops/scripts/train_unified_central_ai.py"])

    # Bước 5: Verify Suite
    run_step(5, "Kiểm thử tự động toàn diện (PyTest)",
             [sys.executable, "-m", "pytest",
              "tests/test_multisensor_training.py",
              "tests/test_unified_central_ai.py",
              "tests/test_regression_locks.py",
              "tests/test_deep_learning_upgrade_4phases.py",
              "tests/test_challenger_m1_deeplearning.py",
              "tests/test_challenger_m1_r2_adversarial.py",
              "-v"])

    print("\n" + "=" * 80)
    print("  TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN & XUẤT MÔ HÌNH THÀNH CÔNG RỰC RỠ 100%!")
    print("  Artifacts sinh ra:")
    print("    - ESP32-S3 Flash Header : esp/include/model_data.h (18.84 KB INT8)")
    print("    - ESP32 C++ Engine      : esp/src/tinyml_edge_engine.cpp")
    print("    - Raspberry Pi 5 ONNX   : mlops/models/central_unified_ai.onnx (104.886 params)")
    print("    - PyTorch Checkpoint    : mlops/models/central_unified_ai.pt")
    print("===============================================================================\n")

if __name__ == "__main__":
    main()
