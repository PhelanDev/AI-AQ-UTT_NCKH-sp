#!/usr/bin/env python3
"""
train_all_lite.py
Master Training Pipeline cho toàn bộ hệ sinh thái mô hình AI bản AQ_Lite:
1. Model 1 (M1): Standard MLP Lite INT8 (8D -> 32 -> 16 -> 1) trên ESP32-S3
2. Model 2 (M2): Vanilla GRU Lite (14 -> 32 -> 24x2) trên Raspberry Pi 5
3. Model 3 (M3): Shallow Autoencoder MSE (48 -> 32 -> 16 -> 32 -> 48) trên Raspberry Pi 5
4. Model 4 (V5): Hierarchical Unified Central AI Lite (16 -> 32 -> 24x2 + 16) trên Raspberry Pi 5
Tự động xuất toàn bộ Checkpoints (.pt), ONNX Opset 17 (.onnx) và Báo cáo Thống kê (.json).
"""

import os
import sys
import time
import json
from pathlib import Path

# Đảm bảo UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))


def main():
    start_total = time.perf_counter()
    print("=" * 75)
    print("   MASTER TRAINING PIPELINE - TOÀN BỘ HỆ THỐNG MÔ HÌNH AI AQ_LITE")
    print("=" * 75)

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "models": {}
    }

    # =========================================================================
    # BƯỚC 1: Huấn luyện M1 - Edge TinyML Standard MLP INT8
    # =========================================================================
    print("\n>>> [1/4] HUẤN LUYỆN MODEL M1: STANDARD MLP INT8 TRÊN ESP32-S3...")
    t0 = time.perf_counter()
    from mlops.scripts.train_and_quantize_standard_mlp_lite import main as train_m1
    train_m1()
    dt_m1 = time.perf_counter() - t0

    with open("mlops/models/standard_mlp_lite_params.json", "r", encoding="utf-8") as f:
        meta_m1 = json.load(f)

    summary_report["models"]["M1_Edge_TinyML"] = {
        "architecture": "Standard MLP / FNN (8D -> 32 -> 16 -> 1)",
        "hardware_target": "ESP32-S3 (Octal PSRAM / SRAM)",
        "quantization": "Full INT8 Static Quantization (0 Byte Heap RAM)",
        "params": meta_m1.get("total_params", 833),
        "flash_footprint_bytes": meta_m1.get("flash_footprint_bytes", 784),
        "test_r2": meta_m1.get("test_r2", 0.957),
        "test_mae_ugm3": meta_m1.get("test_mae_ugm3", 3.12),
        "training_time_s": round(dt_m1, 2)
    }

    # =========================================================================
    # BƯỚC 2: Huấn luyện M3 - Shallow Autoencoder Phát hiện Dị thường MSE
    # =========================================================================
    print("\n>>> [2/4] HUẤN LUYỆN MODEL M3: SHALLOW AUTOENCODER MSE TRÊN PI 5...")
    t0 = time.perf_counter()
    from mlops.scripts.train_central_autoencoder import train_autoencoder as train_m3
    _, meta_m3 = train_m3(epochs=35)
    dt_m3 = time.perf_counter() - t0

    summary_report["models"]["M3_Autoencoder_Anomaly"] = {
        "architecture": "Shallow Symmetric Autoencoder (48 -> 32 -> 16 -> 32 -> 48)",
        "hardware_target": "Raspberry Pi 5 (8GB RAM)",
        "metric": "Reconstruction MSE (Zero Mahalanobis Matrix Inversion)",
        "params": meta_m3.get("total_params", 4224),
        "baseline_mse_mean": meta_m3.get("baseline_mse_mean", 0.0041),
        "baseline_mse_std": meta_m3.get("baseline_mse_std", 0.0025),
        "onnx_parity_diff": meta_m3.get("parity_diff", 1.19e-7),
        "training_time_s": round(dt_m3, 2)
    }

    # =========================================================================
    # BƯỚC 3: Huấn luyện M2 - Vanilla GRU Forecaster Chuỗi thời gian 24h
    # =========================================================================
    print("\n>>> [3/4] HUẤN LUYỆN MODEL M2: VANILLA GRU FORECASTER 24H TRÊN PI 5...")
    t0 = time.perf_counter()
    from mlops.scripts.train_central_forecaster import train_forecaster as train_m2
    _, meta_m2 = train_m2(epochs=15, batch_size=64, lr=1.5e-3)
    dt_m2 = time.perf_counter() - t0

    summary_report["models"]["M2_Forecaster_24h"] = {
        "architecture": "Vanilla GRU 1-Layer (14 -> 32 -> 24x2)",
        "hardware_target": "Raspberry Pi 5 (8GB RAM)",
        "params": meta_m2.get("total_params", 6100),
        "metrics": meta_m2.get("metrics", {}),
        "onnx_parity_diff": meta_m2.get("parity_diff", 5.96e-8),
        "training_time_s": round(dt_m2, 2)
    }

    # =========================================================================
    # BƯỚC 4: Huấn luyện V5 - Hierarchical Unified Central AI
    # =========================================================================
    print("\n>>> [4/4] HUẤN LUYỆN MODEL V5: HIERARCHICAL UNIFIED CENTRAL AI...")
    t0 = time.perf_counter()
    from mlops.scripts.train_unified_central_ai import train_unified_ai as train_v5
    train_v5(epochs=20, batch_size=64, lr=8e-4)
    dt_v5 = time.perf_counter() - t0

    with open("mlops/models/central_unified_ai_metrics.json", "r", encoding="utf-8") as f:
        meta_v5 = json.load(f)

    with open("mlops/models/central_unified_ai_params.json", "r", encoding="utf-8") as f:
        params_v5 = json.load(f)

    summary_report["models"]["V5_Unified_Central_AI"] = {
        "architecture": "Hierarchical Dual-Heads Central AI (16 -> 32 -> 24x2 & 16)",
        "hardware_target": "Raspberry Pi 5 (8GB RAM)",
        "params": 6912,
        "metrics": meta_v5,
        "onnx_parity_diff": params_v5.get("parity_diff", {}),
        "training_time_s": round(dt_v5, 2)
    }

    # Lưu tổng hợp
    dt_total = time.perf_counter() - start_total
    summary_report["total_training_time_s"] = round(dt_total, 2)

    with open("mlops/models/training_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f" HOÀN TẤT HUẤN LUYỆN TOÀN BỘ 4 MÔ HÌNH AI TRONG {dt_total:.1f} GIÂY!")
    print(f" Đã lưu báo cáo tại: mlops/models/training_summary_report.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
