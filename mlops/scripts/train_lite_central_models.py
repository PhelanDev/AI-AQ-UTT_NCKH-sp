#!/usr/bin/env python3
"""
train_lite_central_models.py
Script quản lý và bảo vệ các checkpoint mô hình AI Trung tâm Bản Lite:
- central_forecaster.pt (Vanilla GRU Forecaster)
- central_unified_ai.pt (HierarchicalCentralAI Lite)
- central_autoencoder.pt (MultiStationDeepAutoencoder)

LƯU Ý BẢO MẬT/KIẾN TRÚC:
Tuyệt đối KHÔNG tạo trọng số ngẫu nhiên hoặc gán bias tĩnh để ghi đè checkpoint đang vận hành.
Để huấn luyện lại toàn bộ mô hình với dữ liệu thực, chạy: python mlops/scripts/train_all_lite.py
"""

import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))


def check_or_train_lite_models(force_retrain: bool = False):
    print("=" * 65)
    print(" QUẢN LÝ CHECKPOINT AI TRUNG TÂM LITE CHO RASPBERRY PI 5")
    print("=" * 65)

    os.makedirs("mlops/models", exist_ok=True)
    
    forecaster_checkpoint = Path("mlops/models/central_forecaster.pt")
    unified_pt = Path("mlops/models/central_unified_ai.pt")
    autoencoder_pt = Path("mlops/models/central_autoencoder.pt")

    all_exist = forecaster_checkpoint.exists() and unified_pt.exists() and autoencoder_pt.exists()

    if all_exist and not force_retrain:
        print("[AN TOÀN] Các checkpoint mô hình trung tâm đã tồn tại và được bảo vệ:")
        print(f"  + Forecaster  : {forecaster_checkpoint} ({forecaster_checkpoint.stat().st_size:,} bytes)")
        print(f"  + Unified AI  : {unified_pt} ({unified_pt.stat().st_size:,} bytes)")
        print(f"  + Autoencoder : {autoencoder_pt} ({autoencoder_pt.stat().st_size:,} bytes)")
        print("\n[CHÚ Ý] Không ghi đè trọng số ngẫu nhiên lên artifact đang triển khai.")
        print("Để huấn luyện lại toàn diện trên tập dữ liệu thực, sử dụng lệnh:")
        print("    python mlops/scripts/train_all_lite.py")
        print("Hoặc chạy lại với cờ: --force-retrain")
        return

    print("\n-> Đang kích hoạt quy trình huấn luyện thực thụ qua train_all_lite.py...")
    from mlops.scripts.train_all_lite import main as run_train_all
    run_train_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quản lý và bảo vệ Checkpoint AI Lite")
    parser.add_argument("--force-retrain", action="store_true", help="Buộc huấn luyện lại toàn bộ mô hình với dữ liệu thực")
    args = parser.parse_args()
    check_or_train_lite_models(force_retrain=args.force_retrain)
