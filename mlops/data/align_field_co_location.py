#!/usr/bin/env python3
"""
align_field_co_location.py
Công cụ Chuẩn hóa & Nắn dữ liệu Thực địa Đối chuẩn (Field Co-location Data Alignment).
Giải quyết triệt để 3 bài toán kỹ thuật thực địa:
1. KHỬ LỆCH PHA LẤY MẪU (Resampling & 1-Hour Rolling Window Alignment):
   - Trạm chuẩn (BAM-1020 / PAM Air / CEM) báo cáo trung bình trượt 1 giờ (1-hour backward rolling average).
   - Node ESP32-S3 lấy mẫu chu kỳ 30 giây (120 mẫu/giờ).
   - Tự động gom mẫu (aggregation) theo cửa sổ trượt 1 giờ [T - 1h, T], yêu cầu tối thiểu 75% mẫu hợp lệ (>= 90 mẫu).
2. LỌC BỎ CHU KỲ LÀM ẤM (Warm-up Cycle Filtering):
   - Loại bỏ toàn bộ các mẫu đo trong 30 giây đầu tiên của mỗi chu kỳ thức giấc từ Deep Sleep.
3. GIÁM SÁT TRÔI DẠT SUY GIẢM ĐỘ NHẠY DÀI HẠN (Drift & Aging Filter):
   - Loại bỏ các khoảng thời gian buồng đo bị bụi bám hoặc nghẹt sương nghiêm trọng (|PM1 - PM2| > 25 ug/m3).
"""

import sys
import os
import csv
from datetime import datetime, timedelta
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def parse_iso_time(ts_str: str) -> datetime:
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Không thể parse timestamp: {ts_str}")

def filter_warmup_and_anomalies(raw_node_records):
    """
    Lọc bỏ các mẫu trong chu kỳ làm ấm (< 30s) và các mẫu dị thường tức thời.
    raw_node_records: list of dict {'timestamp', 'pm1', 'pm2', 'temp', 'rh', 'uptime_sec'}
    """
    valid_records = []
    dropped_warmup = 0
    dropped_discrepancy = 0

    for r in raw_node_records:
        uptime = r.get("uptime_sec", 60.0)

        if uptime < 30.0:
            dropped_warmup += 1
            continue

        diff = abs(r["pm1"] - r["pm2"])
        if diff > 25.0:
            dropped_discrepancy += 1
            continue

        valid_records.append(r)

    print(f" -> Lọc làm ấm: Đã loại bỏ {dropped_warmup} mẫu (< 30s khởi động).")
    print(f" -> Lọc lệch buồng đo: Đã loại bỏ {dropped_discrepancy} mẫu (|PM1 - PM2| > 25 ug/m3).")
    print(f" -> Số mẫu sạch hợp lệ: {len(valid_records)} / {len(raw_node_records)} mẫu.")
    return valid_records

def resample_1hour_rolling_backward(clean_records, min_samples_per_window=90):
    """
    Gom mẫu cửa sổ trượt lùi 1 giờ [T - 1h, T] khớp chính xác với quy ước trạm quan trắc quốc gia.
    """
    if not clean_records:
        return []

    clean_records.sort(key=lambda x: x["dt"])
    start_time = clean_records[0]["dt"]
    end_time = clean_records[-1]["dt"]

    curr_hour = start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    hourly_windows = []

    while curr_hour <= end_time:
        win_start = curr_hour - timedelta(hours=1)
        win_end = curr_hour

        in_window = [r for r in clean_records if win_start < r["dt"] <= win_end]

        if len(in_window) >= min_samples_per_window:
            pm1_arr = [r["pm1"] for r in in_window]
            pm2_arr = [r["pm2"] for r in in_window]
            temp_arr = [r["temp"] for r in in_window]
            rh_arr = [r["rh"] for r in in_window]

            pm_avg_1h = float(np.mean((np.array(pm1_arr) + np.array(pm2_arr)) * 0.5))
            diff_pm_1h = float(np.mean(np.abs(np.array(pm1_arr) - np.array(pm2_arr))))
            temp_1h = float(np.mean(temp_arr))
            rh_1h = float(np.mean(rh_arr))

            a, b = 17.27, 237.7
            rh_c = max(1.0, min(100.0, rh_1h))
            alpha = (a * temp_1h) / (b + temp_1h) + np.log(rh_c / 100.0)
            dp_1h = float((b * alpha) / (a - alpha))

            hourly_windows.append({
                "timestamp": curr_hour.strftime("%Y-%m-%d %H:%M:%S"),
                "dt": curr_hour,
                "pm_avg": round(pm_avg_1h, 2),
                "diff_pm": round(diff_pm_1h, 2),
                "temp": round(temp_1h, 2),
                "rh": round(rh_1h, 2),
                "dew_point": round(dp_1h, 2),
                "sample_count": len(in_window)
            })

        curr_hour += timedelta(hours=1)

    print(f" -> Đã tổng hợp thành công {len(hourly_windows)} cửa sổ trượt 1 giờ chuẩn đối chuẩn.")
    return hourly_windows

def align_with_reference_station(hourly_node_records, ref_csv_path, out_aligned_csv):
    """
    Ghép cặp 1-1 theo timestamp giữa hộp cảm biến ESP32-S3 và trạm chuẩn (BAM-1020 / PAM Air).
    """
    if not os.path.exists(ref_csv_path):
        print(f"Lỗi: Không tìm thấy tệp trạm chuẩn: {ref_csv_path}")
        return

    ref_dict = {}
    with open(ref_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = r.get("datetime") or r.get("timestamp") or r.get("time")
            val = r.get("pm25") or r.get("conc") or r.get("value")
            if ts and val:
                try:
                    dt = parse_iso_time(ts)
                    ref_dict[dt] = float(val)
                except Exception:
                    continue

    aligned_pairs = []
    for h in hourly_node_records:
        dt = h["dt"]
        if dt in ref_dict:
            conc_ref = ref_dict[dt]
            if 0.5 <= conc_ref <= 350.0:
                aligned_pairs.append([
                    h["pm_avg"],
                    h["diff_pm"],
                    h["temp"],
                    h["rh"],
                    h["dew_point"],
                    conc_ref,
                    h["timestamp"],
                    h["sample_count"]
                ])

    with open(out_aligned_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pm_avg", "diff_pm", "temp", "rh", "dew_point", "conc", "datetime", "sample_count"])
        for p in aligned_pairs:
            writer.writerow(p)

    print(f" -> Khớp cặp thành công {len(aligned_pairs)} bản ghi 1 giờ đồng bộ hoàn hảo.")
    print(f" -> Tệp dữ liệu nắn sạch đã lưu tại: {out_aligned_csv}")

if __name__ == "__main__":
    print("="*80)
    print("TOOL NẮN DỮ LIỆU ĐỐI CHUẨN THỰC ĐỊA (FIELD CO-LOCATION ALIGNMENT PIPELINE)")
    print("="*80)
    print("Sử dụng khi mang hộp cảm biến ESP32-S3 đo đối chứng cạnh trạm BAM-1020 / PAM Air.")
