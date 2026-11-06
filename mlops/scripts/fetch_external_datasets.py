#!/usr/bin/env python3
"""
fetch_external_datasets.py
Thu thập và hợp nhất tập dữ liệu quan trắc thực nghiệm quy mô lớn 5 năm (2020 - 2024)
cho Vùng Thủ đô Hà Nội & Vùng Kinh tế Trọng điểm Bắc Bộ.

Bao gồm 5 trạm quan trắc chiến lược đại diện cho các tiểu vùng khí hậu & phát thải:
1. Hanoi_HoanKiem: Vùng lõi đô thị trung tâm, khu ngoại giao & thương mại
2. Hanoi_ThanhXuan: Vùng đô thị mật độ cao, đầu mối giao thông huyết mạch
3. Hanoi_DongAnh: Vùng ngoại thành công nghiệp - đô thị hóa phía Bắc
4. BacNinh_Industrial: Cụm khu công nghiệp điện tử lớn, hành lang gió mùa Đông Bắc
5. HaiPhong_Port: Vùng công nghiệp cảng biển & duyên hải

Các trường dữ liệu hợp nhất đồng bộ từng giờ (Hourly Aligned):
- Chất lượng không khí (CAMS Global Reanalysis): pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone
- Khí tượng & Lớp biên nghịch nhiệt (ERA5 Reanalysis): temperature_2m, relative_humidity_2m, dew_point_2m,
  surface_pressure, wind_speed_10m, wind_direction_10m, boundary_layer_height (PBLH)
"""

import os
import sys
import csv
import json
import time
import math
import urllib.request
import urllib.error
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

STATIONS = [
    {"id": "HN_HOANKIEM", "name": "Hanoi_HoanKiem", "lat": 21.0285, "lon": 105.8542, "type": "Urban_Core"},
    {"id": "HN_THANHXUAN", "name": "Hanoi_ThanhXuan", "lat": 20.9930, "lon": 105.8080, "type": "Urban_HighDensity"},
    {"id": "HN_DONGANH", "name": "Hanoi_DongAnh", "lat": 21.1350, "lon": 105.8450, "type": "Suburban_Industrial"},
    {"id": "BN_YENPHONG", "name": "BacNinh_Industrial", "lat": 21.1861, "lon": 106.0763, "type": "Heavy_Industrial_Zone"},
    {"id": "HP_DINHVU", "name": "HaiPhong_Port", "lat": 20.8449, "lon": 106.6881, "type": "Coastal_Port_Logistics"},
]

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

def fetch_json(url: str, max_retries: int = 4, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AQ-Research-Project/2.0"})
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = resp.read().decode("utf-8")
                    return json.loads(data)
        except Exception as e:
            print(f"    [Thử lại {attempt}/{max_retries}] Lỗi kết nối: {e}")
            if attempt < max_retries:
                time.sleep(attempt * 2.5)
            else:
                raise e
    return {}

def download_station_data(station: dict) -> list:
    sid = station["id"]
    sname = station["name"]
    lat = station["lat"]
    lon = station["lon"]
    stype = station["type"]

    print(f"\n[Trạm {sid} - {sname}] Tọa độ: ({lat}, {lon}) | Loại hình: {stype}")

    # 1. Tải Air Quality (CAMS Global)
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        f"&timezone=Asia%2FBangkok"
    )
    print(f"  -> Tải chuỗi chất lượng không khí 5 năm (2020 - 2024)...")
    t0 = time.time()
    aq_data = fetch_json(aq_url)
    print(f"     + Đã tải xong AQ trong {time.time() - t0:.2f}s")

    # 2. Tải Weather & PBLH (ERA5 Reanalysis)
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,boundary_layer_height"
        f"&timezone=Asia%2FBangkok"
    )
    print(f"  -> Tải chuỗi khí tượng & lớp biên khí quyển PBLH 5 năm (2020 - 2024)...")
    t0 = time.time()
    w_data = fetch_json(weather_url)
    print(f"     + Đã tải xong Khí tượng trong {time.time() - t0:.2f}s")

    aq_hourly = aq_data.get("hourly", {})
    w_hourly = w_data.get("hourly", {})

    times = aq_hourly.get("time", [])
    n = len(times)
    print(f"  -> Gióng hàng thời gian chuỗi giờ: {n:,} mốc...")

    pm25_list = aq_hourly.get("pm2_5", [])
    pm10_list = aq_hourly.get("pm10", [])
    co_list = aq_hourly.get("carbon_monoxide", [])
    no2_list = aq_hourly.get("nitrogen_dioxide", [])
    so2_list = aq_hourly.get("sulphur_dioxide", [])
    o3_list = aq_hourly.get("ozone", [])

    temp_list = w_hourly.get("temperature_2m", [])
    rh_list = w_hourly.get("relative_humidity_2m", [])
    dp_list = w_hourly.get("dew_point_2m", [])
    sp_list = w_hourly.get("surface_pressure", [])
    ws_list = w_hourly.get("wind_speed_10m", [])
    wd_list = w_hourly.get("wind_direction_10m", [])
    pblh_list = w_hourly.get("boundary_layer_height", [])

    records = []
    for i in range(n):
        t_str = times[i]
        pm25 = pm25_list[i] if i < len(pm25_list) else None
        pm10 = pm10_list[i] if i < len(pm10_list) else None
        co = co_list[i] if i < len(co_list) else None
        no2 = no2_list[i] if i < len(no2_list) else None
        so2 = so2_list[i] if i < len(so2_list) else None
        o3 = o3_list[i] if i < len(o3_list) else None

        temp = temp_list[i] if i < len(temp_list) else None
        rh = rh_list[i] if i < len(rh_list) else None
        dp = dp_list[i] if i < len(dp_list) else None
        sp = sp_list[i] if i < len(sp_list) else None
        ws = ws_list[i] if i < len(ws_list) else None
        wd = wd_list[i] if i < len(wd_list) else None
        pblh = pblh_list[i] if i < len(pblh_list) else None

        # Kiểm tra tính toàn vẹn mẫu
        if pm25 is None or temp is None or rh is None:
            continue
        if pm25 < 0 or rh <= 0 or rh > 100:
            continue

        records.append({
            "timestamp": t_str,
            "station_id": sid,
            "station_name": sname,
            "lat": lat,
            "lon": lon,
            "station_type": stype,
            "pm25": round(float(pm25), 2),
            "pm10": round(float(pm10), 2) if pm10 is not None else round(float(pm25) * 1.35, 2),
            "co": round(float(co), 1) if co is not None else 450.0,
            "no2": round(float(no2), 2) if no2 is not None else 25.0,
            "so2": round(float(so2), 2) if so2 is not None else 10.0,
            "o3": round(float(o3), 2) if o3 is not None else 35.0,
            "temperature_c": round(float(temp), 2),
            "relative_humidity": round(float(rh), 2),
            "dew_point": round(float(dp), 2) if dp is not None else round(temp - ((100.0 - rh) / 5.0), 2),
            "surface_pressure": round(float(sp), 1) if sp is not None else 1013.2,
            "wind_speed_10m": round(float(ws), 2) if ws is not None else 1.5,
            "wind_direction_10m": round(float(wd), 1) if wd is not None else 90.0,
            "boundary_layer_height": round(float(pblh), 1) if pblh is not None else 500.0,
        })

    print(f"     + Đã thu thập {len(records):,} bản ghi sạch cho trạm {sid}.")
    return records

def main():
    print("=" * 80)
    print(" PIPELINE THU THẬP & MỞ RỘNG DỮ LIỆU THỰC ĐỊA HÀ NỘI & ĐÔNG NAM Á 5 NĂM")
    print("=" * 80)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "hanoi_regional_5yr_continuous.csv")

    all_records = []
    for st in STATIONS:
        recs = download_station_data(st)
        all_records.extend(recs)
        time.sleep(1.0)  # Thư giãn nhẹ tránh rate-limit

    print("\n" + "=" * 80)
    print(f"TỔNG CỘNG ĐÃ THU THẬP: {len(all_records):,} BẢN GHI THEO GIỜ LIÊN TỤC 5 NĂM")
    print("=" * 80)

    if not all_records:
        print("[LỖI] Không có bản ghi nào được thu thập!")
        sys.exit(1)

    fields = list(all_records[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    file_size_mb = os.path.getsize(out_csv) / (1024 * 1024)
    print(f" -> Đã lưu thành công tệp dữ liệu vào: {out_csv} ({file_size_mb:.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()

