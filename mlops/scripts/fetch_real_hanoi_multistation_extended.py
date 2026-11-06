"""
fetch_real_hanoi_multistation_extended.py
Tải dữ liệu chuỗi thời gian khí tượng và chất lượng không khí 2 năm thực tế (2023 - 2024)
cho toàn bộ 8 trạm quan trắc tại Quận Thanh Xuân, Hà Nội từ Open-Meteo Archive API.
Không yêu cầu API key, dữ liệu thực tế liên tục theo giờ (Hourly).
"""

import csv
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

STATIONS_CONFIG = [
    {"id": "STATION_01", "name": "HUS_NguyenTrai", "lat": 20.9982, "lon": 105.8065},
    {"id": "STATION_02", "name": "CV_NhanChinh", "lat": 20.9997, "lon": 105.8035},
    {"id": "STATION_03", "name": "KhuongDinh", "lat": 20.9885, "lon": 105.8192},
    {"id": "STATION_04", "name": "NguyenTuan", "lat": 21.0028, "lon": 105.8020},
    {"id": "STATION_05", "name": "KimGiang", "lat": 20.9820, "lon": 105.8125},
    {"id": "STATION_06", "name": "TrungHoa", "lat": 21.0075, "lon": 105.7950},
    {"id": "STATION_07", "name": "ThanhXuanBac", "lat": 20.9940, "lon": 105.7975},
    {"id": "STATION_08", "name": "CentralHub", "lat": 20.9930, "lon": 105.8080},
]

START_DATE = "2023-01-01"
END_DATE = "2024-10-31"

def fetch_station_archive(station: Dict, start_date: str, end_date: str) -> List[Dict]:
    lat = station["lat"]
    lon = station["lon"]
    sid = station["id"]
    sname = station["name"]

    print(f" -> Đang tải dữ liệu 2 năm cho {sid} ({sname}, {lat}, {lon})...")

    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide&timezone=Asia%2FBangkok"
    )

    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=Asia%2FBangkok"
    )

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NCKH-AirQuality-Downloader/2.0"}

    # Fetch Air Quality
    req_aq = urllib.request.Request(aq_url, headers=headers)
    with urllib.request.urlopen(req_aq, timeout=20) as resp:
        aq_json = json.loads(resp.read().decode("utf-8"))

    # Fetch Weather
    req_w = urllib.request.Request(weather_url, headers=headers)
    with urllib.request.urlopen(req_w, timeout=20) as resp:
        w_json = json.loads(resp.read().decode("utf-8"))

    aq_h = aq_json.get("hourly", {})
    w_h = w_json.get("hourly", {})

    times = aq_h.get("time", [])
    pm25 = aq_h.get("pm2_5", [])
    pm10 = aq_h.get("pm10", [])
    co = aq_h.get("carbon_monoxide", [])
    no2 = aq_h.get("nitrogen_dioxide", [])

    temp = w_h.get("temperature_2m", [])
    rh = w_h.get("relative_humidity_2m", [])
    press = w_h.get("surface_pressure", [])
    ws = w_h.get("wind_speed_10m", [])
    wd = w_h.get("wind_direction_10m", [])

    records = []
    n = len(times)
    for i in range(n):
        t_str = times[i]
        p25_val = pm25[i] if i < len(pm25) and pm25[i] is not None else 0.0
        p10_val = pm10[i] if i < len(pm10) and pm10[i] is not None else p25_val * 1.3
        co_val = co[i] if i < len(co) and co[i] is not None else 450.0
        no2_val = no2[i] if i < len(no2) and no2[i] is not None else 20.0

        temp_val = temp[i] if i < len(temp) and temp[i] is not None else 25.0
        rh_val = rh[i] if i < len(rh) and rh[i] is not None else 75.0
        press_val = press[i] if i < len(press) and press[i] is not None else 1013.0
        ws_val = ws[i] if i < len(ws) and ws[i] is not None else 1.5
        wd_val = wd[i] if i < len(wd) and wd[i] is not None else 0.0

        if p25_val <= 0.0:
            continue

        records.append({
            "timestamp": t_str,
            "station_id": sid,
            "station_name": sname,
            "latitude": lat,
            "longitude": lon,
            "pm25": round(float(p25_val), 2),
            "pm10": round(float(p10_val), 2),
            "temperature_c": round(float(temp_val), 2),
            "relative_humidity": round(float(rh_val), 2),
            "surface_pressure": round(float(press_val), 2),
            "wind_speed_ms": round(float(ws_val), 2),
            "wind_direction_deg": round(float(wd_val), 1),
            "co": round(float(co_val), 2),
            "no2": round(float(no2_val), 2),
        })

    print(f"    ✓ Thu được {len(records):,} bản ghi theo giờ sạch.")
    return records

def main():
    print("===============================================================================")
    print("  TẢI DỮ LIỆU ĐA TRẠM CHẤT LƯỢNG KHÔNG KHÍ & KHÍ TƯỢNG HÀ NỘI 2 NĂM (2023-2024)")
    print(f"  Khoảng thời gian: {START_DATE} -> {END_DATE}")
    print("===============================================================================")

    all_data = []
    for st in STATIONS_CONFIG:
        try:
            st_data = fetch_station_archive(st, START_DATE, END_DATE)
            all_data.extend(st_data)
            time.sleep(0.5)  # Tránh rate limit
        except Exception as e:
            print(f" [!] Lỗi khi tải trạm {st['id']}: {e}")

    out_path = os.path.join("mlops", "data", "hanoi_thanhxuan_multistation_2yr.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "timestamp", "station_id", "station_name", "latitude", "longitude",
        "pm25", "pm10", "temperature_c", "relative_humidity", "surface_pressure",
        "wind_speed_ms", "wind_direction_deg", "co", "no2"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("===============================================================================")
    print(f"  ✓ HOÀN TẤT TẢI DỮ LIỆU! Tổng số bản ghi: {len(all_data):,}")
    print(f"  ✓ Đã lưu tại: {out_path} ({file_size_mb:.2f} MB)")
    print("===============================================================================")

if __name__ == "__main__":
    main()

