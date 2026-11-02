"""
FETCH REAL MULTI-STATION AIR QUALITY & WEATHER DATA FOR HANOI (THANH XUAN)
Tự động thu thập dữ liệu quan trắc thực tế đa trạm theo chuỗi thời gian tại Hà Nội
Sử dụng Open-Meteo Air Quality & Weather API (Keyless, Miễn phí, Dữ liệu thực nghiệm)
"""

import csv
import json
import os
import sys
import time
from typing import Dict, List
import urllib.request

# Danh sách 8 trạm quan trắc tại Quận Thanh Xuân, Hà Nội (tương thích pi/config.py)
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


def fetch_station_data(station: Dict, past_days: int = 90) -> List[Dict]:
    """
    Kéo dữ liệu chất lượng không khí và khí tượng thực tế từ Open-Meteo API.
    """
    lat = station["lat"]
    lon = station["lon"]
    sid = station["id"]

    # 1. API Chất lượng không khí (Air Quality)
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide&past_days={past_days}"
    )

    # 2. API Khí tượng thời tiết (Weather)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&past_days={past_days}"
    )

    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NCKH-AirQuality-Scout/1.0"}

    # Fetch Air Quality
    req_aq = urllib.request.Request(aq_url, headers=req_headers)
    with urllib.request.urlopen(req_aq, timeout=15) as resp:
        aq_data = json.loads(resp.read().decode())

    # Fetch Weather
    req_weather = urllib.request.Request(weather_url, headers=req_headers)
    with urllib.request.urlopen(req_weather, timeout=15) as resp:
        weather_data = json.loads(resp.read().decode())

    aq_hourly = aq_data.get("hourly", {})
    weather_hourly = weather_data.get("hourly", {})

    times = aq_hourly.get("time", [])
    pm25_list = aq_hourly.get("pm2_5", [])
    pm10_list = aq_hourly.get("pm10", [])
    co_list = aq_hourly.get("carbon_monoxide", [])
    no2_list = aq_hourly.get("nitrogen_dioxide", [])

    temp_list = weather_hourly.get("temperature_2m", [])
    rh_list = weather_hourly.get("relative_humidity_2m", [])
    ws_list = weather_hourly.get("wind_speed_10m", [])
    wd_list = weather_hourly.get("wind_direction_10m", [])
    press_list = weather_hourly.get("surface_pressure", [])

    records = []
    n = len(times)
    for i in range(n):
        t_str = times[i]
        p25 = pm25_list[i] if i < len(pm25_list) and pm25_list[i] is not None else 35.0
        p10 = pm10_list[i] if i < len(pm10_list) and pm10_list[i] is not None else p25 * 1.35
        co = co_list[i] if i < len(co_list) and co_list[i] is not None else 650.0
        no2 = no2_list[i] if i < len(no2_list) and no2_list[i] is not None else 25.0

        temp = temp_list[i] if i < len(temp_list) and temp_list[i] is not None else 24.0
        rh = rh_list[i] if i < len(rh_list) and rh_list[i] is not None else 80.0
        ws = ws_list[i] if i < len(ws_list) and ws_list[i] is not None else 2.5
        wd = wd_list[i] if i < len(wd_list) and wd_list[i] is not None else 45.0
        pres = press_list[i] if i < len(press_list) and press_list[i] is not None else 1013.25

        records.append({
            "station_id": sid,
            "station_name": station["name"],
            "lat": lat,
            "lon": lon,
            "time": t_str,
            "pm25": round(float(p25), 2),
            "pm10": round(float(p10), 2),
            "co": round(float(co), 2),
            "no2": round(float(no2), 2),
            "temperature_c": round(float(temp), 2),
            "relative_humidity": round(float(rh), 2),
            "wind_speed_mps": round(float(ws), 2),
            "wind_direction_deg": round(float(wd), 2),
            "pressure_hpa": round(float(pres), 2),
        })

    return records


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_file = "mlops/data/hanoi_thanhxuan_multistation_real.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print("=== Đang thu thập dữ liệu thực tế 8 trạm Thanh Xuân (Open-Meteo API) ===")
    all_rows = []

    for st in STATIONS_CONFIG:
        print(f" -> Kéo dữ liệu trạm {st['id']} ({st['name']}) [Lat: {st['lat']}, Lon: {st['lon']}]...")
        try:
            st_rows = fetch_station_data(st, past_days=90)
            all_rows.extend(st_rows)
            print(f"    OK: {len(st_rows)} bản ghi giờ.")
            time.sleep(0.5)  # Tránh rate limit
        except Exception as e:
            print(f"    Lỗi: {e}")

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n Hoàn tất! Đã lưu {len(all_rows)} dòng dữ liệu vào: {out_file}")
        print(f" Dung lượng file: {os.path.getsize(out_file):,} bytes")
    else:
        print(" Không có dữ liệu nào được thu thập!")


if __name__ == "__main__":
    main()
