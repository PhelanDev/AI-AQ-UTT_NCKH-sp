"""
Xử lý và chuẩn hóa bộ dữ liệu chuỗi thời gian 2 năm Bắc Kinh và Jakarta
Lưu trữ trực tiếp vào thư mục chuẩn hóa: mlops/data/
"""

import os
import sys
import csv
import math

sys.stdout.reconfigure(encoding="utf-8")

def calc_rh(temp_c: float, dewp_c: float) -> float:
    try:
        alpha = ((17.27 * dewp_c) / (237.7 + dewp_c)) - ((17.27 * temp_c) / (237.7 + temp_c))
        rh = 100.0 * math.exp(alpha)
        return round(min(100.0, max(1.0, rh)), 1)
    except Exception:
        return 50.0

def process_beijing(src_path: str, out_path: str):
    print(f" -> Đang xử lý dữ liệu Bắc Kinh từ: {src_path}")
    rows = []
    header = None
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        # Bỏ qua 8 dòng metadata Markdown
        for _ in range(8):
            f.readline()
        reader = csv.reader(f)
        header = next(reader)
        # Tìm index các cột
        col_idx = {name.strip().strip('"'): i for i, name in enumerate(header)}
        for r in reader:
            if not r or len(r) < len(header):
                continue
            try:
                yr = int(r[col_idx["year"]])
                mo = int(r[col_idx["month"]])
                dy = int(r[col_idx["day"]])
                hr = int(r[col_idx["hour"]])
                time_str = f"{yr:04d}-{mo:02d}-{dy:02d} {hr:02d}:00"
                
                pm25 = float(r[col_idx["PM2.5"]]) if r[col_idx["PM2.5"]] != "NA" and r[col_idx["PM2.5"]] != "" else None
                pm10 = float(r[col_idx["PM10"]]) if r[col_idx["PM10"]] != "NA" and r[col_idx["PM10"]] != "" else None
                so2 = float(r[col_idx["SO2"]]) if r[col_idx["SO2"]] != "NA" and r[col_idx["SO2"]] != "" else None
                no2 = float(r[col_idx["NO2"]]) if r[col_idx["NO2"]] != "NA" and r[col_idx["NO2"]] != "" else None
                co = float(r[col_idx["CO"]]) if r[col_idx["CO"]] != "NA" and r[col_idx["CO"]] != "" else None
                o3 = float(r[col_idx["O3"]]) if r[col_idx["O3"]] != "NA" and r[col_idx["O3"]] != "" else None
                temp = float(r[col_idx["TEMP"]]) if r[col_idx["TEMP"]] != "NA" and r[col_idx["TEMP"]] != "" else None
                pres = float(r[col_idx["PRES"]]) if r[col_idx["PRES"]] != "NA" and r[col_idx["PRES"]] != "" else None
                dewp = float(r[col_idx["DEWP"]]) if r[col_idx["DEWP"]] != "NA" and r[col_idx["DEWP"]] != "" else None
                rain = float(r[col_idx["RAIN"]]) if r[col_idx["RAIN"]] != "NA" and r[col_idx["RAIN"]] != "" else 0.0
                wd = r[col_idx["wd"]] if r[col_idx["wd"]] != "NA" else "N"
                wspm = float(r[col_idx["WSPM"]]) if r[col_idx["WSPM"]] != "NA" and r[col_idx["WSPM"]] != "" else 1.0

                rh = calc_rh(temp, dewp) if temp is not None and dewp is not None else 50.0

                rows.append({
                    "time": time_str,
                    "year": yr,
                    "month": mo,
                    "day": dy,
                    "hour": hr,
                    "pm25": pm25 if pm25 is not None else "",
                    "pm10": pm10 if pm10 is not None else "",
                    "so2": so2 if so2 is not None else "",
                    "no2": no2 if no2 is not None else "",
                    "co": co if co is not None else "",
                    "o3": o3 if o3 is not None else "",
                    "temperature": temp if temp is not None else "",
                    "pressure": pres if pres is not None else "",
                    "dew_point": dewp if dewp is not None else "",
                    "relative_humidity": rh,
                    "rain": rain,
                    "wind_dir": wd,
                    "wind_speed": wspm,
                    "city": "Beijing",
                    "station": "Aotizhongxin"
                })
            except Exception:
                continue

    # Lấy 2 năm gần nhất (17.544 dòng) hoặc toàn bộ
    # Ở đây lấy toàn bộ 35.064 dòng (4 năm) nhưng đảm bảo ít nhất 2 năm
    print(f" -> Đọc được tổng cộng {len(rows)} dòng dữ liệu Bắc Kinh.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f" [OK] Đã xuất thành công: {out_path} ({len(rows)} mẫu)")

def process_jakarta(aq_src: str, wx_src: str, out_path: str):
    print(f" -> Đang xử lý dữ liệu Jakarta từ:\n    AQ: {aq_src}\n    WX: {wx_src}")
    
    # 1. Đọc AQ
    aq_data = {}
    with open(aq_src, "r", encoding="utf-8", errors="ignore") as f:
        # Bỏ qua metadata
        lines = [line.strip() for line in f if line.strip()]
        # Tìm dòng bắt đầu bằng time,pm10
        start_idx = 0
        for i, l in enumerate(lines):
            if l.startswith("time,pm10"):
                start_idx = i
                break
        
        reader = csv.DictReader(lines[start_idx:])
        for r in reader:
            t = r.get("time")
            if not t:
                continue
            aq_data[t] = {
                "pm10": r.get("pm10 (μg/m³)", ""),
                "pm25": r.get("pm2_5 (μg/m³)", ""),
                "co": r.get("carbon_monoxide (μg/m³)", ""),
                "no2": r.get("nitrogen_dioxide (μg/m³)", ""),
                "so2": r.get("sulphur_dioxide (μg/m³)", ""),
                "o3": r.get("ozone (μg/m³)", ""),
            }

    # 2. Đọc Weather và gộp
    merged_rows = []
    with open(wx_src, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
        start_idx = 0
        for i, l in enumerate(lines):
            if l.startswith("time,temperature_2m"):
                start_idx = i
                break
        
        reader = csv.DictReader(lines[start_idx:])
        for r in reader:
            t = r.get("time")
            if not t or t not in aq_data:
                continue
            
            # Chuẩn hóa time sang định dạng YYYY-MM-DD HH:00
            t_clean = t.replace("T", " ")
            aq = aq_data[t]
            merged_rows.append({
                "time": t_clean,
                "pm25": aq["pm25"],
                "pm10": aq["pm10"],
                "co": aq["co"],
                "no2": aq["no2"],
                "so2": aq["so2"],
                "o3": aq["o3"],
                "temperature": r.get("temperature_2m (°C)", ""),
                "relative_humidity": r.get("relative_humidity_2m (%)", ""),
                "dew_point": r.get("dew_point_2m (°C)", ""),
                "pressure": r.get("surface_pressure (hPa)", ""),
                "wind_speed": r.get("wind_speed_10m (km/h)", ""),
                "wind_dir": r.get("wind_direction_10m (°)", ""),
                "city": "Jakarta",
                "station": "Jakarta_Central_CAMS_ERA5"
            })

    print(f" -> Đọc và gộp được {len(merged_rows)} dòng dữ liệu 2 năm Jakarta.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(merged_rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f" [OK] Đã xuất thành công: {out_path} ({len(merged_rows)} mẫu)")

def main():
    beijing_src = r"C:\Users\LEGION\.gemini\antigravity\brain\aa0cd030-2bd2-4db2-b7d0-ac74526c20b2\.system_generated\steps\28\content.md"
    jakarta_aq_src = r"C:\Users\LEGION\.gemini\antigravity\brain\aa0cd030-2bd2-4db2-b7d0-ac74526c20b2\.system_generated\steps\94\content.md"
    jakarta_wx_src = r"C:\Users\LEGION\.gemini\antigravity\brain\aa0cd030-2bd2-4db2-b7d0-ac74526c20b2\.system_generated\steps\100\content.md"

    beijing_out = r"d:\AQ\mlops\data\beijing_hourly_2yr.csv"
    jakarta_out = r"d:\AQ\mlops\data\jakarta_hourly_2yr.csv"

    print("===============================================================================")
    print("    XỬ LÝ DỮ LIỆU ĐA NĂM BẮC KINH VÀ JAKARTA CHO MLOPS PIPELINE")
    print("===============================================================================")
    process_beijing(beijing_src, beijing_out)
    process_jakarta(jakarta_aq_src, jakarta_wx_src, jakarta_out)
    print("===============================================================================")
    print("HOÀN TẤT TRÍCH XUẤT 100%!")

if __name__ == "__main__":
    main()
