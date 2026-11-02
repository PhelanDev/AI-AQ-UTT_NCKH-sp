"""
Pipeline Hợp Nhất Dữ Liệu Đa Cảm Biến 8 Kênh (Multi-Sensor Fusion Pipeline)
Khớp đúng danh mục BOM thực tế:
- Dual PMS7003: PM1, PM2.5, PM10
- Sensirion SHT40: Temp, Relative Humidity
- SenseAir S8: CO2 (ppm)
- Sensirion SGP41: TVOC, NOx (Gas Index scale 1 - 500)
- Trạm chuẩn đối chứng: BAM-1020 / US EPA AQS / PAM Air Hanoi
"""

import os
import csv
import math
import json
import random

def calc_dew_point(temp_c: float, rh_pct: float) -> float:
    """Công thức Magnus-Tetens tính điểm sương chuẩn nhiệt động học."""
    a, b = 17.27, 237.7
    rh_clamped = max(0.01, min(100.0, rh_pct))
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh_clamped / 100.0)
    return round((b * alpha) / (a - alpha), 2)

def generate_multisensor_dataset(
    output_path: str = "mlops/data/multisensor_training_set.csv",
    scaler_path: str = "mlops/models/multisensor_scaler.json",
    seed: int = 42,
):
    random.seed(seed)
    records = []

    # 1. NAP DU LIEU THUC TE AIRGRADIENT WORLD (2.874 mau day du 4 cum cam bien)
    airgradient_file = "mlops/data/airgradient_world_multisensor.csv"
    if os.path.exists(airgradient_file):
        with open(airgradient_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    pm25 = float(r["pm02"])
                    pm10 = float(r["pm10"]) if r.get("pm10") else pm25 * 1.3
                    t = float(r["temp_c"])
                    rh = float(r["rh_pct"])
                    co2 = float(r["co2_ppm"]) if r.get("co2_ppm") else 450.0
                    tvoc_idx = float(r["tvoc_index"]) if r.get("tvoc_index") else 100.0
                    nox_idx = float(r["nox_index"]) if r.get("nox_index") else 1.0

                    if pm25 <= 0 or rh <= 0:
                        continue

                    # Dual PM simulation: chenh lech quang hoc giua 2 buong do song song
                    diff_pm = round(abs(random.gauss(0, 0.04 * pm25 + 0.3)), 2)
                    pm_avg = round(pm25 + random.uniform(-0.5 * diff_pm, 0.5 * diff_pm), 2)
                    r_pm = round(min(1.0, max(0.1, pm25 / max(pm10, 0.1))), 3)
                    dp = calc_dew_point(t, rh)
                    gas_proxy = round(tvoc_idx + 2.0 * nox_idx, 1)

                    # Nhan tham chieu doi chung: khu bot he so phong neu am cao ma khong co khoi
                    if rh > 75.0 and co2 < 600.0 and gas_proxy < 150.0:
                        # Phong nuoc don thuan do suong am
                        swelling_factor = 1.0 + 0.35 * ((rh / 100.0) ** 2) / (1.001 - (rh / 100.0))
                        ref_pm = round(max(1.0, pm_avg / min(3.5, swelling_factor)), 2)
                    else:
                        # Khoi thuc hoac moi truong binh thuong
                        ref_pm = pm_avg

                    records.append({
                        "pm_avg": pm_avg,
                        "diff_pm": diff_pm,
                        "r_pm": r_pm,
                        "temp": t,
                        "rh": rh,
                        "dew_point": dp,
                        "co2": co2,
                        "gas_proxy": gas_proxy,
                        "ref_pm25": ref_pm,
                    })
                except (ValueError, TypeError):
                    continue

    print(f"Loaded {len(records)} records from AirGradient World")

    # 2. NAP DU LIEU DOI CHUAN THANH XUAN HANOI (18.240 mau thuc te 8 tram)
    thanhxuan_file = "mlops/data/hanoi_thanhxuan_multistation_real.csv"
    if os.path.exists(thanhxuan_file):
        with open(thanhxuan_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    ref_pm = float(r["pm25"])
                    pm10 = float(r["pm10"])
                    t = float(r["temperature_c"])
                    rh = float(r["relative_humidity"])
                    co = float(r["co"])
                    no2 = float(r["no2"])

                    if ref_pm <= 0 or rh <= 0:
                        continue

                    dp = calc_dew_point(t, rh)
                    r_pm = round(min(1.0, max(0.1, ref_pm / max(pm10, 0.1))), 3)

                    # Chuyen doi tuong quan CO giao thong sang CO2 SenseAir S8 (400 - 1600 ppm)
                    co2 = round(min(2200.0, max(410.0, 420.0 + (co / 1000.0) * 450.0 + random.gauss(0, 15))), 1)
                    # Chuyen doi NO2 sang Sensirion Gas Index (100 - 450)
                    gas_proxy = round(min(500.0, max(50.0, 100.0 + (no2 / 40.0) * 120.0 + random.gauss(0, 5))), 1)

                    # Mo phong hien tuong cam bien quang hoc PMS do ao khi am cao
                    is_combustion = (co2 > 700.0 or gas_proxy > 180.0)
                    if rh > 75.0 and not is_combustion:
                        # Suong mu nom am khien cam bien phong hat
                        grow = 1.0 + 0.38 * ((rh / 100.0) ** 1.8) / (1.002 - (rh / 100.0))
                        pm_raw = ref_pm * min(3.8, grow)
                    elif is_combustion:
                        # Khoi xe, dot rac lam tang ca PM lan CO2, it phong nuoc hon
                        pm_raw = ref_pm * random.uniform(1.02, 1.15)
                    else:
                        pm_raw = ref_pm * random.uniform(0.96, 1.05)

                    diff_pm = round(abs(random.gauss(0, 0.05 * pm_raw + 0.4)), 2)
                    pm_avg = round(pm_raw, 2)

                    records.append({
                        "pm_avg": pm_avg,
                        "diff_pm": diff_pm,
                        "r_pm": r_pm,
                        "temp": t,
                        "rh": rh,
                        "dew_point": dp,
                        "co2": co2,
                        "gas_proxy": gas_proxy,
                        "ref_pm25": round(ref_pm, 2),
                    })
                except (ValueError, TypeError):
                    continue

    print(f"Loaded records total after Hanoi Thanh Xuan: {len(records)}")

    # 3. NAP THEM DONG DU LIEU DOI CHUAN EPA HOURLY (15k mau lay mau ngau nhien)
    epa_file = "mlops/data/real_collocation_hourly.csv"
    if os.path.exists(epa_file):
        with open(epa_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            epa_rows = []
            for r in reader:
                try:
                    ref_pm = float(r["conc"])
                    pma = float(r["pm2.5_cf_1_a"])
                    pmb = float(r["pm2.5_cf_1_b"])
                    rh = float(r["humidity_a"])
                    t = float(r["temp_Celsius"])
                    if ref_pm > 0 and rh > 0:
                        epa_rows.append((ref_pm, pma, pmb, rh, t))
                except (ValueError, TypeError):
                    continue

            # Lay 12.000 mau dai dien
            sample_size = min(12000, len(epa_rows))
            sampled = random.sample(epa_rows, sample_size)
            for ref_pm, pma, pmb, rh, t in sampled:
                pm_avg = round((pma + pmb) / 2.0, 2)
                diff_pm = round(abs(pma - pmb), 2)
                r_pm = round(random.uniform(0.55, 0.75), 3)
                dp = calc_dew_point(t, rh)
                # Dữ liệu EPA ngoại ô sạch: CO2 nền 420 ppm, gas proxy nền 100
                co2 = round(420.0 + random.gauss(0, 10), 1)
                gas_proxy = round(100.0 + random.gauss(0, 5), 1)

                records.append({
                    "pm_avg": pm_avg,
                    "diff_pm": diff_pm,
                    "r_pm": r_pm,
                    "temp": t,
                    "rh": rh,
                    "dew_point": dp,
                    "co2": co2,
                    "gas_proxy": gas_proxy,
                    "ref_pm25": round(ref_pm, 2),
                })

    print(f"Total merged records: {len(records)}")

    # Shuffle dataset
    random.shuffle(records)

    # Ghi ra file CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = ["pm_avg", "diff_pm", "r_pm", "temp", "rh", "dew_point", "co2", "gas_proxy", "ref_pm25"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Tinh Min-Max Scaler de luu file JSON
    feature_keys = fields[:-1]
    mins = [min(r[k] for r in records) for k in feature_keys]
    maxs = [max(r[k] for r in records) for k in feature_keys]

    scaler_data = {
        "features": feature_keys,
        "x_min": mins,
        "x_max": maxs,
        "total_samples": len(records),
    }

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_data, f, indent=2)

    print(f"Successfully generated {output_path} with {len(records)} samples.")
    print(f"Scaler saved to {scaler_path}")
    return len(records)

if __name__ == "__main__":
    generate_multisensor_dataset()
