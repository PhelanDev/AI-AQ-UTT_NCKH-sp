#!/usr/bin/env python3
"""
multi_noise_simulator.py
MÔI TRƯỜNG MÔ PHỎNG ĐA NHIỄU VẬT LÝ THỰC ĐỊA CHO CẢM BIẾN QUANG HỌC GIÁ RẺ
Mô hình hóa 5 nguồn nhiễu loạn vật lý thực tế ngoài đời để triệt tiêu học vẹt:
1. Hysteresis Loop (Hiện tượng trễ ẩm ngậm nước vs nhả nước Köhler)
2. Aerosol Composition Shift (Biến thiên thành phần hạt & chiết suất quang học muội than / khoáng / muối)
3. Dual-Channel Asymmetry & Spikes (Độ lệch kênh A/B, zero offset, gain drift, xung dị vật buồng đo)
4. Thermal Self-Heating & Board Microclimate (Tự gia nhiệt ESP32 và sụt giảm RH buồng đo)
5. Optical Chamber Turbulence & Brownian Fluctuations (Nhiễu loạn luồng khí quạt hút 5V)
"""

import sys
import os
import math
import numpy as np
from typing import Dict, Tuple, List

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class MultiNoiseEnvironment:
    """
    Bộ tạo môi trường đa nhiễu vật lý (Physical Multi-Noise Injector).
    Chuyển đổi nồng độ bụi chuẩn (BAM-1020 FEM) và khí tượng thực tế thành
    tín hiệu tán xạ thô rung lắc thực tế của cảm biến kép Plantower PMS7003.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Trạng thái trôi dạt chậm theo chuỗi thời gian (Brownian Drift States)
        self.channel_a_offset = 0.0
        self.channel_b_offset = 0.0
        self.aerosol_factor = 1.0
        self.chamber_fouling = 1.0  # Bám bụi thấu kính dài hạn

    def reset_sensor_state(self):
        """Khởi tạo lại trạng thái vật lý của cặp cảm biến."""
        self.channel_a_offset = self.rng.normal(0.0, 1.8)
        self.channel_b_offset = self.rng.normal(0.0, 1.8)
        self.aerosol_factor = self.rng.uniform(0.85, 1.20)
        self.chamber_fouling = 1.0

    def calc_dew_point(self, temp_c: float, rh_pct: float) -> float:
        """Công thức Magnus-Tetens tính điểm sương nhiệt động học."""
        a, b = 17.27, 237.7
        rh = max(0.05, min(100.0, rh_pct))
        alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
        return (b * alpha) / (a - alpha)

    def simulate_sample(
        self,
        ref_pm25: float,
        temp_c: float,
        rh_pct: float,
        d_rh_dt: float = 0.0,
        co2_ppm: float = 450.0,
        tvoc_ppb: float = 80.0,
        is_combustion: bool = False,
    ) -> Dict[str, float]:
        """
        Tiêm đa nhiễu vật lý vào 1 mốc đo thời gian thực.
        
        Args:
            ref_pm25: Nồng độ bụi chuẩn BAM-1020 (ug/m3)
            temp_c: Nhiệt độ môi trường bên ngoài (C)
            rh_pct: Độ ẩm môi trường bên ngoài (%)
            d_rh_dt: Tốc độ biến thiên ẩm (độ trễ dRH/dt, %/giờ)
            co2_ppm: Nồng độ CO2 SenseAir S8
            tvoc_ppb: Chỉ số khí Sensirion SGP41
            is_combustion: Có khói đốt rác/xe cộ nồng độ cao không
        """
        ref_pm25 = max(0.5, float(ref_pm25))
        temp_c = float(temp_c)
        rh_pct = max(1.0, min(100.0, float(rh_pct)))

        # -------------------------------------------------------------
        # NHIỄU 1: TỰ GIA NHIỆT BO MẠCH (Board Microclimate Coupling)
        # ESP32-S3 + Quạt 5V làm buồng đo ấm hơn bên ngoài 1.2 - 2.5 độ C
        # -------------------------------------------------------------
        self_heating = self.rng.uniform(1.2, 2.4)
        t_chamber = temp_c + self_heating
        # Khi nhiệt độ buồng tăng, áp suất hơi bão hòa tăng -> RH cục bộ buồng giảm
        rh_ratio = rh_pct / 100.0
        p_sat_env = 6.1078 * math.exp((17.27 * temp_c) / (237.3 + temp_c))
        p_sat_cham = 6.1078 * math.exp((17.27 * t_chamber) / (237.3 + t_chamber))
        actual_vapor_p = rh_ratio * p_sat_env
        rh_chamber = min(99.5, max(5.0, (actual_vapor_p / p_sat_cham) * 100.0))

        # -------------------------------------------------------------
        # NHIỄU 2: MÔ HÌNH PHỒNG ẨM QUANG HỌC THỂ TÍCH & VÒNG TRỄ KÖHLER
        # Sử dụng trực tiếp độ ẩm môi trường (rh_pct) vì thời gian hạt qua buồng đo (<150ms)
        # không đủ để nước ngưng tụ trên hạt bay hơi tức thời.
        # -------------------------------------------------------------
        s = min(0.985, max(0.01, rh_pct / 100.0))
        if not is_combustion:
            # Sol khí hòa tan (sunfat/nitrat) phồng mạnh theo mô hình thể tích / tán xạ Mie khi RH > 75%
            kappa_vol = self.rng.uniform(0.35, 0.65)
            volumetric_growth = 1.0 + kappa_vol * (s / max(0.015, 1.0 - s)) ** 1.10
            sig_rh = 1.0 / (1.0 + math.exp(-(rh_pct - 76.5) / 1.6))
            base_growth = 1.0 + (min(4.8, max(1.0, volumetric_growth)) - 1.0) * sig_rh
        else:
            # Sol khí bồ hóng/muội than kỵ nước
            kappa_soot = self.rng.uniform(0.05, 0.12)
            sig_rh = 1.0 / (1.0 + math.exp(-(rh_pct - 76.5) / 1.6))
            base_growth = 1.0 + (min(1.6, max(1.0, 1.0 + kappa_soot * (s / max(0.05, 1.0 - s)))) - 1.0) * sig_rh

        # Hiệu ứng trễ pha Hysteresis:
        if d_rh_dt < -1.0:  # Đang khô đi (nhánh nhả nước Efflorescence chậm)
            hysteresis_factor = 1.0 + self.rng.uniform(0.15, 0.35) * (rh_pct / 100.0)
        elif d_rh_dt > 1.0: # Đang dâng ẩm (nhánh ngậm nước chậm)
            hysteresis_factor = 1.0 - self.rng.uniform(0.05, 0.15) * ((100.0 - rh_pct) / 100.0)
        else:
            hysteresis_factor = self.rng.uniform(0.96, 1.04)

        growth_factor = min(4.8, max(1.0, base_growth * hysteresis_factor))

        # -------------------------------------------------------------
        # NHIỄU 3: BIẾN THIÊN THÀNH PHẦN HẠT (Aerosol Composition Drift)
        # Muội than giao thông (Black Carbon): Tán xạ quang thấp hơn khối lượng
        # Bụi khoáng cát: Tán xạ to hơn
        # -------------------------------------------------------------
        self.aerosol_factor += self.rng.normal(0.0, 0.02)  # Quá trình ngẫu nhiên liên tục
        self.aerosol_factor = float(np.clip(self.aerosol_factor, 0.82, 1.25))

        if is_combustion or co2_ppm > 850.0 or tvoc_ppb > 200.0:
            # Khói đốt thực: Tán xạ quang bị hấp thụ bởi hạt đen bồ hóng
            optical_gain = self.aerosol_factor * self.rng.uniform(0.88, 1.08)
            # Khói thực ít bị phồng ẩm hơn muối hòa tan
            growth_factor = 1.0 + (growth_factor - 1.0) * self.rng.uniform(0.20, 0.40)
        else:
            optical_gain = self.aerosol_factor * self.rng.uniform(0.95, 1.15)

        optical_pm_base = ref_pm25 * growth_factor * optical_gain

        # -------------------------------------------------------------
        # NHIỄU 4: LỆCH KÊNH ĐÔI & NHIỄU BUỒNG QUANG (Dual PMS7003 Mismatch)
        # Hai buồng laser độc lập có độ lệch gain riêng, baseline riêng và nhiễu quạt
        # -------------------------------------------------------------
        # Trôi dạt chậm (Chamber Drift)
        self.channel_a_offset = float(np.clip(self.channel_a_offset + self.rng.normal(0.0, 0.05), -4.0, 4.0))
        self.channel_b_offset = float(np.clip(self.channel_b_offset + self.rng.normal(0.0, 0.05), -4.0, 4.0))

        # Nhiễu quạt hút khí động học (Turbulence / Shot Noise)
        turb_a = self.rng.normal(0.0, 0.04 * optical_pm_base + 1.2)
        turb_b = self.rng.normal(0.0, 0.04 * optical_pm_base + 1.2)

        # Xung dị vật đột biến (Spike anomaly: 1.5% xác suất hạt bụi to hoặc xơ vải lướt qua buồng A hoặc B)
        spike_a = self.rng.exponential(25.0) if self.rng.random() < 0.015 else 0.0
        spike_b = self.rng.exponential(25.0) if self.rng.random() < 0.015 else 0.0

        gain_a = self.rng.uniform(0.94, 1.06)
        gain_b = self.rng.uniform(0.94, 1.06)

        pma = max(0.5, optical_pm_base * gain_a + self.channel_a_offset + turb_a + spike_a)
        pmb = max(0.5, optical_pm_base * gain_b + self.channel_b_offset + turb_b + spike_b)

        diff_pm = abs(pma - pmb)
        pm_avg = (pma + pmb) * 0.5
        dp = self.calc_dew_point(t_chamber, rh_chamber)

        # Giả lập PM10 và tỷ số hạt R_pm
        coarse_ratio = self.rng.uniform(1.25, 1.85) if not is_combustion else self.rng.uniform(1.10, 1.35)
        pm10_raw = max(pm_avg * 1.05, pm_avg * coarse_ratio + self.rng.normal(0.0, 3.0))
        r_pm = float(np.clip(pm_avg / max(pm10_raw, 0.1), 0.15, 0.98))

        # Proxy khí hỗn hợp
        gas_proxy = float(np.clip(tvoc_ppb + 1.5 * self.rng.uniform(5.0, 40.0), 10.0, 600.0))

        return {
            "pm_avg": round(float(pm_avg), 2),
            "diff_pm": round(float(diff_pm), 2),
            "r_pm": round(float(r_pm), 3),
            "temp": round(float(t_chamber), 2),
            "rh": round(float(rh_chamber), 2),
            "dew_point": round(float(dp), 2),
            "co2": round(float(co2_ppm), 1),
            "gas_proxy": round(float(gas_proxy), 1),
            "ref_pm25": round(float(ref_pm25), 2),
            "growth_applied": round(float(growth_factor), 3),
            "is_combustion": 1.0 if is_combustion else 0.0,
        }


def generate_multi_noise_dataset_from_real_records(
    input_csv: str,
    output_csv: str,
    samples_per_station: int = 4000,
    seed: int = 101,
) -> int:
    """
    Đọc chuỗi khí tượng thực tế và tiêm môi trường đa nhiễu vật lý liên tục theo dòng thời gian.
    Bảo toàn 100% trật tự thời gian (Chronological Ordering).
    """
    import csv

    print(f" -> Đang khởi tạo MultiNoiseEnvironment (Seed={seed})...")
    env = MultiNoiseEnvironment(seed=seed)

    rows = []
    with open(input_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f" -> Nạp {len(rows):,} bản ghi từ: {input_csv}")
    
    # Gom theo trạm để giữ liên tục chuỗi thời gian dRH/dt và trôi dạt cảm biến
    from collections import defaultdict
    station_data = defaultdict(list)
    for r in rows:
        st_id = r.get("station_id") or r.get("station_name") or "DEFAULT"
        station_data[st_id].append(r)

    out_records = []
    for st_id, st_rows in station_data.items():
        print(f"    + Xử lý đa nhiễu vật lý cho trạm {st_id} ({len(st_rows)} mốc thời gian)...")
        env.reset_sensor_state()
        prev_rh = 60.0

        for idx, r in enumerate(st_rows[:samples_per_station]):
            try:
                ref_pm = float(r.get("pm25") or r.get("conc") or r.get("PM25") or 25.0)
                temp = float(r.get("temperature_c") or r.get("temp") or r.get("Temperature") or 24.0)
                rh = float(r.get("relative_humidity") or r.get("rh") or r.get("Relative Humidity") or 75.0)
                
                # Tốc độ biến thiên ẩm thực tế (%/h)
                d_rh_dt = (rh - prev_rh)
                prev_rh = rh

                # Nồng độ CO và khí
                co_raw = float(r.get("co") or 500.0)
                co2_proxy = 420.0 + (co_raw / 1000.0) * 450.0 + env.rng.normal(0, 15.0)
                tvoc_proxy = 80.0 + float(r.get("no2") or 20.0) * 3.0 + env.rng.normal(0, 10.0)

                # Nhận diện sự kiện khói giao thông hoặc đốt rác
                is_combustion = (co2_proxy > 800.0 or tvoc_proxy > 180.0 or (ref_pm > 80.0 and rh < 65.0))

                sim = env.simulate_sample(
                    ref_pm25=ref_pm,
                    temp_c=temp,
                    rh_pct=rh,
                    d_rh_dt=d_rh_dt,
                    co2_ppm=co2_proxy,
                    tvoc_ppb=tvoc_proxy,
                    is_combustion=is_combustion,
                )
                sim["station_id"] = st_id
                sim["time_idx"] = idx
                out_records.append(sim)
            except Exception:
                continue

    # Xuất ra file CSV
    if out_records:
        fieldnames = list(out_records[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_records)
        print(f" -> HOÀN TẤT: Đã ghi {len(out_records):,} bản ghi đa nhiễu thực nghiệm vào: {output_csv}")

    return len(out_records)


if __name__ == "__main__":
    import os
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    in_csv = os.path.join(base_dir, "hanoi_thanhxuan_multistation_real.csv")
    out_csv = os.path.join(base_dir, "multi_noise_realistic_dataset.csv")
    generate_multi_noise_dataset_from_real_records(in_csv, out_csv)
