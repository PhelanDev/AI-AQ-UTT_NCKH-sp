"""
ai/central/simulator.py
Spatial-Temporal Multi-Station Air Quality Simulator for Thanh Xuan, Hanoi.
Physics: 2D Advection-Diffusion, Discrete Gaussian Puff, Ornstein-Uhlenbeck Wind Field,
Diurnal Planetary Boundary Layer Height (PBLH) Inversion, Bimodal Traffic Corridors,
Optical Sensor Hygroscopic Swelling (Hänel / κ-Köhler), and 3 Anomaly Scenarios.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from pi.config import (
    STATIONS,
    STATION_COORDS_UTM,
    NUM_STATIONS,
    AtmosphericConstants,
    StationInfo,
)

class AnomalyType(IntEnum):
    """Categorical classification of physical anomaly signatures."""

    CLEAN = 0
    FIRE = 1
    SENSOR_DRIFT = 2
    MICRO_BURST = 3

@dataclass
class AnomalyScenario:
    """Specification of an injected physical anomaly event."""

    anomaly_type: AnomalyType
    start_time_s: float
    duration_s: float
    target_station_id: Optional[str] = None
    source_coords_utm: Optional[Tuple[float, float]] = None
    emission_rate_pm25: float = 350.0
    emission_rate_voc: float = 300.0
    emission_rate_co2: float = 400.0
    drift_slope_per_day: float = 1.2

class WindFieldOU:
    """Dynamic 2D wind field via Ornstein-Uhlenbeck stochastic mean-reverting process.

    du = theta * (u_mean - u) * dt + sigma * sqrt(dt) * dW_u
    dv = theta * (v_mean - v) * dt + sigma * sqrt(dt) * dW_v
    """

    def __init__(
        self,
        mean_speed: float = 2.4,
        prevailing_bearing_deg: float = 45.0,
        theta: float = AtmosphericConstants.OU_THETA_WIND,
        sigma: float = AtmosphericConstants.OU_SIGMA_WIND,
        seed: Optional[int] = None,
    ):
        self.mean_speed = mean_speed
        self.prevailing_bearing_deg = prevailing_bearing_deg
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

        rad = math.radians(prevailing_bearing_deg)
        self.u_mean = mean_speed * math.sin(rad)
        self.v_mean = mean_speed * math.cos(rad)

        self.u = self.u_mean
        self.v = self.v_mean

    def step(self, dt: float) -> Tuple[float, float, float, float]:
        """Advance wind field by dt seconds using the exact analytical OU transition formula.

        Returns (u, v, wind_speed, bearing_deg).
        """

        decay = math.exp(-self.theta * dt)
        var = (self.sigma**2 / (2.0 * self.theta)) * max(0.0, 1.0 - math.exp(-2.0 * self.theta * dt))
        std = math.sqrt(max(1e-12, var))

        dW_u = self.rng.randn()
        dW_v = self.rng.randn()

        self.u = self.u_mean + (self.u - self.u_mean) * decay + std * dW_u
        self.v = self.v_mean + (self.v - self.v_mean) * decay + std * dW_v

        speed = math.hypot(self.u, self.v)
        if speed < AtmosphericConstants.MIN_WIND_SPEED:
            factor = AtmosphericConstants.MIN_WIND_SPEED / max(speed, 1e-6)
            self.u *= factor
            self.v *= factor
            speed = AtmosphericConstants.MIN_WIND_SPEED

        bearing = math.degrees(math.atan2(self.u, self.v)) % 360.0
        return self.u, self.v, speed, bearing

class DiurnalPBLH:
    """Diurnal Planetary Boundary Layer Height, Temperature, and Relative Humidity cycle.

    - Thermal inversion layer at night (H_min ~ 250m) traps pollutants.
    - Convective boundary layer during day (H_max ~ 1600m) enhances dilution.
    - Temperature peaks at ~14:00, troughs at ~05:00.
    - Relative Humidity is inverted with temperature (85-95% at night, 50-65% at midday).
    """

    def __init__(
        self,
        h_min: float = AtmosphericConstants.PBLH_MIN,
        h_max: float = AtmosphericConstants.PBLH_MAX,
        temp_mean: float = 23.5,
        temp_amp: float = 5.5,
        rh_mean: float = 76.0,
        rh_amp: float = 18.0,
    ):
        self.h_min = h_min
        self.h_max = h_max
        self.temp_mean = temp_mean
        self.temp_amp = temp_amp
        self.rh_mean = rh_mean
        self.rh_amp = rh_amp

    def get_state(self, time_s: float) -> Tuple[float, float, float]:
        """Compute (PBLH_m, temp_c, rh_percent) at elapsed simulation time in seconds."""
        time_hours = (time_s / 3600.0) % 24.0

        if 6.0 <= time_hours <= 20.0:
            solar_phase = math.sin(math.pi * (time_hours - 6.0) / 14.0)
            pblh = self.h_min + (self.h_max - self.h_min) * (max(0.0, solar_phase) ** 1.6)
        else:
            pblh = self.h_min

        temp_phase = math.sin(math.pi * (time_hours - 8.0) / 12.0)
        temp = self.temp_mean + self.temp_amp * temp_phase

        rh = self.rh_mean - self.rh_amp * temp_phase
        rh = max(20.0, min(98.0, rh))

        return float(pblh), float(temp), float(rh)

class TrafficEmissionModel:
    """Bimodal rush-hour traffic emissions along Nguyen Trai and Ring Road 3 corridors.

    Peaks at 08:00 AM (rush hour 1) and 18:00 PM (rush hour 2).
    Weekend traffic attenuated by factor 0.65.
    """

    def __init__(
        self,
        base_emission: float = 18.0,
        morn_peak_emission: float = 85.0,
        eve_peak_emission: float = 95.0,
    ):
        self.base_emission = base_emission
        self.morn_peak_emission = morn_peak_emission
        self.eve_peak_emission = eve_peak_emission

    def get_emission_rate(self, time_s: float) -> float:
        """Compute traffic emission intensity in ug/s."""
        time_hours = (time_s / 3600.0) % 24.0
        day_of_week = int(time_s / 86400.0) % 7
        is_weekend = day_of_week in [5, 6]

        morn_peak = math.exp(
            -((time_hours - AtmosphericConstants.RUSH_HOUR_MORNING_PEAK) ** 2)
            / (2.0 * AtmosphericConstants.RUSH_HOUR_MORNING_WIDTH**2)
        )
        eve_peak = math.exp(
            -((time_hours - AtmosphericConstants.RUSH_HOUR_EVENING_PEAK) ** 2)
            / (2.0 * AtmosphericConstants.RUSH_HOUR_EVENING_WIDTH**2)
        )

        q = self.base_emission + self.morn_peak_emission * morn_peak + self.eve_peak_emission * eve_peak

        if is_weekend:
            q *= AtmosphericConstants.WEEKEND_TRAFFIC_FACTOR

        return float(q)

@dataclass
class GaussianPuff:
    """Discrete Gaussian puff for atmospheric transport."""

    x: float
    y: float
    release_time_s: float
    mass_pm25: float
    mass_pm10: float
    mass_co2: float
    mass_voc: float
    initial_sigma: float = AtmosphericConstants.INITIAL_PUFF_SIGMA
    source_tag: str = "general"

    def advance(self, u: float, v: float, dt: float, lambda_loss: float = AtmosphericConstants.LAMBDA_LOSS):
        """Advect puff with wind and apply exponential decay."""
        self.x += u * dt
        self.y += v * dt
        decay = math.exp(-lambda_loss * dt)
        self.mass_pm25 *= decay
        self.mass_pm10 *= decay
        self.mass_voc *= decay

    def get_dispersion_sigma(self, current_time_s: float, wind_speed: float) -> float:
        """Calculate Pasquill-Gifford urban dispersion width sigma(t)."""
        age = max(1.0, current_time_s - self.release_time_s)
        dist_traveled = wind_speed * age
        sigma = self.initial_sigma + AtmosphericConstants.DISPERSION_A * (
            dist_traveled**AtmosphericConstants.DISPERSION_B
        )
        return float(sigma)

    def compute_concentration(
        self,
        station_x: float,
        station_y: float,
        current_time_s: float,
        wind_speed: float,
        pblh: float,
    ) -> Tuple[float, float, float, float]:
        """Compute concentration contribution at (station_x, station_y).

        Returns (c_pm25, c_pm10, c_co2, c_voc) in ug/m3 or respective units.
        """
        sigma = self.get_dispersion_sigma(current_time_s, wind_speed)
        dist_sq = (station_x - self.x) ** 2 + (station_y - self.y) ** 2

        exponent = -dist_sq / (2.0 * sigma**2)
        if exponent < -40.0:
            return 0.0, 0.0, 0.0, 0.0

        gaussian_factor = math.exp(exponent) / (2.0 * math.pi * sigma**2 * pblh)

        c_pm25 = self.mass_pm25 * gaussian_factor
        c_pm10 = self.mass_pm10 * gaussian_factor
        c_co2 = self.mass_co2 * gaussian_factor
        c_voc = self.mass_voc * gaussian_factor

        return c_pm25, c_pm10, c_co2, c_voc

class HygroscopicSwellingModel:
    """Hänel / κ-Köhler hygroscopic swelling of optical particulate matter.

    Optical sensors (e.g. Plantower PMS7003) measure light scattering, which swells
    rapidly when RH > 75%, leading to false high PM2.5 readings if uncalibrated.
    """

    def __init__(
        self,
        ah: float = AtmosphericConstants.HANEL_AH,
        bh: float = AtmosphericConstants.HANEL_BH,
        max_factor: float = AtmosphericConstants.HANEL_MAX_GROWTH_FACTOR,
    ):
        self.ah = ah
        self.bh = bh
        self.max_factor = max_factor

    def growth_factor(self, rh_percent: float) -> float:
        """Compute optical particle growth factor f_growth(RH)."""
        rh_clamped = max(10.0, min(99.0, rh_percent))
        if rh_clamped <= 60.0:
            return 1.0

        rh_ratio = (rh_clamped / 100.0) / (1.0 - (rh_clamped / 100.0))
        factor = 1.0 + self.ah * (rh_ratio**self.bh)
        return float(min(self.max_factor, factor))

    def simulate_dual_sensor_raw(
        self,
        pm25_true: float,
        pm10_true: float,
        rh_percent: float,
        rng: np.random.RandomState,
    ) -> Tuple[float, float, float]:
        """Simulate dual PMS7003 raw sensor readings (pm25_raw1, pm25_raw2, pm10_raw).

        Incorporates hygroscopic optical swelling and hardware chamber noise.
        """
        growth = self.growth_factor(rh_percent)
        swollen_pm25 = pm25_true * growth
        swollen_pm10 = pm10_true * growth

        noise_pm1 = rng.normal(0.0, 1.2)
        noise_pm2 = rng.normal(0.0, 1.2)
        noise_pm10 = rng.normal(0.0, 2.5)

        pm1_raw = max(0.5, swollen_pm25 + noise_pm1)
        pm2_raw = max(0.5, swollen_pm25 + noise_pm2)
        pm10_raw = max(pm1_raw, swollen_pm10 + noise_pm10)

        if rh_percent > 92.0 and rng.rand() < 0.04:

            if rng.rand() < 0.5:
                pm1_raw += rng.uniform(28.0, 45.0)
            else:
                pm2_raw += rng.uniform(28.0, 45.0)

        return float(pm1_raw), float(pm2_raw), float(pm10_raw)

class AtmosphericSimulator:
    """Integrated Spatial-Temporal Multi-Station Air Quality Simulator."""

    def __init__(
        self,
        stations: Optional[List[StationInfo]] = None,
        dt_s: float = 60.0,
        seed: int = 42,
    ):
        self.stations = stations or STATIONS
        self.num_stations = len(self.stations)
        self.dt_s = dt_s
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.wind_model = WindFieldOU(seed=seed)
        self.pblh_model = DiurnalPBLH()
        self.traffic_model = TrafficEmissionModel()
        self.swelling_model = HygroscopicSwellingModel()

        self.active_puffs: List[GaussianPuff] = []
        self.max_puffs: int = 400

        self.bg_pm25_mean = 28.0
        self.bg_pm10_mean = 42.0
        self.bg_co2_mean = 420.0
        self.bg_voc_mean = 85.0

        self.anomalies: List[AnomalyScenario] = []

    def add_anomaly(self, anomaly: AnomalyScenario):
        """Register an anomaly scenario for injection during simulation."""
        self.anomalies.append(anomaly)

    def _emit_background_and_traffic(self, current_time_s: float):
        """Release new puffs from urban corridors and background traffic."""
        q_traffic = self.traffic_model.get_emission_rate(current_time_s)

        traffic_scale = 5e4
        p1_x, p1_y = self.stations[0].utm_x + self.rng.uniform(-200, 200), self.stations[0].utm_y + self.rng.uniform(-200, 200)
        puff_pm25_1 = q_traffic * traffic_scale * self.dt_s * 0.55
        self.active_puffs.append(
            GaussianPuff(
                x=p1_x,
                y=p1_y,
                release_time_s=current_time_s,
                mass_pm25=puff_pm25_1,
                mass_pm10=puff_pm25_1 * 1.5,
                mass_co2=puff_pm25_1 * 8.0,
                mass_voc=puff_pm25_1 * 3.5,
                source_tag="traffic_nguyen_trai",
            )
        )

        p2_x, p2_y = self.stations[1].utm_x + self.rng.uniform(-200, 200), self.stations[1].utm_y + self.rng.uniform(-200, 200)
        puff_pm25_2 = q_traffic * traffic_scale * self.dt_s * 0.45
        self.active_puffs.append(
            GaussianPuff(
                x=p2_x,
                y=p2_y,
                release_time_s=current_time_s,
                mass_pm25=puff_pm25_2,
                mass_pm10=puff_pm25_2 * 1.6,
                mass_co2=puff_pm25_2 * 9.0,
                mass_voc=puff_pm25_2 * 4.0,
                source_tag="traffic_ring_road_3",
            )
        )

    def _emit_anomalies(self, current_time_s: float):
        """Inject active Type A fire emission puffs."""
        for anom in self.anomalies:
            if anom.anomaly_type == AnomalyType.FIRE:
                if anom.start_time_s <= current_time_s <= (anom.start_time_s + anom.duration_s):

                    coords = anom.source_coords_utm or (self.stations[0].utm_x - 300, self.stations[0].utm_y - 200)

                    scale_fire = 1e6
                    mass_pm25 = anom.emission_rate_pm25 * scale_fire * self.dt_s
                    self.active_puffs.append(
                        GaussianPuff(
                            x=coords[0] + self.rng.normal(0, 15),
                            y=coords[1] + self.rng.normal(0, 15),
                            release_time_s=current_time_s,
                            mass_pm25=mass_pm25,
                            mass_pm10=mass_pm25 * 1.7,
                            mass_co2=anom.emission_rate_co2 * scale_fire * self.dt_s * 25.0,
                            mass_voc=anom.emission_rate_voc * scale_fire * self.dt_s * 10.0,
                            initial_sigma=40.0,
                            source_tag="fire_plume",
                        )
                    )

    def step(self, current_time_s: float) -> Dict[str, np.ndarray]:
        """Advance simulation by 1 time step (dt_s).

        Returns state dictionary containing arrays for all stations.
        """

        u, v, speed, bearing = self.wind_model.step(self.dt_s)

        pblh, temp_c, rh_percent = self.pblh_model.get_state(current_time_s)

        self._emit_background_and_traffic(current_time_s)

        self._emit_anomalies(current_time_s)

        for puff in self.active_puffs:
            puff.advance(u, v, self.dt_s)

        self.active_puffs = [
            p
            for p in self.active_puffs
            if p.mass_pm25 > 1e-2 and (current_time_s - p.release_time_s) < 14400.0
        ]
        if len(self.active_puffs) > self.max_puffs:
            self.active_puffs = self.active_puffs[-self.max_puffs :]

        pm25_true = np.zeros(self.num_stations, dtype=np.float64)
        pm10_true = np.zeros(self.num_stations, dtype=np.float64)
        co2_ambient = np.zeros(self.num_stations, dtype=np.float64)
        voc_ambient = np.zeros(self.num_stations, dtype=np.float64)

        for i, station in enumerate(self.stations):

            env_factor = 1.25 if "traffic" in station.environment or "highway" in station.environment else 1.0
            station_pm25_bg = (self.bg_pm25_mean + self.rng.normal(0, 1.5)) * env_factor
            station_pm10_bg = (self.bg_pm10_mean + self.rng.normal(0, 2.0)) * env_factor
            station_co2_bg = self.bg_co2_mean + (120.0 if env_factor > 1.1 else 30.0) + self.rng.normal(0, 5.0)
            station_voc_bg = self.bg_voc_mean + (40.0 if env_factor > 1.1 else 10.0) + self.rng.normal(0, 3.0)

            sum_pm25 = station_pm25_bg
            sum_pm10 = station_pm10_bg
            sum_co2 = station_co2_bg
            sum_voc = station_voc_bg

            for puff in self.active_puffs:
                cp25, cp10, cco2, cvoc = puff.compute_concentration(
                    station.utm_x, station.utm_y, current_time_s, speed, pblh
                )
                sum_pm25 += cp25
                sum_pm10 += cp10
                sum_co2 += cco2
                sum_voc += cvoc

            pm25_true[i] = max(1.0, sum_pm25)
            pm10_true[i] = max(pm25_true[i] * 1.15, sum_pm10)
            co2_ambient[i] = max(380.0, sum_co2)
            voc_ambient[i] = max(10.0, sum_voc)

        anomaly_flags = np.zeros(self.num_stations, dtype=np.int32)
        drift_offsets = np.zeros(self.num_stations, dtype=np.float64)

        for anom in self.anomalies:

            if anom.anomaly_type == AnomalyType.FIRE:
                if anom.start_time_s <= current_time_s <= (anom.start_time_s + anom.duration_s + 1800.0):
                    for i in range(self.num_stations):
                        if pm25_true[i] > (self.bg_pm25_mean * 2.2):
                            anomaly_flags[i] = int(AnomalyType.FIRE)

            elif anom.anomaly_type == AnomalyType.SENSOR_DRIFT:
                if current_time_s >= anom.start_time_s:
                    for i, st in enumerate(self.stations):
                        if st.id == anom.target_station_id or anom.target_station_id is None and i == 3:
                            elapsed_days = (current_time_s - anom.start_time_s) / 86400.0
                            drift_val = anom.drift_slope_per_day * elapsed_days
                            drift_offsets[i] = drift_val
                            anomaly_flags[i] = int(AnomalyType.SENSOR_DRIFT)

            elif anom.anomaly_type == AnomalyType.MICRO_BURST:
                if anom.start_time_s <= current_time_s <= (anom.start_time_s + anom.duration_s):
                    for i, st in enumerate(self.stations):
                        if st.id == anom.target_station_id or anom.target_station_id is None and i == 2:
                            midpoint = anom.start_time_s + anom.duration_s / 2.0
                            sigma_t = max(180.0, anom.duration_s / 5.0)
                            burst_pulse = anom.emission_rate_pm25 * math.exp(
                                -((current_time_s - midpoint) ** 2) / (2.0 * sigma_t**2)
                            )
                            pm25_true[i] += burst_pulse
                            pm10_true[i] += burst_pulse * 1.4
                            voc_ambient[i] += anom.emission_rate_voc * math.exp(
                                -((current_time_s - midpoint) ** 2) / (2.0 * sigma_t**2)
                            )
                            anomaly_flags[i] = int(AnomalyType.MICRO_BURST)

        pm25_raw_1 = np.zeros(self.num_stations, dtype=np.float64)
        pm25_raw_2 = np.zeros(self.num_stations, dtype=np.float64)
        pm10_raw = np.zeros(self.num_stations, dtype=np.float64)

        for i in range(self.num_stations):

            r1, r2, r10 = self.swelling_model.simulate_dual_sensor_raw(
                pm25_true[i], pm10_true[i], rh_percent, self.rng
            )

            pm25_raw_1[i] = r1 + drift_offsets[i]
            pm25_raw_2[i] = r2 + drift_offsets[i]
            pm10_raw[i] = r10 + drift_offsets[i]

        return {
            "time_s": current_time_s,
            "wind_u": u,
            "wind_v": v,
            "wind_speed": speed,
            "wind_bearing": bearing,
            "pblh": pblh,
            "temp_c": temp_c,
            "rh_percent": rh_percent,
            "pm25_true": pm25_true,
            "pm10_true": pm10_true,
            "pm25_raw_1": pm25_raw_1,
            "pm25_raw_2": pm25_raw_2,
            "pm10_raw": pm10_raw,
            "co2": co2_ambient,
            "voc": voc_ambient,
            "anomaly_flags": anomaly_flags,
        }

    def run_simulation(
        self,
        duration_days: float = 3.0,
        sample_interval_s: float = 600.0,
        start_time_s: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Run full multi-day simulation and return concatenated arrays.

        Total steps = duration_days * 86400 / sample_interval_s
        """
        total_time_s = duration_days * 86400.0
        num_steps = int(math.ceil(total_time_s / sample_interval_s))

        times = np.zeros(num_steps, dtype=np.float64)
        wind_u = np.zeros(num_steps, dtype=np.float64)
        wind_v = np.zeros(num_steps, dtype=np.float64)
        wind_speed = np.zeros(num_steps, dtype=np.float64)
        wind_bearing = np.zeros(num_steps, dtype=np.float64)
        pblh = np.zeros(num_steps, dtype=np.float64)
        temp_c = np.zeros(num_steps, dtype=np.float64)
        rh_percent = np.zeros(num_steps, dtype=np.float64)

        pm25_true = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        pm10_true = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        pm25_raw_1 = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        pm25_raw_2 = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        pm10_raw = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        co2 = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        voc = np.zeros((num_steps, self.num_stations), dtype=np.float64)
        anomaly_flags = np.zeros((num_steps, self.num_stations), dtype=np.int32)

        curr_t = start_time_s
        for step_idx in range(num_steps):
            state = self.step(curr_t)

            times[step_idx] = curr_t
            wind_u[step_idx] = state["wind_u"]
            wind_v[step_idx] = state["wind_v"]
            wind_speed[step_idx] = state["wind_speed"]
            wind_bearing[step_idx] = state["wind_bearing"]
            pblh[step_idx] = state["pblh"]
            temp_c[step_idx] = state["temp_c"]
            rh_percent[step_idx] = state["rh_percent"]

            pm25_true[step_idx] = state["pm25_true"]
            pm10_true[step_idx] = state["pm10_true"]
            pm25_raw_1[step_idx] = state["pm25_raw_1"]
            pm25_raw_2[step_idx] = state["pm25_raw_2"]
            pm10_raw[step_idx] = state["pm10_raw"]
            co2[step_idx] = state["co2"]
            voc[step_idx] = state["voc"]
            anomaly_flags[step_idx] = state["anomaly_flags"]

            curr_t += sample_interval_s

        return {
            "time_s": times,
            "wind_u": wind_u,
            "wind_v": wind_v,
            "wind_speed": wind_speed,
            "wind_bearing": wind_bearing,
            "pblh": pblh,
            "temp_c": temp_c,
            "rh_percent": rh_percent,
            "pm25_true": pm25_true,
            "pm10_true": pm10_true,
            "pm25_raw_1": pm25_raw_1,
            "pm25_raw_2": pm25_raw_2,
            "pm10_raw": pm10_raw,
            "co2": co2,
            "voc": voc,
            "anomaly_flags": anomaly_flags,
        }
