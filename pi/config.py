"""
ai/central/config.py
Configuration, Prototype Station, Topology, and Atmospheric Constants
for the Central Neural Network AI Engine on Raspberry Pi 5 (8GB RAM).
Hỗ trợ chế độ thử nghiệm Prototype 1 trạm thực tế (Hardware Demo Node)
kết hợp cơ chế mở rộng mô phỏng đa điểm (Scalability Simulation).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple
import numpy as np

def wgs84_to_utm48n(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS84 latitude and longitude to UTM Zone 48N (Hanoi, Vietnam) coordinates in meters.

    Reference ellipsoid: WGS84 (a = 6378137.0 m, f = 1 / 298.257223563)
    UTM Zone 48 central meridian = 105.0° E, false easting = 500,000 m.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f**2
    k0 = 0.9996
    lon0 = math.radians(105.0)

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    N = a / math.sqrt(1.0 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = (e2 / (1.0 - e2)) * math.cos(lat_r) ** 2
    A = (lon_r - lon0) * math.cos(lat_r)

    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0) * lat_r
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0) * math.sin(2.0 * lat_r)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0) * math.sin(4.0 * lat_r)
        - (35.0 * e2**3 / 3072.0) * math.sin(6.0 * lat_r)
    )

    x = k0 * N * (
        A
        + (1.0 - T + C) * A**3 / 6.0
        + (5.0 - 18.0 * T + T**2 + 72.0 * C - 58.0 * (e2 / (1.0 - e2))) * A**5 / 120.0
    ) + 500000.0

    y = k0 * (
        M
        + N
        * math.tan(lat_r)
        * (
            A**2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C**2) * A**4 / 24.0
            + (61.0 - 58.0 * T + T**2 + 600.0 * C - 330.0 * (e2 / (1.0 - e2))) * A**6 / 720.0
        )
    )

    return float(x), float(y)

def utm48n_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """Chuyển đổi ngược tọa độ UTM Zone 48N sang WGS84 (Vĩ độ/Kinh độ).
    Đạt độ chính xác milimet phục vụ căn chỉnh bản đồ nhiệt và định vị trạm.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f**2
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))
    k0 = 0.9996
    lon0 = math.radians(105.0)

    x_adj = x - 500000.0
    y_adj = y

    M = y_adj / k0
    mu = M / (a * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))

    phi1 = mu + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu) \
              + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu) \
              + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu) \
              + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)

    C1 = (e2 / (1.0 - e2)) * math.cos(phi1)**2
    T1 = math.tan(phi1)**2
    N1 = a / math.sqrt(1.0 - e2 * math.sin(phi1)**2)
    R1 = a * (1.0 - e2) / ((1.0 - e2 * math.sin(phi1)**2)**1.5)
    D = x_adj / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1**2 - 9.0 * (e2 / (1.0 - e2))) * D**4 / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1**2 - 252.0 * (e2 / (1.0 - e2)) - 3.0 * C1**2) * D**6 / 720.0
    )
    lon = lon0 + (
        D
        - (1.0 + 2.0 * T1 + C1) * D**3 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1**2 + 8.0 * (e2 / (1.0 - e2)) + 24.0 * T1**2) * D**5 / 120.0
    ) / math.cos(phi1)

    return float(math.degrees(lat)), float(math.degrees(lon))

@dataclass(frozen=True)
class StationInfo:
    """Information and spatial attributes of an air quality monitoring station."""

    id: str
    name: str
    display_name: str
    lat: float
    lon: float
    utm_x: float
    utm_y: float
    alt_m: float
    environment: str

# ==============================================================================
# DANH MỤC TRẠM QUAN TRẮC:
# - STATION_01: Nút cảm biến Prototype phần cứng thực tế (Hardware Demo Node)
# - STATION_02 đến STATION_08: Tọa độ mô phỏng phục vụ nghiên cứu mở rộng quy mô
# ==============================================================================
PROTOTYPE_STATION_ID: str = "STATION_01"
IS_PROTOTYPE_DEMO: bool = True

_RAW_STATIONS_DATA = [
    (
        "STATION_01",
        "UTT_TrieuKhuc",
        "Trạm Cảm Biến Prototype (54 Triều Khúc - ĐH Công Nghệ GTVT)",
        20.983688,
        105.797477,
        18.0,
        "traffic_corridor",
    ),
    (
        "STATION_02",
        "CV_NhanChinh",
        "Điểm mô phỏng quy hoạch (CV Nhân Chính)",
        20.9997,
        105.8035,
        16.0,
        "urban_park_highway",
    ),
    (
        "STATION_03",
        "KhuongDinh",
        "Điểm mô phỏng quy hoạch (Khương Đình)",
        20.9885,
        105.8192,
        14.0,
        "river_residential",
    ),
    (
        "STATION_04",
        "NguyenTuan",
        "Điểm mô phỏng quy hoạch (Nguyễn Tuân)",
        21.0028,
        105.8020,
        20.0,
        "urban_canyon",
    ),
    (
        "STATION_05",
        "KimGiang",
        "Điểm mô phỏng quy hoạch (Kim Giang)",
        20.9820,
        105.8125,
        15.0,
        "residential_suburban",
    ),
    (
        "STATION_06",
        "TrungHoa",
        "Điểm mô phỏng quy hoạch (Trung Hòa)",
        21.0075,
        105.7950,
        17.0,
        "highway_intersection",
    ),
    (
        "STATION_07",
        "ThanhXuanBac",
        "Điểm mô phỏng quy hoạch (Thanh Xuân Bắc)",
        20.9940,
        105.7975,
        16.0,
        "high_density_residential",
    ),
    (
        "STATION_08",
        "CentralHub",
        "Trạm Trung Tâm Điều Hành (Gateway Pi 5)",
        20.9930,
        105.8080,
        22.0,
        "central_hub",
    ),
]

STATIONS: List[StationInfo] = []
for sid, sname, dname, lat, lon, alt, env in _RAW_STATIONS_DATA:
    ux, uy = wgs84_to_utm48n(lat, lon)
    STATIONS.append(
        StationInfo(
            id=sid,
            name=sname,
            display_name=dname,
            lat=lat,
            lon=lon,
            utm_x=ux,
            utm_y=uy,
            alt_m=alt,
            environment=env,
        )
    )

STATION_IDS: List[str] = [s.id for s in STATIONS]
STATION_NAMES: List[str] = [s.name for s in STATIONS]
STATION_MAP: Dict[str, StationInfo] = {s.id: s for s in STATIONS}

STATION_COORDS_WGS84: np.ndarray = np.array(
    [[s.lat, s.lon] for s in STATIONS], dtype=np.float64
)

STATION_COORDS_UTM: np.ndarray = np.array(
    [[s.utm_x, s.utm_y] for s in STATIONS], dtype=np.float64
)

def compute_distance_matrix(coords_utm: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix D_ij in meters."""
    diff = coords_utm[:, np.newaxis, :] - coords_utm[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def compute_spatial_adjacency_matrix(
    dist_matrix_m: np.ndarray, sigma_m: float = 1200.0
) -> np.ndarray:
    """Compute spatial adjacency matrix A_ij = exp(-D_ij^2 / (2 * sigma^2))."""
    return np.exp(-(dist_matrix_m**2) / (2.0 * (sigma_m**2)))

DISTANCE_MATRIX: np.ndarray = compute_distance_matrix(STATION_COORDS_UTM)
DISTANCE_MATRIX_KM: np.ndarray = DISTANCE_MATRIX / 1000.0
ADJACENCY_MATRIX: np.ndarray = compute_spatial_adjacency_matrix(
    DISTANCE_MATRIX, sigma_m=1200.0
)

@dataclass
class AtmosphericConstants:
    """Atmospheric dispersion, boundary layer, traffic, and sensor parameters."""

    LAMBDA_LOSS: float = 1.5e-5

    PBLH_MIN: float = 250.0
    PBLH_MAX: float = 1600.0

    RUSH_HOUR_MORNING_PEAK: float = 8.0
    RUSH_HOUR_MORNING_WIDTH: float = 1.2
    RUSH_HOUR_EVENING_PEAK: float = 18.0
    RUSH_HOUR_EVENING_WIDTH: float = 1.5
    WEEKEND_TRAFFIC_FACTOR: float = 0.65

    SIGMA_SPATIAL_KERNEL: float = 1200.0

    OU_THETA_WIND: float = 0.05
    OU_SIGMA_WIND: float = 0.35
    MIN_WIND_SPEED: float = 0.4

    DIFFUSION_DX: float = 10.0
    DIFFUSION_DY: float = 10.0

    DISPERSION_A: float = 0.35
    DISPERSION_B: float = 0.85
    INITIAL_PUFF_SIGMA: float = 25.0

    HANEL_AH: float = 0.28
    HANEL_BH: float = 0.92
    HANEL_RH_THRESHOLD: float = 75.0
    HANEL_MAX_GROWTH_FACTOR: float = 4.5

FEATURE_NAMES: List[str] = [
    "pm25_calib",
    "pm10_raw",
    "temp",
    "rh",
    "dew_point",
    "co2",
    "voc",
    "wind_u",
    "wind_v",
    "hour_sin",
    "hour_cos",
]

NUM_STATIONS: int = len(STATIONS)
NUM_FEATURES: int = len(FEATURE_NAMES)

# ==============================================================================
# DUTY CYCLE & TEMPORAL RESOLUTION CONSTANTS (15-MINUTE CYCLE)
# Chu kỳ quan trắc tối ưu năng lượng 15 phút đồng bộ ESP32-S3 và Raspberry Pi 5
# ==============================================================================
TELEMETRY_INTERVAL_S: float = 900.0      # 15 phút (900.0 giây)
ESP_WARMUP_SECONDS: float = 30.0         # 30 giây làm ấm quạt/laser PMS7003
ESP_ACTIVE_PROCESS_SECONDS: float = 2.0  # 2 giây đọc & xử lý cảm biến
ESP_DEEP_SLEEP_SECONDS: float = 868.0    # 14.47 phút ngủ sâu (868.0 giây)
STEPS_PER_HOUR: int = 4                  # 4 bước / giờ
SUBHOURLY_RING_BUFFER_LEN: int = 96      # 96 bước 15 phút = 24 giờ
WLS_MAX_LAG_STEPS: int = 12              # Tương quan trễ tối đa 12 bước (3 giờ)

