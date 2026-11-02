"""
ai/central/config.py
Configuration, Station Coordinates, Spatial Topology, and Atmospheric Constants
for the Central Neural Network AI Engine on Raspberry Pi 5 (8GB RAM).
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

_RAW_STATIONS_DATA = [
    (
        "STATION_01",
        "HUS_NguyenTrai",
        "ĐH Khoa học Tự nhiên (334 Nguyễn Trãi)",
        20.9982,
        105.8065,
        18.0,
        "traffic_corridor",
    ),
    (
        "STATION_02",
        "CV_NhanChinh",
        "Công viên Nhân Chính (Khuất Duy Tiến / Lê Văn Lương)",
        20.9997,
        105.8035,
        16.0,
        "urban_park_highway",
    ),
    (
        "STATION_03",
        "KhuongDinh",
        "PAM Air Khương Đình (Vũ Tông Phan / Sông Tô Lịch)",
        20.9885,
        105.8192,
        14.0,
        "river_residential",
    ),
    (
        "STATION_04",
        "NguyenTuan",
        "Ngã tư Nguyễn Tuân - Ngụy Như Kon Tum",
        21.0028,
        105.8020,
        20.0,
        "urban_canyon",
    ),
    (
        "STATION_05",
        "KimGiang",
        "Kim Giang - Hoàng Đạo Thành",
        20.9820,
        105.8125,
        15.0,
        "residential_suburban",
    ),
    (
        "STATION_06",
        "TrungHoa",
        "Nút giao Trung Hòa - Trần Duy Hưng",
        21.0075,
        105.7950,
        17.0,
        "highway_intersection",
    ),
    (
        "STATION_07",
        "ThanhXuanBac",
        "KĐT Thanh Xuân Bắc (Khuất Duy Tiến)",
        20.9940,
        105.7975,
        16.0,
        "high_density_residential",
    ),
    (
        "STATION_08",
        "CentralHub",
        "Trung tâm Điều hành AI (UBND Quận / Hub Server)",
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
