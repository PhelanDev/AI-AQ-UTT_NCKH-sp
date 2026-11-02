"""
pi package
Central Neural Network AI Engine on Raspberry Pi 5 (8GB RAM)
NCKH Multi-Station Air Quality Monitoring & Spatial-Temporal Edge-Central System
"""

from pi.config import (
    STATIONS,
    STATION_NAMES,
    STATION_IDS,
    STATION_COORDS_WGS84,
    STATION_COORDS_UTM,
    DISTANCE_MATRIX,
    ADJACENCY_MATRIX,
    FEATURE_NAMES,
    AtmosphericConstants,
    StationInfo,
    compute_distance_matrix,
    compute_spatial_adjacency_matrix,
    wgs84_to_utm48n,
)
from pi.simulator import (
    AtmosphericSimulator,
    WindFieldOU,
    DiurnalPBLH,
    GaussianPuff,
    HygroscopicSwellingModel,
    TrafficEmissionModel,
    AnomalyScenario,
    AnomalyType,
)
from pi.data_pipeline import (
    AirQualityDataPipeline,
    SpatialTemporalAirQualityDataset,
    create_dataloaders,
    calibrate_pm25_edge,
    calc_dew_point_magnus_tetens,
    filter_dual_pm_sensor,
    calc_vn_aqi_pm25,
)
from pi.spatial_temporal_engine import (
    SpatialTemporalEngine,
    compute_cross_correlation_matrix,
    estimate_wind_and_dispersion,
    localize_emission_source,
)
from pi.forecasting_engine import (
    DilatedTCNBiGRUForecaster,
    AsymmetricHuberLoss,
    calculate_metrics,
    train_central_forecaster,
)
from pi.anomaly_engine import (
    MultiStationDeepAutoencoder,
    AdaptiveAnomalyDetector,
    PhysicalSignatureClassifier,
    AnomalyEngine,
)
from pi.central_engine import (
    CentralAIEngine,
)
from pi.benchmark_pi5 import (
    run_full_pi5_benchmark,
    benchmark_onnx_model,
)

__all__ = [

    "STATIONS",
    "STATION_NAMES",
    "STATION_IDS",
    "STATION_COORDS_WGS84",
    "STATION_COORDS_UTM",
    "DISTANCE_MATRIX",
    "ADJACENCY_MATRIX",
    "FEATURE_NAMES",
    "AtmosphericConstants",
    "StationInfo",
    "compute_distance_matrix",
    "compute_spatial_adjacency_matrix",
    "wgs84_to_utm48n",

    "AtmosphericSimulator",
    "WindFieldOU",
    "DiurnalPBLH",
    "GaussianPuff",
    "HygroscopicSwellingModel",
    "TrafficEmissionModel",
    "AnomalyScenario",
    "AnomalyType",

    "AirQualityDataPipeline",
    "SpatialTemporalAirQualityDataset",
    "create_dataloaders",
    "calibrate_pm25_edge",
    "calc_dew_point_magnus_tetens",
    "filter_dual_pm_sensor",
    "calc_vn_aqi_pm25",

    "SpatialTemporalEngine",
    "compute_cross_correlation_matrix",
    "estimate_wind_and_dispersion",
    "localize_emission_source",

    "DilatedTCNBiGRUForecaster",
    "AsymmetricHuberLoss",
    "calculate_metrics",
    "train_central_forecaster",

    "MultiStationDeepAutoencoder",
    "AdaptiveAnomalyDetector",
    "PhysicalSignatureClassifier",
    "AnomalyEngine",

    "CentralAIEngine",
    "run_full_pi5_benchmark",
    "benchmark_onnx_model",
]
