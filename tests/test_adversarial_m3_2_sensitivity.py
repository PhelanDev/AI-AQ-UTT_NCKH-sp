"""
tests/test_adversarial_m3_2_sensitivity.py
Adversarial Stress-Testing Suite for Milestone 3 (M3_2):
Empirical Sensitivity, Score Margin, Fault Localization, Cold-Start Window Robustness,
and Multi-Station Physical Signature Classification for Task 3 (Phase 4).

Target:
- CentralAIEngine.process_telemetry_frame
- AdaptiveAnomalyDetector & PhysicalSignatureClassifier
"""

import math
import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.central_engine import CentralAIEngine
from pi.anomaly_engine import PhysicalSignatureClassifier


def _create_clean_synthetic_data(seed: int = 123):
    rng = np.random.default_rng(seed)
    hist_clean = np.full((24, 8), 42.0, dtype=np.float32) + rng.normal(0, 3.0, size=(24, 8)).astype(np.float32)
    curr_clean = np.zeros((8, 6), dtype=np.float32)
    for i in range(8):
        curr_clean[i] = [
            42.0 + rng.normal(0, 3.5),
            58.0 + rng.normal(0, 4.5),
            26.0 + rng.normal(0, 0.8),
            74.0 + rng.normal(0, 3.0),
            2.2,
            115.0,
        ]
    return hist_clean, curr_clean


class TestAdversarialStationSensitivityM3_2:
    """Stress tests for station-by-station anomaly detection across all 8 stations."""

    @pytest.mark.parametrize("station_idx", list(range(8)))
    @pytest.mark.parametrize("pm25_fault", [180.0, 230.0, 280.0])
    def test_all_8_stations_pm25_jump_detected(self, station_idx: int, pm25_fault: float):
        """
        Verify that for EVERY station i in [1..8] and PM2.5 in [180..280] ug/m3:
        1. Anomaly is triggered (is_anomaly is True).
        2. Score jump ratio (fault / clean) is strictly > 2.5x.
        """
        engine_clean = CentralAIEngine()
        hist_clean, curr_clean = _create_clean_synthetic_data(seed=123 + station_idx)

        res_clean = engine_clean.process_telemetry_frame(hist_clean, curr_clean.ravel())
        score_clean = res_clean["anomaly"]["anomaly_score"]
        assert not res_clean["anomaly"]["is_anomaly"], f"False alarm on clean frame for seed {station_idx}"

        # Inject localized sensor defect at station_idx
        engine_fault = CentralAIEngine()
        curr_fault = curr_clean.copy()
        curr_fault[station_idx, 0] = pm25_fault
        curr_fault[station_idx, 1] = pm25_fault * 1.3

        res_fault = engine_fault.process_telemetry_frame(hist_clean, curr_fault.ravel())
        score_fault = res_fault["anomaly"]["anomaly_score"]
        is_anom = res_fault["anomaly"]["is_anomaly"]
        ratio = score_fault / max(score_clean, 1e-9)

        assert is_anom is True, (
            f"FAILED: Station {station_idx + 1} with PM2.5={pm25_fault} did NOT trigger anomaly! "
            f"Score={score_fault:.6f}, Thresh={res_fault['anomaly']['dynamic_threshold']:.6f}"
        )
        assert ratio > 2.5, (
            f"FAILED: Station {station_idx + 1} with PM2.5={pm25_fault} score ratio={ratio:.2f}x <= 2.5x!"
        )

    @pytest.mark.parametrize("station_idx", list(range(8)))
    def test_pure_pm25_single_sensor_defect_without_pm10_change(self, station_idx: int):
        """
        Test single-sensor photodiode defect: ONLY PM2.5 jumps to 220 ug/m3,
        while PM10 and all other channels stay untouched at baseline (~58 ug/m3).
        """
        engine_clean = CentralAIEngine()
        hist_clean, curr_clean = _create_clean_synthetic_data(seed=200 + station_idx)

        res_clean = engine_clean.process_telemetry_frame(hist_clean, curr_clean.ravel())
        score_clean = res_clean["anomaly"]["anomaly_score"]

        engine_fault = CentralAIEngine()
        curr_fault = curr_clean.copy()
        curr_fault[station_idx, 0] = 220.0  # Only PM2.5 jumps

        res_fault = engine_fault.process_telemetry_frame(hist_clean, curr_fault.ravel())
        score_fault = res_fault["anomaly"]["anomaly_score"]
        is_anom = res_fault["anomaly"]["is_anomaly"]
        ratio = score_fault / max(score_clean, 1e-9)

        assert is_anom is True, (
            f"Station {station_idx + 1} pure PM2.5 defect not flagged as anomaly! Score={score_fault:.6f}"
        )
        assert ratio > 2.5, f"Station {station_idx + 1} pure PM2.5 defect ratio {ratio:.2f}x <= 2.5x"


class TestAdversarialColdStartWindowM3_2:
    """Stress tests for cold-start history window lengths T in [1, 2, 4, 8, 16, 24]."""

    @pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 24])
    def test_cold_start_history_window_lengths_no_crash_and_functional(self, T: int):
        """
        Verify that for cold-start history window length T in [1, 2, 4, 8, 16, 24]:
        1. CentralAIEngine runs without crash/exception.
        2. Clean frame produces is_anomaly == False.
        3. Localized fault produces is_anomaly == True with ratio > 2.5x.
        """
        hist_clean_full, curr_clean = _create_clean_synthetic_data(seed=777)
        hist_cold = hist_clean_full[:T, :]  # Shape (T, 8)

        engine_clean = CentralAIEngine()
        res_clean = engine_clean.process_telemetry_frame(hist_cold, curr_clean.ravel())
        assert res_clean["anomaly"]["is_anomaly"] is False, f"Cold-start T={T} false alarm on clean data"
        score_clean = res_clean["anomaly"]["anomaly_score"]

        # Inject fault at station 3 (index 2)
        engine_fault = CentralAIEngine()
        curr_fault = curr_clean.copy()
        curr_fault[2, 0] = 240.0
        curr_fault[2, 1] = 290.0

        res_fault = engine_fault.process_telemetry_frame(hist_cold, curr_fault.ravel())
        assert res_fault["anomaly"]["is_anomaly"] is True, f"Cold-start T={T} failed to detect station 3 fault"
        score_fault = res_fault["anomaly"]["anomaly_score"]
        ratio = score_fault / max(score_clean, 1e-9)

        assert ratio > 2.5, f"Cold-start T={T} score ratio {ratio:.2f}x <= 2.5x"


class TestAdversarialMultiStationFireSignatureM3_2:
    """Stress tests for multi-station combustion scenario and physical signature classification."""

    def test_multi_station_fire_combustion_detected_and_classified(self):
        """
        Test multi-station fire/combustion (>= 2 stations elevated, e.g. stations 1, 2, 4 with PM2.5 > 150).
        Verify:
        1. is_anomaly is True.
        2. signature_class_id == 1 (Fire/Combustion).
        3. signature_class_name == "Fire/Combustion".
        4. Fire probability >= 0.80.
        """
        hist_clean, curr_clean = _create_clean_synthetic_data(seed=888)
        engine = CentralAIEngine()

        # Clean baseline check
        res_clean = engine.process_telemetry_frame(hist_clean, curr_clean.ravel())
        assert res_clean["anomaly"]["is_anomaly"] is False
        assert res_clean["anomaly"]["signature_class_id"] == 0
        assert res_clean["anomaly"]["signature_class_name"] == "Clean"

        # Multi-station fire: 3 stations elevated
        curr_fire = curr_clean.copy()
        curr_fire[0, 0] = 210.0  # Station 1
        curr_fire[0, 1] = 260.0
        curr_fire[1, 0] = 185.0  # Station 2
        curr_fire[1, 1] = 230.0
        curr_fire[3, 0] = 160.0  # Station 4
        curr_fire[3, 1] = 200.0

        engine_fire = CentralAIEngine()
        res_fire = engine_fire.process_telemetry_frame(hist_clean, curr_fire.ravel())

        anom = res_fire["anomaly"]
        assert anom["is_anomaly"] is True, "Fire scenario did not trigger is_anomaly=True"
        assert anom["signature_class_id"] == 1, (
            f"Expected class 1 (Fire/Combustion), got {anom['signature_class_id']}: {anom['signature_class_name']}"
        )
        assert anom["signature_class_name"] == "Fire/Combustion"
        assert anom["class_probabilities"][1] >= 0.80, (
            f"Fire probability {anom['class_probabilities'][1]} < 0.80"
        )

    def test_single_station_defect_not_misclassified_as_fire(self):
        """
        Verify that a single station fault (even extreme PM2.5 = 280 ug/m3) is NOT misclassified as Fire/Combustion.
        It should be classified as Local Spike (class 3) or Sensor Drift (class 2).
        """
        hist_clean, curr_clean = _create_clean_synthetic_data(seed=999)
        engine = CentralAIEngine()

        curr_single_fault = curr_clean.copy()
        curr_single_fault[4, 0] = 280.0  # Station 5
        curr_single_fault[4, 1] = 340.0

        res = engine.process_telemetry_frame(hist_clean, curr_single_fault.ravel())
        anom = res["anomaly"]

        assert anom["is_anomaly"] is True
        assert anom["signature_class_id"] != 1, (
            f"Single-station fault falsely classified as Fire/Combustion! {anom['signature_class_name']}"
        )
        assert anom["signature_class_id"] in (2, 3), (
            f"Expected Local Spike (3) or Sensor Drift (2), got {anom['signature_class_id']}"
        )
