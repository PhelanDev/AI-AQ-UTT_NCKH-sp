"""
tests/test_challenger_m3_phase4_adversarial.py
Adversarial Stress-Testing Harness for Task 3 (Phase 4):
Empirical verification of SVD Rank-6 Collinearity Breaking and False Alarm Rate
under Natural Wind Oscillations for Multi-Station Autoencoder and Central Engine.
"""

import os
import sys
import math
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlops.scripts.train_central_autoencoder import prepare_48d_multistation_dataset
from pi.central_engine import CentralAIEngine


class TestM3Phase4EmpiricalAdversarial:
    """Adversarial stress-test suite for Milestone 3 (Phase 4)."""

    def test_svd_rank6_collinearity_breaking_across_diverse_seeds(self):
        """
        Adversarially challenge SVD variance ratio across diverse random seeds.
        Assert that top-6 singular value variance ratio on centered X_train is strictly < 0.90
        across all seeds (target: 75% - 88%), proving complete breaking of the previous 98.17% rank-6 degeneracy.
        """
        test_seeds = [42, 100, 2026, 9999, 0, 777, 12345, 88888, 314159, 2024]
        ratios_train = []
        eff_ranks = []

        print("\n--- SVD Variance Ratio Stress on X_train Across 10 Diverse Seeds ---")
        for seed in test_seeds:
            X_train, X_val, min_48, max_48 = prepare_48d_multistation_dataset(seed=seed)

            # SVD on centered X_train
            c_train = X_train - np.mean(X_train, axis=0)
            _, s_train, _ = np.linalg.svd(c_train, full_matrices=False)
            var_train = s_train ** 2
            top6_ratio_tr = float(np.sum(var_train[:6]) / np.sum(var_train))
            ratios_train.append(top6_ratio_tr)

            # Calculate effective rank (Shannon entropy of singular values)
            p_tr = var_train / np.sum(var_train)
            entropy_tr = -np.sum(p_tr * np.log(p_tr + 1e-12))
            eff_rank_tr = float(np.exp(entropy_tr))
            eff_ranks.append(eff_rank_tr)

            print(
                f"Seed {seed:6d} -> X_train Top-6: {top6_ratio_tr*100:.2f}% | "
                f"Eff-Rank: {eff_rank_tr:.2f} | s[0..5]: {np.round(s_train[:6], 2)}"
            )

            # Strict assertion per seed (< 0.90 and >= 0.70)
            assert top6_ratio_tr < 0.90, (
                f"Seed {seed} failed SVD rank-6 breaking! top6_ratio={top6_ratio_tr:.4f} >= 0.90"
            )
            assert top6_ratio_tr >= 0.70, (
                f"Seed {seed} variance ratio {top6_ratio_tr:.4f} abnormally collapsed below 0.70"
            )

        mean_tr = float(np.mean(ratios_train))
        min_tr = float(np.min(ratios_train))
        max_tr = float(np.max(ratios_train))
        std_tr = float(np.std(ratios_train))

        print(
            f"\nSVD Top-6 Ratio Summary Across {len(test_seeds)} Seeds: "
            f"Mean={mean_tr*100:.2f}%, Min={min_tr*100:.2f}%, Max={max_tr*100:.2f}%, Std={std_tr*100:.2f}%"
        )

        assert max_tr < 0.90, f"Max train ratio {max_tr:.4f} exceeds 0.90 threshold"
        assert min_tr >= 0.75, f"Min train ratio {min_tr:.4f} below target range (75%-88%)"
        assert max_tr <= 0.88, f"Max train ratio {max_tr:.4f} above target range (75%-88%)"

    def test_natural_wind_oscillations_false_alarm_rate_1000_independent_frames(self):
        """
        Simulate 1,000 independent frames of natural clean atmospheric wind variations
        with +-8% to +-15% cross-station fluctuation across 8 stations.
        Each frame is evaluated independently against the calibrated baseline detector state.
        Empirically count false positives where res["anomaly"]["is_anomaly"] is True.
        Assert false alarm rate is <= 1.0% (target: 0.0%).
        """
        engine = CentralAIEngine()
        rng = np.random.default_rng(2026)

        n_trials = 1000
        false_alarms_count = 0
        scores = []
        thresholds = []

        # Baseline parameters from calibrated central_autoencoder_params.json
        base_mu = engine.anomaly_detector.mu_ema
        base_var = engine.anomaly_detector.var_ema

        # Atmospheric conditions: natural baseline variations [25, 55] ug/m3
        base_pm_levels = [28.0, 35.0, 42.0, 50.0]

        for i in range(n_trials):
            # Reset detector state to calibrated baseline for pure independent frame assessment
            engine.anomaly_detector.mu_ema = base_mu
            engine.anomaly_detector.var_ema = base_var

            base_pm = base_pm_levels[i % len(base_pm_levels)]
            fluct_pct = rng.uniform(0.08, 0.15)

            # 24h history with mild spatial variation
            hist = np.full((24, 8), base_pm, dtype=np.float32) + rng.normal(
                0, base_pm * 0.07, size=(24, 8)
            ).astype(np.float32)

            curr = np.zeros((8, 6), dtype=np.float32)
            wind_spd = rng.uniform(1.2, 4.5)
            wind_dir = rng.uniform(0.0, 360.0)

            for s in range(8):
                st_delta = rng.uniform(-fluct_pct, fluct_pct)
                st_pm25 = max(5.0, base_pm * (1.0 + st_delta) + rng.normal(0, 1.5))
                st_pm10 = st_pm25 * rng.uniform(1.2, 1.45)
                st_temp = 25.0 + rng.normal(0, 1.0)
                st_rh = np.clip(72.0 + rng.normal(0, 4.0), 30.0, 95.0)

                curr[s] = [st_pm25, st_pm10, st_temp, st_rh, wind_spd, wind_dir]

            res = engine.process_telemetry_frame(hist, curr.ravel())
            is_anom = res["anomaly"]["is_anomaly"]
            score = res["anomaly"]["anomaly_score"]
            thresh = res["anomaly"]["dynamic_threshold"]

            scores.append(score)
            thresholds.append(thresh)

            if is_anom:
                false_alarms_count += 1

        far_pct = (false_alarms_count / n_trials) * 100.0
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        mean_thresh = float(np.mean(thresholds))

        print(f"\n--- 1,000 Natural Wind Fluctuation Frames (Independent) ---")
        print(f"Total Trials: {n_trials}")
        print(f"False Alarms Count: {false_alarms_count}")
        print(f"Empirical False Alarm Rate: {far_pct:.2f}% (Requirement <= 1.0%, Target 0.0%)")
        print(f"Mean Anomaly Score: {mean_score:.6f} | Max Score: {max_score:.6f}")
        print(f"Mean Dynamic Threshold: {mean_thresh:.6f}")

        assert far_pct <= 1.0, f"False alarm rate {far_pct:.2f}% exceeds strict 1.0% tolerance!"

    def test_natural_wind_oscillations_streaming_600_sequential_frames(self):
        """
        Simulate continuous operational deployment: 600 sequential frames streaming
        through CentralAIEngine with continuous temporal transitions and wind oscillations.
        Verify that the adaptive threshold maintains stability and false alarm rate <= 1.0%.
        """
        engine = CentralAIEngine()
        rng = np.random.default_rng(9999)

        n_frames = 600
        false_alarms_count = 0
        scores = []
        thresholds = []

        bg_pm = 38.0
        wind_spd = 2.5
        wind_dir = 90.0

        hist_buffer = np.full((24, 8), bg_pm, dtype=np.float32)

        for step in range(n_frames):
            bg_pm = np.clip(bg_pm + rng.normal(0, 0.4), 18.0, 70.0)
            wind_spd = np.clip(wind_spd + rng.normal(0, 0.1), 0.8, 6.0)
            wind_dir = (wind_dir + rng.normal(0, 3.0)) % 360.0

            curr_frame = np.zeros((8, 6), dtype=np.float32)
            for s in range(8):
                fluct = rng.uniform(-0.12, 0.12)
                s_pm = max(5.0, bg_pm * (1.0 + fluct) + rng.normal(0, 1.2))
                s_pm10 = s_pm * rng.uniform(1.22, 1.40)
                s_temp = 24.0 + rng.normal(0, 0.5)
                s_rh = 75.0 + rng.normal(0, 2.0)
                curr_frame[s] = [s_pm, s_pm10, s_temp, s_rh, wind_spd, wind_dir]

            hist_buffer = np.vstack([hist_buffer[1:], curr_frame[:, 0]])

            res = engine.process_telemetry_frame(hist_buffer, curr_frame.ravel())
            is_anom = res["anomaly"]["is_anomaly"]
            score = res["anomaly"]["anomaly_score"]
            thresh = res["anomaly"]["dynamic_threshold"]

            scores.append(score)
            thresholds.append(thresh)

            if is_anom:
                false_alarms_count += 1

        far_pct = (false_alarms_count / n_frames) * 100.0
        print(f"\n--- {n_frames} Continuous Streaming Frames ---")
        print(f"Total Sequential Frames: {n_frames}")
        print(f"False Alarms Count: {false_alarms_count}")
        print(f"Streaming False Alarm Rate: {far_pct:.2f}% (Requirement <= 1.0%)")
        print(f"Mean Score: {np.mean(scores):.6f} | Mean Threshold: {np.mean(thresholds):.6f}")

        assert far_pct <= 1.0, f"Sequential false alarm rate {far_pct:.2f}% exceeds 1.0% tolerance!"

    def test_positive_control_anomaly_detection_power(self):
        """
        Positive Controls: Ensure the model is NOT merely predicting 'False' unconditionally.
        Test critical real anomaly scenarios:
        1. Single-station sensor fault / catastrophic drift (+180 ug/m3).
        2. Real localized combustion/fire event across multiple downwind stations.
        Assert that true anomalies trigger is_anomaly=True with high jump ratio (> 2.5x).
        """
        engine = CentralAIEngine()
        rng = np.random.default_rng(42)

        hist_clean = np.full((24, 8), 40.0, dtype=np.float32)
        clean_curr = np.zeros((8, 6), dtype=np.float32)
        for s in range(8):
            clean_curr[s] = [40.0 + rng.normal(0, 2.0), 55.0, 25.0, 75.0, 2.5, 90.0]

        res_clean = engine.process_telemetry_frame(hist_clean, clean_curr.ravel())
        clean_score = res_clean["anomaly"]["anomaly_score"]
        assert res_clean["anomaly"]["is_anomaly"] is False

        # Positive Control 1: Single station catastrophic sensor drift (Station 3 = 260 ug/m3)
        fault_curr = clean_curr.copy()
        fault_curr[2, 0] = 260.0
        fault_curr[2, 1] = 310.0
        res_fault1 = engine.process_telemetry_frame(hist_clean, fault_curr.ravel())

        print(f"\n--- Positive Control 1: Single Station Sensor Drift ---")
        print(f"Clean Score: {clean_score:.6f} | Fault Score: {res_fault1['anomaly']['anomaly_score']:.6f}")
        print(f"Score Ratio: {res_fault1['anomaly']['anomaly_score'] / clean_score:.2f}x")
        assert res_fault1["anomaly"]["is_anomaly"] is True
        assert res_fault1["anomaly"]["anomaly_score"] > clean_score * 2.5

        # Positive Control 2: Localized Combustion / Fire plume (Stations 0, 1, 3 elevated)
        fire_curr = clean_curr.copy()
        for st_idx in [0, 1, 3]:
            fire_curr[st_idx, 0] = 165.0  # PM2.5 spike
            fire_curr[st_idx, 1] = 230.0  # PM10 spike
        res_fire = engine.process_telemetry_frame(hist_clean, fire_curr.ravel())

        print(f"\n--- Positive Control 2: Multi-Station Combustion Signature ---")
        print(f"Clean Score: {clean_score:.6f} | Fire Score: {res_fire['anomaly']['anomaly_score']:.6f}")
        print(f"Score Ratio: {res_fire['anomaly']['anomaly_score'] / clean_score:.2f}x")
        assert res_fire["anomaly"]["is_anomaly"] is True
        assert res_fire["anomaly"]["anomaly_score"] > clean_score * 3.0
