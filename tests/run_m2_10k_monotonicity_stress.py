"""
tests/run_m2_10k_monotonicity_stress.py
Adversarial Stress Test: 10,000 Synthetic and Extreme Edge Telemetry Frames
Verifying Strict Monotonicity Invariant: 0 <= P10 <= P50 <= P90 across all 24 horizons.
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi.central_engine import CentralAIEngine


def generate_frame(i: int, rng: np.random.Generator):
    """Generate diverse synthetic and extreme edge frames."""
    case = i % 10

    if case == 0:
        # Standard clean / normal range
        hist = rng.uniform(5.0, 75.0, size=(24, 8)).astype(np.float32)
        curr = rng.uniform(5.0, 75.0, size=48).astype(np.float32)
    elif case == 1:
        # Industrial surge: steep ramp up to 450 ug/m3
        base = rng.uniform(10.0, 40.0)
        peak = rng.uniform(180.0, 450.0)
        ramp = np.linspace(base, peak, 24, dtype=np.float32)[:, None].repeat(8, axis=1)
        hist = ramp + rng.normal(0, 5, size=(24, 8)).astype(np.float32)
        curr = np.tile(np.array([peak, peak * 1.3, 28.0, 70.0, 2.5, 150.0], dtype=np.float32), 8)
    elif case == 2:
        # Wildfire smoke plume: extreme values up to 800 ug/m3
        hist = rng.uniform(200.0, 800.0, size=(24, 8)).astype(np.float32)
        curr = rng.uniform(200.0, 800.0, size=48).astype(np.float32)
    elif case == 3:
        # Zeros and near-zero clean alpine air
        hist = rng.uniform(0.0, 3.0, size=(24, 8)).astype(np.float32)
        curr = rng.uniform(0.0, 3.0, size=48).astype(np.float32)
    elif case == 4:
        # Saturated humidity and mist / rain (RH = 100%)
        hist = rng.uniform(30.0, 120.0, size=(24, 8)).astype(np.float32)
        curr = np.tile(np.array([60.0, 80.0, 22.0, 100.0, 1.0, 90.0], dtype=np.float32), 8)
    elif case == 5:
        # Extreme desert heat and dry dust (RH = 10%, Temp = 45C)
        hist = rng.uniform(50.0, 250.0, size=(24, 8)).astype(np.float32)
        curr = np.tile(np.array([150.0, 350.0, 45.0, 10.0, 6.0, 110.0], dtype=np.float32), 8)
    elif case == 6:
        # Freezing winter temperature inversion (Temp = -10C)
        hist = rng.uniform(40.0, 220.0, size=(24, 8)).astype(np.float32)
        curr = np.tile(np.array([120.0, 160.0, -10.0, 85.0, 0.5, 300.0], dtype=np.float32), 8)
    elif case == 7:
        # Spatial anomaly / single station point-source flare
        hist = rng.uniform(15.0, 30.0, size=(24, 8)).astype(np.float32)
        hist[:, 3] = rng.uniform(300.0, 600.0, size=24) # station 3 flares
        curr = np.tile(np.array([25.0, 35.0, 25.0, 65.0, 1.5, 100.0], dtype=np.float32), 8)
        curr[3 * 6 : 4 * 6] = np.array([550.0, 700.0, 25.0, 65.0, 1.5, 100.0], dtype=np.float32)
    elif case == 8:
        # Sensor faults: negative values, extreme spikes, high noise
        hist = rng.uniform(-50.0, 250.0, size=(24, 8)).astype(np.float32)
        curr = rng.uniform(-50.0, 300.0, size=48).astype(np.float32)
    else:
        # Step jumps and alternating pulses
        v1 = rng.uniform(20.0, 80.0)
        v2 = rng.uniform(150.0, 400.0)
        seq = np.concatenate([np.full(12, v1), np.full(12, v2)])
        hist = seq[:, None].repeat(8, axis=1).astype(np.float32)
        curr = np.tile(np.array([v2, v2 * 1.3, 26.0, 72.0, 2.0, 130.0], dtype=np.float32), 8)

    # In 2% of samples, inject NaNs and Infs to stress error resilience
    if rng.random() < 0.02:
        nan_mask_h = rng.random(hist.shape) < 0.05
        hist[nan_mask_h] = np.nan
        inf_mask_h = rng.random(hist.shape) < 0.02
        hist[inf_mask_h] = np.inf
        nan_mask_c = rng.random(curr.shape) < 0.05
        curr[nan_mask_c] = np.nan

    return hist, curr


def worker_task(worker_id: int, start_idx: int, count: int):
    """Worker process evaluating a chunk of frames."""
    engine = CentralAIEngine()
    rng = np.random.default_rng(seed=10000 + worker_id)

    quantile_crossings = 0
    negative_p10_count = 0
    nan_inf_count = 0
    unhandled_exceptions = 0
    total_horizons = 0

    max_p90_observed = 0.0
    min_p10_observed = float("inf")

    for i in range(count):
        idx = start_idx + i
        hist, curr = generate_frame(idx, rng)
        try:
            res = engine.process_telemetry_frame(hist, curr)
            fc = res.get("forecast_24h", {})
            p10 = fc.get("pm25_p10", [])
            p50 = fc.get("pm25_p50", [])
            p90 = fc.get("pm25_p90", [])

            if len(p10) != 24 or len(p50) != 24 or len(p90) != 24:
                unhandled_exceptions += 1
                continue

            for h in range(24):
                total_horizons += 1
                v10 = p10[h]
                v50 = p50[h]
                v90 = p90[h]

                if np.isnan(v10) or np.isnan(v50) or np.isnan(v90) or np.isinf(v10) or np.isinf(v50) or np.isinf(v90):
                    nan_inf_count += 1
                    continue

                if v10 < 0.0:
                    negative_p10_count += 1

                # Strict Monotonicity: P10 <= P50 <= P90 (allow machine precision epsilon 1e-6)
                if v10 > v50 + 1e-6 or v50 > v90 + 1e-6:
                    quantile_crossings += 1

                if v90 > max_p90_observed:
                    max_p90_observed = v90
                if v10 < min_p10_observed:
                    min_p10_observed = v10

        except Exception as e:
            unhandled_exceptions += 1

    return {
        "worker_id": worker_id,
        "count": count,
        "total_horizons": total_horizons,
        "quantile_crossings": quantile_crossings,
        "negative_p10_count": negative_p10_count,
        "nan_inf_count": nan_inf_count,
        "unhandled_exceptions": unhandled_exceptions,
        "min_p10_observed": min_p10_observed,
        "max_p90_observed": max_p90_observed,
    }


def main():
    total_frames = 10000
    n_workers = min(8, mp.cpu_count())
    chunk_size = total_frames // n_workers
    remainder = total_frames % n_workers

    print(f"Starting Monotonicity Stress Harness: {total_frames} frames across {n_workers} workers.")
    t0 = time.time()

    tasks = []
    curr_idx = 0
    for w in range(n_workers):
        cnt = chunk_size + (1 if w < remainder else 0)
        tasks.append((w, curr_idx, cnt))
        curr_idx += cnt

    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(worker_task, tasks)

    t1 = time.time()
    elapsed = t1 - t0

    tot_checked = sum(r["count"] for r in results)
    tot_horizons = sum(r["total_horizons"] for r in results)
    tot_crossings = sum(r["quantile_crossings"] for r in results)
    tot_neg = sum(r["negative_p10_count"] for r in results)
    tot_nan = sum(r["nan_inf_count"] for r in results)
    tot_exc = sum(r["unhandled_exceptions"] for r in results)
    min_p10 = min(r["min_p10_observed"] for r in results)
    max_p90 = max(r["max_p90_observed"] for r in results)

    print("\n==================================================================")
    print("      10,000 TELEMETRY FRAMES MONOTONICITY STRESS REPORT         ")
    print("==================================================================")
    print(f"Total Telemetry Frames Processed: {tot_checked:,}")
    print(f"Total Forecast Horizons Audited: {tot_horizons:,} (24 per frame)")
    print(f"Total Elapsed Time: {elapsed:.2f}s ({elapsed/tot_checked*1000:.2f} ms/frame)")
    print(f"Quantile Crossings (P10 > P50 or P50 > P90): {tot_crossings} (Target: 0)")
    print(f"Negative P10 Predictions (P10 < 0.0): {tot_neg} (Target: 0)")
    print(f"NaN / Inf Quantile Outputs: {tot_nan} (Target: 0)")
    print(f"Unhandled Runtime Exceptions: {tot_exc} (Target: 0)")
    print(f"Min P10 observed: {min_p10:.4f} ug/m3")
    print(f"Max P90 observed: {max_p90:.4f} ug/m3")
    print("==================================================================")

    assert tot_checked == total_frames, f"Expected {total_frames} frames, got {tot_checked}"
    assert tot_crossings == 0, f"FAILED: {tot_crossings} quantile crossings detected!"
    assert tot_neg == 0, f"FAILED: {tot_neg} negative P10 values detected!"
    assert tot_nan == 0, f"FAILED: {tot_nan} NaN/Inf values detected!"
    assert tot_exc == 0, f"FAILED: {tot_exc} unhandled exceptions detected!"

    print("VERDICT: PASSED 100% - STRICT MONOTONICITY (P10 <= P50 <= P90) CONFIRMED FOR ALL 10,000 SAMPLES!")


if __name__ == "__main__":
    main()
