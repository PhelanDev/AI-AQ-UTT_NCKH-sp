"""
Adversarial Empirical Test Harness for Model 1 (Köhler-PINN SE-ResMLP)
Focus: US EPA AQS Collocation Accuracy, Humidity Bins, Chronological Data Leakage, and True Out-of-Time Generalization.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import csv
import numpy as np
import torch
from datetime import datetime, timedelta

from mlops.scripts.train_multisensor_resmlp import (
    MultiSensorResMLP,
    load_dataset_from_csv,
    compute_dew_point,
    predict_calibrated_pm25,
)

def compute_metrics(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return {"n": 0, "r2": 0.0, "mae": 0.0, "rmse": 0.0}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-6)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"n": n, "r2": r2, "mae": mae, "rmse": rmse}

def evaluate_model_on_data(model, X, PM, Ref, scaler):
    x_min = np.array(scaler["x_min"], dtype=np.float32)
    x_max = np.array(scaler["x_max"], dtype=np.float32)
    X_norm = np.clip((X - x_min) / np.maximum(x_max - x_min, 1e-6), 0.0, 1.0)
    with torch.no_grad():
        c_pred = model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(-1).numpy()
    pred = PM * c_pred
    return pred, c_pred

def test_epa_current_test_set_bins():
    print("=" * 80)
    print("TEST 1: EPA Test Set (3,000 samples) Performance across Humidity Bins")
    print("=" * 80)
    
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    with open("mlops/models/multisensor_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)
        
    X_test, PM_test, Ref_test, RH_test = load_dataset_from_csv("mlops/data/multisensor_test_set.csv")
    
    # The last 3000 rows are EPA
    X_epa = X_test[-3000:]
    PM_epa = PM_test[-3000:]
    Ref_epa = Ref_test[-3000:]
    RH_epa = RH_test[-3000:]
    
    pred_epa, c_epa = evaluate_model_on_data(model, X_epa, PM_epa, Ref_epa, scaler)
    overall = compute_metrics(Ref_epa, pred_epa)
    print(f"OVERALL EPA Test Set (3000 samples): R2 = {overall['r2']:.4f}, MAE = {overall['mae']:.2f}, RMSE = {overall['rmse']:.2f}")
    
    # Humidity bins
    rh_bins = [(0, 35), (35, 50), (50, 65), (65, 80), (80, 100)]
    print("\n--- Humidity Bins Breakdown ---")
    for low, high in rh_bins:
        mask = (RH_epa >= low) & (RH_epa < high)
        if np.sum(mask) > 0:
            m = compute_metrics(Ref_epa[mask], pred_epa[mask])
            c_mean = float(np.mean(c_epa[mask]))
            print(f"RH [{low:2d}%, {high:2d}%): n = {m['n']:4d} | R2 = {m['r2']:7.4f} | MAE = {m['mae']:5.2f} | RMSE = {m['rmse']:5.2f} | mean_C = {c_mean:.3f}")
        else:
            print(f"RH [{low:2d}%, {high:2d}%): n = 0")

    # PM concentration bins
    pm_bins = [(0, 10), (10, 25), (25, 50), (50, 500)]
    print("\n--- PM Concentration Bins Breakdown ---")
    for low, high in pm_bins:
        mask = (Ref_epa >= low) & (Ref_epa < high)
        if np.sum(mask) > 0:
            m = compute_metrics(Ref_epa[mask], pred_epa[mask])
            print(f"Ref PM [{low:3d}, {high:3d}): n = {m['n']:4d} | R2 = {m['r2']:7.4f} | MAE = {m['mae']:5.2f} | RMSE = {m['rmse']:5.2f}")

def test_chronological_data_leakage():
    print("\n" + "=" * 80)
    print("TEST 2: Chronological Ordering & Data Leakage Audit in EPA AQS Dataset")
    print("=" * 80)
    
    raw_path = "mlops/data/real_collocation_hourly.csv"
    
    # Parse all rows with timestamp, site, sensor, values
    rows = []
    fmt1 = "%Y-%m-%d %H:%M:%S"
    fmt2 = "%Y-%m-%d"
    
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            try:
                ref_pm = float(r["conc"])
                pma = float(r["pm2.5_cf_1_a"])
                pmb = float(r["pm2.5_cf_1_b"])
                rh = float(r["humidity_a"])
                t = float(r["temp_Celsius"])
                d_str = r.get("date2", "").strip()
                site = r.get("AQSID", "").strip()
                sensor = r.get("sensor_index", "").strip()
                
                if ref_pm > 0.5 and rh > 5.0 and pma > 0.5 and pmb > 0.5 and d_str:
                    try:
                        dt = datetime.strptime(d_str, fmt1)
                    except ValueError:
                        dt = datetime.strptime(d_str, fmt2)
                    rows.append({
                        "file_idx": idx,
                        "dt": dt,
                        "site": site,
                        "sensor": sensor,
                        "ref_pm": ref_pm,
                        "pma": pma,
                        "pmb": pmb,
                        "rh": rh,
                        "temp": t,
                    })
            except Exception:
                continue

    n = len(rows)
    print(f"Total valid parsed EPA rows: {n}")
    
    # Slicing as done in data_fusion_multisensor.py
    split_train = int(0.75 * n)
    split_test_start = min(n, split_train + 48)
    
    naive_train = rows[:split_train]
    naive_test = rows[split_test_start:]
    
    train_dates = [r["dt"] for r in naive_train]
    test_dates = [r["dt"] for r in naive_test]
    
    print(f"File index slice: Train [0 : {split_train}], Test [{split_test_start} : {n}]")
    print(f"Train Date Range: {min(train_dates)} -> {max(train_dates)}")
    print(f"Test Date Range:  {min(test_dates)} -> {max(test_dates)}")
    
    # Overlapping dates
    train_date_set = set(r["dt"].strftime("%Y-%m-%d") for r in naive_train)
    test_date_set = set(r["dt"].strftime("%Y-%m-%d") for r in naive_test)
    overlap_days = train_date_set.intersection(test_date_set)
    print(f"Calendar Days Overlap: {len(overlap_days)} days out of {len(test_date_set)} test days ({len(overlap_days)/len(test_date_set)*100:.1f}%)")
    
    # Check sensor and site overlap
    train_sites = set(r["site"] for r in naive_train)
    test_sites = set(r["site"] for r in naive_test)
    print(f"Train Sites: {train_sites}")
    print(f"Test Sites: {test_sites}")
    print(f"Common Sites: {train_sites.intersection(test_sites)}")
    
    train_sensors = set(r["sensor"] for r in naive_train)
    test_sensors = set(r["sensor"] for r in naive_test)
    print(f"Train Sensors: {train_sensors}")
    print(f"Test Sensors: {test_sensors}")
    print(f"Common Sensors: {train_sensors.intersection(test_sensors)}")

    # Check exact sample leakage
    # Are any identical (dt, site, sensor) tuples in both train and test?
    train_keys = set((r["dt"], r["site"], r["sensor"]) for r in naive_train)
    test_keys = set((r["dt"], r["site"], r["sensor"]) for r in naive_test)
    exact_leak = train_keys.intersection(test_keys)
    print(f"Exact (dt, site, sensor) tuple leakage: {len(exact_leak)} samples")

    # Now check what happens under a TRUE chronological split
    print("\n--- Evaluating under Strict Chronological Split (Sorted by dt) ---")
    sorted_rows = sorted(rows, key=lambda x: x["dt"])
    chrono_train_end_idx = int(0.75 * n)
    train_cutoff_dt = sorted_rows[chrono_train_end_idx]["dt"]
    buffer_48h_dt = train_cutoff_dt + timedelta(hours=48)
    
    true_train_rows = [r for r in sorted_rows if r["dt"] <= train_cutoff_dt]
    true_test_rows = [r for r in sorted_rows if r["dt"] > buffer_48h_dt]
    
    print(f"Chronological Train Cutoff Date: {train_cutoff_dt}")
    print(f"48-hour Buffer End Date:        {buffer_48h_dt}")
    print(f"True Out-of-Time Train Count:   {len(true_train_rows)}")
    print(f"True Out-of-Time Test Count:    {len(true_test_rows)}")
    
    # Evaluate current trained model on True Out-of-Time test set!
    model = MultiSensorResMLP(in_features=8, hidden_dim=96)
    model.load_state_dict(torch.load("mlops/models/multisensor_resmlp.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    with open("mlops/models/multisensor_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    # Let's sample e.g. 5,000 points from true_test_rows and evaluate
    np.random.seed(42)
    sample_indices = np.random.choice(len(true_test_rows), min(5000, len(true_test_rows)), replace=False)
    sampled_test = [true_test_rows[i] for i in sample_indices]
    
    # Convert to feature vectors matching data_fusion_multisensor
    X_list, PM_list, Ref_list, RH_list = [], [], [], []
    for r in sampled_test:
        pm_avg = (r["pma"] + r["pmb"]) / 2.0
        diff_pm = abs(r["pma"] - r["pmb"])
        r_pm = np.clip(0.76 - 0.14 * (r["rh"] / 100.0) - 0.05 * (diff_pm / max(pm_avg, 1.0)), 0.45, 0.85)
        dp = compute_dew_point(r["temp"], r["rh"])
        co2 = 420.0
        gas_proxy = 100.0
        
        X_list.append([pm_avg, diff_pm, r_pm, r["temp"], r["rh"], dp, co2, gas_proxy])
        PM_list.append(pm_avg)
        Ref_list.append(r["ref_pm"])
        RH_list.append(r["rh"])
        
    X_arr = np.array(X_list, dtype=np.float32)
    PM_arr = np.array(PM_list, dtype=np.float32)
    Ref_arr = np.array(Ref_list, dtype=np.float32)
    RH_arr = np.array(RH_list, dtype=np.float32)
    
    pred_oot, c_oot = evaluate_model_on_data(model, X_arr, PM_arr, Ref_arr, scaler)
    oot_metrics = compute_metrics(Ref_arr, pred_oot)
    
    print(f"\n--> Model Performance on TRUE Out-of-Time EPA Test Set (5,000 unseen future samples):")
    print(f"    R2 = {oot_metrics['r2']:.4f} | MAE = {oot_metrics['mae']:.2f} | RMSE = {oot_metrics['rmse']:.2f}")

    rh_bins = [(0, 35), (35, 50), (50, 65), (65, 80), (80, 100)]
    for low, high in rh_bins:
        mask = (RH_arr >= low) & (RH_arr < high)
        if np.sum(mask) > 0:
            m = compute_metrics(Ref_arr[mask], pred_oot[mask])
            print(f"    OOT RH [{low:2d}%, {high:2d}%): n = {m['n']:4d} | R2 = {m['r2']:7.4f} | MAE = {m['mae']:5.2f} | RMSE = {m['rmse']:5.2f}")

if __name__ == "__main__":
    test_epa_current_test_set_bins()
    test_chronological_data_leakage()
