"""
Measurement script for Challenger M3_2 empirical report.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from pi.central_engine import CentralAIEngine

def create_clean_data(seed=123):
    rng = np.random.default_rng(seed)
    hist = np.full((24, 8), 42.0, dtype=np.float32) + rng.normal(0, 3.0, size=(24, 8)).astype(np.float32)
    curr = np.zeros((8, 6), dtype=np.float32)
    for i in range(8):
        curr[i] = [
            42.0 + rng.normal(0, 3.5),
            58.0 + rng.normal(0, 4.5),
            26.0 + rng.normal(0, 0.8),
            74.0 + rng.normal(0, 3.0),
            2.2,
            115.0,
        ]
    return hist, curr

hist_clean, curr_clean = create_clean_data(123)
eng_clean = CentralAIEngine()
res_clean = eng_clean.process_telemetry_frame(hist_clean, curr_clean.ravel())
score_base = res_clean["anomaly"]["anomaly_score"]
thresh_base = res_clean["anomaly"]["dynamic_threshold"]
is_anom_base = res_clean["anomaly"]["is_anomaly"]

print(f"BASELINE CLEAN: score={score_base:.6f}, thresh={thresh_base:.6f}, is_anomaly={is_anom_base}")
print("=" * 95)
print(f"| Station | Fault PM2.5 (ug/m3) | Clean Score | Fault Score | Dynamic Thresh | Ratio   | Detected? |")
print("|" + "-"*9 + "|" + "-"*21 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*16 + "|" + "-"*9 + "|" + "-"*11 + "|")

all_ratios = []
for st in range(8):
    for pm_val in [180.0, 230.0, 280.0]:
        eng = CentralAIEngine()
        curr_f = curr_clean.copy()
        curr_f[st, 0] = pm_val
        curr_f[st, 1] = pm_val * 1.3
        res_f = eng.process_telemetry_frame(hist_clean, curr_f.ravel())
        sc = res_f["anomaly"]["anomaly_score"]
        th = res_f["anomaly"]["dynamic_threshold"]
        anom = res_f["anomaly"]["is_anomaly"]
        ratio = sc / score_base
        all_ratios.append(ratio)
        print(f"| St {st+1:<4} | {pm_val:>12.1f}        | {score_base:>9.6f}   | {sc:>9.6f}   | {th:>12.6f}   | {ratio:>5.2f}x  | {str(anom):<9} |")

print("=" * 95)
print(f"Score Ratio Distribution across all 24 evaluations:")
print(f"Min: {min(all_ratios):.2f}x")
print(f"Mean: {np.mean(all_ratios):.2f}x")
print(f"Median: {np.median(all_ratios):.2f}x")
print(f"Max: {max(all_ratios):.2f}x")

print("\n" + "=" * 95)
print("COLD-START WINDOW LENGTH EVALUATION:")
print(f"| Window T | Clean Score | Clean Anom? | Fault Score | Fault Anom? | Ratio   | Status  |")
print("|" + "-"*10 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*9 + "|" + "-"*9 + "|")

for T in [1, 2, 4, 8, 16, 24]:
    hist_cold = hist_clean[:T, :]
    eng_c = CentralAIEngine()
    res_c = eng_c.process_telemetry_frame(hist_cold, curr_clean.ravel())
    sc_c = res_c["anomaly"]["anomaly_score"]
    an_c = res_c["anomaly"]["is_anomaly"]

    eng_f = CentralAIEngine()
    curr_f = curr_clean.copy()
    curr_f[2, 0] = 240.0
    curr_f[2, 1] = 290.0
    res_f = eng_f.process_telemetry_frame(hist_cold, curr_f.ravel())
    sc_f = res_f["anomaly"]["anomaly_score"]
    an_f = res_f["anomaly"]["is_anomaly"]
    r_cold = sc_f / sc_c

    status = "PASS" if (not an_c and an_f and r_cold > 2.5) else "FAIL"
    print(f"| T={T:<6} | {sc_c:>9.6f}   | {str(an_c):<11} | {sc_f:>9.6f}   | {str(an_f):<11} | {r_cold:>5.2f}x  | {status:<7} |")

print("=" * 95)
print("\nMULTI-STATION FIRE SCENARIO EVALUATION:")
curr_fire = curr_clean.copy()
curr_fire[0, 0] = 210.0
curr_fire[0, 1] = 260.0
curr_fire[1, 0] = 185.0
curr_fire[1, 1] = 230.0
curr_fire[3, 0] = 160.0
curr_fire[3, 1] = 200.0

eng_fire = CentralAIEngine()
res_fire = eng_fire.process_telemetry_frame(hist_clean, curr_fire.ravel())
an_fire = res_fire["anomaly"]

print(f"Elevated Stations: St 1 (210 ug/m3), St 2 (185 ug/m3), St 4 (160 ug/m3)")
print(f"is_anomaly: {an_fire['is_anomaly']}")
print(f"Anomaly Score: {an_fire['anomaly_score']:.6f} (vs Baseline {score_base:.6f}) -> Ratio: {an_fire['anomaly_score']/score_base:.2f}x")
print(f"Dynamic Threshold: {an_fire['dynamic_threshold']:.6f}")
print(f"Signature Class ID: {an_fire['signature_class_id']}")
print(f"Signature Class Name: {an_fire['signature_class_name']}")
print(f"Class Probabilities: {an_fire['class_probabilities']}")
