#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <new>
#include "tinyml_edge_engine.h"

// Heap Allocation Interception to verify 0B Dynamic Heap Allocation
static size_t g_heap_allocations = 0;
static size_t g_heap_bytes = 0;

void* operator new(size_t size) {
    g_heap_allocations++;
    g_heap_bytes += size;
    return malloc(size);
}
void* operator new[](size_t size) {
    g_heap_allocations++;
    g_heap_bytes += size;
    return malloc(size);
}
void operator delete(void* p) noexcept {
    free(p);
}
void operator delete[](void* p) noexcept {
    free(p);
}
void operator delete(void* p, size_t) noexcept {
    free(p);
}
void operator delete[](void* p, size_t) noexcept {
    free(p);
}

int main() {
    printf("================================================================================\n");
    printf("ADVERSARIAL STRESS TEST & FIRMWARE MEMORY BOUNDS (50,000 INFERENCES)\n");
    printf("================================================================================\n");

    // 1. Extreme Out-of-Range Sensor Testing
    printf("\n[PHASE 1: Extreme Out-of-Range Sensor Values]\n");
    {
        bool is_anom = false;
        TinyML_AnomalyCode code;

        // Extreme PM=5000, RH=120%, Temp=-30 C
        float pred1 = tinyml_predict_pm25_multisensor(5000.0f, 4980.0f, 6000.0f, -30.0f, 120.0f, 450.0f, 100.0f, &is_anom);
        printf(" -> [Extreme 1] PM=5000, RH=120%%, Temp=-30C: Out=%.2f ug/m3 (isnan=%d, isinf=%d)\n",
               pred1, isnan(pred1), isinf(pred1));

        // Extreme Fog (RH=95%, CO2=420, Gas=0): Check C factor
        float c_fog = tinyml_infer_c_factor_multisensor(120.0f, 1.5f, 0.72f, 19.0f, 95.0f, 18.2f, 420.0f, 0.0f);
        printf(" -> [Extreme 2 - Fog] RH=95%%, CO2=420, Gas=0: C_factor=%.4f (Target: <= 0.65)\n", c_fog);

        // Extreme Smoke/Fire (RH=95%, CO2=2000, Gas=800): Check C factor
        float c_smoke = tinyml_infer_c_factor_multisensor(120.0f, 3.0f, 0.75f, 20.0f, 95.0f, 19.1f, 2000.0f, 800.0f);
        printf(" -> [Extreme 3 - Smoke] RH=95%%, CO2=2000, Gas=800: C_factor=%.4f (Target: >= 0.85)\n", c_smoke);

        // Advanced API extreme warm-up and drift check
        float pred_adv = tinyml_predict_pm25_advanced(5000.0f, 4980.0f, -30.0f, 120.0f, 45.0f, &code);
        printf(" -> [Extreme 4 - Advanced] PM=5000, RH=120%%, Temp=-30C, t=45s: Out=%.2f ug/m3, Code=%d\n",
               pred_adv, code);

        if (isnan(pred1) || isinf(pred1)) {
            printf("FAIL: Out-of-range inputs produced NaN/Inf!\n");
            return 1;
        }
        if (c_fog > 0.65f) {
            printf("FAIL: C_fog = %.4f > 0.65f!\n", c_fog);
            return 1;
        }
        if (c_smoke < 0.85f) {
            printf("FAIL: C_smoke = %.4f < 0.85f!\n", c_smoke);
            return 1;
        }
    }

    // 2. Continuous 50,000 Inferences Stress Test
    printf("\n[PHASE 2: 50,000 Continuous Adversarial Inferences]\n");
    const int NUM_INFERENCES = 50000;
    
    // Reset heap counter to verify zero dynamic heap allocations during inference loop
    g_heap_allocations = 0;
    g_heap_bytes = 0;

    clock_t start = clock();
    size_t nan_count = 0;
    size_t inf_count = 0;
    double dummy_sum = 0.0;

    for (int i = 0; i < NUM_INFERENCES; i++) {
        // Pseudo-random deterministic noise generation without stdlib rand overhead
        uint32_t seed = (uint32_t)(i * 1664525u + 1013904223u);
        float r1 = (float)(seed & 0xFFFF) / 65535.0f;
        float r2 = (float)((seed >> 16) & 0xFFFF) / 65535.0f;

        // Generate sweeping adversarial variations across cycles:
        // Cycle types:
        // 0: Clean baseline
        // 1: Heavy humidity (80% - 105%)
        // 2: Extreme temperature (-30 C to +65 C)
        // 3: Wild PM concentrations (0 to 5000 ug/m3)
        // 4: Smoke & fire plumes (CO2 up to 4000 ppm, Gas up to 1200)
        float pm1, pm2, pm10, temp, rh, co2, gas;
        int mode = i % 5;
        if (mode == 0) {
            pm1 = 15.0f + r1 * 40.0f;
            pm2 = pm1 + (r2 - 0.5f) * 4.0f;
            pm10 = pm1 * 1.3f;
            temp = 20.0f + r1 * 10.0f;
            rh = 50.0f + r2 * 25.0f;
            co2 = 400.0f + r1 * 100.0f;
            gas = 50.0f + r2 * 50.0f;
        } else if (mode == 1) {
            pm1 = 80.0f + r1 * 150.0f;
            pm2 = pm1 + (r2 - 0.5f) * 10.0f;
            pm10 = pm1 * 1.25f;
            temp = 15.0f + r1 * 15.0f;
            rh = 80.0f + r2 * 25.0f; // up to 105%
            co2 = 420.0f + r1 * 50.0f;
            gas = 30.0f + r2 * 40.0f;
        } else if (mode == 2) {
            pm1 = 50.0f + r1 * 100.0f;
            pm2 = pm1 + (r2 - 0.5f) * 6.0f;
            pm10 = pm1 * 1.4f;
            temp = -30.0f + r1 * 95.0f; // -30 C to +65 C
            rh = 10.0f + r2 * 90.0f;
            co2 = 450.0f + r1 * 200.0f;
            gas = 80.0f + r2 * 100.0f;
        } else if (mode == 3) {
            pm1 = r1 * 5000.0f; // 0 to 5000 ug/m3
            pm2 = r2 * 5000.0f;
            pm10 = pm1 * 1.2f + 5.0f;
            temp = -10.0f + r1 * 50.0f;
            rh = r2 * 120.0f; // 0 to 120%
            co2 = 300.0f + r1 * 3000.0f;
            gas = r2 * 1000.0f;
        } else {
            // Smoke / Fire mode
            pm1 = 200.0f + r1 * 1500.0f;
            pm2 = pm1 + (r2 - 0.5f) * 20.0f;
            pm10 = pm1 * 1.15f;
            temp = 25.0f + r1 * 20.0f;
            rh = 80.0f + r2 * 18.0f;
            co2 = 1200.0f + r1 * 2800.0f; // 1200 to 4000 ppm
            gas = 300.0f + r2 * 900.0f;   // 300 to 1200
        }

        bool anom = false;
        float out = tinyml_predict_pm25_multisensor(pm1, pm2, pm10, temp, rh, co2, gas, &anom);
        
        if (isnan(out)) nan_count++;
        if (isinf(out)) inf_count++;
        dummy_sum += out;
    }
    clock_t end = clock();

    double total_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double us_per_inference = (total_sec / NUM_INFERENCES) * 1e6;
    double inferences_per_sec = NUM_INFERENCES / total_sec;

    printf(" -> Hoan thanh 50,000 chu ky suy luan lien tuc.\n");
    printf(" -> Tong thoi gian: %.4f giay\n", total_sec);
    printf(" -> Do tre trung binh moi lan suy luan: %.2f micro-giay (us)\n", us_per_inference);
    printf(" -> Toc do xu ly: %.0f inferences/sec\n", inferences_per_sec);
    printf(" -> Tong so loi NaN phat sinh: %zu\n", nan_count);
    printf(" -> Tong so loi Inf phat sinh: %zu\n", inf_count);
    printf(" -> Tong bo nho Heap duoc cap phat dong trong vong lap: %zu bytes (Allocations: %zu)\n",
           g_heap_bytes, g_heap_allocations);
    printf(" -> Dummy checksum sum = %.2f\n", dummy_sum);

    printf("\n[PHASE 3: Floating Point NaN & Inf Boundary Injection]\n");
    {
        // Testing explicit NaN and Inf safety
        float nan_val = 0.0f / 0.0f;
        float inf_val = 1.0f / 0.0f;
        bool anom = false;

        // Note: checking that edge engine does not segfault or crash when encountering NaN inputs
        printf(" -> Testing with explicit NaN in temp: ");
        float out_nan_temp = tinyml_predict_pm25_multisensor(50.0f, 52.0f, 65.0f, nan_val, 85.0f, 420.0f, 100.0f, &anom);
        printf("Completed without crash (Out=%.2f, isnan=%d)\n", out_nan_temp, isnan(out_nan_temp));

        printf(" -> Testing with explicit +Inf in PM: ");
        float out_inf_pm = tinyml_predict_pm25_multisensor(inf_val, inf_val, inf_val, 25.0f, 70.0f, 420.0f, 100.0f, &anom);
        printf("Completed without crash (Out=%.2f, isinf=%d)\n", out_inf_pm, isinf(out_inf_pm));
    }

    printf("\n[PHASE 4: Deliquescence Boundary Continuity Check (RH=59.9% vs 60.1%)]\n");
    bool pass_boundary = false;
    {
        bool anom = false;
        // Kich ban kiem tra theo tieu chuan nghiem thu: PM=150 ug/m3, CO2=600 ppm, T=26C, Gas=110
        float pm_599 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 26.0f, 59.9f, 600.0f, 110.0f, &anom);
        float pm_601 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 26.0f, 60.1f, 600.0f, 110.0f, &anom);
        float diff_boundary = fabsf(pm_599 - pm_601);

        printf(" -> PM(RH=59.9%%) = %.2f ug/m3 | PM(RH=60.1%%) = %.2f ug/m3\n", pm_599, pm_601);
        printf(" -> Chenh lech buoc nhay bien: %.2f ug/m3 (Yeu cau nghiem thu: < 2.0 ug/m3)\n", diff_boundary);

        if (diff_boundary < 2.0f && pm_601 >= 135.0f) {
            pass_boundary = true;
            printf(" -> Boundary continuity check: PASS\n");
        } else {
            printf("FAIL: Phat hien dut gay bien hoac du doan qua thap: diff=%.2f (can < 2.0), pm_601=%.2f (can >= 135.0)\n",
                   diff_boundary, pm_601);
        }
    }

    printf("\n================================================================================\n");
    printf("VERDICT EVALUATION:\n");
    bool pass_heap = (g_heap_bytes == 0 && g_heap_allocations == 0);
    bool pass_latency = (us_per_inference < 50.0);
    bool pass_nan = (nan_count == 0 && inf_count == 0);

    printf("1. No Segfaults: PASS (All 50,000 runs survived)\n");
    printf("2. No NaN/Inf in valid domain & out-of-range sweeps: %s (%zu NaNs, %zu Infs)\n",
           pass_nan ? "PASS" : "FAIL", nan_count, inf_count);
    printf("3. 0B Dynamic Heap Allocation: %s (%zu bytes)\n",
           pass_heap ? "PASS" : "FAIL", g_heap_bytes);
    printf("4. Latency < 50 us: %s (Actual: %.2f us)\n",
           pass_latency ? "PASS" : "FAIL", us_per_inference);
    printf("5. Deliquescence Boundary Continuity: %s\n",
           pass_boundary ? "PASS" : "FAIL");
    printf("================================================================================\n");

    if (pass_heap && pass_latency && pass_nan && pass_boundary) {
        printf("RESULT: ALL FIRMWARE CHECKS PASSED!\n");
        return 0;
    } else {
        printf("RESULT: ADVERSARIAL FIRMWARE CHECK FAILED!\n");
        return 1;
    }
}
