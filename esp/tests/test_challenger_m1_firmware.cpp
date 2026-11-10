#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <new>
#include "tinyml_edge_engine.h"

// Heap interception
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
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }

int main() {
    printf("================================================================================\n");
    printf("CHALLENGER 1: EMPIRICAL ADVERSARIAL STRESS TEST FOR M1 C++ FIRMWARE\n");
    printf("================================================================================\n\n");

    int total_failures = 0;

    // TEST SUITE 1: Fine sweep of RH around 60% (55.0% to 65.0%, step 0.05%)
    printf("[CHALLENGE 1: Fine Sweep of RH around 60%% (55.0%% to 65.0%%, step 0.05%%)]\n");
    struct SweepCase {
        float pm;
        float co2;
        float temp;
        float gas;
        const char* name;
    } cases[] = {
        {150.0f, 600.0f, 26.0f, 110.0f, "PM=150, CO2=600 (Spec Benchmark)"},
        {50.0f,  400.0f, 20.0f, 80.0f,  "PM=50, CO2=400 (Clean Ambient)"},
        {25.0f,  500.0f, 15.0f, 50.0f,  "PM=25, CO2=500 (Low PM / Cool)"},
        {300.0f, 700.0f, 30.0f, 150.0f, "PM=300, CO2=700 (High Pollution)"},
        {500.0f, 650.0f, 25.0f, 120.0f, "PM=500, CO2=650 (Extreme Ambient)"}
    };

    for (int c = 0; c < 5; c++) {
        float pm = cases[c].pm;
        float co2 = cases[c].co2;
        float temp = cases[c].temp;
        float gas = cases[c].gas;
        bool is_anom = false;

        float max_step_delta = 0.0f;
        float worst_rh_start = 0.0f;
        float prev_pred = -1.0f;
        float first_pred = 0.0f;
        float last_pred = 0.0f;
        int step_count = 0;

        for (float rh = 55.0f; rh <= 65.001f; rh += 0.05f) {
            float pred = tinyml_predict_pm25_multisensor(pm, pm, pm * 1.3f, temp, rh, co2, gas, &is_anom);
            if (step_count == 0) first_pred = pred;
            last_pred = pred;

            if (prev_pred >= 0.0f) {
                float delta = fabsf(pred - prev_pred);
                if (delta > max_step_delta) {
                    max_step_delta = delta;
                    worst_rh_start = rh - 0.05f;
                }
            }
            prev_pred = pred;
            step_count++;
        }

        printf(" -> Case %d [%s]:\n", c + 1, cases[c].name);
        printf("    Steps: %d, Start: %.2f, End: %.2f ug/m3\n", step_count, first_pred, last_pred);
        printf("    Max step delta: %.4f ug/m3 at RH=%.2f%% -> %.2f%%\n",
               max_step_delta, worst_rh_start, worst_rh_start + 0.05f);

        if (max_step_delta >= 2.0f) {
            printf("    FAIL: Cliff jump detected (%.4f >= 2.0 ug/m3)!\n", max_step_delta);
            total_failures++;
        } else {
            printf("    PASS (Max delta < 2.0 ug/m3)\n");
        }
    }

    // TEST SUITE 2: Extended sweep across RH 50% to 75% (step 0.05%)
    printf("\n[CHALLENGE 2: Extended Sweep RH 50.0%% to 75.0%% (step 0.05%%)]\n");
    {
        float pm = 150.0f;
        float co2 = 600.0f;
        float temp = 26.0f;
        float gas = 110.0f;
        bool is_anom = false;
        float max_step_delta = 0.0f;
        float worst_rh = 0.0f;
        float prev_pred = -1.0f;

        for (float rh = 50.0f; rh <= 75.001f; rh += 0.05f) {
            float pred = tinyml_predict_pm25_multisensor(pm, pm, pm * 1.3f, temp, rh, co2, gas, &is_anom);
            if (prev_pred >= 0.0f) {
                float delta = fabsf(pred - prev_pred);
                if (delta > max_step_delta) {
                    max_step_delta = delta;
                    worst_rh = rh - 0.05f;
                }
            }
            prev_pred = pred;
        }

        printf(" -> Extended PM=150, CO2=600: Max step delta = %.4f ug/m3 at RH=%.2f%% -> %.2f%%\n",
               max_step_delta, worst_rh, worst_rh + 0.05f);
        if (max_step_delta >= 2.0f) {
            printf("    FAIL: Extended cliff jump (%.4f >= 2.0 ug/m3)!\n", max_step_delta);
            total_failures++;
        } else {
            printf("    PASS (Extended max delta < 2.0 ug/m3)\n");
        }
    }

    // TEST SUITE 3: Smoke discrimination & Gate activation (CO2 >= 800)
    printf("\n[CHALLENGE 3: Smoke Discrimination & Gate Activation (CO2 >= 800)]\n");
    {
        // Fog condition: RH=95%, CO2=420, Gas=0
        float c_fog = tinyml_infer_c_factor_multisensor(120.0f, 1.5f, 0.72f, 19.0f, 95.0f, 18.2f, 420.0f, 0.0f);
        // Smoke condition: RH=95%, CO2=2000, Gas=800
        float c_smoke = tinyml_infer_c_factor_multisensor(120.0f, 3.0f, 0.75f, 20.0f, 95.0f, 19.1f, 2000.0f, 800.0f);

        printf(" -> Fog C-factor:   %.4f (Target <= 0.65)\n", c_fog);
        printf(" -> Smoke C-factor: %.4f (Target >= 0.85)\n", c_smoke);

        if (c_fog > 0.65f) {
            printf("    FAIL: Fog C-factor too high!\n");
            total_failures++;
        } else {
            printf("    PASS: Fog discrimination verified\n");
        }

        if (c_smoke < 0.85f) {
            printf("    FAIL: Smoke C-factor too low!\n");
            total_failures++;
        } else {
            printf("    PASS: Smoke discrimination verified\n");
        }

        // Verify gate opens fully at CO2 >= 800
        bool is_anom = false;
        float pm_in = 100.0f;
        float pred_smoke_gate = tinyml_predict_pm25_multisensor(pm_in, pm_in, pm_in * 1.3f, 20.0f, 40.0f, 800.0f, 200.0f, &is_anom);
        float c_pred_raw = tinyml_infer_c_factor_multisensor(pm_in, 0.0f, 100.0f / (pm_in * 1.3f), 20.0f, 40.0f, tinyml_calc_dew_point(20.0f, 40.0f), 800.0f, 200.0f);
        float expected_pm = pm_in * c_pred_raw;

        printf(" -> Gate check at CO2=800, RH=40%%: pred=%.4f, expected=%.4f (delta=%.6f)\n",
               pred_smoke_gate, expected_pm, fabsf(pred_smoke_gate - expected_pm));

        if (fabsf(pred_smoke_gate - expected_pm) > 1e-3f) {
            printf("    FAIL: Gate did not open fully (w_del != 1.0)!\n");
            total_failures++;
        } else {
            printf("    PASS: Gate opens fully (w_del = 1.0) when CO2 >= 800\n");
        }
    }

    // TEST SUITE 4: Extreme values stress test
    printf("\n[CHALLENGE 4: Extreme Temperature, Gas, PM, and Humidity Invariants]\n");
    {
        float test_temps[] = {-30.0f, -15.0f, 0.0f, 25.0f, 50.0f, 65.0f};
        float test_gases[] = {0.0f, 10.0f, 100.0f, 500.0f, 1200.0f, 3000.0f};
        float test_pms[]   = {0.0f, 5.0f, 50.0f, 150.0f, 600.0f, 2000.0f, 5000.0f};
        float test_rhs[]   = {0.0f, 1.0f, 20.0f, 60.0f, 95.0f, 100.0f, 120.0f};

        int nan_count = 0;
        int inf_count = 0;
        int oob_c_count = 0;
        int count = 0;

        for (int t = 0; t < 6; t++) {
            for (int g = 0; g < 6; g++) {
                for (int p = 0; p < 7; p++) {
                    for (int r = 0; r < 7; r++) {
                        bool is_anom = false;
                        float pm = test_pms[p];
                        float rh = test_rhs[r];
                        float temp = test_temps[t];
                        float gas = test_gases[g];

                        float dp = tinyml_calc_dew_point(temp, rh);
                        float c = tinyml_infer_c_factor_multisensor(pm, 1.0f, 0.70f, temp, rh, dp, 450.0f, gas);
                        float pred = tinyml_predict_pm25_multisensor(pm, pm, pm * 1.3f, temp, rh, 450.0f, gas, &is_anom);

                        if (isnan(c) || isnan(pred)) nan_count++;
                        if (isinf(c) || isinf(pred)) inf_count++;
                        if (c < 0.25f || c > 1.05f) oob_c_count++;
                        count++;
                    }
                }
            }
        }

        printf(" -> Tested %d combinatorial extreme points:\n", count);
        printf("    NaNs: %d, Infs: %d, C out of [0.25, 1.05]: %d\n", nan_count, inf_count, oob_c_count);

        if (nan_count > 0 || inf_count > 0 || oob_c_count > 0) {
            printf("    FAIL: Numerical invariant violated under extreme inputs!\n");
            total_failures++;
        } else {
            printf("    PASS: Strict numerical stability under all extreme inputs\n");
        }
    }

    // TEST SUITE 5: Zero Heap & Latency Benchmark
    printf("\n[CHALLENGE 5: Zero-Heap Allocation Check]\n");
    {
        g_heap_allocations = 0;
        g_heap_bytes = 0;

        bool is_anom = false;
        for (int i = 0; i < 10000; i++) {
            tinyml_predict_pm25_multisensor(120.0f, 122.0f, 150.0f, 25.0f, 62.0f, 450.0f, 100.0f, &is_anom);
        }

        printf(" -> 10,000 inferences executed. Dynamic heap bytes: %zu, allocations: %zu\n",
               g_heap_bytes, g_heap_allocations);

        if (g_heap_bytes > 0 || g_heap_allocations > 0) {
            printf("    FAIL: Dynamic heap allocation detected (%zu bytes)!\n", g_heap_bytes);
            total_failures++;
        } else {
            printf("    PASS: 0B Dynamic Heap Allocation strictly verified\n");
        }
    }

    printf("\n================================================================================\n");
    if (total_failures == 0) {
        printf("OVERALL FIRMWARE VERDICT: APPROVE (All %d challenge suites PASSED!)\n", 5);
        printf("================================================================================\n");
        return 0;
    } else {
        printf("OVERALL FIRMWARE VERDICT: REJECT (%d challenge suites FAILED!)\n", total_failures);
        printf("================================================================================\n");
        return 1;
    }
}
