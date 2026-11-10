/**
 * esp/tests/test_challenger_m1_r2_firmware.cpp
 * Independent Challenger 1 Firmware Stress Test - Milestone 1 Iteration 2
 * Focus: Model 1 Edge AI Köhler-PINN + Smooth Deliquescence Sigmoid Gate
 *
 * Requirements:
 * 1. Boundary continuity:
 *    - RH = 59.9% vs 60.1% across CO2 = {400, 800, 1500 ppm} (PM=150): diff <= 2.0 ug/m3.
 *    - CO2 = 795 vs 805 ppm across RH = {30%, 35%, 55%, 85%} (PM=150): diff <= 2.0 ug/m3.
 *    - Old 7.16 ug/m3 cliff at RH=35% completely gone.
 * 2. Extreme air conditions:
 *    - Dry dust storm (RH=20%, PM=250): c_eff in [0.95, 1.05].
 *    - Dense fog (RH=95%, CO2=420, Gas=0): c <= 0.65.
 *    - High-CO2 smoke (RH=95%, CO2=2000, Gas=800): c >= 0.85.
 * 3. Closed dry AC rooms:
 *    - RH=35%, CO2=850 ppm: c_eff >= 0.98.
 * 4. Microcontroller constraints:
 *    - 0B Dynamic Heap Allocation.
 *    - Latency < 50 us across 50,000 randomized inferences.
 *    - 0 NaN, 0 Inf.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
    printf("CHALLENGER 1 (M1-R2) INDEPENDENT FIRMWARE ADVERSARIAL STRESS TEST\n");
    printf("================================================================================\n");

    int failures = 0;

    // -------------------------------------------------------------------------
    // 1. Boundary Continuity: RH = 59.9% vs 60.1% across CO2 = {400, 800, 1500}
    // -------------------------------------------------------------------------
    printf("\n[SECTION 1: RH Boundary Transitions (59.9%% -> 60.1%%, PM=150 ug/m3)]\n");
    float test_co2s[] = {400.0f, 800.0f, 1500.0f};
    for (int i = 0; i < 3; i++) {
        float co2 = test_co2s[i];
        bool anom = false;
        float pm_599 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 25.0f, 59.9f, co2, 110.0f, &anom);
        float pm_601 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 25.0f, 60.1f, co2, 110.0f, &anom);
        float delta = fabsf(pm_599 - pm_601);
        printf(" -> CO2=%.0f ppm: PM(59.9%%)=%.2f, PM(60.1%%)=%.2f, Delta=%.4f ug/m3 (Target <= 2.0)\n",
               co2, pm_599, pm_601, delta);
        if (delta > 2.0f) {
            printf("    FAIL: Delta %.4f > 2.0 ug/m3!\n", delta);
            failures++;
        }
    }

    // -------------------------------------------------------------------------
    // 2. Boundary Continuity: CO2 = 795 vs 805 ppm across RH = {30%, 35%, 55%, 85%}
    // -------------------------------------------------------------------------
    printf("\n[SECTION 2: CO2 Boundary Transitions (795 -> 805 ppm, PM=150 ug/m3)]\n");
    float test_rhs[] = {30.0f, 35.0f, 55.0f, 85.0f};
    for (int i = 0; i < 4; i++) {
        float rh = test_rhs[i];
        bool anom = false;
        float pm_795 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 25.0f, rh, 795.0f, 110.0f, &anom);
        float pm_805 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 25.0f, rh, 805.0f, 110.0f, &anom);
        float delta = fabsf(pm_795 - pm_805);
        printf(" -> RH=%.0f%%: PM(795)=%.2f, PM(805)=%.2f, Delta=%.4f ug/m3 (Target <= 2.0)\n",
               rh, pm_795, pm_805, delta);
        if (delta > 2.0f) {
            printf("    FAIL: Delta %.4f > 2.0 ug/m3!\n", delta);
            failures++;
        }
        if (rh == 35.0f && delta > 0.05f) {
            printf("    FAIL: Old 7.16 cliff at RH=35%% not eliminated! Delta=%.4f\n", delta);
            failures++;
        }
    }

    // -------------------------------------------------------------------------
    // 3. Extreme Air Conditions
    // -------------------------------------------------------------------------
    printf("\n[SECTION 3: Extreme Air Conditions]\n");
    {
        // 3a. Dry Dust Storm: RH=20%, PM=250
        bool anom = false;
        float pm_dust = tinyml_predict_pm25_multisensor(250.0f, 250.0f, 357.14f, 30.0f, 20.0f, 420.0f, 100.0f, &anom);
        float c_eff_dust = pm_dust / 250.0f;
        printf(" -> Dry Dust Storm (RH=20%%, PM=250): Out=%.2f ug/m3, c_eff=%.4f (Target: [0.95, 1.05])\n",
               pm_dust, c_eff_dust);
        if (c_eff_dust < 0.95f || c_eff_dust > 1.05f) {
            printf("    FAIL: Dry dust storm c_eff out of bounds!\n");
            failures++;
        }

        // 3b. Dense Fog: RH=95%, CO2=420 ppm, Gas=0
        float c_fog = tinyml_infer_c_factor_multisensor(120.0f, 1.5f, 0.72f, 19.0f, 95.0f, 18.2f, 420.0f, 0.0f);
        float pm_fog = tinyml_predict_pm25_multisensor(120.0f, 120.0f, 166.67f, 19.0f, 95.0f, 420.0f, 0.0f, &anom);
        float c_eff_fog = pm_fog / 120.0f;
        printf(" -> Dense Fog (RH=95%%, CO2=420, Gas=0): raw_c=%.4f, c_eff=%.4f (Target: <= 0.65)\n",
               c_fog, c_eff_fog);
        if (c_fog > 0.65f || c_eff_fog > 0.65f) {
            printf("    FAIL: Fog suppression failed (c > 0.65)!\n");
            failures++;
        }

        // 3c. High-CO2 Smoke: RH=95%, CO2=2000, Gas=800
        float c_smoke = tinyml_infer_c_factor_multisensor(120.0f, 3.0f, 0.75f, 22.0f, 95.0f, 21.1f, 2000.0f, 800.0f);
        float pm_smoke = tinyml_predict_pm25_multisensor(120.0f, 120.0f, 160.0f, 22.0f, 95.0f, 2000.0f, 800.0f, &anom);
        float c_eff_smoke = pm_smoke / 120.0f;
        printf(" -> High-CO2 Smoke (RH=95%%, CO2=2000, Gas=800): raw_c=%.4f, c_eff=%.4f (Target: >= 0.85)\n",
               c_smoke, c_eff_smoke);
        if (c_smoke < 0.85f || c_eff_smoke < 0.85f) {
            printf("    FAIL: Smoke preservation failed (c < 0.85)!\n");
            failures++;
        }
    }

    // -------------------------------------------------------------------------
    // 4. Closed Dry AC Rooms
    // -------------------------------------------------------------------------
    printf("\n[SECTION 4: Closed Dry AC Room (RH=35%%, CO2=850 ppm, PM=50 ug/m3)]\n");
    {
        bool anom = false;
        float pm_ac = tinyml_predict_pm25_multisensor(50.0f, 50.0f, 71.43f, 24.0f, 35.0f, 850.0f, 120.0f, &anom);
        float c_eff_ac = pm_ac / 50.0f;
        printf(" -> Dry AC Room (RH=35%%, CO2=850 ppm): Out=%.2f ug/m3, c_eff=%.4f (Target: >= 0.98)\n",
               pm_ac, c_eff_ac);
        if (c_eff_ac < 0.98f) {
            printf("    FAIL: False PM suppression in dry AC room! c_eff=%.4f < 0.98\n", c_eff_ac);
            failures++;
        }
    }

    // -------------------------------------------------------------------------
    // 5. 50,000 Inferences Stress Test: 0B Heap, < 50 us Latency, 0 NaN/Inf
    // -------------------------------------------------------------------------
    printf("\n[SECTION 5: 50,000 Continuous Inferences Stress Test]\n");
    const int NUM_RUNS = 50000;
    g_heap_allocations = 0;
    g_heap_bytes = 0;

    size_t nan_count = 0;
    size_t inf_count = 0;
    double sum = 0.0;
    clock_t start = clock();

    for (int i = 0; i < NUM_RUNS; i++) {
        uint32_t seed = (uint32_t)(i * 1664525u + 1013904223u);
        float r1 = (float)(seed & 0xFFFF) / 65535.0f;
        float r2 = (float)((seed >> 16) & 0xFFFF) / 65535.0f;

        float pm = 5.0f + r1 * 400.0f;
        float diff = r2 * 5.0f;
        float r_pm = 0.5f + r1 * 0.4f;
        float pm10 = pm / r_pm;
        float temp = 10.0f + r2 * 35.0f;
        float rh = 10.0f + r1 * 85.0f;
        float co2 = 350.0f + r2 * 2000.0f;
        float gas = 20.0f + r1 * 500.0f;

        bool anom = false;
        float out = tinyml_predict_pm25_multisensor(pm, pm + diff, pm10, temp, rh, co2, gas, &anom);
        if (isnan(out)) nan_count++;
        if (isinf(out)) inf_count++;
        sum += out;
    }
    clock_t end = clock();

    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double us_per_run = (total_time / NUM_RUNS) * 1e6;

    printf(" -> Completed 50,000 inferences in %.4f s\n", total_time);
    printf(" -> Average latency: %.2f us (Target < 50 us)\n", us_per_run);
    printf(" -> Total heap allocated: %zu bytes (Target: 0B)\n", g_heap_bytes);
    printf(" -> Total NaNs: %zu, Infs: %zu (Target: 0)\n", nan_count, inf_count);
    printf(" -> Checksum: %.2f\n", sum);

    if (g_heap_bytes > 0 || g_heap_allocations > 0) {
        printf("    FAIL: Dynamic heap allocation detected!\n");
        failures++;
    }
    if (us_per_run >= 50.0) {
        printf("    FAIL: Latency >= 50 us!\n");
        failures++;
    }
    if (nan_count > 0 || inf_count > 0) {
        printf("    FAIL: NaN/Inf detected!\n");
        failures++;
    }

    printf("\n================================================================================\n");
    if (failures == 0) {
        printf("CHALLENGER 1 FIRMWARE VERDICT: APPROVE (All checks PASSED!)\n");
        printf("================================================================================\n");
        return 0;
    } else {
        printf("CHALLENGER 1 FIRMWARE VERDICT: REJECT (%d checks FAILED!)\n", failures);
        printf("================================================================================\n");
        return 1;
    }
}
