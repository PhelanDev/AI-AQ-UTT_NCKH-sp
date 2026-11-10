#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <new>
#include <chrono>
#include <vector>
#include <algorithm>
#include "tinyml_edge_engine.h"

// ============================================================================
// 1. HEAP ALLOCATION INTERCEPTION (STRICT ZERO-HEAP ENFORCEMENT)
// ============================================================================
static size_t g_heap_allocations = 0;
static size_t g_heap_bytes = 0;
static bool g_heap_intercept_enabled = false;

void* operator new(size_t size) {
    if (g_heap_intercept_enabled) {
        g_heap_allocations++;
        g_heap_bytes += size;
    }
    return malloc(size);
}
void* operator new[](size_t size) {
    if (g_heap_intercept_enabled) {
        g_heap_allocations++;
        g_heap_bytes += size;
    }
    return malloc(size);
}
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }

// ============================================================================
// 2. DYNAMIC STACK WATERMARKING
// ============================================================================
__attribute__((noinline)) size_t measure_stack_watermark_predict() {
    volatile uint8_t stack_canvas[4096];
    for (size_t i = 0; i < sizeof(stack_canvas); i++) {
        stack_canvas[i] = 0xA5;
    }

    bool is_anom = false;
    volatile float res = tinyml_predict_pm25_multisensor(
        150.0f, 150.0f, 200.0f, 26.0f, 60.0f, 600.0f, 110.0f, &is_anom
    );
    (void)res;

    size_t deepest = 0;
    for (size_t i = 0; i < sizeof(stack_canvas); i++) {
        if (stack_canvas[i] != 0xA5) {
            deepest = sizeof(stack_canvas) - i;
            break;
        }
    }
    return deepest;
}

__attribute__((noinline)) size_t measure_stack_watermark_c_factor() {
    volatile uint8_t stack_canvas[4096];
    for (size_t i = 0; i < sizeof(stack_canvas); i++) {
        stack_canvas[i] = 0xA5;
    }

    volatile float res = tinyml_infer_c_factor_multisensor(
        150.0f, 2.0f, 0.70f, 26.0f, 60.0f, 18.0f, 600.0f, 110.0f
    );
    (void)res;

    size_t deepest = 0;
    for (size_t i = 0; i < sizeof(stack_canvas); i++) {
        if (stack_canvas[i] != 0xA5) {
            deepest = sizeof(stack_canvas) - i;
            break;
        }
    }
    return deepest;
}

// ============================================================================
// MAIN ADVERSARIAL CHALLENGER SUITE
// ============================================================================
int main() {
    printf("================================================================================\n");
    printf("CHALLENGER M4-2: PHYSICAL BOUNDARIES & HARDWARE BUDGET STRESS HARNESS\n");
    printf("================================================================================\n\n");

    int failures = 0;

    // ------------------------------------------------------------------------
    // CHALLENGE 1: DYNAMIC STACK USAGE < 1024 BYTES (1 KB)
    // ------------------------------------------------------------------------
    printf("[CHALLENGE 1: Dynamic Stack Usage Verification]\n");
    size_t stack_c = measure_stack_watermark_c_factor();
    size_t stack_predict = measure_stack_watermark_predict();
    printf(" -> Stack watermark for tinyml_infer_c_factor_multisensor: %zu bytes\n", stack_c);
    printf(" -> Stack watermark for tinyml_predict_pm25_multisensor:   %zu bytes\n", stack_predict);

    if (stack_predict > 1024) {
        printf(" -> FAIL: Dynamic stack usage (%zu bytes) exceeds 1024 bytes limit!\n", stack_predict);
        failures++;
    } else {
        printf(" -> PASS: Dynamic stack usage (%zu bytes) is strictly < 1024 bytes (1 KB)\n", stack_predict);
    }

    // ------------------------------------------------------------------------
    // CHALLENGE 2: ZERO-HEAP ALLOCATION UNDER 100,000 ADVERSARIAL CYCLES
    // ------------------------------------------------------------------------
    const int NUM_CYCLES = 100000;
    printf("\n[CHALLENGE 2: 100,000 Cycles Zero-Heap Allocation & Latency]\n");
    g_heap_allocations = 0;
    g_heap_bytes = 0;
    g_heap_intercept_enabled = true;

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t nan_count = 0;
    size_t inf_count = 0;
    double dummy_acc = 0.0;

    for (int i = 0; i < NUM_CYCLES; i++) {
        uint32_t seed = (uint32_t)(i * 1664525u + 1013904223u);
        float r1 = (float)(seed & 0xFFFF) / 65535.0f;
        float r2 = (float)((seed >> 16) & 0xFFFF) / 65535.0f;

        float pm1, pm2, pm10, temp, rh, co2, gas;
        int mode = i % 5;
        if (mode == 0) {
            pm1 = 10.0f + r1 * 50.0f;
            pm2 = pm1 + (r2 - 0.5f) * 2.0f;
            pm10 = pm1 * 1.3f;
            temp = 20.0f + r1 * 10.0f;
            rh = 55.0f + r2 * 10.0f;
            co2 = 400.0f + r1 * 200.0f;
            gas = 50.0f + r2 * 50.0f;
        } else if (mode == 1) {
            pm1 = 80.0f + r1 * 200.0f;
            pm2 = pm1 + (r2 - 0.5f) * 10.0f;
            pm10 = pm1 * 1.25f;
            temp = 15.0f + r1 * 15.0f;
            rh = 80.0f + r2 * 25.0f; // High humidity
            co2 = 420.0f + r1 * 100.0f;
            gas = 30.0f + r2 * 40.0f;
        } else if (mode == 2) {
            pm1 = 50.0f + r1 * 100.0f;
            pm2 = pm1 + (r2 - 0.5f) * 6.0f;
            pm10 = pm1 * 1.4f;
            temp = -30.0f + r1 * 95.0f; // -30 C to +65 C
            rh = 10.0f + r2 * 90.0f;
            co2 = 450.0f + r1 * 500.0f;
            gas = 80.0f + r2 * 100.0f;
        } else if (mode == 3) {
            pm1 = r1 * 5000.0f; // Massive range
            pm2 = r2 * 5000.0f;
            pm10 = pm1 * 1.2f + 5.0f;
            temp = -10.0f + r1 * 50.0f;
            rh = r2 * 105.0f;
            co2 = 300.0f + r1 * 3500.0f;
            gas = r2 * 1000.0f;
        } else {
            // Smoke / Fire mode
            pm1 = 200.0f + r1 * 1500.0f;
            pm2 = pm1 + (r2 - 0.5f) * 20.0f;
            pm10 = pm1 * 1.15f;
            temp = 25.0f + r1 * 20.0f;
            rh = 80.0f + r2 * 18.0f;
            co2 = 1200.0f + r1 * 2800.0f;
            gas = 300.0f + r2 * 900.0f;
        }

        bool anom = false;
        float pred = tinyml_predict_pm25_multisensor(pm1, pm2, pm10, temp, rh, co2, gas, &anom);
        if (isnan(pred)) nan_count++;
        if (isinf(pred)) inf_count++;
        dummy_acc += pred;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_heap_intercept_enabled = false;

    double total_sec = std::chrono::duration<double>(t1 - t0).count();
    double us_per_call = (total_sec / NUM_CYCLES) * 1e6;
    double throughput = NUM_CYCLES / total_sec;

    printf(" -> Cycles completed:          %d\n", NUM_CYCLES);
    printf(" -> Total duration:            %.4f seconds\n", total_sec);
    printf(" -> Mean latency:              %.2f us\n", us_per_call);
    printf(" -> Throughput:                %.0f calls/sec\n", throughput);
    printf(" -> NaN count:                 %zu\n", nan_count);
    printf(" -> Inf count:                 %zu\n", inf_count);
    printf(" -> Intercepted allocations:   %zu\n", g_heap_allocations);
    printf(" -> Intercepted heap bytes:    %zu bytes\n", g_heap_bytes);
    printf(" -> Checksum accumulator:      %.2f\n", dummy_acc);

    if (g_heap_bytes > 0 || g_heap_allocations > 0) {
        printf(" -> FAIL: Heap allocation detected during inference! (%zu bytes, %zu allocs)\n",
               g_heap_bytes, g_heap_allocations);
        failures++;
    } else {
        printf(" -> PASS: Strictly 0B heap memory allocated across 100,000 cycles\n");
    }

    if (nan_count > 0 || inf_count > 0) {
        printf(" -> FAIL: Unexpected NaN/Inf outputs in valid domain!\n");
        failures++;
    } else {
        printf(" -> PASS: 0 NaNs and 0 Infs produced across 100,000 cycles\n");
    }

    if (us_per_call > 50.0) {
        printf(" -> FAIL: Mean latency %.2f us exceeds 50 us threshold!\n", us_per_call);
        failures++;
    } else {
        printf(" -> PASS: Mean latency %.2f us is well below 50 us budget\n", us_per_call);
    }

    // ------------------------------------------------------------------------
    // CHALLENGE 3: DENSE RH BOUNDARY SWEEP [58.0%, 62.0%], STEP 0.05%
    // ------------------------------------------------------------------------
    printf("\n[CHALLENGE 3: Ultra-Dense Deliquescence Boundary Sweep]\n");
    printf(" -> Sweeping RH in [58.0%%, 62.0%%] with step 0.05%% (80 steps)\n");
    printf(" -> Requirement: Max consecutive step delta < 0.20 ug/m3 (far below 2.0 ug/m3)\n");

    const float pm_levels[] = {35.0f, 75.0f, 150.0f, 300.0f};
    const int num_pm_levels = sizeof(pm_levels) / sizeof(pm_levels[0]);

    float overall_max_delta = 0.0f;

    for (int p = 0; p < num_pm_levels; p++) {
        float pm_test = pm_levels[p];
        float prev_pred = -1.0f;
        float max_step_delta = 0.0f;
        float worst_rh = 0.0f;

        int step_count = 0;
        for (float rh = 58.0f; rh <= 62.0001f; rh += 0.05f) {
            bool anom = false;
            float pred = tinyml_predict_pm25_multisensor(
                pm_test, pm_test, pm_test * 1.4f, 26.0f, rh, 600.0f, 110.0f, &anom
            );

            if (step_count > 0) {
                float delta = fabsf(pred - prev_pred);
                if (delta > max_step_delta) {
                    max_step_delta = delta;
                    worst_rh = rh;
                }
            }
            prev_pred = pred;
            step_count++;
        }

        if (max_step_delta > overall_max_delta) {
            overall_max_delta = max_step_delta;
        }

        printf(" -> PM = %5.1f ug/m3: %d steps, Max consecutive delta = %.4f ug/m3 (at RH = %.2f%%)\n",
               pm_test, step_count, max_step_delta, worst_rh);

        if (max_step_delta >= 0.20f) {
            printf("    FAIL: Max step delta %.4f >= 0.20 ug/m3!\n", max_step_delta);
            failures++;
        } else {
            printf("    PASS: Max step delta %.4f < 0.20 ug/m3 threshold (smooth transition confirmed)\n", max_step_delta);
        }
    }

    // Check specific boundary jump at 59.9% vs 60.1% for PM=150
    {
        bool anom = false;
        float pm_599 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 26.0f, 59.9f, 600.0f, 110.0f, &anom);
        float pm_601 = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 26.0f, 60.1f, 600.0f, 110.0f, &anom);
        float diff_60 = fabsf(pm_599 - pm_601);
        printf(" -> Reference delta PM(60.1%%) - PM(59.9%%) @ PM=150: %.4f ug/m3 (Limit < 2.0 ug/m3)\n", diff_60);
        if (diff_60 >= 2.0f) {
            printf("    FAIL: Boundary delta %.4f >= 2.0 ug/m3!\n", diff_60);
            failures++;
        }
    }

    printf("\n================================================================================\n");
    printf("CHALLENGER M4-2 FIRMWARE TEST SUMMARY:\n");
    printf(" -> Dynamic stack usage:        %zu bytes (< 1024 bytes limit)\n", stack_predict);
    printf(" -> Heap allocations:           %zu bytes (0B required)\n", g_heap_bytes);
    printf(" -> 100,000 Cycles Mean Latency: %.2f us (< 50 us limit)\n", us_per_call);
    printf(" -> Max Step Delta (RH sweep):  %.4f ug/m3 (< 0.20 ug/m3 limit)\n", overall_max_delta);
    printf(" -> Total Failures:             %d\n", failures);
    printf("================================================================================\n");

    if (failures == 0) {
        printf("RESULT: ALL PHYSICAL BOUNDARY & FIRMWARE BUDGET CHECKS PASSED!\n");
        return 0;
    } else {
        printf("RESULT: CHECKS FAILED!\n");
        return 1;
    }
}
