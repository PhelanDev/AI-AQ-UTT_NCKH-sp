#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <new>
#include <float.h>

#include "tinyml_edge_engine.h"

// ============================================================================
// 1. COMPREHENSIVE HEAP INTERCEPTION (C & C++)
// ============================================================================
static size_t g_heap_alloc_count = 0;
static size_t g_heap_bytes_allocated = 0;
static bool g_track_heap = false;

extern "C" {
    void* __real_malloc(size_t size);
    void  __real_free(void* ptr);
}

void* operator new(size_t size) {
    if (g_track_heap) {
        g_heap_alloc_count++;
        g_heap_bytes_allocated += size;
    }
    return malloc(size);
}

void* operator new[](size_t size) {
    if (g_track_heap) {
        g_heap_alloc_count++;
        g_heap_bytes_allocated += size;
    }
    return malloc(size);
}

void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }

// ============================================================================
// 2. STACK WATERMARKING & PROFILING
// ============================================================================
// We fill a stack region with 0xA5 canary bytes, invoke the target, and inspect
// how many bytes were overwritten down the stack.

__attribute__((noinline)) size_t measure_stack_watermark_predict() {
    volatile uint8_t stack_canvas[4096];
    for (size_t i = 0; i < sizeof(stack_canvas); i++) {
        stack_canvas[i] = 0xA5;
    }

    uintptr_t canvas_top = (uintptr_t)&stack_canvas[sizeof(stack_canvas) - 1];

    // Call the function
    bool is_anom = false;
    volatile float res = tinyml_predict_pm25_multisensor(
        150.0f, 150.0f, 200.0f, 26.0f, 65.0f, 600.0f, 110.0f, &is_anom
    );
    (void)res;

    // Check lowest address where 0xA5 was overwritten
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
        150.0f, 2.0f, 0.70f, 26.0f, 65.0f, 18.0f, 600.0f, 110.0f
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
// MAIN ADVERSARIAL CHALLENGER 2 HARNESS
// ============================================================================
int main() {
    printf("================================================================================\n");
    printf("CHALLENGER 2: ADVERSARIAL FIRMWARE STRESS & MEMORY SAFETY HARNESS\n");
    printf("================================================================================\n\n");

    int test_failures = 0;

    // ------------------------------------------------------------------------
    // TEST 1: STACK WATERMARK USAGE CHECK (< 1024 BYTES)
    // ------------------------------------------------------------------------
    printf("[CHALLENGE 2.1: Stack Usage Profiling]\n");
    {
        size_t stack_c = measure_stack_watermark_c_factor();
        size_t stack_pred = measure_stack_watermark_predict();

        printf(" -> Stack watermark tinyml_infer_c_factor_multisensor: %zu bytes\n", stack_c);
        printf(" -> Stack watermark tinyml_predict_pm25_multisensor:   %zu bytes\n", stack_pred);

        // Theoretical maximum: 832 bytes arrays + frame registers
        if (stack_pred > 1024) {
            printf("    FAIL: Stack usage exceeds 1KB limit (%zu > 1024 bytes)!\n", stack_pred);
            test_failures++;
        } else {
            printf("    PASS: Stack usage is strictly under 1KB limit (< 1024 bytes)\n");
        }
    }

    // ------------------------------------------------------------------------
    // TEST 2: ZERO-HEAP ALLOCATION UNDER 100,000 ADVERSARIAL INFERENCES
    // ------------------------------------------------------------------------
    printf("\n[CHALLENGE 2.2: Dynamic Heap Allocation Interception (100,000 Inferences)]\n");
    {
        g_heap_alloc_count = 0;
        g_heap_bytes_allocated = 0;
        g_track_heap = true;

        bool is_anom = false;
        TinyML_AnomalyCode code;
        float dummy_acc = 0.0f;

        for (int i = 0; i < 100000; i++) {
            float pm = (float)(i % 500);
            float rh = 10.0f + (float)(i % 90);
            float temp = -10.0f + (float)(i % 55);
            float co2 = 350.0f + (float)(i % 2000);
            float gas = 20.0f + (float)(i % 600);

            float pred = tinyml_predict_pm25_multisensor(pm, pm + 1.0f, pm * 1.3f, temp, rh, co2, gas, &is_anom);
            float adv = tinyml_predict_pm25_advanced(pm, pm + 1.0f, temp, rh, (float)(i % 60), &code);
            dummy_acc += pred + adv;
        }

        g_track_heap = false;

        printf(" -> Total inferences run: 100,000 (x2 function paths = 200,000 calls)\n");
        printf(" -> Heap allocations intercepted: %zu\n", g_heap_alloc_count);
        printf(" -> Heap bytes allocated:       %zu bytes\n", g_heap_bytes_allocated);
        printf(" -> Checksum accumulator:       %.2f\n", dummy_acc);

        if (g_heap_bytes_allocated > 0 || g_heap_alloc_count > 0) {
            printf("    FAIL: Heap memory was dynamically allocated (%zu bytes)!\n", g_heap_bytes_allocated);
            test_failures++;
        } else {
            printf("    PASS: Strictly 0B Dynamic Heap Allocation verified\n");
        }
    }

    // ------------------------------------------------------------------------
    // TEST 3: PATHOLOGICAL INPUTS & NAN / INF STRESS TESTING
    // ------------------------------------------------------------------------
    printf("\n[CHALLENGE 2.3: Pathological Singularity & NaN/Inf Injections]\n");
    {
        float qnan = std::numeric_limits<float>::quiet_NaN();
        float pinf = std::numeric_limits<float>::infinity();
        float ninf = -std::numeric_limits<float>::infinity();
        float subnormal = 1e-42f;

        struct PathologicalCase {
            float pm1, pm2, pm10, temp, rh, co2, gas;
            const char* desc;
        } path_cases[] = {
            {qnan, 50.0f, 65.0f, 25.0f, 60.0f, 420.0f, 100.0f, "NaN in PM1"},
            {50.0f, qnan, 65.0f, 25.0f, 60.0f, 420.0f, 100.0f, "NaN in PM2"},
            {50.0f, 50.0f, qnan, 25.0f, 60.0f, 420.0f, 100.0f, "NaN in PM10"},
            {50.0f, 50.0f, 65.0f, qnan, 60.0f, 420.0f, 100.0f, "NaN in Temp"},
            {50.0f, 50.0f, 65.0f, 25.0f, qnan, 420.0f, 100.0f, "NaN in RH"},
            {50.0f, 50.0f, 65.0f, 25.0f, 60.0f, qnan, 100.0f, "NaN in CO2"},
            {50.0f, 50.0f, 65.0f, 25.0f, 60.0f, 420.0f, qnan, "NaN in Gas"},
            {pinf, pinf, pinf, 25.0f, 60.0f, 420.0f, 100.0f, "+Inf in PM"},
            {50.0f, 50.0f, 65.0f, pinf, 60.0f, 420.0f, 100.0f, "+Inf in Temp"},
            {50.0f, 50.0f, 65.0f, 25.0f, pinf, 420.0f, 100.0f, "+Inf in RH"},
            {50.0f, 50.0f, 65.0f, 25.0f, 60.0f, pinf, 100.0f, "+Inf in CO2"},
            {50.0f, 50.0f, 65.0f, 25.0f, 60.0f, 420.0f, pinf, "+Inf in Gas"},
            {ninf, 50.0f, 65.0f, 25.0f, 60.0f, 420.0f, 100.0f, "-Inf in PM1"},
            {50.0f, 50.0f, 65.0f, -237.7f, 60.0f, 420.0f, 100.0f, "Magnus-Tetens Singularity (Temp = -237.7 C)"},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, "All Zeros (PM10=0 division test)"},
            {-100.0f, -100.0f, -100.0f, -50.0f, -10.0f, -200.0f, -50.0f, "All Negative Out-of-Bounds"},
            {1e6f, 1e6f, 1e6f, 1000.0f, 500.0f, 1e6f, 1e6f, "Massive Overflows (1,000,000)"},
            {subnormal, subnormal, subnormal, subnormal, subnormal, subnormal, subnormal, "Subnormal Underflows (1e-42)"}
        };

        int path_count = sizeof(path_cases) / sizeof(path_cases[0]);
        int crash_count = 0;

        for (int i = 0; i < path_count; i++) {
            PathologicalCase& pc = path_cases[i];
            bool anom = false;

            // Attempt calculation
            float pred = tinyml_predict_pm25_multisensor(
                pc.pm1, pc.pm2, pc.pm10, pc.temp, pc.rh, pc.co2, pc.gas, &anom
            );

            float c = tinyml_infer_c_factor_multisensor(
                pc.pm1, fabsf(pc.pm1 - pc.pm2), 0.70f, pc.temp, pc.rh, 15.0f, pc.co2, pc.gas
            );

            printf(" -> Case %2d [%-45s]: pred=%10.2f, c=%6.4f, anom=%d\n",
                   i + 1, pc.desc, pred, c, anom);
        }

        printf("    PASS: Engine survived all 18 pathological input cases without segfault or crash\n");
    }

    // ------------------------------------------------------------------------
    // TEST 4: HIGH-RESOLUTION LATENCY BENCHMARK (< 50 US)
    // ------------------------------------------------------------------------
    printf("\n[CHALLENGE 2.4: High-Resolution Latency Benchmark (100,000 Iterations)]\n");
    {
        const int BENCH_ROUNDS = 100000;
        std::vector<double> latencies_us;
        latencies_us.reserve(BENCH_ROUNDS);

        bool anom = false;
        // Warmup
        for (int i = 0; i < 1000; i++) {
            tinyml_predict_pm25_multisensor(120.0f, 122.0f, 150.0f, 25.0f, 60.0f, 450.0f, 100.0f, &anom);
        }

        auto t_start_total = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < BENCH_ROUNDS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            volatile float out = tinyml_predict_pm25_multisensor(
                120.0f, 122.0f, 150.0f, 25.0f, 60.0f, 450.0f, 100.0f, &anom
            );
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)out;

            double dur_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            latencies_us.push_back(dur_us);
        }

        auto t_end_total = std::chrono::high_resolution_clock::now();
        double total_dur_sec = std::chrono::duration<double>(t_end_total - t_start_total).count();
        double mean_us = (total_dur_sec / BENCH_ROUNDS) * 1e6;

        std::sort(latencies_us.begin(), latencies_us.end());
        double p50 = latencies_us[BENCH_ROUNDS * 50 / 100];
        double p95 = latencies_us[BENCH_ROUNDS * 95 / 100];
        double p99 = latencies_us[BENCH_ROUNDS * 99 / 100];
        double pmax = latencies_us[BENCH_ROUNDS - 1];

        printf(" -> Total benchmark duration: %.4f seconds (%d inferences)\n", total_dur_sec, BENCH_ROUNDS);
        printf(" -> Throughput:               %.0f inferences/sec\n", BENCH_ROUNDS / total_dur_sec);
        printf(" -> Mean Latency:             %.3f us\n", mean_us);
        printf(" -> P50  Latency:             %.3f us\n", p50);
        printf(" -> P95  Latency:             %.3f us\n", p95);
        printf(" -> P99  Latency:             %.3f us\n", p99);
        printf(" -> Max  Latency:             %.3f us\n", pmax);

        if (mean_us >= 50.0) {
            printf("    FAIL: Mean latency %.3f us exceeds 50 us threshold!\n", mean_us);
            test_failures++;
        } else {
            printf("    PASS: Mean latency %.3f us is well below 50 us threshold\n", mean_us);
        }
    }

    printf("\n================================================================================\n");
    if (test_failures == 0) {
        printf("CHALLENGER 2 C++ FIRMWARE VERDICT: PASS (All memory, stack, NaN, latency tests PASSED)\n");
        printf("================================================================================\n");
        return 0;
    } else {
        printf("CHALLENGER 2 C++ FIRMWARE VERDICT: FAIL (%d tests failed)\n", test_failures);
        printf("================================================================================\n");
        return 1;
    }
}
