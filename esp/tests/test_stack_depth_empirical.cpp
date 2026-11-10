#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "tinyml_edge_engine.h"

static uintptr_t g_caller_sp = 0;
static uintptr_t g_callee_sp = 0;

__attribute__((noinline)) void sample_callee_sp() {
    register uintptr_t sp asm("rsp");
    g_callee_sp = sp;
}

int main() {
    register uintptr_t sp asm("rsp");
    g_caller_sp = sp;

    bool is_anom = false;
    float pred = tinyml_predict_pm25_multisensor(150.0f, 150.0f, 214.28f, 26.0f, 60.0f, 600.0f, 110.0f, &is_anom);

    printf("Caller RSP: 0x%zx\n", g_caller_sp);
    printf("Result: %.2f\n", pred);

    return 0;
}
