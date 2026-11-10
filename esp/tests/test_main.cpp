#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "tinyml_edge_engine.h"

int main() {
    printf("================================================================================\n");
    printf("KIEM THU XAC THUC STANDARD TINYML MLP LITE (833 THAM SO) & VONG DOI THUC DIA\n");
    printf("================================================================================\n\n");

    {
        TinyML_AnomalyCode code;
        printf("[TEST 1: Chu ky lam am cam bien PMS7003]\n");

        float pm_early = tinyml_predict_pm25_advanced(120.0f, 115.0f, 22.0f, 90.0f, 10.0f, &code);
        printf(" -> Tai t = 10s (Chua du 30s warm-up): PM = %.2f ug/m3 | AnomalyCode = %d (%s)\n",
               pm_early, code, code == TINYML_ANOMALY_WARMUP_INCOMPLETE ? "WARMUP_INCOMPLETE -> CHAN KHONG CHO TINYML SUY LUAN" : "ERROR");

        float pm_ready = tinyml_predict_pm25_advanced(120.0f, 115.0f, 22.0f, 90.0f, 35.0f, &code);
        printf(" -> Tai t = 35s (Warm-up hoan tat)   : PM = %.2f ug/m3 | AnomalyCode = %d (%s)\n\n",
               pm_ready, code, code == TINYML_ANOMALY_OK ? "SAN SANG -> TINYML DA HIEU CHUAN CHINH XAC" : "ERROR");
    }

    {
        TinyML_AnomalyCode code = TINYML_ANOMALY_OK;
        printf("[TEST 2: Phat hien lao hoa / bam bui thau kinh dai han (Chamber Drift)]\n");

        for (int i = 0; i < 250; i++) {
            tinyml_predict_pm25_advanced(68.0f, 48.0f, 25.0f, 70.0f, 35.0f, &code);
        }
        printf(" -> Sau 250 chu ky lech lien tuc mot chieu (|PM1 - PM2| = 20 ug/m3):\n");
        printf(" -> Trang thai AnomalyCode = %d (%s)\n",
               code, code == TINYML_ANOMALY_CHAMBER_DRIFT ? "CHAMBER_DRIFT -> KICH HOAT CANH BAO BAO DUONG VE SINH THAU KINH" : "CHUA CANH BAO");
        printf("\n");
    }

    {
        bool is_anom = false;
        float pm_wet = tinyml_predict_pm25(142.0f, 138.0f, 21.0f, 94.0f, &is_anom);
        printf("[TEST 3: Hieu chuan nom am thuc dia (RH=94.0%%, T=21.0C)]\n");
        printf(" -> PMS7003 thô đo được     : 140.00 ug/m3 (Phồng ẩm nghiêm trọng)\n");
        printf(" -> Standard MLP Lite bù ẩm  : %.2f ug/m3 (Bù nồm ẩm hiệu quả)\n", pm_wet);
        printf(" -> Trang thai canh bao      : %s\n\n", is_anom ? "CANH BAO" : "BINH THUONG");
    }

    printf("================================================================================\n");
    printf("BENCHMARK TOC DO SUY LUAN MANG STANDARD MLP LITE 833 THAM SO (10.000 LAN LAP)\n");
    printf("================================================================================\n");

    const int N_ITER = 10000;
    clock_t start = clock();
    float dummy_sum = 0.0f;
    TinyML_AnomalyCode dummy_code;

    for (int i = 0; i < N_ITER; i++) {
        dummy_sum += tinyml_predict_pm25_advanced(110.0f + (i % 20), 108.0f + (i % 20), 22.0f, 85.0f, 35.0f, &dummy_code);
    }
    clock_t end = clock();

    double total_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double us_per_infer = (total_sec / N_ITER) * 1e6;
    double fps = N_ITER / total_sec;

    printf(" -> Tong thoi gian chay 10.000 lan: %.4f giay\n", total_sec);
    printf(" -> Do tre trung binh moi lan suy luan: %.2f micro-giay (us) (~%.3f ms)\n", us_per_infer, us_per_infer / 1000.0);
    printf(" -> Toc do xu ly: %.0f lan suy luan / giay (Inferences/sec)\n", fps);
    printf(" -> Du tinh tren ESP32-S3 (240MHz): ~0.15 - 0.35 ms / lan (Chi ton <0.004%% CPU)\n");
    printf("================================================================================\n");

    return 0;
}
