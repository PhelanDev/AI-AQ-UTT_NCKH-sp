#ifndef TINYML_EDGE_ENGINE_H
#define TINYML_EDGE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
  #include "esp_attr.h"
  #define TINYML_RTC_ATTR RTC_DATA_ATTR
#else
  #define TINYML_RTC_ATTR
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TINYML_ANOMALY_OK = 0,
    TINYML_ANOMALY_WARMUP_INCOMPLETE = 1,
    TINYML_ANOMALY_TRANSIENT_SPIKE = 2,
    TINYML_ANOMALY_CHAMBER_DRIFT = 3
} TinyML_AnomalyCode;

#define PMS7003_WARMUP_SECONDS 30.0f
#define PMS7003_DRIFT_THRESHOLD 15.0f

float tinyml_calc_dew_point(float temp_c, float rh_percent);

float tinyml_filter_dual_pm(float pm1, float pm2, bool *is_anomaly);

TinyML_AnomalyCode tinyml_update_drift_monitor(float pm1, float pm2, float *ema_drift);

void tinyml_reset_drift_monitor(void);
float tinyml_get_current_ema_drift(void);
uint32_t tinyml_get_boot_cycle_count(void);
void tinyml_increment_boot_cycle_count(void);
void tinyml_set_boot_cycle_count(uint32_t count);

// Suy luận hệ số bù C với đầy đủ 8 cảm biến BOM: Dual PMS7003, SHT40, SenseAir S8 (CO2), Sensirion SGP41 (TVOC/NOx)
float tinyml_infer_c_factor_multisensor(float pm_avg, float diff_pm, float r_pm,
                                       float temp_c, float rh_percent, float dew_point,
                                       float co2_ppm, float gas_proxy);

// Hàm tương thích ngược 5 tham số (tự suy đoán giá trị nền cho CO2 và Gas nếu thiếu)
float tinyml_infer_c_factor(float pm_avg, float diff_pm, float temp_c, float rh_percent, float dew_point);

// Suy luận hoàn chỉnh nồng độ PM2.5 đã bù nồm ẩm & khử khói
float tinyml_predict_pm25_multisensor(float pm1_raw, float pm2_raw, float pm10_raw,
                                      float temp_c, float rh_percent,
                                      float co2_ppm, float gas_proxy,
                                      bool *is_anomaly);

float tinyml_predict_pm25(float pm1_raw, float pm2_raw,
                          float temp_c, float rh_percent,
                          bool *is_anomaly);

float tinyml_predict_pm25_advanced(float pm1_raw, float pm2_raw,
                                   float temp_c, float rh_percent,
                                   float warmup_elapsed_sec,
                                   TinyML_AnomalyCode *out_anomaly_code);

#ifdef __cplusplus
}
#endif

#endif // TINYML_EDGE_ENGINE_H
