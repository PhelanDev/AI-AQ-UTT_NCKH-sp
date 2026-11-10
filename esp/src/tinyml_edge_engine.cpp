#include "tinyml_edge_engine.h"
#include "model_data.h"
#include <math.h>

TINYML_RTC_ATTR static float s_ema_drift = 0.0f;
TINYML_RTC_ATTR static bool s_drift_initialized = false;
TINYML_RTC_ATTR static uint32_t s_boot_cycle_count = 0;

static inline float relu(float v) {
    return (v > 0.0f) ? v : 0.0f;
}

static inline float sigmoid(float v) {
    return 1.0f / (1.0f + expf(-v));
}

float tinyml_calc_dew_point(float temp_c, float rh_percent) {
    const float a = 17.27f;
    const float b = 237.7f;
    float rh = rh_percent;
    if (rh < 1.0f) rh = 1.0f;
    if (rh > 100.0f) rh = 100.0f;
    float alpha = ((a * temp_c) / (b + temp_c)) + logf(rh / 100.0f);
    return (b * alpha) / (a - alpha);
}

float tinyml_filter_dual_pm(float pm1, float pm2, bool *is_anomaly) {
    float diff = fabsf(pm1 - pm2);
    if (diff > 25.0f) {
        if (is_anomaly) *is_anomaly = true;
        return (pm1 < pm2) ? pm1 : pm2;
    }
    if (is_anomaly) *is_anomaly = false;
    return (pm1 + pm2) * 0.5f;
}

TinyML_AnomalyCode tinyml_update_drift_monitor(float pm1, float pm2, float *ema_drift) {
    float instant_diff = pm1 - pm2;
    const float alpha = 0.01f;

    if (!s_drift_initialized) {
        s_ema_drift = instant_diff;
        s_drift_initialized = true;
    } else {
        s_ema_drift = (1.0f - alpha) * s_ema_drift + alpha * instant_diff;
    }

    if (ema_drift) {
        *ema_drift = s_ema_drift;
    }

    if (fabsf(s_ema_drift) > PMS7003_DRIFT_THRESHOLD) {
        return TINYML_ANOMALY_CHAMBER_DRIFT;
    }

    return TINYML_ANOMALY_OK;
}

void tinyml_reset_drift_monitor(void) {
    s_ema_drift = 0.0f;
    s_drift_initialized = false;
}

float tinyml_get_current_ema_drift(void) {
    return s_ema_drift;
}

uint32_t tinyml_get_boot_cycle_count(void) {
    return s_boot_cycle_count;
}

void tinyml_increment_boot_cycle_count(void) {
    s_boot_cycle_count++;
}

void tinyml_set_boot_cycle_count(uint32_t count) {
    s_boot_cycle_count = count;
}

float tinyml_infer_c_factor_multisensor(float pm_avg, float diff_pm, float r_pm,
                                       float temp_c, float rh_percent, float dew_point,
                                       float co2_ppm, float gas_proxy) {
    float raw_in[TINYML_INPUT_DIM] = {
        pm_avg, diff_pm, r_pm, temp_c, rh_percent, dew_point, co2_ppm, gas_proxy
    };
    float norm_in[TINYML_INPUT_DIM];

    // Chuan hoa Min-Max
    for (int i = 0; i < TINYML_INPUT_DIM; i++) {
        float range = TINYML_X_MAX[i] - TINYML_X_MIN[i];
        if (range <= 1e-6f) range = 1.0f;
        float val = (raw_in[i] - TINYML_X_MIN[i]) / range;
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        norm_in[i] = val;
    }

    // Tang 1: fc1 (8 -> 32)
    float h0[TINYML_HIDDEN1_DIM];
    for (int i = 0; i < TINYML_HIDDEN1_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TINYML_INPUT_DIM; j++) {
            sum += norm_in[j] * ((float)FC1_WEIGHT[i][j]);
        }
        h0[i] = relu(sum * FC1_SCALE + FC1_BIAS[i]);
    }

    // Tang 2: fc2 (32 -> 16)
    float h1[TINYML_HIDDEN2_DIM];
    for (int i = 0; i < TINYML_HIDDEN2_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TINYML_HIDDEN1_DIM; j++) {
            sum += h0[j] * ((float)FC2_WEIGHT[i][j]);
        }
        h1[i] = relu(sum * FC2_SCALE + FC2_BIAS[i]);
    }

    // Tang Output: fc_out (16 -> 1)
    float raw_out = 0.0f;
    for (int j = 0; j < TINYML_HIDDEN2_DIM; j++) {
        raw_out += h1[j] * ((float)FC_OUT_WEIGHT[0][j]);
    }
    raw_out = raw_out * FC_OUT_SCALE + FC_OUT_BIAS[0];

    // Bounded Physics Multiplier: C in [0.25, 1.05]
    float c_mult = 0.25f + 0.80f * sigmoid(raw_out);
    return c_mult;
}

float tinyml_infer_c_factor(float pm_avg, float diff_pm, float temp_c, float rh_percent, float dew_point) {
    // Gia tri mac dinh SenseAir S8 = 420 ppm, SGP41 Gas = 100
    float r_pm = 0.70f;
    float co2_ppm = 420.0f;
    float gas_proxy = 100.0f;
    return tinyml_infer_c_factor_multisensor(pm_avg, diff_pm, r_pm, temp_c, rh_percent, dew_point, co2_ppm, gas_proxy);
}

float tinyml_predict_pm25_multisensor(float pm1_raw, float pm2_raw, float pm10_raw,
                                      float temp_c, float rh_percent,
                                      float co2_ppm, float gas_proxy,
                                      bool *is_anomaly) {
    float diff_pm = fabsf(pm1_raw - pm2_raw);
    float pm_avg = tinyml_filter_dual_pm(pm1_raw, pm2_raw, is_anomaly);
    float dew_point = tinyml_calc_dew_point(temp_c, rh_percent);
    float r_pm = (pm10_raw > 0.1f) ? (pm_avg / pm10_raw) : 0.70f;
    if (r_pm > 1.0f) r_pm = 1.0f;
    if (r_pm < 0.1f) r_pm = 0.1f;

    float c_factor = tinyml_infer_c_factor_multisensor(
        pm_avg, diff_pm, r_pm, temp_c, rh_percent, dew_point, co2_ppm, gas_proxy
    );

    // CỔNG CHUYỂN PHA DELIQUESCENCE KHẢ VI 2 CHIỀU (SMOOTH 2D SIGMOID GATE)
    float rh_safe = rh_percent;
    if (rh_safe < 0.0f) rh_safe = 0.0f;
    if (rh_safe > 100.0f) rh_safe = 100.0f;
    float sig_rh = sigmoid((rh_safe - 68.0f) / 7.0f);

    float co2_safe = co2_ppm;
    if (co2_safe < 0.0f) co2_safe = 0.0f;
    float sig_co2 = sigmoid((co2_safe - 800.0f) / 240.0f);

    float w_del = sig_rh * (1.0f - sig_co2);
    float c_eff = (1.0f - w_del) * 1.0f + w_del * c_factor;

    return pm_avg * c_eff;
}

float tinyml_predict_pm25(float pm1_raw, float pm2_raw,
                          float temp_c, float rh_percent,
                          bool *is_anomaly) {
    return tinyml_predict_pm25_multisensor(pm1_raw, pm2_raw, pm1_raw * 1.3f, temp_c, rh_percent, 420.0f, 100.0f, is_anomaly);
}

float tinyml_predict_pm25_advanced(float pm1_raw, float pm2_raw,
                                   float temp_c, float rh_percent,
                                   float warmup_elapsed_sec,
                                   TinyML_AnomalyCode *out_anomaly_code) {
    if (warmup_elapsed_sec < PMS7003_WARMUP_SECONDS) {
        if (out_anomaly_code) *out_anomaly_code = TINYML_ANOMALY_WARMUP_INCOMPLETE;
        return (pm1_raw + pm2_raw) * 0.5f;
    }

    TinyML_AnomalyCode drift_status = tinyml_update_drift_monitor(pm1_raw, pm2_raw, 0);
    if (drift_status == TINYML_ANOMALY_CHAMBER_DRIFT) {
        if (out_anomaly_code) *out_anomaly_code = TINYML_ANOMALY_CHAMBER_DRIFT;
        return (pm1_raw < pm2_raw) ? pm1_raw : pm2_raw;
    }

    bool is_spike = false;
    float pred = tinyml_predict_pm25(pm1_raw, pm2_raw, temp_c, rh_percent, &is_spike);

    if (out_anomaly_code) {
        *out_anomaly_code = is_spike ? TINYML_ANOMALY_TRANSIENT_SPIKE : TINYML_ANOMALY_OK;
    }

    return pred;
}
