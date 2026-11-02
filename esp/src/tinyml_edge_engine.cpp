#include "tinyml_edge_engine.h"
#include "model_data.h"
#include <math.h>

static float s_ema_drift = 0.0f;
static bool s_drift_initialized = false;

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

    // Tang 1: fc_in (8 -> 96)
    float h0[TINYML_HIDDEN_DIM];
    for (int i = 0; i < TINYML_HIDDEN_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TINYML_INPUT_DIM; j++) {
            sum += norm_in[j] * ((float)FC_IN_WEIGHT[i][j]);
        }
        h0[i] = relu(sum * FC_IN_SCALE + FC_IN_BIAS[i]);
    }

    // Residual Block 1: block1 (96 -> 96) + Skip Connection h0
    float h1[TINYML_HIDDEN_DIM];
    for (int i = 0; i < TINYML_HIDDEN_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TINYML_HIDDEN_DIM; j++) {
            sum += h0[j] * ((float)FC_BLOCK1_WEIGHT[i][j]);
        }
        h1[i] = relu(sum * FC_BLOCK1_SCALE + FC_BLOCK1_BIAS[i] + h0[i]);
    }

    // Residual Block 2: block2 (96 -> 96) + Skip Connection h1 (tai su dung mang h0 lam bo dem Stack)
    float* h2 = h0;
    for (int i = 0; i < TINYML_HIDDEN_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TINYML_HIDDEN_DIM; j++) {
            sum += h1[j] * ((float)FC_BLOCK2_WEIGHT[i][j]);
        }
        h2[i] = relu(sum * FC_BLOCK2_SCALE + FC_BLOCK2_BIAS[i] + h1[i]);
    }

    // Tang Output: fc_out (96 -> 1)
    float raw_out = 0.0f;
    for (int j = 0; j < TINYML_HIDDEN_DIM; j++) {
        raw_out += h2[j] * ((float)FC_OUT_WEIGHT[0][j]);
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

    // Neu RH < 60% khong co phong am thi tra ve so do goc
    if (rh_percent < 60.0f && co2_ppm < 800.0f) {
        return pm_avg;
    }

    return pm_avg * c_factor;
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
