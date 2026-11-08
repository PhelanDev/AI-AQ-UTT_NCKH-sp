#include "power_scheduler.h"
#include <string.h>
#include <stdio.h>

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#include "esp_sleep.h"
#include "driver/gpio.h"
#include "driver/rtc_io.h"
#endif

static EdgeFsmState s_current_fsm_state = EDGE_STATE_WAKEUP;

EdgeFsmState power_scheduler_get_state(void) {
    return s_current_fsm_state;
}

void power_scheduler_set_state(EdgeFsmState state) {
    s_current_fsm_state = state;
}

const char *power_scheduler_state_str(EdgeFsmState state) {
    switch (state) {
        case EDGE_STATE_WAKEUP:          return "EDGE_STATE_WAKEUP";
        case EDGE_STATE_WARMUP:          return "EDGE_STATE_WARMUP";
        case EDGE_STATE_ACQUIRE_INFER:   return "EDGE_STATE_ACQUIRE_INFER";
        case EDGE_STATE_DISPLAY_PUBLISH: return "EDGE_STATE_DISPLAY_PUBLISH";
        case EDGE_STATE_PREPARE_SLEEP:   return "EDGE_STATE_PREPARE_SLEEP";
        case EDGE_STATE_DEEP_SLEEP:      return "EDGE_STATE_DEEP_SLEEP";
        default:                         return "EDGE_STATE_UNKNOWN";
    }
}

EdgeCycleResult edge_execute_15min_duty_cycle(const RawSensorSample *raw_sample,
                                             float warmup_elapsed_sec,
                                             bool mqtt_connected) {
    EdgeCycleResult res;
    memset(&res, 0, sizeof(res));

    if (!raw_sample) {
        res.anomaly_code = TINYML_ANOMALY_WARMUP_INCOMPLETE;
        return res;
    }

    // ========================================================================
    // STATE 1: WARMUP TIME VALIDATION (< 30s blocks measurement)
    // ========================================================================
    if (warmup_elapsed_sec < PMS7003_WARMUP_SECONDS) {
        s_current_fsm_state = EDGE_STATE_WARMUP;
        res.anomaly_code = TINYML_ANOMALY_WARMUP_INCOMPLETE;
        res.pm25_calibrated = (raw_sample->pm1_raw + raw_sample->pm2_raw) * 0.5f;
        res.vn_aqi = 0;
        res.lcd_updated = false;
        res.telemetry_sent = false;
        res.entered_deep_sleep = false;
        res.deep_sleep_duration_sec = 0;
        res.pms_set_pin_level = true; // PMS_SET stays HIGH: keep fan active to complete warmup
        return res;
    }

    // ========================================================================
    // STATE 0: WAKEUP & PERIPHERAL INIT
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_WAKEUP;
    tinyml_increment_boot_cycle_count();
    uint32_t boot_cycle = tinyml_get_boot_cycle_count();

    // ========================================================================
    // STATE 1: WARMUP CONFIRMED (30 seconds elapsed)
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_WARMUP;
    res.pms_set_pin_level = true;

    // ========================================================================
    // STATE 2: ACQUIRE SENSOR SAMPLES & RUN TINYML INFERENCE (< 0.35ms)
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_ACQUIRE_INFER;

    // 1. Update long-term dual PMS7003 chamber drift monitor in RTC memory
    float current_drift = 0.0f;
    TinyML_AnomalyCode drift_status = tinyml_update_drift_monitor(
        raw_sample->pm1_raw, raw_sample->pm2_raw, &current_drift
    );

    // 2. Execute TinyML 2D Deliquescence gated neural network
    bool is_spike_anomaly = false;
    float pm25_cal = tinyml_predict_pm25_multisensor(
        raw_sample->pm1_raw, raw_sample->pm2_raw, raw_sample->pm10_raw,
        raw_sample->temp_c, raw_sample->rh_percent,
        raw_sample->co2_ppm, raw_sample->gas_proxy,
        &is_spike_anomaly
    );

    int aqi = tinyml_compute_vn_aqi(pm25_cal);

    if (drift_status == TINYML_ANOMALY_CHAMBER_DRIFT) {
        res.anomaly_code = TINYML_ANOMALY_CHAMBER_DRIFT;
    } else if (is_spike_anomaly) {
        res.anomaly_code = TINYML_ANOMALY_TRANSIENT_SPIKE;
    } else {
        res.anomaly_code = TINYML_ANOMALY_OK;
    }

    res.pm25_calibrated = pm25_cal;
    res.vn_aqi = aqi;

    // ========================================================================
    // STATE 3: DISPLAY TO LCD & PUBLISH MQTT TELEMETRY
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_DISPLAY_PUBLISH;

    // 1. Update 2.8" SPI TFT LCD (ST7789) Screen Buffer
    EdgeDisplayPayload display_payload;
    display_payload.pm25_calibrated = res.pm25_calibrated;
    display_payload.pm25_raw = (raw_sample->pm1_raw + raw_sample->pm2_raw) * 0.5f;
    display_payload.aqi = res.vn_aqi;
    display_payload.temp_c = raw_sample->temp_c;
    display_payload.rh_percent = raw_sample->rh_percent;
    display_payload.co2_ppm = raw_sample->co2_ppm;
    display_payload.gas_proxy = raw_sample->gas_proxy;
    display_payload.boot_cycle = boot_cycle;
    display_payload.status_msg = (res.anomaly_code == TINYML_ANOMALY_OK) ? "HEALTHY" : "ANOMALY";

    char lcd_buffer[256];
    lcd_format_screen_buffer(&display_payload, lcd_buffer, sizeof(lcd_buffer));
    res.lcd_updated = true;

    // 2. Prepare 32-byte Telemetry Packet
    EdgeTelemetryPacket pkt;
    pkt.timestamp = raw_sample->timestamp;
    pkt.pm1_raw = raw_sample->pm1_raw;
    pkt.pm2_raw = raw_sample->pm2_raw;
    pkt.pm25_calibrated = res.pm25_calibrated;
    pkt.temp_c = raw_sample->temp_c;
    pkt.rh_percent = raw_sample->rh_percent;
    pkt.co2_ppm = raw_sample->co2_ppm;
    pkt.aqi = (uint16_t)res.vn_aqi;
    pkt.anomaly_code = (uint8_t)res.anomaly_code;
    pkt.flags = mqtt_connected ? 0x01 : 0x00;

    if (mqtt_connected) {
        char json_buf[512];
        int json_len = mqtt_format_telemetry_json(&pkt, json_buf, sizeof(json_buf));
        res.telemetry_sent = (json_len > 0);

        // Opportunistically drain any buffered packets queued during network outages
        EdgeTelemetryPacket queued_pkt;
        while (offline_ringbuf_pop(&queued_pkt)) {
            // In live deployment, queued_pkt would be published to MQTT broker
        }
    } else {
        res.telemetry_sent = false;
        // Network offline: Enqueue into static 32-byte ring buffer (Zero Dynamic Allocation)
        offline_ringbuf_push(&pkt);
    }

    // ========================================================================
    // STATE 4: PREPARE DEEP SLEEP (Pull PMS_SET LOW, hold pin, set RTC timer)
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_PREPARE_SLEEP;
    res.pms_set_pin_level = false; // Turn OFF fan and laser
    res.deep_sleep_duration_sec = ESP_DEEP_SLEEP_SEC; // 868 seconds

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
    gpio_set_level((gpio_num_t)PIN_PMS_SET, 0);
    gpio_hold_en((gpio_num_t)PIN_PMS_SET);
    esp_sleep_enable_timer_wakeup(((uint64_t)ESP_DEEP_SLEEP_SEC) * 1000000ULL);
#endif

    // ========================================================================
    // STATE 5: ENTER DEEP SLEEP (868 seconds)
    // ========================================================================
    s_current_fsm_state = EDGE_STATE_DEEP_SLEEP;
    res.entered_deep_sleep = true;

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
    esp_deep_sleep_start();
#endif

    return res;
}
