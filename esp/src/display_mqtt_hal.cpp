#include "display_mqtt_hal.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// ============================================================================
// VN-AQI Calculation (QCVN 05:2023 / Decision 1459/QD-TCMT)
// ============================================================================

struct AQIBreakpoint {
    float c_lo, c_hi;
    float i_lo, i_hi;
};

static const AQIBreakpoint QCVN_BREAKPOINTS[] = {
    {0.0f,   25.0f,   0.0f,   50.0f},  // Tốt (Good)
    {25.0f,  50.0f,  50.0f,  100.0f},  // Trung bình (Moderate)
    {50.0f,  80.0f, 100.0f,  150.0f},  // Kém (Unhealthy for sensitive)
    {80.0f, 150.0f, 150.0f,  200.0f},  // Xấu (Unhealthy)
    {150.0f, 250.0f, 200.0f, 300.0f},  // Rất xấu (Very Unhealthy)
    {250.0f, 500.0f, 300.0f, 500.0f}   // Nguy hại (Hazardous)
};

int tinyml_compute_vn_aqi(float pm25_ugm3) {
    if (pm25_ugm3 <= 0.0f) {
        return 0;
    }
    if (pm25_ugm3 >= 500.0f) {
        return 500;
    }

    size_t num_bp = sizeof(QCVN_BREAKPOINTS) / sizeof(QCVN_BREAKPOINTS[0]);
    for (size_t i = 0; i < num_bp; ++i) {
        if (pm25_ugm3 <= QCVN_BREAKPOINTS[i].c_hi) {
            float c_lo = QCVN_BREAKPOINTS[i].c_lo;
            float c_hi = QCVN_BREAKPOINTS[i].c_hi;
            float i_lo = QCVN_BREAKPOINTS[i].i_lo;
            float i_hi = QCVN_BREAKPOINTS[i].i_hi;

            float aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25_ugm3 - c_lo) + i_lo;
            int aqi_int = (int)roundf(aqi);
            if (aqi_int < 0) aqi_int = 0;
            if (aqi_int > 500) aqi_int = 500;
            return aqi_int;
        }
    }

    return 500;
}

const char *tinyml_get_aqi_category(int aqi) {
    if (aqi <= 50)  return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for Sensitive";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very Unhealthy";
    return "Hazardous";
}

// ============================================================================
// LCD SPI Screen Buffer Formatter
// ============================================================================

void lcd_format_screen_buffer(const EdgeDisplayPayload *payload, char *out_buf, size_t buf_size) {
    if (!out_buf || buf_size == 0) {
        return;
    }

    if (!payload) {
        snprintf(out_buf, buf_size, "[LCD] NO PAYLOAD");
        return;
    }

    const char *cat = tinyml_get_aqi_category(payload->aqi);
    const char *st = (payload->status_msg && payload->status_msg[0] != '\0')
                     ? payload->status_msg : "NORMAL";

    snprintf(out_buf, buf_size,
        "=== AIR QUALITY NODE ===\n"
        "Cycle: #%u | Status: %s\n"
        "PM2.5 Calib: %.1f ug/m3 (Raw: %.1f)\n"
        "VN-AQI: %d [%s]\n"
        "Temp: %.1f C | RH: %.1f %%\n"
        "CO2: %.0f ppm | Gas: %.0f\n",
        (unsigned int)payload->boot_cycle,
        st,
        payload->pm25_calibrated,
        payload->pm25_raw,
        payload->aqi,
        cat,
        payload->temp_c,
        payload->rh_percent,
        payload->co2_ppm,
        payload->gas_proxy
    );
}

// ============================================================================
// MQTT Telemetry JSON Formatter
// ============================================================================

int mqtt_format_telemetry_json(const EdgeTelemetryPacket *pkt, char *out_json, size_t max_len) {
    if (!pkt || !out_json || max_len == 0) {
        return -1;
    }

    bool is_backfill = (pkt->flags & TELEMETRY_FLAG_BACKFILL) != 0;

    int written = snprintf(out_json, max_len,
        "{\"station_id\":\"esp32s3_node_01\","
        "\"timestamp\":%u,"
        "\"pm1_raw\":%.2f,"
        "\"pm2_raw\":%.2f,"
        "\"pm25_calibrated\":%.2f,"
        "\"temp_c\":%.2f,"
        "\"rh_percent\":%.2f,"
        "\"co2_ppm\":%.1f,"
        "\"aqi\":%u,"
        "\"anomaly_code\":%u,"
        "\"flags\":%u,"
        "\"is_backfill\":%s}",
        (unsigned int)pkt->timestamp,
        pkt->pm1_raw,
        pkt->pm2_raw,
        pkt->pm25_calibrated,
        pkt->temp_c,
        pkt->rh_percent,
        pkt->co2_ppm,
        (unsigned int)pkt->aqi,
        (unsigned int)pkt->anomaly_code,
        (unsigned int)pkt->flags,
        is_backfill ? "true" : "false"
    );

    if (written < 0 || (size_t)written >= max_len) {
        return -1;
    }

    return written;
}

// ============================================================================
// Offline Circular Ring Buffer (Static Memory, Zero Dynamic Allocation)
// ============================================================================

static EdgeTelemetryPacket s_ringbuf[OFFLINE_RINGBUF_CAPACITY];
static size_t s_head = 0;
static size_t s_tail = 0;
static size_t s_count = 0;
static size_t s_dropped_count = 0;

bool offline_ringbuf_push(const EdgeTelemetryPacket *pkt) {
    if (!pkt || s_count >= OFFLINE_RINGBUF_CAPACITY) {
        return false;
    }
    s_ringbuf[s_head] = *pkt;
    s_head = (s_head + 1) % OFFLINE_RINGBUF_CAPACITY;
    s_count++;
    return true;
}

bool offline_ringbuf_push_overwrite(const EdgeTelemetryPacket *pkt) {
    if (!pkt) {
        return false;
    }

    if (s_count >= OFFLINE_RINGBUF_CAPACITY) {
        // Buffer full: drop oldest sample to maintain latest telemetry stream
        s_tail = (s_tail + 1) % OFFLINE_RINGBUF_CAPACITY;
        s_dropped_count++;
        s_count--;
    }

    s_ringbuf[s_head] = *pkt;
    if (s_dropped_count > 0) {
        s_ringbuf[s_head].flags |= TELEMETRY_FLAG_OVERFLOW_DROP;
    }
    s_head = (s_head + 1) % OFFLINE_RINGBUF_CAPACITY;
    s_count++;
    return true;
}

bool offline_ringbuf_pop(EdgeTelemetryPacket *out_pkt) {
    if (!out_pkt || s_count == 0) {
        return false;
    }
    *out_pkt = s_ringbuf[s_tail];
    s_tail = (s_tail + 1) % OFFLINE_RINGBUF_CAPACITY;
    s_count--;
    return true;
}

size_t offline_ringbuf_count(void) {
    return s_count;
}

size_t offline_ringbuf_capacity(void) {
    return OFFLINE_RINGBUF_CAPACITY;
}

bool offline_ringbuf_is_empty(void) {
    return (s_count == 0);
}

bool offline_ringbuf_is_full(void) {
    return (s_count >= OFFLINE_RINGBUF_CAPACITY);
}

void offline_ringbuf_clear(void) {
    s_head = 0;
    s_tail = 0;
    s_count = 0;
    s_dropped_count = 0;
}

size_t offline_ringbuf_dropped_count(void) {
    return s_dropped_count;
}

void offline_ringbuf_reset_dropped(void) {
    s_dropped_count = 0;
}
