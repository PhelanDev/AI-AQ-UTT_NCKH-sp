#ifndef POWER_SCHEDULER_H
#define POWER_SCHEDULER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "tinyml_edge_engine.h"
#include "display_mqtt_hal.h"
#include "board_pins.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 15-Minute Ultra-Low-Power Finite State Machine (FSM) States
// ============================================================================

typedef enum {
    EDGE_STATE_WAKEUP          = 0, // Wake up from RTC timer, initialize peripherals, increment boot counter
    EDGE_STATE_WARMUP          = 1, // PIN_PMS_SET (GPIO6) = HIGH (fan & laser ON), 30s laminar flow stabilization
    EDGE_STATE_ACQUIRE_INFER   = 2, // Acquire sensor samples, execute TinyML inference (<0.35ms), update drift
    EDGE_STATE_DISPLAY_PUBLISH = 3, // Render to ST7789 SPI LCD and publish JSON telemetry (or offline ring buffer)
    EDGE_STATE_PREPARE_SLEEP   = 4, // PIN_PMS_SET = LOW, rtc_gpio_hold_en, configure RTC wakeup timer for 868s
    EDGE_STATE_DEEP_SLEEP      = 5  // Invoke esp_deep_sleep_start()
} EdgeFsmState;

// Raw multisensor sample acquired during ACTIVE state
typedef struct {
    uint32_t timestamp;  // Unix epoch or uptime timestamp (seconds)
    float pm1_raw;       // Primary PMS7003 atmospheric PM2.5 (ug/m3)
    float pm2_raw;       // Cross-checking PMS7003 atmospheric PM2.5 (ug/m3)
    float pm10_raw;      // Atmospheric PM10 (ug/m3)
    float temp_c;        // Sensirion SHT40 Temperature (deg C)
    float rh_percent;    // Sensirion SHT40 Relative Humidity (%)
    float co2_ppm;       // SenseAir S8 NDIR CO2 (ppm)
    float gas_proxy;     // Sensirion SGP41 Raw Gas Index / VOC proxy
} RawSensorSample;

// ============================================================================
// Burst Sampling & Robust Outlier Denoising (30 Samples / 20% Trimmed Mean)
// ============================================================================
#define SENSOR_BURST_SAMPLE_COUNT 30
#define SENSOR_BURST_TRIM_COUNT   3

typedef struct {
    RawSensorSample samples[SENSOR_BURST_SAMPLE_COUNT];
    uint8_t count;
    bool is_full;
} SensorBurstAccumulator;

/**
 * @brief Resets and initializes the burst sample accumulator (0 Byte Heap).
 */
void burst_accumulator_init(SensorBurstAccumulator *acc);

/**
 * @brief Pushes a raw sensor sample into the burst accumulator.
 * @return true if accumulator reached SENSOR_BURST_SAMPLE_COUNT; false otherwise.
 */
bool burst_accumulator_push(SensorBurstAccumulator *acc, const RawSensorSample *sample);

/**
 * @brief Robust Outlier Denoising via 20% Trimmed Mean:
 * For each sensor channel, sorts all samples, discards trim_count min drops and
 * trim_count max spikes, and calculates arithmetic mean of the core remaining samples.
 *
 * @param acc Pointer to accumulator containing samples.
 * @param trim_count Number of minimum and maximum outliers to discard per channel.
 * @param out_filtered_sample Output pointer receiving clean representative sample.
 * @return true if filtering succeeded; false if insufficient samples (count <= 2*trim_count).
 */
bool burst_filter_trimmed_mean(const SensorBurstAccumulator *acc,
                               uint8_t trim_count,
                               RawSensorSample *out_filtered_sample);

// Complete result output of 15-minute duty cycle execution
typedef struct {
    TinyML_AnomalyCode anomaly_code;    // Anomaly status from warmup, drift, or spike checks
    float pm25_calibrated;              // TinyML calibrated PM2.5 concentration (ug/m3)
    int vn_aqi;                         // Vietnam Air Quality Index (0..500)
    bool lcd_updated;                   // Whether ST7789 LCD frame buffer was updated
    bool telemetry_sent;                // Whether telemetry packet was sent to MQTT broker
    bool entered_deep_sleep;            // Whether system configured and entered deep sleep
    uint32_t deep_sleep_duration_sec;   // Programmed deep sleep duration (nominally 868s)
    bool pms_set_pin_level;             // Final logic state of PIN_PMS_SET (GPIO6)
    uint16_t offline_buffered_count;    // Number of packets currently buffered in offline ring buffer
    uint16_t offline_backfilled_count;  // Number of buffered packets flushed/backfilled upon reconnection
} EdgeCycleResult;

/**
 * @brief Executes one full 15-minute ultra-low-power duty cycle through the FSM.
 */
EdgeCycleResult edge_execute_15min_duty_cycle(const RawSensorSample *raw_sample,
                                             float warmup_elapsed_sec,
                                             bool mqtt_connected);

/**
 * @brief Executes one full 15-minute cycle with automatic 30-sample burst denoising.
 * Applies 20% trimmed mean to filter outliers before invoking FSM.
 */
EdgeCycleResult edge_execute_15min_duty_cycle_burst(const SensorBurstAccumulator *burst_acc,
                                                    float warmup_elapsed_sec,
                                                    bool mqtt_connected);

/**
 * @brief Returns the current state of the power scheduler FSM.
 */
EdgeFsmState power_scheduler_get_state(void);

/**
 * @brief Sets the current state of the power scheduler FSM.
 */
void power_scheduler_set_state(EdgeFsmState state);

/**
 * @brief Returns string description for an FSM state.
 */
const char *power_scheduler_state_str(EdgeFsmState state);

#ifdef __cplusplus
}
#endif

#endif // POWER_SCHEDULER_H
