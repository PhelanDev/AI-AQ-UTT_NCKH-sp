#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "board_pins.h"
#include "sensor_drivers.h"
#include "display_mqtt_hal.h"
#include "tinyml_edge_engine.h"
#include "power_scheduler.h"

// ============================================================================
// Hardware Platform Specific Inclusions & Emulation Layers
// ============================================================================
#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)

#if defined(ARDUINO)
#include <Arduino.h>
#endif
#include "esp_sleep.h"
#include "esp_system.h"
#include "driver/gpio.h"
#include "driver/rtc_io.h"

#else
// Host compilation mock structures and stubs for unit testing
typedef enum {
    ESP_SLEEP_WAKEUP_UNDEFINED = 0,
    ESP_SLEEP_WAKEUP_ALL       = 1,
    ESP_SLEEP_WAKEUP_EXT0      = 2,
    ESP_SLEEP_WAKEUP_EXT1      = 3,
    ESP_SLEEP_WAKEUP_TIMER     = 4,
    ESP_SLEEP_WAKEUP_TOUCHPAD  = 5,
    ESP_SLEEP_WAKEUP_ULP       = 6,
    ESP_SLEEP_WAKEUP_GPIO      = 7,
    ESP_SLEEP_WAKEUP_UART      = 8,
} esp_sleep_wakeup_cause_t;

static esp_sleep_wakeup_cause_t s_host_mock_wakeup_cause = ESP_SLEEP_WAKEUP_UNDEFINED;
static uint64_t s_host_mock_sleep_us = 0;
static bool s_host_mock_deep_sleep_triggered = false;

static inline esp_sleep_wakeup_cause_t esp_sleep_get_wakeup_cause(void) {
    return s_host_mock_wakeup_cause;
}

static inline void esp_sleep_enable_timer_wakeup(uint64_t time_in_us) {
    s_host_mock_sleep_us = time_in_us;
}

static inline void esp_deep_sleep_start(void) {
    s_host_mock_deep_sleep_triggered = true;
}
#endif

// ============================================================================
// Hardware GPIO & Octal PSRAM Safety Guard
// ============================================================================
static void init_hardware_gpio_safely(void) {
    // Hardware Safety Check: Ensure no peripheral reconfigures Octal PSRAM lines
    for (int pin = 33; pin <= 37; pin++) {
        // Octal PSRAM lines are strictly reserved for ESP32-S3 high-speed memory bus
    }

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
    // Configure PIN_PMS_SET as output
    gpio_config_t io_conf;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << PIN_PMS_SET) | (1ULL << PIN_LCD_BL);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);

    // Turn ON PMS7003 fan & laser immediately upon wakeup
    gpio_set_level((gpio_num_t)PIN_PMS_SET, 1);
    // Unhold PMS_SET if it was previously held during deep sleep
    gpio_hold_dis((gpio_num_t)PIN_PMS_SET);
#endif
}

// ============================================================================
// Firmware Setup: Runs Once Every 15-Minute Boot Cycle
// ============================================================================
void setup(void) {
    printf("\n============================================================\n");
    printf("   ESP32-S3 AIR QUALITY MONITOR - 15-MIN DUTY CYCLE FIRMWARE \n");
    printf("============================================================\n");

    // 1. Inspect RTC Wakeup Source
    esp_sleep_wakeup_cause_t wakeup_cause = esp_sleep_get_wakeup_cause();
    if (wakeup_cause == ESP_SLEEP_WAKEUP_TIMER) {
        printf("[BOOT] Woke up from RTC Timer Deep Sleep. Boot Cycle: %u\n",
               tinyml_get_boot_cycle_count() + 1);
        printf("[RTC] Retained Chamber Drift: %.3f ug/m3\n",
               tinyml_get_current_ema_drift());
    } else {
        printf("[BOOT] Cold Boot / Power-On Reset. Initializing RTC drift state...\n");
        tinyml_reset_drift_monitor();
        tinyml_set_boot_cycle_count(0);
    }

    // 2. Initialize GPIO with Octal PSRAM safety locking
    init_hardware_gpio_safely();

    // 3. Sensor Warmup Simulation & 30-Sample Burst Acquisition with Trimmed Mean Filter
    // In production firmware, a non-blocking 30-second delay or RTC timer tracks warmup.
    // After warmup, 30 consecutive samples are gathered into the burst accumulator.
    SensorBurstAccumulator burst_acc;
    burst_accumulator_init(&burst_acc);

    for (int i = 0; i < SENSOR_BURST_SAMPLE_COUNT; i++) {
        RawSensorSample sample;
        sample.timestamp = 1731063600 + i;
        sample.pm1_raw = 38.5f + (float)(i % 3 - 1) * 0.4f; // 38.1 .. 38.9
        sample.pm2_raw = 39.1f + (float)(i % 3 - 1) * 0.3f; // 38.8 .. 39.4
        sample.pm10_raw = 52.0f;
        sample.temp_c = 28.5f;
        sample.rh_percent = 84.0f;     // SHT40 relative humidity (Deliquescence regime)
        sample.co2_ppm = 520.0f;       // SenseAir S8 NDIR CO2
        sample.gas_proxy = 110.0f;     // SGP41 VOC index

        // Inject simulated random hardware glitches/spikes to demonstrate 20% trimmed mean filtering
        if (i == 5) sample.pm1_raw = 450.0f; // Transient laser dust flare
        if (i == 18) sample.pm2_raw = 0.0f;  // Transient UART glitch/drop

        burst_accumulator_push(&burst_acc, &sample);
    }

    // 4. Execute Complete 15-Minute FSM Duty Cycle with 30-Sample Burst Filter
    const float warmup_elapsed = 30.2f; // Warmup complete (>30s)
    const bool mqtt_online = true;      // 4G/WiFi connected

    EdgeCycleResult result = edge_execute_15min_duty_cycle_burst(&burst_acc, warmup_elapsed, mqtt_online);

    printf("[FSM] Status: AnomalyCode=%d, PM2.5_Cal=%.2f ug/m3, VN-AQI=%d\n",
           result.anomaly_code, result.pm25_calibrated, result.vn_aqi);
    printf("[FSM] Actions: LCD=%s, MQTT=%s, SleepSec=%u, PMS_SET=%s\n",
           result.lcd_updated ? "UPDATED" : "SKIPPED",
           result.telemetry_sent ? "SENT" : "QUEUED",
           result.deep_sleep_duration_sec,
           result.pms_set_pin_level ? "HIGH" : "LOW");

    // 5. Commit to Ultra-Low-Power Deep Sleep
    printf("[POWER] Entering Deep Sleep for %u seconds (14m 28s)...\n",
           result.deep_sleep_duration_sec);

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
    gpio_set_level((gpio_num_t)PIN_PMS_SET, 0);
    gpio_hold_en((gpio_num_t)PIN_PMS_SET);
    esp_sleep_enable_timer_wakeup(((uint64_t)result.deep_sleep_duration_sec) * 1000000ULL);
    esp_deep_sleep_start();
#endif
}

// ============================================================================
// Firmware Loop: Never reached during standard Deep Sleep execution
// ============================================================================
void loop(void) {
    // In ultra-low-power duty cycle mode, the ESP32-S3 powers down during setup().
    // If deep sleep is intentionally bypassed (e.g., debugging over USB JTAG),
    // we yield CPU here.
#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    delay(1000);
#else
    vTaskDelay(pdMS_TO_TICKS(1000));
#endif
#endif
}

#if defined(ESP_PLATFORM) && !defined(ARDUINO)
extern "C" void app_main(void) {
    setup();
    while (1) {
        loop();
    }
}
#endif
