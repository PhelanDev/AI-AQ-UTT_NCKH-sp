#ifndef BOARD_PINS_H
#define BOARD_PINS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ESP32-S3-WROOM-1 / N16R8 Hardware Pinout Definitions (BOM Locked)
// ============================================================================

// Dual Plantower PMS7003 Optical Laser Dust Sensors
#define PIN_PMS1_RX     17  // Hardware UART1 RX
#define PIN_PMS1_TX     18  // Hardware UART1 TX
#define PIN_PMS2_RX     15  // Hardware UART2 RX (Cross-checking channel)
#define PIN_PMS2_TX     16  // Hardware UART2 TX
#define PIN_PMS_SET      6  // RTC GPIO6: Sleep/Wakeup control for PMS fan & laser

// SenseAir S8 LP/0053 CO2 NDIR Sensor (Modbus RTU)
#define PIN_S8_RX        8  // UART3 / Modbus RX
#define PIN_S8_TX        9  // UART3 / Modbus TX

// Sensirion SHT40 (Temp/RH) & SGP41 (TVOC/NOx) Environmental Sensors
#define PIN_I2C_SDA      4  // I2C0 SDA
#define PIN_I2C_SCL      5  // I2C0 SCL

// 2.8" SPI TFT LCD (ST7789)
#define PIN_LCD_CS      10  // SPI Chip Select
#define PIN_LCD_MOSI    11  // SPI MOSI
#define PIN_LCD_SCK     12  // SPI Clock
#define PIN_LCD_DC      13  // Data / Command
#define PIN_LCD_RST     14  // Hardware Reset
#define PIN_LCD_BL       7  // RTC GPIO7: Backlight control (PWM / Digital)

// SIMCOM A7677S 4G LTE Cat-1 + GNSS Module
#define PIN_SIM_RX      19  // UART AT RX
#define PIN_SIM_TX      20  // UART AT TX

// Onboard WS2812B Addressable RGB LED (AQI Status Indicator)
#define PIN_LED_RGB     48  // RMT 1-Wire RGB LED

// ============================================================================
// 15-Minute Ultra-Low-Power Duty Cycle Timing Constants (900 seconds total)
// ============================================================================
#define ESP_CYCLE_TOTAL_SEC     900  // 15 minutes total cycle
#define ESP_WARMUP_SEC           30  // 30s fan & laser warmup for laminar airflow
#define ESP_ACTIVE_PROCESS_SEC    2  // 2s measurement, TinyML inference, LCD & MQTT
#define ESP_DEEP_SLEEP_SEC      868  // 14m 28s deep sleep (RTC Timer wakeup)

// Display Operating Mode (1 = Always-On 24/7 Display for 5V Mains/Demo; 0 = Battery Sleep)
#define CONFIG_LCD_ALWAYS_ON    1

#ifdef __cplusplus
}

// ============================================================================
// Hardware Static Assertions & Octal PSRAM (GPIO33..37) Conflict Prevention
// ============================================================================
#define IS_OCTAL_PSRAM_PIN(pin) ((pin) >= 33 && (pin) <= 37)

static_assert(!IS_OCTAL_PSRAM_PIN(PIN_PMS1_RX), "PIN_PMS1_RX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_PMS1_TX), "PIN_PMS1_TX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_PMS2_RX), "PIN_PMS2_RX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_PMS2_TX), "PIN_PMS2_TX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_PMS_SET), "PIN_PMS_SET collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_S8_RX),   "PIN_S8_RX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_S8_TX),   "PIN_S8_TX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_I2C_SDA), "PIN_I2C_SDA collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_I2C_SCL), "PIN_I2C_SCL collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_CS),  "PIN_LCD_CS collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_MOSI),"PIN_LCD_MOSI collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_SCK), "PIN_LCD_SCK collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_DC),  "PIN_LCD_DC collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_RST), "PIN_LCD_RST collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LCD_BL),  "PIN_LCD_BL collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_SIM_RX),  "PIN_SIM_RX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_SIM_TX),  "PIN_SIM_TX collides with Octal PSRAM GPIO33..37");
static_assert(!IS_OCTAL_PSRAM_PIN(PIN_LED_RGB), "PIN_LED_RGB collides with Octal PSRAM GPIO33..37");

static_assert(ESP_WARMUP_SEC + ESP_ACTIVE_PROCESS_SEC + ESP_DEEP_SLEEP_SEC == ESP_CYCLE_TOTAL_SEC,
              "Duty cycle components must sum exactly to ESP_CYCLE_TOTAL_SEC (900s)");

#endif // __cplusplus

#endif // BOARD_PINS_H
