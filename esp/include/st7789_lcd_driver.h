#ifndef ST7789_LCD_DRIVER_H
#define ST7789_LCD_DRIVER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "display_mqtt_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ST7789 SPI Display Dimensions & Configuration
// ============================================================================
#define ST7789_LCD_WIDTH         240
#define ST7789_LCD_HEIGHT        280  // Standard 2.8" or 240x280/240x320 panel

// ============================================================================
// ST7789 Command Set (Standard Sitronix Controller)
// ============================================================================
#define ST7789_CMD_NOP           0x00
#define ST7789_CMD_SWRESET       0x01
#define ST7789_CMD_SLPIN         0x10
#define ST7789_CMD_SLPOUT        0x11
#define ST7789_CMD_INVOFF        0x20
#define ST7789_CMD_INVON         0x21
#define ST7789_CMD_DISPOFF       0x28
#define ST7789_CMD_DISPON        0x29
#define ST7789_CMD_CASET         0x2A
#define ST7789_CMD_RASET         0x2B
#define ST7789_CMD_RAMWR         0x2C
#define ST7789_CMD_MADCTL        0x36
#define ST7789_CMD_COLMOD        0x3A

// ============================================================================
// Standard RGB565 Color Palette (16-bit: R5 G6 B5)
// ============================================================================
#define RGB565_BLACK             0x0000
#define RGB565_WHITE             0xFFFF
#define RGB565_DARKGREY          0x39E7
#define RGB565_NAVY              0x000F
#define RGB565_CYAN              0x07FF

// Vietnam VN-AQI Indicator Colors (QCVN 05:2023)
#define RGB565_AQI_GOOD          0x07E0  // Green (0..50)
#define RGB565_AQI_MODERATE      0xFFE0  // Yellow (51..100)
#define RGB565_AQI_UNHEALTHY_S   0xFD20  // Orange (101..150)
#define RGB565_AQI_UNHEALTHY     0xF800  // Red (151..200)
#define RGB565_AQI_VERY_UNHEALTHY 0x780F // Purple (201..300)
#define RGB565_AQI_HAZARDOUS     0x8200  // Maroon / Brown (301..500)

// ============================================================================
// Host PC Mock State for Zero-Hardware Unit Testing
// ============================================================================
typedef struct {
    bool is_initialized;
    bool backlight_on;
    bool in_sleep_mode;
    uint16_t last_window_x0;
    uint16_t last_window_y0;
    uint16_t last_window_x1;
    uint16_t last_window_y1;
    uint16_t last_fill_color;
    uint32_t total_pixels_written;
    int rendered_aqi;
    float rendered_pm25;
} ST7789MockState;

// ============================================================================
// ST7789 Public API (Zero Dynamic Heap Memory Allocation)
// ============================================================================

/**
 * @brief Returns 16-bit RGB565 color corresponding to VN-AQI breakpoint.
 *
 * @param aqi VN-AQI value [0..500].
 * @return 16-bit RGB565 color code.
 */
uint16_t st7789_get_aqi_color_rgb565(int aqi);

/**
 * @brief Initializes SPI2 bus and sends ST7789 initialization command sequence.
 */
void st7789_init(void);

/**
 * @brief Controls LCD backlight power via PIN_LCD_BL (GPIO7).
 *
 * @param on true to turn on backlight, false to turn off for power saving.
 */
void st7789_set_backlight(bool on);

/**
 * @brief Sets active drawing window coordinates (CASET / RASET).
 */
void st7789_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1);

/**
 * @brief Fills rectangular region with specified RGB565 color.
 */
void st7789_fill_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color);

/**
 * @brief Renders the complete real-time air quality dashboard onto ST7789 display.
 *
 * Layout:
 *  - Top bar: Node ID & RTC boot cycle
 *  - Central Banner: VN-AQI value with corresponding dynamic background color
 *  - Data Grid: PM2.5 (Calib & Raw), SHT40 (Temp & RH), SenseAir S8 (CO2)
 *  - Bottom Status bar: System health ("HEALTHY" / "WARMUP" / "DRIFT")
 *
 * @param payload Pointer to display payload data.
 */
void st7789_render_dashboard(const EdgeDisplayPayload *payload);

/**
 * @brief Puts display into ultra-low-power sleep mode (Turn off backlight, SLPIN 0x10).
 *        Called immediately before ESP32-S3 deep sleep entry.
 */
void st7789_enter_sleep(void);

/**
 * @brief Gets access to host mock state for unit test inspection.
 */
const ST7789MockState *st7789_get_mock_state(void);

/**
 * @brief Resets host mock state.
 */
void st7789_reset_mock_state(void);

#ifdef __cplusplus
}
#endif

#endif // ST7789_LCD_DRIVER_H
