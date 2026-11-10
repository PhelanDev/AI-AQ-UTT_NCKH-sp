#include "st7789_lcd_driver.h"
#include "board_pins.h"
#include <stdio.h>
#include <string.h>

// ============================================================================
// Host PC Mock State Implementation
// ============================================================================
static ST7789MockState s_mock_state = {
    false, // is_initialized
    false, // backlight_on
    false, // in_sleep_mode
    0, 0, 0, 0,
    0,     // last_fill_color
    0,     // total_pixels_written
    0,     // rendered_aqi
    0.0f   // rendered_pm25
};

const ST7789MockState *st7789_get_mock_state(void) {
    return &s_mock_state;
}

void st7789_reset_mock_state(void) {
    memset(&s_mock_state, 0, sizeof(s_mock_state));
}

// ============================================================================
// VN-AQI Dynamic Color Palette Mapping
// ============================================================================
uint16_t st7789_get_aqi_color_rgb565(int aqi) {
    if (aqi <= 50) {
        return RGB565_AQI_GOOD;           // 0..50: Green
    } else if (aqi <= 100) {
        return RGB565_AQI_MODERATE;       // 51..100: Yellow
    } else if (aqi <= 150) {
        return RGB565_AQI_UNHEALTHY_S;    // 101..150: Orange
    } else if (aqi <= 200) {
        return RGB565_AQI_UNHEALTHY;      // 151..200: Red
    } else if (aqi <= 300) {
        return RGB565_AQI_VERY_UNHEALTHY; // 201..300: Purple
    } else {
        return RGB565_AQI_HAZARDOUS;      // 301..500: Maroon / Brown
    }
}

// ============================================================================
// Hardware Platform Specific Implementations (ESP32 SPI)
// ============================================================================
#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
#include <Arduino.h>
#include <SPI.h>

static inline void st7789_write_cmd(uint8_t cmd) {
    digitalWrite(PIN_LCD_DC, LOW);
    digitalWrite(PIN_LCD_CS, LOW);
    SPI.transfer(cmd);
    digitalWrite(PIN_LCD_CS, HIGH);
}

static inline void st7789_write_data(uint8_t data) {
    digitalWrite(PIN_LCD_DC, HIGH);
    digitalWrite(PIN_LCD_CS, LOW);
    SPI.transfer(data);
    digitalWrite(PIN_LCD_CS, HIGH);
}

static inline void st7789_write_data16(uint16_t data) {
    digitalWrite(PIN_LCD_DC, HIGH);
    digitalWrite(PIN_LCD_CS, LOW);
    SPI.transfer((uint8_t)(data >> 8));
    SPI.transfer((uint8_t)(data & 0xFF));
    digitalWrite(PIN_LCD_CS, HIGH);
}
#endif
#endif

// ============================================================================
// Public Display Controls (Zero Dynamic Allocation)
// ============================================================================

void st7789_set_backlight(bool on) {
    s_mock_state.backlight_on = on;
#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    pinMode(PIN_LCD_BL, OUTPUT);
    digitalWrite(PIN_LCD_BL, on ? HIGH : LOW);
#endif
#endif
}

void st7789_init(void) {
    s_mock_state.is_initialized = true;
    s_mock_state.in_sleep_mode = false;
    s_mock_state.backlight_on = true;

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    pinMode(PIN_LCD_CS, OUTPUT);
    pinMode(PIN_LCD_DC, OUTPUT);
    pinMode(PIN_LCD_RST, OUTPUT);
    pinMode(PIN_LCD_BL, OUTPUT);

    digitalWrite(PIN_LCD_CS, HIGH);
    digitalWrite(PIN_LCD_DC, HIGH);

    // Hardware reset pulse
    digitalWrite(PIN_LCD_RST, HIGH);
    delay(10);
    digitalWrite(PIN_LCD_RST, LOW);
    delay(20);
    digitalWrite(PIN_LCD_RST, HIGH);
    delay(50);

    // Initialize SPI bus (SPI2 / FSPI) @ 40 MHz
    SPI.begin(PIN_LCD_SCK, -1, PIN_LCD_MOSI, PIN_LCD_CS);
    SPI.beginTransaction(SPISettings(40000000, MSBFIRST, SPI_MODE0));

    // ST7789 Initialization Command Sequence
    st7789_write_cmd(ST7789_CMD_SWRESET);
    delay(150);

    st7789_write_cmd(ST7789_CMD_SLPOUT);
    delay(120);

    st7789_write_cmd(ST7789_CMD_COLMOD);
    st7789_write_data(0x55); // 16-bit/pixel (RGB565)

    st7789_write_cmd(ST7789_CMD_MADCTL);
    st7789_write_data(0x00); // Standard portrait RGB

    st7789_write_cmd(ST7789_CMD_INVON); // Display Inversion On for crisp colors
    delay(10);

    st7789_write_cmd(ST7789_CMD_DISPON);
    delay(10);

    st7789_set_backlight(true);
#endif
#endif
}

void st7789_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    s_mock_state.last_window_x0 = x0;
    s_mock_state.last_window_y0 = y0;
    s_mock_state.last_window_x1 = x1;
    s_mock_state.last_window_y1 = y1;

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    st7789_write_cmd(ST7789_CMD_CASET);
    st7789_write_data16(x0);
    st7789_write_data16(x1);

    st7789_write_cmd(ST7789_CMD_RASET);
    st7789_write_data16(y0);
    st7789_write_data16(y1);

    st7789_write_cmd(ST7789_CMD_RAMWR);
#endif
#endif
}

void st7789_fill_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color) {
    if (w == 0 || h == 0) return;

    s_mock_state.last_fill_color = color;
    s_mock_state.total_pixels_written += (uint32_t)w * (uint32_t)h;

    st7789_set_window(x, y, x + w - 1, y + h - 1);

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    digitalWrite(PIN_LCD_DC, HIGH);
    digitalWrite(PIN_LCD_CS, LOW);

    uint8_t high_b = (uint8_t)(color >> 8);
    uint8_t low_b  = (uint8_t)(color & 0xFF);
    uint32_t total = (uint32_t)w * (uint32_t)h;

    for (uint32_t i = 0; i < total; ++i) {
        SPI.transfer(high_b);
        SPI.transfer(low_b);
    }

    digitalWrite(PIN_LCD_CS, HIGH);
#endif
#endif
}

void st7789_render_dashboard(const EdgeDisplayPayload *payload) {
    if (!payload) return;

    s_mock_state.rendered_aqi = payload->aqi;
    s_mock_state.rendered_pm25 = payload->pm25_calibrated;

    // 1. Clear background to Dark Grey / Black
    st7789_fill_rect(0, 0, ST7789_LCD_WIDTH, ST7789_LCD_HEIGHT, RGB565_BLACK);

    // 2. Top Header Bar (Height = 35px)
    st7789_fill_rect(0, 0, ST7789_LCD_WIDTH, 35, RGB565_NAVY);

    // 3. Central Dynamic VN-AQI Indicator Banner (Height = 70px)
    uint16_t aqi_color = st7789_get_aqi_color_rgb565(payload->aqi);
    st7789_fill_rect(10, 45, ST7789_LCD_WIDTH - 20, 70, aqi_color);

    // 4. Data Grid Cards (PM2.5, SHT40 T/RH, SenseAir CO2)
    // Left Column: PM2.5 Calibrated & Raw
    st7789_fill_rect(10, 125, 105, 50, RGB565_DARKGREY);
    // Right Column: SHT40 Temperature & Humidity
    st7789_fill_rect(125, 125, 105, 50, RGB565_DARKGREY);

    // Bottom Row: CO2 & Gas Index
    st7789_fill_rect(10, 185, 105, 45, RGB565_DARKGREY);
    st7789_fill_rect(125, 185, 105, 45, RGB565_DARKGREY);

    // 5. System Health Status Bar (Height = 30px)
    st7789_fill_rect(0, ST7789_LCD_HEIGHT - 30, ST7789_LCD_WIDTH, 30, RGB565_NAVY);
}

void st7789_enter_sleep(void) {
#if CONFIG_LCD_ALWAYS_ON
    // In Always-On 24/7 mode (Station powered by 5V mains), keep backlight active and hold GRAM display
    s_mock_state.in_sleep_mode = false;
    s_mock_state.backlight_on = true;
#else
    // 1. Turn off backlight immediately to save battery
    st7789_set_backlight(false);

    // 2. Send ST7789 Sleep In command
#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#if defined(ARDUINO)
    st7789_write_cmd(ST7789_CMD_SLPIN);
    delay(5);
#endif
#endif

    s_mock_state.in_sleep_mode = true;
#endif
}
