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

// Embedded 5x7 Standard ASCII Bitmap Font (ASCII 32 ' ' to 90 'Z')
static const uint8_t s_font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, // 32 ' '
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // 33 '!'
    {0x00, 0x07, 0x00, 0x07, 0x00}, // 34 '"'
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, // 35 '#'
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // 36 '$'
    {0x23, 0x13, 0x08, 0x64, 0x62}, // 37 '%'
    {0x36, 0x49, 0x55, 0x22, 0x50}, // 38 '&'
    {0x00, 0x05, 0x03, 0x00, 0x00}, // 39 '\''
    {0x00, 0x1C, 0x22, 0x41, 0x00}, // 40 '('
    {0x00, 0x41, 0x22, 0x1C, 0x00}, // 41 ')'
    {0x08, 0x2A, 0x1C, 0x2A, 0x08}, // 42 '*'
    {0x08, 0x08, 0x3E, 0x08, 0x08}, // 43 '+'
    {0x00, 0x50, 0x30, 0x00, 0x00}, // 44 ','
    {0x08, 0x08, 0x08, 0x08, 0x08}, // 45 '-'
    {0x00, 0x60, 0x60, 0x00, 0x00}, // 46 '.'
    {0x20, 0x10, 0x08, 0x04, 0x02}, // 47 '/'
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 48 '0'
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 49 '1'
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 50 '2'
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 51 '3'
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 52 '4'
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 53 '5'
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 54 '6'
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 55 '7'
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 56 '8'
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 57 '9'
    {0x00, 0x36, 0x36, 0x00, 0x00}, // 58 ':'
    {0x00, 0x56, 0x36, 0x00, 0x00}, // 59 ';'
    {0x08, 0x14, 0x22, 0x41, 0x00}, // 60 '<'
    {0x14, 0x14, 0x14, 0x14, 0x14}, // 61 '='
    {0x00, 0x41, 0x22, 0x14, 0x08}, // 62 '>'
    {0x02, 0x01, 0x51, 0x09, 0x06}, // 63 '?'
    {0x32, 0x49, 0x79, 0x41, 0x3E}, // 64 '@'
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, // 65 'A'
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // 66 'B'
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // 67 'C'
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // 68 'D'
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // 69 'E'
    {0x7F, 0x09, 0x09, 0x09, 0x01}, // 70 'F'
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, // 71 'G'
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // 72 'H'
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // 73 'I'
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // 74 'J'
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // 75 'K'
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // 76 'L'
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // 77 'M'
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // 78 'N'
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // 79 'O'
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // 80 'P'
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // 81 'Q'
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // 82 'R'
    {0x46, 0x49, 0x49, 0x49, 0x31}, // 83 'S'
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // 84 'T'
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // 85 'U'
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // 86 'V'
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, // 87 'W'
    {0x63, 0x14, 0x08, 0x14, 0x63}, // 88 'X'
    {0x07, 0x08, 0x70, 0x08, 0x07}, // 89 'Y'
    {0x61, 0x51, 0x49, 0x45, 0x43}, // 90 'Z'
};

void st7789_draw_char(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg, uint8_t size) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c < 32 || c > 90) return;
    const uint8_t *glyph = s_font5x7[c - 32];
    for (uint8_t col = 0; col < 5; col++) {
        uint8_t line = glyph[col];
        for (uint8_t row = 0; row < 7; row++) {
            if (line & (1 << row)) {
                st7789_fill_rect(x + col * size, y + row * size, size, size, color);
            }
        }
    }
}

void st7789_draw_string(uint16_t x, uint16_t y, const char *str, uint16_t color, uint16_t bg, uint8_t size) {
    if (!str) return;
    uint16_t glyph_x = x;
    while (*str) {
        st7789_draw_char(glyph_x, y, *str, color, bg, size);
        glyph_x += 6 * size;
        str++;
    }
}

void st7789_render_dashboard(const EdgeDisplayPayload *display_data) {
    if (!display_data) return;

    s_mock_state.rendered_aqi = display_data->aqi;
    s_mock_state.rendered_pm25 = display_data->pm25_calibrated;

    // 1. Clear background to Dark Grey / Black
    st7789_fill_rect(0, 0, ST7789_LCD_WIDTH, ST7789_LCD_HEIGHT, RGB565_BLACK);

    // 2. Top Header Bar (Height = 35px)
    st7789_fill_rect(0, 0, ST7789_LCD_WIDTH, 35, RGB565_NAVY);
    st7789_draw_string(15, 10, "AQ-LITE UTT-54TK", RGB565_WHITE, RGB565_NAVY, 2);

    // 3. Central Dynamic VN-AQI Indicator Banner (Height = 70px)
    uint16_t aqi_color = st7789_get_aqi_color_rgb565(display_data->aqi);
    st7789_fill_rect(10, 45, ST7789_LCD_WIDTH - 20, 70, aqi_color);
    char aqi_str[32];
    snprintf(aqi_str, sizeof(aqi_str), "VN-AQI: %d", display_data->aqi);
    st7789_draw_string(25, 68, aqi_str, RGB565_BLACK, aqi_color, 3);

    // 4. Data Grid Cards (PM2.5, SHT40 T/RH, SenseAir CO2)
    // Left Column: PM2.5 Calibrated & Raw
    st7789_fill_rect(10, 125, 105, 50, RGB565_DARKGREY);
    char pm_str[32];
    snprintf(pm_str, sizeof(pm_str), "PM:%.1f", display_data->pm25_calibrated);
    st7789_draw_string(15, 140, pm_str, RGB565_WHITE, RGB565_DARKGREY, 2);

    // Right Column: SHT40 Temperature & Humidity
    st7789_fill_rect(125, 125, 105, 50, RGB565_DARKGREY);
    char env_str[32];
    snprintf(env_str, sizeof(env_str), "%.0fC/%.0f%%", display_data->temp_c, display_data->rh_percent);
    st7789_draw_string(130, 140, env_str, RGB565_CYAN, RGB565_DARKGREY, 2);

    // Bottom Row: CO2 & Gas Index
    st7789_fill_rect(10, 185, 105, 45, RGB565_DARKGREY);
    char co2_str[32];
    snprintf(co2_str, sizeof(co2_str), "CO2:%.0f", display_data->co2_ppm);
    st7789_draw_string(15, 198, co2_str, RGB565_WHITE, RGB565_DARKGREY, 2);

    st7789_fill_rect(125, 185, 105, 45, RGB565_DARKGREY);
    st7789_draw_string(135, 198, "GAS:OK", RGB565_WHITE, RGB565_DARKGREY, 2);

    // 5. System Health Status Bar (Height = 30px)
    st7789_fill_rect(0, ST7789_LCD_HEIGHT - 30, ST7789_LCD_WIDTH, 30, RGB565_NAVY);
    st7789_draw_string(20, ST7789_LCD_HEIGHT - 22, "STATUS: HEALTHY", RGB565_WHITE, RGB565_NAVY, 1);
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
