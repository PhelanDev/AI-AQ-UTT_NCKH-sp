#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "board_pins.h"
#include "sensor_drivers.h"
#include "display_mqtt_hal.h"

// ============================================================================
// TEST 1: Board Pins & Hardware Protection (Octal PSRAM GPIO33..37 Safety)
// ============================================================================
static void test_board_pins_and_psram_safety(void) {
    printf("[TEST 1] Testing board pins & Octal PSRAM safety...\n");

    // PMS7003 Dual PM UARTS
    assert(PIN_PMS1_RX == 17);
    assert(PIN_PMS1_TX == 18);
    assert(PIN_PMS2_RX == 15);
    assert(PIN_PMS2_TX == 16);
    assert(PIN_PMS_SET == 6);

    // SenseAir S8 Modbus UART
    assert(PIN_S8_RX == 8);
    assert(PIN_S8_TX == 9);

    // I2C Sensors (SHT40 & SGP41)
    assert(PIN_I2C_SDA == 4);
    assert(PIN_I2C_SCL == 5);

    // LCD SPI (ST7789) & Backlight
    assert(PIN_LCD_CS == 10);
    assert(PIN_LCD_MOSI == 11);
    assert(PIN_LCD_SCK == 12);
    assert(PIN_LCD_DC == 13);
    assert(PIN_LCD_RST == 14);
    assert(PIN_LCD_BL == 7);

    // SIMCOM 4G & Status LED
    assert(PIN_SIM_RX == 19);
    assert(PIN_SIM_TX == 20);
    assert(PIN_LED_RGB == 48);

    // 15-Minute Duty Cycle Constants
    assert(ESP_CYCLE_TOTAL_SEC == 900);
    assert(ESP_WARMUP_SEC == 30);
    assert(ESP_ACTIVE_PROCESS_SEC == 2);
    assert(ESP_DEEP_SLEEP_SEC == 868);
    assert(ESP_WARMUP_SEC + ESP_ACTIVE_PROCESS_SEC + ESP_DEEP_SLEEP_SEC == ESP_CYCLE_TOTAL_SEC);

    // Ensure NO defined pin falls in the forbidden Octal PSRAM range [33..37]
    const int all_pins[] = {
        PIN_PMS1_RX, PIN_PMS1_TX, PIN_PMS2_RX, PIN_PMS2_TX, PIN_PMS_SET,
        PIN_S8_RX, PIN_S8_TX, PIN_I2C_SDA, PIN_I2C_SCL,
        PIN_LCD_CS, PIN_LCD_MOSI, PIN_LCD_SCK, PIN_LCD_DC, PIN_LCD_RST, PIN_LCD_BL,
        PIN_SIM_RX, PIN_SIM_TX, PIN_LED_RGB
    };
    for (size_t i = 0; i < sizeof(all_pins) / sizeof(all_pins[0]); ++i) {
        assert(all_pins[i] < 33 || all_pins[i] > 37);
    }

    printf("  -> PASS: All pins and timing constants verified without PSRAM collision.\n");
}

// ============================================================================
// TEST 2: PMS7003 32-Byte Frame Parser & Checksum
// ============================================================================
static void test_pms7003_parser(void) {
    printf("[TEST 2] Testing PMS7003 frame parser...\n");

    uint8_t frame[32];
    memset(frame, 0, sizeof(frame));

    // Header 0x42, 0x4D
    frame[0] = 0x42;
    frame[1] = 0x4D;
    // Frame length = 28 bytes (0x001C)
    frame[2] = 0x00;
    frame[3] = 0x1C;

    // Standard particle CF=1
    frame[4] = 0x00; frame[5] = 10; // PM1.0 = 10
    frame[6] = 0x00; frame[7] = 20; // PM2.5 = 20
    frame[8] = 0x00; frame[9] = 30; // PM10  = 30

    // Atmospheric environment (ambient)
    frame[10] = 0x00; frame[11] = 12; // PM1.0 atm = 12
    frame[12] = 0x00; frame[13] = 25; // PM2.5 atm = 25
    frame[14] = 0x00; frame[15] = 40; // PM10 atm  = 40

    // Particle counts
    frame[16] = 0x05; frame[17] = 0x00; // >0.3um
    frame[18] = 0x01; frame[19] = 0x00; // >0.5um
    frame[20] = 0x00; frame[21] = 0x50; // >1.0um
    frame[22] = 0x00; frame[23] = 0x10; // >2.5um
    frame[24] = 0x00; frame[25] = 0x02; // >5.0um
    frame[26] = 0x00; frame[27] = 0x01; // >10um
    frame[28] = 0x91;                   // Version
    frame[29] = 0x00;                   // Error code

    // Compute checksum = sum of bytes 0..29
    uint16_t csum = 0;
    for (size_t i = 0; i < 30; ++i) {
        csum += frame[i];
    }
    frame[30] = (uint8_t)(csum >> 8);
    frame[31] = (uint8_t)(csum & 0xFF);

    float pm1 = 0.0f, pm25 = 0.0f, pm10 = 0.0f;
    bool ok = pms7003_parse_frame(frame, sizeof(frame), &pm1, &pm25, &pm10);
    assert(ok == true);
    assert(fabsf(pm1 - 12.0f) < 1e-3f);
    assert(fabsf(pm25 - 25.0f) < 1e-3f);
    assert(fabsf(pm10 - 40.0f) < 1e-3f);

    // Negative tests:
    // 1. Corrupt Checksum
    frame[31] ^= 0xFF;
    ok = pms7003_parse_frame(frame, sizeof(frame), &pm1, &pm25, &pm10);
    assert(ok == false);
    frame[31] ^= 0xFF; // restore

    // 2. Corrupt Header
    frame[0] = 0x00;
    ok = pms7003_parse_frame(frame, sizeof(frame), &pm1, &pm25, &pm10);
    assert(ok == false);
    frame[0] = 0x42; // restore

    // 3. Invalid length field
    frame[3] = 0x20;
    ok = pms7003_parse_frame(frame, sizeof(frame), &pm1, &pm25, &pm10);
    assert(ok == false);
    frame[3] = 0x1C; // restore

    // 4. Short buffer
    ok = pms7003_parse_frame(frame, 31, &pm1, &pm25, &pm10);
    assert(ok == false);

    // 5. Null buffer
    ok = pms7003_parse_frame(NULL, sizeof(frame), &pm1, &pm25, &pm10);
    assert(ok == false);

    printf("  -> PASS: PMS7003 frame parser verified (valid, checksum error, header error, bounds).\n");
}

// ============================================================================
// TEST 3: Modbus CRC-16 & SenseAir S8 Response Parser
// ============================================================================
static void test_senseair_s8_parser(void) {
    printf("[TEST 3] Testing Modbus CRC-16 & SenseAir S8 parser...\n");

    // 1. Verify Modbus CRC-16 on standard SenseAir S8 Read CO2 request
    // Command: FE 04 00 03 00 01 -> CRC is 0xC5D5 (transmits D5 C5)
    const uint8_t req[] = {0xFE, 0x04, 0x00, 0x03, 0x00, 0x01};
    uint16_t crc_req = modbus_crc16(req, sizeof(req));
    // Standard Modbus CRC-16 sends Low byte first (0xD5), High byte second (0xC5)
    assert((crc_req & 0xFF) == 0xD5);
    assert((crc_req >> 8) == 0xC5);

    // 2. Test SenseAir S8 2-byte data response (Read 1 register)
    // Response: FE 04 02 [HI LO] [CRC_LO CRC_HI]
    // 450 ppm = 0x01C2
    uint8_t resp2[7] = {0xFE, 0x04, 0x02, 0x01, 0xC2, 0x00, 0x00};
    uint16_t crc_resp2 = modbus_crc16(resp2, 5);
    resp2[5] = (uint8_t)(crc_resp2 & 0xFF);
    resp2[6] = (uint8_t)(crc_resp2 >> 8);

    float co2_ppm = 0.0f;
    bool ok = senseair_s8_parse_response(resp2, sizeof(resp2), &co2_ppm);
    assert(ok == true);
    assert(fabsf(co2_ppm - 450.0f) < 1e-3f);

    // 3. Test SenseAir S8 4-byte data response (Read 2 registers, e.g. status + CO2 or CO2 + temp)
    // Response: FE 04 04 [03 20] [00 00] [CRC_LO CRC_HI] -> 0x0320 = 800 ppm
    uint8_t resp4[9] = {0xFE, 0x04, 0x04, 0x03, 0x20, 0x00, 0x00, 0x00, 0x00};
    uint16_t crc_resp4 = modbus_crc16(resp4, 7);
    resp4[7] = (uint8_t)(crc_resp4 & 0xFF);
    resp4[8] = (uint8_t)(crc_resp4 >> 8);

    ok = senseair_s8_parse_response(resp4, sizeof(resp4), &co2_ppm);
    assert(ok == true);
    assert(fabsf(co2_ppm - 800.0f) < 1e-3f);

    // 4. Test S8 response with function code 0x03 (Holding register compatibility)
    uint8_t resp_fn3[7] = {0xFE, 0x03, 0x02, 0x04, 0x57, 0x00, 0x00}; // 0x0457 = 1111 ppm
    uint16_t crc_fn3 = modbus_crc16(resp_fn3, 5);
    resp_fn3[5] = (uint8_t)(crc_fn3 & 0xFF);
    resp_fn3[6] = (uint8_t)(crc_fn3 >> 8);

    ok = senseair_s8_parse_response(resp_fn3, sizeof(resp_fn3), &co2_ppm);
    assert(ok == true);
    assert(fabsf(co2_ppm - 1111.0f) < 1e-3f);

    // 5. Negative tests:
    // Corrupted CRC
    resp2[6] ^= 0x55;
    ok = senseair_s8_parse_response(resp2, sizeof(resp2), &co2_ppm);
    assert(ok == false);

    // Modbus error response (Function code 0x84)
    uint8_t err_resp[5] = {0xFE, 0x84, 0x02, 0x00, 0x00};
    uint16_t crc_err = modbus_crc16(err_resp, 3);
    err_resp[3] = (uint8_t)(crc_err & 0xFF);
    err_resp[4] = (uint8_t)(crc_err >> 8);
    ok = senseair_s8_parse_response(err_resp, sizeof(err_resp), &co2_ppm);
    assert(ok == false);

    // Buffer too short
    ok = senseair_s8_parse_response(resp2, 6, &co2_ppm);
    assert(ok == false);

    // NULL pointer
    ok = senseair_s8_parse_response(NULL, 7, &co2_ppm);
    assert(ok == false);

    printf("  -> PASS: Modbus CRC-16 & SenseAir S8 parser verified.\n");
}

// ============================================================================
// TEST 4: VN-AQI Computation (QCVN 05:2023 / Decision 1459/QD-TCMT)
// ============================================================================
static void test_vn_aqi_computation(void) {
    printf("[TEST 4] Testing VN-AQI calculation according to QCVN...\n");

    // Zero & Negative edge cases
    assert(tinyml_compute_vn_aqi(-10.0f) == 0);
    assert(tinyml_compute_vn_aqi(0.0f) == 0);

    // Range 1: 0 - 25 ug/m3 -> AQI 0 - 50 (Good / Tot)
    assert(tinyml_compute_vn_aqi(12.5f) == 25);
    assert(tinyml_compute_vn_aqi(25.0f) == 50);

    // Range 2: 25 - 50 ug/m3 -> AQI 50 - 100 (Moderate / Trung binh)
    assert(tinyml_compute_vn_aqi(37.5f) == 75);
    assert(tinyml_compute_vn_aqi(50.0f) == 100);

    // Range 3: 50 - 80 ug/m3 -> AQI 100 - 150 (Unhealthy for sensitive / Kem)
    // 65.0 is exactly midpoint: 100 + (15/30)*50 = 125
    assert(tinyml_compute_vn_aqi(65.0f) == 125);
    assert(tinyml_compute_vn_aqi(80.0f) == 150);

    // Range 4: 80 - 150 ug/m3 -> AQI 150 - 200 (Unhealthy / Xau)
    // 115.0 is midpoint: 150 + (35/70)*50 = 175
    assert(tinyml_compute_vn_aqi(115.0f) == 175);
    assert(tinyml_compute_vn_aqi(150.0f) == 200);

    // Range 5: 150 - 250 ug/m3 -> AQI 200 - 300 (Very Unhealthy / Rat xau)
    // 200.0 is midpoint: 200 + (50/100)*100 = 250
    assert(tinyml_compute_vn_aqi(200.0f) == 250);
    assert(tinyml_compute_vn_aqi(250.0f) == 300);

    // Range 6: 250 - 500 ug/m3 -> AQI 300 - 500 (Hazardous / Nguy hai)
    // 375.0 is midpoint: 300 + (125/250)*200 = 400
    assert(tinyml_compute_vn_aqi(375.0f) == 400);
    assert(tinyml_compute_vn_aqi(500.0f) == 500);

    // Above 500 ug/m3 -> Capped at 500
    assert(tinyml_compute_vn_aqi(550.0f) == 500);
    assert(tinyml_compute_vn_aqi(1000.0f) == 500);

    printf("  -> PASS: VN-AQI calculation matches QCVN breakpoints.\n");
}

// ============================================================================
// TEST 5: LCD Screen Buffer Formatting
// ============================================================================
static void test_lcd_formatting(void) {
    printf("[TEST 5] Testing LCD screen buffer formatting...\n");

    EdgeDisplayPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.pm25_calibrated = 28.5f;
    payload.pm25_raw = 45.2f;
    payload.aqi = 57;
    payload.temp_c = 24.8f;
    payload.rh_percent = 82.5f;
    payload.co2_ppm = 490.0f;
    payload.gas_proxy = 120.0f;
    payload.boot_cycle = 42;
    payload.status_msg = "HEALTHY";

    char buf[512];
    memset(buf, 0, sizeof(buf));
    lcd_format_screen_buffer(&payload, buf, sizeof(buf));

    // Verify key fields are rendered into screen buffer
    assert(strstr(buf, "28.5") != NULL);
    assert(strstr(buf, "45.2") != NULL);
    assert(strstr(buf, "57") != NULL);
    assert(strstr(buf, "24.8") != NULL);
    assert(strstr(buf, "82.5") != NULL);
    assert(strstr(buf, "490") != NULL);
    assert(strstr(buf, "42") != NULL);
    assert(strstr(buf, "HEALTHY") != NULL);

    // Buffer too small safety test (should not crash or buffer overflow)
    char tiny[16];
    lcd_format_screen_buffer(&payload, tiny, sizeof(tiny));
    assert(tiny[sizeof(tiny) - 1] == '\0');

    // NULL pointer safety
    lcd_format_screen_buffer(NULL, buf, sizeof(buf));

    printf("  -> PASS: LCD screen buffer format verified.\n");
}

// ============================================================================
// TEST 6: MQTT JSON Telemetry Formatter
// ============================================================================
static void test_mqtt_json_formatter(void) {
    printf("[TEST 6] Testing MQTT JSON telemetry packet formatting...\n");

    EdgeTelemetryPacket pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.timestamp = 1731063600;
    pkt.pm1_raw = 32.1f;
    pkt.pm2_raw = 33.4f;
    pkt.pm25_calibrated = 22.8f;
    pkt.temp_c = 25.5f;
    pkt.rh_percent = 79.0f;
    pkt.co2_ppm = 510.0f;
    pkt.aqi = 46;
    pkt.anomaly_code = 0;
    pkt.flags = 1;

    // Verify packet size is exactly 32 bytes for PSRAM storage
    assert(sizeof(EdgeTelemetryPacket) == 32);

    char json[256];
    memset(json, 0, sizeof(json));
    int len = mqtt_format_telemetry_json(&pkt, json, sizeof(json));
    assert(len > 0);
    assert((size_t)len < sizeof(json));

    // Check JSON contains expected key-value pairs
    assert(strstr(json, "\"timestamp\": 1731063600") != NULL || strstr(json, "\"timestamp\":1731063600") != NULL);
    assert(strstr(json, "\"pm25_calibrated\":") != NULL);
    assert(strstr(json, "22.8") != NULL);
    assert(strstr(json, "\"aqi\": 46") != NULL || strstr(json, "\"aqi\":46") != NULL);
    assert(strstr(json, "\"co2_ppm\":") != NULL);

    // Test buffer too small
    char tiny[10];
    int tiny_len = mqtt_format_telemetry_json(&pkt, tiny, sizeof(tiny));
    assert(tiny_len < 0);

    // Test NULL handling
    assert(mqtt_format_telemetry_json(NULL, json, sizeof(json)) < 0);
    assert(mqtt_format_telemetry_json(&pkt, NULL, sizeof(json)) < 0);

    printf("  -> PASS: MQTT JSON telemetry formatting verified.\n");
}

// ============================================================================
// TEST 7: Offline Ring Buffer (FIFO, Overflow, Pop/Push, Zero-Heap)
// ============================================================================
static void test_offline_ringbuf(void) {
    printf("[TEST 7] Testing offline ring buffer (FIFO & Zero-Heap)...\n");

    offline_ringbuf_clear();
    assert(offline_ringbuf_is_empty() == true);
    assert(offline_ringbuf_count() == 0);

    // Pop on empty returns false
    EdgeTelemetryPacket out_pkt;
    memset(&out_pkt, 0, sizeof(out_pkt));
    assert(offline_ringbuf_pop(&out_pkt) == false);

    // Push 10 packets
    for (uint32_t i = 1; i <= 10; ++i) {
        EdgeTelemetryPacket pkt;
        memset(&pkt, 0, sizeof(pkt));
        pkt.timestamp = 1000 + i;
        pkt.pm25_calibrated = (float)(i * 5);
        pkt.aqi = (uint16_t)(i * 10);
        bool pushed = offline_ringbuf_push(&pkt);
        assert(pushed == true);
    }
    assert(offline_ringbuf_count() == 10);
    assert(offline_ringbuf_is_empty() == false);

    // Pop all 10 packets in FIFO order
    for (uint32_t i = 1; i <= 10; ++i) {
        bool popped = offline_ringbuf_pop(&out_pkt);
        assert(popped == true);
        assert(out_pkt.timestamp == 1000 + i);
        assert(fabsf(out_pkt.pm25_calibrated - (float)(i * 5)) < 1e-3f);
        assert(out_pkt.aqi == (uint16_t)(i * 10));
    }
    assert(offline_ringbuf_count() == 0);
    assert(offline_ringbuf_is_empty() == true);

    // Test buffer full condition
    size_t cap = offline_ringbuf_capacity();
    assert(cap >= 16);
    for (size_t i = 0; i < cap; ++i) {
        EdgeTelemetryPacket pkt;
        memset(&pkt, 0, sizeof(pkt));
        pkt.timestamp = (uint32_t)(2000 + i);
        assert(offline_ringbuf_push(&pkt) == true);
    }
    assert(offline_ringbuf_is_full() == true);

    // One more push when full should fail (no overwriting without pop)
    EdgeTelemetryPacket extra_pkt;
    memset(&extra_pkt, 0, sizeof(extra_pkt));
    extra_pkt.timestamp = 99999;
    assert(offline_ringbuf_push(&extra_pkt) == false);

    // Pop first, now can push
    assert(offline_ringbuf_pop(&out_pkt) == true);
    assert(out_pkt.timestamp == 2000);
    assert(offline_ringbuf_push(&extra_pkt) == true);

    offline_ringbuf_clear();
    assert(offline_ringbuf_is_empty() == true);

    printf("  -> PASS: Offline ring buffer verified (FIFO, capacity, full protection).\n");
}

int main(void) {
    printf("====================================================================\n");
    printf("   ESP32-S3 FIRMWARE & 15-MIN DUTY CYCLE TEST SUITE (TASK 1)       \n");
    printf("====================================================================\n");

    test_board_pins_and_psram_safety();
    test_pms7003_parser();
    test_senseair_s8_parser();
    test_vn_aqi_computation();
    test_lcd_formatting();
    test_mqtt_json_formatter();
    test_offline_ringbuf();

    printf("\n>>> ALL TASK 1 TESTS PASSED SUCCESSFULLY! <<<\n");
    return 0;
}
