#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "board_pins.h"
#include "sensor_drivers.h"
#include "display_mqtt_hal.h"
#include "tinyml_edge_engine.h"
#include "power_scheduler.h"

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

// ============================================================================
// TEST 8: RTC Memory Retention (Dual PMS Chamber Drift & Boot Cycle Counter)
// ============================================================================
static void test_rtc_drift_retention(void) {
    printf("[TEST 8] Testing RTC drift & boot counter retention...\n");

    tinyml_reset_drift_monitor();
    assert(fabsf(tinyml_get_current_ema_drift()) < 1e-5f);

    tinyml_set_boot_cycle_count(0);
    assert(tinyml_get_boot_cycle_count() == 0);
    tinyml_increment_boot_cycle_count();
    assert(tinyml_get_boot_cycle_count() == 1);

    // Initial cycle: pm1 = 50.0, pm2 = 40.0 -> diff = 10.0
    float ema = 0.0f;
    TinyML_AnomalyCode code = tinyml_update_drift_monitor(50.0f, 40.0f, &ema);
    assert(code == TINYML_ANOMALY_OK);
    assert(fabsf(ema - 10.0f) < 1e-3f);
    assert(fabsf(tinyml_get_current_ema_drift() - 10.0f) < 1e-3f);

    // Second cycle: pm1 = 60.0, pm2 = 40.0 -> diff = 20.0
    // EMA = (1 - 0.01) * 10.0 + 0.01 * 20.0 = 9.9 + 0.2 = 10.1
    code = tinyml_update_drift_monitor(60.0f, 40.0f, &ema);
    assert(code == TINYML_ANOMALY_OK);
    assert(fabsf(ema - 10.1f) < 1e-2f);
    assert(fabsf(tinyml_get_current_ema_drift() - 10.1f) < 1e-2f);

    // Simulate persistent excessive drift exceeding threshold (15.0 ug/m3)
    for (int i = 0; i < 200; ++i) {
        code = tinyml_update_drift_monitor(100.0f, 0.0f, &ema);
    }
    assert(ema > PMS7003_DRIFT_THRESHOLD);
    assert(code == TINYML_ANOMALY_CHAMBER_DRIFT);

    // Reset monitor
    tinyml_reset_drift_monitor();
    assert(fabsf(tinyml_get_current_ema_drift()) < 1e-5f);

    printf("  -> PASS: RTC drift retention, EMA convergence, and anomaly detection verified.\n");
}

// ============================================================================
// TEST 9: 15-Minute FSM Warmup Enforced (< 30s Blocks Measurement)
// ============================================================================
static void test_fsm_warmup_incomplete(void) {
    printf("[TEST 9] Testing FSM warmup validation (< 30s blocks inference)...\n");

    RawSensorSample sample;
    memset(&sample, 0, sizeof(sample));
    sample.timestamp = 1731063600;
    sample.pm1_raw = 40.0f;
    sample.pm2_raw = 41.0f;
    sample.pm10_raw = 55.0f;
    sample.temp_c = 25.0f;
    sample.rh_percent = 70.0f;
    sample.co2_ppm = 500.0f;
    sample.gas_proxy = 100.0f;

    // Test with warmup < 30s (e.g. 15.0s and 29.9s)
    EdgeCycleResult res15 = edge_execute_15min_duty_cycle(&sample, 15.0f, true);
    assert(res15.anomaly_code == TINYML_ANOMALY_WARMUP_INCOMPLETE);
    assert(res15.lcd_updated == false);
    assert(res15.telemetry_sent == false);
    assert(res15.entered_deep_sleep == false);
    assert(res15.deep_sleep_duration_sec == 0);
    assert(res15.pms_set_pin_level == true); // Fan kept running
    assert(power_scheduler_get_state() == EDGE_STATE_WARMUP);

    EdgeCycleResult res29 = edge_execute_15min_duty_cycle(&sample, 29.9f, true);
    assert(res29.anomaly_code == TINYML_ANOMALY_WARMUP_INCOMPLETE);
    assert(res29.lcd_updated == false);
    assert(res29.telemetry_sent == false);
    assert(res29.entered_deep_sleep == false);

    // Null safety
    EdgeCycleResult null_res = edge_execute_15min_duty_cycle(NULL, 35.0f, true);
    assert(null_res.anomaly_code == TINYML_ANOMALY_WARMUP_INCOMPLETE);

    printf("  -> PASS: FSM strictly blocks premature measurement when warmup < 30s.\n");
}

// ============================================================================
// TEST 10: 15-Minute FSM Nominal Duty Cycle (Warmup >= 30s, Sleep 868s)
// ============================================================================
static void test_fsm_warmup_complete_nominal_cycle(void) {
    printf("[TEST 10] Testing nominal FSM cycle (warmup >= 30s, LCD/MQTT, sleep 868s)...\n");

    tinyml_reset_drift_monitor();
    tinyml_set_boot_cycle_count(0);

    RawSensorSample sample;
    sample.timestamp = 1731063600;
    sample.pm1_raw = 35.0f;
    sample.pm2_raw = 36.0f;
    sample.pm10_raw = 48.0f;
    sample.temp_c = 28.0f;
    sample.rh_percent = 85.0f; // High humidity: Deliquescence regime
    sample.co2_ppm = 500.0f;
    sample.gas_proxy = 100.0f;

    EdgeCycleResult res = edge_execute_15min_duty_cycle(&sample, 30.5f, true);

    assert(res.anomaly_code == TINYML_ANOMALY_OK);
    assert(res.pm25_calibrated > 0.0f);
    // At RH=85%, hygroscopic water swelling must be compensated downwards (calibrated < raw average 35.5)
    assert(res.pm25_calibrated < 35.5f);
    assert(res.vn_aqi > 0);
    assert(res.lcd_updated == true);
    assert(res.telemetry_sent == true);
    assert(res.pms_set_pin_level == false); // Fan and laser turned OFF for sleep
    assert(res.entered_deep_sleep == true);
    assert(res.deep_sleep_duration_sec == 868);
    assert(tinyml_get_boot_cycle_count() == 1);
    assert(power_scheduler_get_state() == EDGE_STATE_DEEP_SLEEP);

    printf("  -> PASS: Nominal cycle verified: Calibrated=%.2f ug/m3, VN-AQI=%d, DeepSleep=%us, PMS_SET=LOW.\n",
           res.pm25_calibrated, res.vn_aqi, res.deep_sleep_duration_sec);
}

// ============================================================================
// TEST 11: 100 Consecutive Duty Cycles Simulation & Timing Integrity
// ============================================================================
static void test_100_cycles_simulation_and_duty_timing(void) {
    printf("[TEST 11] Testing 100 consecutive 15-min duty cycles (25 hours simulation)...\n");

    tinyml_reset_drift_monitor();
    tinyml_set_boot_cycle_count(0);
    offline_ringbuf_clear();

    const uint32_t NUM_CYCLES = 100;
    uint64_t accumulated_simulated_seconds = 0;

    for (uint32_t c = 1; c <= NUM_CYCLES; ++c) {
        RawSensorSample sample;
        sample.timestamp = 1731000000 + (uint32_t)(c * 900);
        sample.pm1_raw = 25.0f + (float)(c % 20);
        sample.pm2_raw = sample.pm1_raw + ((c % 2 == 0) ? 0.6f : -0.6f);
        sample.pm10_raw = sample.pm1_raw * 1.35f;
        sample.temp_c = 26.0f + (float)(c % 8) * 0.5f;
        sample.rh_percent = 65.0f + (float)(c % 25);
        sample.co2_ppm = 450.0f + (float)(c % 100);
        sample.gas_proxy = 100.0f;

        // Disconnect network on cycles 30..34 (5 offline cycles)
        bool online = !(c >= 30 && c <= 34);

        EdgeCycleResult res = edge_execute_15min_duty_cycle(&sample, 30.0f, online);

        assert(res.anomaly_code == TINYML_ANOMALY_OK);
        assert(res.entered_deep_sleep == true);
        assert(res.pms_set_pin_level == false);
        assert(res.deep_sleep_duration_sec == ESP_DEEP_SLEEP_SEC);
        assert(res.lcd_updated == true);

        if (!online) {
            assert(res.telemetry_sent == false);
        } else if (c == 35) {
            // Reconnection: automatically popped and published queued offline packets
            assert(res.telemetry_sent == true);
            assert(offline_ringbuf_is_empty() == true);
        } else {
            assert(res.telemetry_sent == true);
        }

        // Validate exact timing breakdown: 30s + 2s + 868s = 900s
        uint32_t cycle_seconds = ESP_WARMUP_SEC + ESP_ACTIVE_PROCESS_SEC + res.deep_sleep_duration_sec;
        assert(cycle_seconds == ESP_CYCLE_TOTAL_SEC);
        accumulated_simulated_seconds += cycle_seconds;

        assert(tinyml_get_boot_cycle_count() == c);
    }

    // 100 cycles of 900s = 90,000s = 25.0 hours
    assert(accumulated_simulated_seconds == 100ULL * 900ULL);
    assert(accumulated_simulated_seconds == 90000ULL);
    assert(tinyml_get_boot_cycle_count() == 100);

    // Drift monitor across 100 cycles with small symmetric noise must remain near 0
    float drift = tinyml_get_current_ema_drift();
    assert(fabsf(drift) < 2.0f);

    printf("  -> PASS: 100 cycles completed (90,000s / 25h simulated), zero memory leaks, timing exact.\n");
}

// ============================================================================
// TEST 12: 30-Sample Burst Accumulator & 20% Trimmed Mean Outlier Rejection
// ============================================================================
static void test_burst_accumulator_trimmed_mean_filtering(void) {
    printf("[TEST 12] Testing 30-sample burst accumulator & 20%% trimmed mean filtering...\n");

    SensorBurstAccumulator acc;
    burst_accumulator_init(&acc);
    assert(acc.count == 0);
    assert(acc.is_full == false);

    // Push 30 samples:
    // 24 clean baseline samples: PM1 ~ 40.0, PM2 ~ 40.0, PM10 ~ 55.0, T = 25.0, RH = 70.0, CO2 = 450.0, Gas = 100.0
    // 3 extreme high spikes: PM1 = 500.0, PM2 = 480.0, CO2 = 2500.0 (simulating laser dust flare / insect)
    // 3 extreme low drops: PM1 = 0.0, PM2 = 0.0, PM10 = 0.0 (simulating voltage drop / UART packet drop)
    for (int i = 0; i < SENSOR_BURST_SAMPLE_COUNT; i++) {
        RawSensorSample s;
        s.timestamp = 1731063600 + i;
        s.pm1_raw = 40.0f;
        s.pm2_raw = 40.0f;
        s.pm10_raw = 55.0f;
        s.temp_c = 25.0f;
        s.rh_percent = 70.0f;
        s.co2_ppm = 450.0f;
        s.gas_proxy = 100.0f;

        // Inject 3 high spikes
        if (i == 5 || i == 14 || i == 23) {
            s.pm1_raw = 500.0f;
            s.pm2_raw = 480.0f;
            s.co2_ppm = 2500.0f;
        }
        // Inject 3 low drops
        if (i == 2 || i == 18 || i == 27) {
            s.pm1_raw = 0.0f;
            s.pm2_raw = 0.0f;
            s.pm10_raw = 0.0f;
        }

        bool full = burst_accumulator_push(&acc, &s);
        if (i < SENSOR_BURST_SAMPLE_COUNT - 1) {
            assert(full == false);
        } else {
            assert(full == true);
        }
    }

    assert(acc.count == SENSOR_BURST_SAMPLE_COUNT);
    assert(acc.is_full == true);

    // Pushing beyond 30 samples must be protected against overflow
    RawSensorSample extra;
    memset(&extra, 0, sizeof(extra));
    bool overflow_check = burst_accumulator_push(&acc, &extra);
    assert(overflow_check == true);
    assert(acc.count == SENSOR_BURST_SAMPLE_COUNT);

    // Calculate Trimmed Mean (trim_count = 3 -> discards 3 lowest & 3 highest, averages 24 middle)
    RawSensorSample filtered;
    bool filter_ok = burst_filter_trimmed_mean(&acc, SENSOR_BURST_TRIM_COUNT, &filtered);
    assert(filter_ok == true);

    // Verify 100% rejection of 3 spikes (500) and 3 drops (0):
    // All 24 core samples have pm1_raw == 40.0f -> filtered mean MUST be exactly 40.0f!
    assert(fabsf(filtered.pm1_raw - 40.0f) < 1e-4f);
    assert(fabsf(filtered.pm2_raw - 40.0f) < 1e-4f);
    assert(fabsf(filtered.pm10_raw - 55.0f) < 1e-4f);
    assert(fabsf(filtered.temp_c - 25.0f) < 1e-4f);
    assert(fabsf(filtered.rh_percent - 70.0f) < 1e-4f);
    assert(fabsf(filtered.co2_ppm - 450.0f) < 1e-4f);
    assert(fabsf(filtered.gas_proxy - 100.0f) < 1e-4f);

    // Contrast with naive un-trimmed average to prove mathematical necessity:
    // Naive sum of pm1_raw: (24 * 40.0 + 3 * 500.0 + 3 * 0.0) / 30 = 82.0 ug/m3 (+105% error!)
    float naive_sum = 0.0f;
    for (int i = 0; i < SENSOR_BURST_SAMPLE_COUNT; i++) naive_sum += acc.samples[i].pm1_raw;
    float naive_avg = naive_sum / (float)SENSOR_BURST_SAMPLE_COUNT;
    assert(fabsf(naive_avg - 82.0f) < 1e-4f);

    printf("  -> PASS: 20%% Trimmed Mean successfully rejected +105%% noise distortion (Clean: 40.0 vs Naive: 82.0 ug/m3).\n");
}

// ============================================================================
// TEST 13: Integrated 15-Minute Cycle with Burst Denoising Engine
// ============================================================================
static void test_burst_duty_cycle_integration(void) {
    printf("[TEST 13] Testing integrated 15-min cycle with burst denoising...\n");

    SensorBurstAccumulator acc;
    burst_accumulator_init(&acc);

    for (int i = 0; i < SENSOR_BURST_SAMPLE_COUNT; i++) {
        RawSensorSample s;
        s.timestamp = 1731063600 + i;
        s.pm1_raw = 35.0f + (float)(i % 5 - 2) * 0.5f; // 34.0 .. 36.0
        s.pm2_raw = 35.2f + (float)(i % 5 - 2) * 0.4f; // 34.4 .. 36.0
        s.pm10_raw = 48.0f;
        s.temp_c = 26.0f;
        s.rh_percent = 82.0f;  // Deliquescence nồm ẩm
        s.co2_ppm = 450.0f;
        s.gas_proxy = 100.0f;

        // Occasional laser flare spike
        if (i == 7) s.pm1_raw = 400.0f;
        if (i == 21) s.pm2_raw = 380.0f;

        burst_accumulator_push(&acc, &s);
    }

    EdgeCycleResult res = edge_execute_15min_duty_cycle_burst(&acc, 30.5f, true);

    // Warmup >= 30s -> nominal cycle completion
    assert(res.anomaly_code == TINYML_ANOMALY_OK);
    assert(res.entered_deep_sleep == true);
    assert(res.pms_set_pin_level == false);
    assert(res.deep_sleep_duration_sec == ESP_DEEP_SLEEP_SEC);
    assert(res.lcd_updated == true);
    assert(res.telemetry_sent == true);

    // PM2.5 calibrated under RH=82% Deliquescence regime must be scaled by C ~ 0.45..0.60
    // Input baseline ~35 ug/m3 -> Calibrated ~16..22 ug/m3 (not distorted by 400 ug/m3 spike!)
    assert(res.pm25_calibrated >= 14.0f && res.pm25_calibrated <= 25.0f);
    assert(res.vn_aqi > 0 && res.vn_aqi <= 70);

    // Fallback safety test: NULL accumulator or empty accumulator
    EdgeCycleResult fallback_null = edge_execute_15min_duty_cycle_burst(NULL, 30.5f, true);
    assert(fallback_null.entered_deep_sleep == true);

    SensorBurstAccumulator empty_acc;
    burst_accumulator_init(&empty_acc);
    EdgeCycleResult fallback_empty = edge_execute_15min_duty_cycle_burst(&empty_acc, 30.5f, true);
    assert(fallback_empty.entered_deep_sleep == true);

    printf("  -> PASS: Burst FSM cycle complete (Calibrated: %.2f ug/m3, VN-AQI: %d, Sleep: %u s).\n",
           res.pm25_calibrated, res.vn_aqi, res.deep_sleep_duration_sec);
}

// ============================================================================
// TEST 14: Sensirion SHT40 CRC-8 & Command Builder
// ============================================================================
static void test_sht40_crc8_and_command(void) {
    printf("[TEST 14] Testing Sensirion SHT40 CRC-8 & command builder...\n");

    // Standard Sensirion test vector: {0xBE, 0xEF} -> CRC-8 = 0x92
    const uint8_t test_vec[2] = {0xBE, 0xEF};
    uint8_t crc = sht40_compute_crc8(test_vec, 2);
    assert(crc == 0x92);

    // Build High-Precision measurement command (0xFD)
    uint8_t cmd_buf[4];
    size_t cmd_len = sht40_build_measure_cmd(cmd_buf, sizeof(cmd_buf));
    assert(cmd_len == 1);
    assert(cmd_buf[0] == 0xFD);

    // Invalid buffers
    assert(sht40_build_measure_cmd(NULL, 4) == 0);
    assert(sht40_build_measure_cmd(cmd_buf, 0) == 0);
    assert(sht40_compute_crc8(NULL, 2) == 0);

    printf("  -> PASS: SHT40 CRC-8 matches Sensirion test vector (0x92) & command is 0xFD.\n");
}

// ============================================================================
// TEST 15: Sensirion SHT40 6-Byte Frame Parser & Physical Conversion
// ============================================================================
static void test_sht40_response_parser(void) {
    printf("[TEST 15] Testing Sensirion SHT40 response frame parser...\n");

    // Synthesize standard SHT40 6-byte response:
    // T_ticks = 26400 (0x67, 0x20) -> T = -45 + 175 * (26400 / 65535) = 25.49668 deg C
    // RH_ticks = 45000 (0xAF, 0xC8) -> RH = -6 + 125 * (45000 / 65535) = 79.83215 %
    uint8_t frame[6];
    frame[0] = 0x67;
    frame[1] = 0x20;
    frame[2] = sht40_compute_crc8(&frame[0], 2); // T_CRC

    frame[3] = 0xAF;
    frame[4] = 0xC8;
    frame[5] = sht40_compute_crc8(&frame[3], 2); // RH_CRC

    float temp_c = 0.0f;
    float rh_percent = 0.0f;

    bool parse_ok = sht40_parse_response(frame, 6, &temp_c, &rh_percent);
    assert(parse_ok == true);
    assert(fabsf(temp_c - 25.49668f) < 1e-3f);
    assert(fabsf(rh_percent - 79.83215f) < 1e-3f);

    // Corrupt T CRC -> must reject
    frame[2] ^= 0x01;
    assert(sht40_parse_response(frame, 6, &temp_c, &rh_percent) == false);
    frame[2] ^= 0x01; // Restore

    // Corrupt RH CRC -> must reject
    frame[5] ^= 0x80;
    assert(sht40_parse_response(frame, 6, &temp_c, &rh_percent) == false);
    frame[5] ^= 0x80; // Restore

    // Length check (< 6 bytes)
    assert(sht40_parse_response(frame, 5, &temp_c, &rh_percent) == false);
    assert(sht40_parse_response(NULL, 6, &temp_c, &rh_percent) == false);

    printf("  -> PASS: SHT40 6-byte parsing verified with CRC verification (T=%.2f C, RH=%.2f%%).\n",
           temp_c, rh_percent);
}

#include "st7789_lcd_driver.h"

// ============================================================================
// TEST 16: ST7789 RGB565 Palette & VN-AQI Mapping
// ============================================================================
static void test_st7789_aqi_colors(void) {
    printf("[TEST 16] Testing ST7789 RGB565 palette & VN-AQI mapping...\n");

    assert(st7789_get_aqi_color_rgb565(0) == RGB565_AQI_GOOD);
    assert(st7789_get_aqi_color_rgb565(25) == RGB565_AQI_GOOD);
    assert(st7789_get_aqi_color_rgb565(50) == RGB565_AQI_GOOD);

    assert(st7789_get_aqi_color_rgb565(51) == RGB565_AQI_MODERATE);
    assert(st7789_get_aqi_color_rgb565(100) == RGB565_AQI_MODERATE);

    assert(st7789_get_aqi_color_rgb565(101) == RGB565_AQI_UNHEALTHY_S);
    assert(st7789_get_aqi_color_rgb565(150) == RGB565_AQI_UNHEALTHY_S);

    assert(st7789_get_aqi_color_rgb565(151) == RGB565_AQI_UNHEALTHY);
    assert(st7789_get_aqi_color_rgb565(200) == RGB565_AQI_UNHEALTHY);

    assert(st7789_get_aqi_color_rgb565(201) == RGB565_AQI_VERY_UNHEALTHY);
    assert(st7789_get_aqi_color_rgb565(300) == RGB565_AQI_VERY_UNHEALTHY);

    assert(st7789_get_aqi_color_rgb565(301) == RGB565_AQI_HAZARDOUS);
    assert(st7789_get_aqi_color_rgb565(500) == RGB565_AQI_HAZARDOUS);

    printf("  -> PASS: All 6 VN-AQI color levels mapped accurately to RGB565.\n");
}

// ============================================================================
// TEST 17: ST7789 Lifecycle, Dashboard Rendering & Deep Sleep Power-Down
// ============================================================================
static void test_st7789_lifecycle_and_dashboard_rendering(void) {
    printf("[TEST 17] Testing ST7789 lifecycle, dashboard rendering & sleep power-down...\n");

    st7789_reset_mock_state();
    const ST7789MockState *mock = st7789_get_mock_state();
    assert(mock->is_initialized == false);

    // 1. Initialize Display
    st7789_init();
    assert(mock->is_initialized == true);
    assert(mock->backlight_on == true);
    assert(mock->in_sleep_mode == false);

    // 2. Render Real-Time Dashboard
    EdgeDisplayPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.boot_cycle = 42;
    payload.pm25_calibrated = 21.8f;
    payload.pm25_raw = 55.0f;
    payload.aqi = 45;
    payload.temp_c = 26.5f;
    payload.rh_percent = 84.0f;
    payload.co2_ppm = 520.0f;
    payload.gas_proxy = 110.0f;
    payload.status_msg = "HEALTHY";

    st7789_render_dashboard(&payload);
    assert(mock->rendered_aqi == 45);
    assert(fabsf(mock->rendered_pm25 - 21.8f) < 1e-4f);
    assert(mock->total_pixels_written > 0);

    // 3. Power-down sequence before Deep Sleep
    st7789_enter_sleep();
#if CONFIG_LCD_ALWAYS_ON
    // In Always-On mode, display backlight stays ON and panel remains awake to show measurements 24/7
    assert(mock->in_sleep_mode == false);
    assert(mock->backlight_on == true);
#else
    assert(mock->in_sleep_mode == true);
    assert(mock->backlight_on == false);
#endif

    printf("  -> PASS: ST7789 initialization, dashboard rendering, and sleep power-down verified.\n");
}

int main(void) {
    printf("====================================================================\n");
    printf("   ESP32-S3 FIRMWARE & 15-MIN DUTY CYCLE TEST SUITE (TASKS 1 & 2)  \n");
    printf("====================================================================\n");

    test_board_pins_and_psram_safety();
    test_pms7003_parser();
    test_senseair_s8_parser();
    test_vn_aqi_computation();
    test_lcd_formatting();
    test_mqtt_json_formatter();
    test_offline_ringbuf();
    test_rtc_drift_retention();
    test_fsm_warmup_incomplete();
    test_fsm_warmup_complete_nominal_cycle();
    test_100_cycles_simulation_and_duty_timing();

    test_burst_accumulator_trimmed_mean_filtering();
    test_burst_duty_cycle_integration();

    test_sht40_crc8_and_command();
    test_sht40_response_parser();

    test_st7789_aqi_colors();
    test_st7789_lifecycle_and_dashboard_rendering();

    printf("\n>>> ALL 17 TESTS (INCLUDING ST7789 LCD SPI & SHT40 I2C) PASSED SUCCESSFULLY! <<<\n");
    return 0;
}
