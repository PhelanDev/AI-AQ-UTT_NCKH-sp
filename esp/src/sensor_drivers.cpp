#include "sensor_drivers.h"

// ============================================================================
// Plantower PMS7003 Laser Dust Sensor Driver
// ============================================================================

bool pms7003_parse_frame(const uint8_t *buf, size_t len,
                         float *out_pm1_0, float *out_pm2_5, float *out_pm10) {
    if (!buf || len < PMS7003_FRAME_SIZE) {
        return false;
    }

    // Check Start bytes 0x42, 0x4D
    if (buf[0] != PMS7003_FRAME_HEADER_1 || buf[1] != PMS7003_FRAME_HEADER_2) {
        return false;
    }

    // Check frame length field (standard Plantower frame length is 28 bytes)
    uint16_t frame_len = ((uint16_t)buf[2] << 8) | buf[3];
    if (frame_len != PMS7003_PAYLOAD_LEN) {
        return false;
    }

    // Checksum = sum of first 30 bytes (bytes 0 to 29)
    uint16_t calc_csum = 0;
    for (size_t i = 0; i < 30; ++i) {
        calc_csum += buf[i];
    }
    uint16_t expected_csum = ((uint16_t)buf[30] << 8) | buf[31];
    if (calc_csum != expected_csum) {
        return false;
    }

    // Plantower PMS7003 Data Mapping:
    // Bytes  4.. 5: PM1.0 (CF=1, standard particle) [ug/m3]
    // Bytes  6.. 7: PM2.5 (CF=1, standard particle) [ug/m3]
    // Bytes  8.. 9: PM10  (CF=1, standard particle) [ug/m3]
    // Bytes 10..11: PM1.0 (under atmospheric environment) [ug/m3]
    // Bytes 12..13: PM2.5 (under atmospheric environment) [ug/m3]
    // Bytes 14..15: PM10  (under atmospheric environment) [ug/m3]
    uint16_t raw_pm1_0 = ((uint16_t)buf[10] << 8) | buf[11];
    uint16_t raw_pm2_5 = ((uint16_t)buf[12] << 8) | buf[13];
    uint16_t raw_pm10  = ((uint16_t)buf[14] << 8) | buf[15];

    if (out_pm1_0) {
        *out_pm1_0 = (float)raw_pm1_0;
    }
    if (out_pm2_5) {
        *out_pm2_5 = (float)raw_pm2_5;
    }
    if (out_pm10) {
        *out_pm10 = (float)raw_pm10;
    }

    return true;
}

// ============================================================================
// SenseAir S8 LP/0053 CO2 NDIR Sensor Driver
// ============================================================================

uint16_t modbus_crc16(const uint8_t *data, size_t len) {
    if (!data) {
        return 0;
    }
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i];
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc = crc >> 1;
            }
        }
    }
    return crc;
}

bool senseair_s8_parse_response(const uint8_t *buf, size_t len, float *out_co2_ppm) {
    if (!buf || len < 7) {
        return false;
    }

    // Check for Modbus error response (exception code bit 7 set)
    if (buf[1] & 0x80) {
        return false;
    }

    // SenseAir S8 supports Read Input Registers (0x04) or Read Holding (0x03)
    if (buf[1] != 0x04 && buf[1] != 0x03) {
        return false;
    }

    // Verify CRC-16 (Modbus sends CRC Low byte first, then High byte)
    uint16_t received_crc = (uint16_t)buf[len - 2] | ((uint16_t)buf[len - 1] << 8);
    uint16_t calculated_crc = modbus_crc16(buf, len - 2);
    if (calculated_crc != received_crc) {
        return false;
    }

    uint8_t byte_count = buf[2];
    if (len < (size_t)(byte_count + 5)) {
        return false;
    }

    uint16_t co2 = 0;
    if (byte_count == 2) {
        // Standard 1-register response: bytes 3..4
        co2 = ((uint16_t)buf[3] << 8) | buf[4];
    } else if (byte_count >= 4) {
        // 2-register response: check word 1 (bytes 3..4) or word 2 (bytes 5..6)
        uint16_t word1 = ((uint16_t)buf[3] << 8) | buf[4];
        uint16_t word2 = ((uint16_t)buf[5] << 8) | buf[6];
        if (word1 >= 300 && word1 <= 10000) {
            co2 = word1;
        } else {
            co2 = word2;
        }
    } else {
        return false;
    }

    if (out_co2_ppm) {
        *out_co2_ppm = (float)co2;
    }

    return true;
}

size_t senseair_s8_build_read_co2_cmd(uint8_t *out_cmd, size_t max_len) {
    if (!out_cmd || max_len < 8) {
        return 0;
    }
    // Command: Slave ID 0xFE, Func 0x04, Reg 0x0003, Count 0x0001, CRC (0xD5 0xC5)
    out_cmd[0] = 0xFE;
    out_cmd[1] = 0x04;
    out_cmd[2] = 0x00;
    out_cmd[3] = 0x03;
    out_cmd[4] = 0x00;
    out_cmd[5] = 0x01;

    uint16_t crc = modbus_crc16(out_cmd, 6);
    out_cmd[6] = (uint8_t)(crc & 0xFF);         // CRC Low
    out_cmd[7] = (uint8_t)((crc >> 8) & 0xFF);  // CRC High

    return 8;
}


// ============================================================================
// Sensirion SHT40 Temperature & Relative Humidity I2C Driver
// ============================================================================

uint8_t sht40_compute_crc8(const uint8_t *data, size_t len) {
    if (!data) {
        return 0;
    }
    uint8_t crc = 0xFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 8; bit > 0; --bit) {
            if (crc & 0x80) {
                crc = (crc << 1) ^ 0x31;
            } else {
                crc = (crc << 1);
            }
        }
    }
    return crc;
}

size_t sht40_build_measure_cmd(uint8_t *out_cmd, size_t max_len) {
    if (!out_cmd || max_len < 1) {
        return 0;
    }
    out_cmd[0] = SHT40_CMD_MEASURE_HIGH_PRECISION;
    return 1;
}

bool sht40_parse_response(const uint8_t *buf, size_t len, float *out_temp_c, float *out_rh_percent) {
    if (!buf || len < SHT40_RESPONSE_SIZE) {
        return false;
    }

    // Verify Temperature word CRC (bytes 0..1 vs byte 2)
    uint8_t calc_temp_crc = sht40_compute_crc8(&buf[0], 2);
    if (calc_temp_crc != buf[2]) {
        return false;
    }

    // Verify Relative Humidity word CRC (bytes 3..4 vs byte 5)
    uint8_t calc_rh_crc = sht40_compute_crc8(&buf[3], 2);
    if (calc_rh_crc != buf[5]) {
        return false;
    }

    uint16_t t_raw = ((uint16_t)buf[0] << 8) | buf[1];
    uint16_t rh_raw = ((uint16_t)buf[3] << 8) | buf[4];

    if (out_temp_c) {
        *out_temp_c = -45.0f + 175.0f * ((float)t_raw / 65535.0f);
    }
    if (out_rh_percent) {
        float rh = -6.0f + 125.0f * ((float)rh_raw / 65535.0f);
        if (rh < 0.0f) {
            rh = 0.0f;
        }
        if (rh > 100.0f) {
            rh = 100.0f;
        }
        *out_rh_percent = rh;
    }

    return true;
}
