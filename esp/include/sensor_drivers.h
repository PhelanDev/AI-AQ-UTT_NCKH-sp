#ifndef SENSOR_DRIVERS_H
#define SENSOR_DRIVERS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Plantower PMS7003 Laser Dust Sensor Driver (32-byte UART Frame)
// ============================================================================

#define PMS7003_FRAME_HEADER_1  0x42
#define PMS7003_FRAME_HEADER_2  0x4D
#define PMS7003_PAYLOAD_LEN     28
#define PMS7003_FRAME_SIZE      32

/**
 * @brief Parses and validates a 32-byte UART frame from Plantower PMS7003.
 *
 * Checks:
 *  - Buffer non-null and length >= 32
 *  - Header 0x42 0x4D
 *  - Payload length == 28 (0x001C)
 *  - Checksum (sum of first 30 bytes equals 16-bit sum in bytes 30..31)
 *
 * @param buf Pointer to raw UART byte buffer.
 * @param len Length of the buffer in bytes.
 * @param out_pm1_0 Output pointer for atmospheric PM1.0 (ug/m3). Can be NULL.
 * @param out_pm2_5 Output pointer for atmospheric PM2.5 (ug/m3). Can be NULL.
 * @param out_pm10 Output pointer for atmospheric PM10 (ug/m3). Can be NULL.
 * @return true if valid frame and checksum matched; false otherwise.
 */
bool pms7003_parse_frame(const uint8_t *buf, size_t len,
                         float *out_pm1_0, float *out_pm2_5, float *out_pm10);

// ============================================================================
// SenseAir S8 LP/0053 CO2 NDIR Sensor Driver (Modbus RTU)
// ============================================================================

/**
 * @brief Calculates standard Modbus RTU CRC-16 (polynomial 0xA001, init 0xFFFF).
 *
 * @param data Pointer to byte buffer.
 * @param len Number of bytes to compute CRC over.
 * @return 16-bit CRC value.
 */
uint16_t modbus_crc16(const uint8_t *data, size_t len);

/**
 * @brief Parses and validates Modbus response from SenseAir S8 (Input register 0x0003).
 *
 * Checks:
 *  - CRC-16 validity
 *  - Function code (0x04 or 0x03)
 *  - Extracts CO2 concentration in ppm (from 2-byte or 4-byte payload)
 *
 * @param buf Pointer to received Modbus RTU response frame.
 * @param len Length of received frame.
 * @param out_co2_ppm Output pointer for CO2 concentration in ppm.
 * @return true if CRC matches and CO2 is parsed; false on error.
 */
bool senseair_s8_parse_response(const uint8_t *buf, size_t len, float *out_co2_ppm);

/**
 * @brief Builds standard Modbus RTU read command for SenseAir S8 CO2 register 0x0003.
 *
 * @param out_cmd Output buffer (at least 8 bytes).
 * @param max_len Size of output buffer.
 * @return Number of bytes written into out_cmd (typically 8 bytes), or 0 on error.
 */
size_t senseair_s8_build_read_co2_cmd(uint8_t *out_cmd, size_t max_len);

// ============================================================================
// Sensirion SHT40 High-Precision Temperature & Humidity Driver (I2C)
// ============================================================================

#define SHT40_I2C_ADDR                   0x44
#define SHT40_CMD_MEASURE_HIGH_PRECISION 0xFD
#define SHT40_RESPONSE_SIZE              6

uint8_t sht40_compute_crc8(const uint8_t *data, size_t len);
bool sht40_parse_response(const uint8_t *buf, size_t len, float *out_temp_c, float *out_rh_percent);
size_t sht40_build_measure_cmd(uint8_t *out_cmd, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif // SENSOR_DRIVERS_H
