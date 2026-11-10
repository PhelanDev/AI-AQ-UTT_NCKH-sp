#ifndef DISPLAY_MQTT_HAL_H
#define DISPLAY_MQTT_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Display & Telemetry Data Structures (Zero Dynamic Memory Allocation)
// ============================================================================

typedef struct {
    float pm25_calibrated; // Calibrated PM2.5 (ug/m3)
    float pm25_raw;        // Raw PM2.5 from sensor (ug/m3)
    int aqi;               // VN-AQI index (0..500)
    float temp_c;          // Temperature from SHT40 (deg C)
    float rh_percent;      // Relative Humidity from SHT40 (%)
    float co2_ppm;         // Carbon Dioxide from SenseAir S8 (ppm)
    float gas_proxy;       // Gas / VOC Index proxy from SGP41
    uint32_t boot_cycle;   // RTC Boot cycle counter
    const char *status_msg;// Diagnostics message ("HEALTHY", "ANOMALY", etc.)
} EdgeDisplayPayload;

// 32-byte packed telemetry packet optimized for PSRAM offline ring buffer
#pragma pack(push, 1)
typedef struct {
    uint32_t timestamp;      // 4 bytes: Unix epoch or boot timestamp
    float pm1_raw;           // 4 bytes: PMS7003 #1 raw PM2.5 (ug/m3)
    float pm2_raw;           // 4 bytes: PMS7003 #2 raw PM2.5 (ug/m3)
    float pm25_calibrated;   // 4 bytes: TinyML calibrated PM2.5 (ug/m3)
    float temp_c;            // 4 bytes: Temperature (deg C)
    float rh_percent;        // 4 bytes: Relative humidity (%)
    float co2_ppm;           // 4 bytes: SenseAir S8 CO2 (ppm)
    uint16_t aqi;            // 2 bytes: VN-AQI (0..500)
    uint8_t anomaly_code;    // 1 byte: TinyML anomaly code
    uint8_t flags;           // 1 byte: Connectivity & health flags
} EdgeTelemetryPacket;
#pragma pack(pop)

// Compile-time assertion for 32-byte packet
#ifdef __cplusplus
static_assert(sizeof(EdgeTelemetryPacket) == 32, "EdgeTelemetryPacket must be exactly 32 bytes for PSRAM storage");
#endif

// ============================================================================
// Telemetry Packet Bitwise Flags (Connectivity & Store-and-Forward Status)
// ============================================================================
#define TELEMETRY_FLAG_REALTIME        (1 << 0) // 0x01: Gói tin gửi trực tiếp thời gian thực
#define TELEMETRY_FLAG_BACKFILL        (1 << 1) // 0x02: Gói tin gửi bù sau khi mạng phục hồi (Store-and-Forward)
#define TELEMETRY_FLAG_OVERFLOW_DROP   (1 << 2) // 0x04: Từng bị tràn bộ đệm (đã loại bỏ mẫu cũ nhất để ưu tiên mẫu mới)

// ============================================================================
// VN-AQI Calculation (QCVN 05:2023 / Decision 1459/QD-TCMT)
// ============================================================================

/**
 * @brief Computes Vietnam Air Quality Index (VN-AQI) from PM2.5 concentration.
 *
 * Range breakpoints:
 *    0.0 ..  25.0 ug/m3 -> AQI   0 ..  50 (Good / Tốt)
 *   25.0 ..  50.0 ug/m3 -> AQI  50 .. 100 (Moderate / Trung bình)
 *   50.0 ..  80.0 ug/m3 -> AQI 100 .. 150 (Unhealthy for sensitive / Kém)
 *   80.0 .. 150.0 ug/m3 -> AQI 150 .. 200 (Unhealthy / Xấu)
 *  150.0 .. 250.0 ug/m3 -> AQI 200 .. 300 (Very Unhealthy / Rất xấu)
 *  250.0 .. 500.0 ug/m3 -> AQI 300 .. 500 (Hazardous / Nguy hại)
 *  > 500.0 ug/m3        -> AQI 500 (Capped)
 *
 * @param pm25_ugm3 Calibrated PM2.5 concentration in ug/m3.
 * @return Integer VN-AQI value in range [0, 500].
 */
int tinyml_compute_vn_aqi(float pm25_ugm3);

/**
 * @brief Returns category string in English or Vietnamese for a given VN-AQI.
 *
 * @param aqi VN-AQI value.
 * @return Constant string descriptor.
 */
const char *tinyml_get_aqi_category(int aqi);

// ============================================================================
// LCD SPI Screen Buffer Formatter
// ============================================================================

/**
 * @brief Formats text representation of telemetry for 2.8" SPI TFT LCD (ST7789).
 *
 * @param payload Pointer to display payload data.
 * @param out_buf Output character buffer.
 * @param buf_size Maximum capacity of output buffer.
 */
void lcd_format_screen_buffer(const EdgeDisplayPayload *payload, char *out_buf, size_t buf_size);

// ============================================================================
// MQTT Telemetry JSON Formatter
// ============================================================================

/**
 * @brief Formats EdgeTelemetryPacket into standard JSON telemetry string for MQTT broker.
 *
 * @param pkt Pointer to telemetry packet.
 * @param out_json Output character buffer.
 * @param max_len Maximum length of out_json.
 * @return Number of characters written on success, or -1 on buffer overflow or error.
 */
int mqtt_format_telemetry_json(const EdgeTelemetryPacket *pkt, char *out_json, size_t max_len);

// ============================================================================
// Offline Circular Ring Buffer (Fault-Tolerant PSRAM/SRAM Storage)
// ============================================================================

#define OFFLINE_RINGBUF_CAPACITY 256

/**
 * @brief Pushes a 32-byte telemetry packet into the static offline ring buffer.
 *
 * @param pkt Packet to insert.
 * @return true if pushed successfully; false if ring buffer is full.
 */
bool offline_ringbuf_push(const EdgeTelemetryPacket *pkt);

/**
 * @brief Pushes a 32-byte telemetry packet into the static offline ring buffer.
 * If buffer is full, automatically drops the oldest sample (FIFO circular overwrite)
 * and increments dropped counter to maintain newest time series history.
 *
 * @param pkt Packet to insert.
 * @return true if pushed successfully (either free slot or overwrite); false if pkt is NULL.
 */
bool offline_ringbuf_push_overwrite(const EdgeTelemetryPacket *pkt);

/**
 * @brief Pops the oldest 32-byte telemetry packet from the static offline ring buffer (FIFO).
 *
 * @param out_pkt Destination packet buffer.
 * @return true if an item was popped; false if buffer is empty.
 */
bool offline_ringbuf_pop(EdgeTelemetryPacket *out_pkt);

/**
 * @brief Returns the number of packets currently stored in the ring buffer.
 */
size_t offline_ringbuf_count(void);

/**
 * @brief Returns maximum capacity of the ring buffer.
 */
size_t offline_ringbuf_capacity(void);

/**
 * @brief Returns true if ring buffer is empty.
 */
bool offline_ringbuf_is_empty(void);

/**
 * @brief Returns true if ring buffer is full.
 */
bool offline_ringbuf_is_full(void);

/**
 * @brief Clears all entries in the ring buffer.
 */
void offline_ringbuf_clear(void);

/**
 * @brief Returns total count of packets dropped due to buffer overflow since last reset.
 */
size_t offline_ringbuf_dropped_count(void);

/**
 * @brief Resets the dropped packets counter to zero.
 */
void offline_ringbuf_reset_dropped(void);

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_MQTT_HAL_H
