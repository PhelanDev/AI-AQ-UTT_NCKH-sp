# Air Quality Monitoring and Forecasting System (AQ_Lite)

A two-tier AIoT system for real-time microclimate air quality monitoring and 24-hour $PM_{2.5}$ forecasting. Developed and evaluated at the University of Transport Technology (UTT), 54 Trieu Khuc, Thanh Xuan, Hanoi, Vietnam.

The system combines on-device TinyML inference on an ESP32-S3 microcontroller with edge computing and spatial interpolation on a Raspberry Pi 5 gateway.

---

## 1. System Overview

AQ_Lite uses a distributed architecture designed for low power consumption, deterministic memory allocation, and verifiable machine learning models.

```text
[ Edge Station: ESP32-S3 WROOM-1 ]
  ├── Sensors: Plantower PMS7003 (UART) + Sensirion SHT40 (I2C)
  ├── Edge TinyML: Standard MLP (833 parameters, Symmetric INT8, 0-Byte Heap)
  ├── Local Display: 2.0" ST7789 IPS LCD via SPI @ 20 MHz (VN-AQI Color Scale)
  └── Telemetry: MQTT over Wi-Fi / ESP-NOW
            │
            │ (15-Minute FSM Duty Cycle)
            ▼
[ Central Gateway: Raspberry Pi 5 ]
  ├── Data Ingestion & Validation: Rolling 3-Sigma (Z > 3) + QCVN 05:2023/BTNMT
  ├── Time-Series Forecaster: 1-Layer Vanilla GRU (32 hidden units, 6,192 parameters)
  ├── Spatial Mapping Engine: Inverse Distance Weighting (IDW, p = 2.0)
  └── Web Dashboard: FastAPI backend with Leaflet.js real-time geospatial UI
```

### Core Operational Principles

- **Zero Dynamic Heap Allocation**: The firmware allocates all tensor buffers, rolling sample arrays, and quantized model weights statically (`static int8_t` / `int32_t`). It avoids `malloc()`, `calloc()`, and `new`, preventing runtime memory leaks and heap fragmentation during unattended outdoor operation.
- **Microsecond Edge Inference**: The quantized INT8 Multi-Layer Perceptron runs in approximately 0.30 microseconds on the 240 MHz Xtensa dual-core processor, executing over 3,300,000 inferences per second.
- **15-Minute Power Management FSM**: A finite-state machine cycles through sensor preheating (30 s), burst sampling (10 s), TinyML calibration (0.3 µs), telemetry and display refresh (5 s), and deep sleep (14 m 15 s). Deep sleep current drops to approximately 10 µA, supporting long-term battery and solar operation.
- **Statistical and Regulatory Validation**: Gateway filtering combines national ambient air quality standards (QCVN 05:2023/BTNMT) with rolling 3-Sigma z-score thresholds to eliminate optical noise from high relative humidity ($RH > 90\%$) and transient sensor faults.

---

## 2. Technical Specifications

| Parameter | Edge Node (ESP32-S3) | Gateway Server (Raspberry Pi 5) |
| :--- | :--- | :--- |
| **Hardware** | ESP32-S3 WROOM-1 (Xtensa LX7 @ 240 MHz) | Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz |
| **System Memory** | 512 KB SRAM, 384 KB ROM, Octal PSRAM isolated | 4 GB / 8 GB LPDDR4X SDRAM |
| **Primary Sensors** | Plantower PMS7003 (UART), Sensirion SHT40 (I2C) | Real-time MQTT telemetry feed |
| **Model Architecture** | Standard MLP: $8 \to 32 \to 16 \to 1$ (ReLU) | Vanilla GRU: 1 Layer, 32 units ($14 \to 32 \to 24\times2$) |
| **Model Parameters** | 833 parameters | 6,192 parameters |
| **Quantization & Format** | Symmetric INT8 (`model_data.h`, 784 bytes) | Float32 ONNX Runtime / PyTorch (`.pt`) |
| **Memory Allocation** | 0 Bytes Dynamic Heap (100% static arrays) | Linux virtual memory |
| **Inference Latency** | ~0.30 µs per sample | 1.6 ms per frame (CPU load < 1%) |
| **Accuracy / Performance** | $R^2 = 0.8182$, $\text{RMSE} = 11.23\ \mu\text{g/m}^3$ | 24-hour horizon point and interval forecast |
| **Anomaly Detection** | Physical boundary sanity check | Shallow Autoencoder MSE (4,208 params) + 3-Sigma |
| **Spatial Mapping** | N/A | Inverse Distance Weighting (IDW, $p = 2.0$) |
| **Testing Coverage** | 17/17 C++ Native Tests PASS | 46/46 Python Pytest Suite PASS |

---

## 3. Architecture Trade-Off: Academic Master vs. Production Lite

The repository represents the applied production edition of the system, configured to eliminate unnecessary complexity while retaining full neural network functionality:

| Evaluation Metric | Academic Master Edition (`research/academic-full`) | Production Lite Edition (`main`) | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Edge Neural Model** | Residual MLP (ResMLP, LayerNorm, Skip Connections) | Standard MLP (2 hidden layers, ReLU) | Avoids overfitting on an 8-dimensional tabular feature vector. |
| **Edge Parameter Count** | 19,297 parameters | 833 parameters | Reduces parameter footprint by 23x. |
| **Weight Table Size** | ~19.3 KB Flash | 784 Bytes Flash (< 1 KB) | The entire weight array fits inside L1 CPU cache. |
| **Edge Inference Latency** | 10.60 µs (94,340 inf/s) | 0.30 µs (> 3,330,000 inf/s) | 35x faster execution speed. |
| **Edge Calibration Accuracy** | $R^2 \approx 0.88$, $\text{RMSE} \approx 8.9\ \mu\text{g/m}^3$ | $R^2 = 0.8182$, $\text{RMSE} = 11.23\ \mu\text{g/m}^3$ | Accuracy difference (~2.3 µg/m³) stays well within PMS7003 sensor tolerance ($\pm 10\ \mu\text{g/m}^3$). |
| **Time-Series Forecaster** | Dilated TCN + Bi-GRU + Dual Heads (~43k params) | Vanilla GRU 1-Layer (6,192 params) | Minimizes training sample demands and short-horizon overfitting. |
| **Spatial Interpolation** | Ordinary Kriging (Semivariogram matrix inversion) | Inverse Distance Weighting (IDW, $p = 2.0$) | Eliminates singular matrix crashes when active node counts are small ($N < 4$). |
| **Outlier Detection** | 16D Deep Autoencoder + Mahalanobis Distance | 3-Sigma ($Z > 3$) + QCVN 05:2023 Thresholds | Replaces covariance matrix inversions with direct statistical checks. |

---

## 4. Repository Structure

```text
E:\AQ_Lite\
├── esp/                              # ESP32-S3 C++ firmware and TinyML engine
│   ├── include/
│   │   ├── config.h                  # Hardware pinout, SPI LCD pins, Octal PSRAM lock
│   │   ├── model_data.h              # Quantized INT8 weight table (784 Bytes)
│   │   └── tinyml_edge_engine.h      # 0-Heap inference engine declarations
│   ├── src/
│   │   ├── main.cpp                  # FreeRTOS FSM cycle and sleep scheduling
│   │   ├── tinyml_edge_engine.cpp    # Symmetric INT8 matrix multiplication engine
│   │   └── display_mqtt_hal.cpp      # ST7789 SPI driver and MQTT telemetry buffer
│   └── tests/                        # Native C++ unit tests (17/17 tests passing)
├── pi/                               # Central Gateway services on Raspberry Pi 5
│   ├── central_engine.py             # Data ingestion, preprocessing, and scheduling
│   ├── forecasting_engine.py         # 24-hour Vanilla GRU forecasting model
│   ├── anomaly_engine.py             # 3-Sigma statistical and QCVN boundary filter
│   ├── spatial_temporal_engine.py    # IDW spatial heatmapping engine (p = 2.0)
│   └── web_server.py                 # FastAPI REST API and Leaflet web dashboard
├── mlops/                            # Training and model quantization pipeline
│   ├── data/                         # Field co-location datasets (Hanoi, Beijing, Jakarta)
│   ├── models/                       # Checkpoints (.pt) and exported ONNX models
│   └── scripts/
│       ├── train_and_quantize_standard_mlp_lite.py # Edge MLP training and C header export
│       ├── train_lite_central_models.py           # Central GRU forecaster training
│       └── train_all_lite.py                      # Master pipeline for all four models
├── docs/                             # Engineering specifications and comparison reports
│   ├── SO_SANH_BAN_FULL_VA_LITE.md   # Architectural comparison and defense guide
│   └── reports/                      # Benchmark data and analytical records
└── tests/                            # Pytest test suite for gateway services (46/46 passing)
```

---

## 5. Build, Test, and Execution Guide

### 5.1. Native C++ Firmware Testing

The firmware unit tests compile on any standard C++17 compiler (MinGW, GCC, Clang) without requiring physical hardware:

```powershell
# Navigate to ESP test directory
Set-Location E:\AQ_Lite\esp\tests

# Compile and run the 15-minute power management FSM test
g++ -std=c++17 -I../include test_firmware_15min_cycle.cpp ../src/display_mqtt_hal.cpp -o test_firmware_15min_cycle.exe
.\test_firmware_15min_cycle.exe

# Compile and run the TinyML INT8 engine test (latency and numerical equivalence)
g++ -std=c++17 -I../include test_edge_engine.cpp ../src/tinyml_edge_engine.cpp -o test_edge_engine.exe
.\test_edge_engine.exe
```

### 5.2. Python Environment Setup and Gateway Tests

Install required dependencies and execute the automated test suite:

```powershell
Set-Location E:\AQ_Lite

# Install dependencies
pip install -r requirements.txt

# Run all gateway and MLOps tests
pytest tests/ -v
```

### 5.3. Retraining and Model Export

To retrain the neural network models and regenerate the embedded C header:

```powershell
Set-Location E:\AQ_Lite

# Train and quantize the Edge Standard MLP model, outputting esp/include/model_data.h
python mlops/scripts/train_and_quantize_standard_mlp_lite.py

# Train the 24-hour Vanilla GRU forecasting model for Raspberry Pi 5
python mlops/scripts/train_lite_central_models.py

# Or run the complete automated pipeline for all models
python mlops/scripts/train_all_lite.py
```

### 5.4. Launching the Gateway Web Server

Start the FastAPI backend with the Leaflet.js real-time monitoring map:

```powershell
Set-Location E:\AQ_Lite
python -m uvicorn pi.web_server:app --host 0.0.0.0 --port 8000 --reload
```

Open a web browser and navigate to `http://localhost:8000` to inspect real-time sensor streams, forecasted 24-hour trends, and spatial pollution distribution.

---

## 6. Key Engineering Defenses

### Why use a Standard MLP rather than CNN or Transformer on the ESP32-S3?
The input vector consists of 8 physical variables (temperature, relative humidity, raw optical dust counts from dual channels, and derived dew point coefficients). It has no 2D spatial locality like images, nor temporal grammatical dependencies like text. A 2-layer MLP with 833 parameters satisfies the Universal Approximation Theorem for nonlinear hygroscopic growth compensation, fitting inside 784 bytes of Flash ROM with zero dynamic memory overhead.

### How does symmetric INT8 quantization preserve accuracy?
Floating-point weights and activations are mapped to 8-bit signed integers using a uniform scaling factor $S$:

$$x_{\text{int8}} = \text{clamp}\left(\left\lfloor \frac{x_{\text{float}}}{S} + 0.5 \right\rfloor, -128, 127\right)$$

Quantization loss on the empirical test set is under 0.5% ($R^2$ decreases from 0.822 to 0.818), while inference latency drops to 0.30 µs and memory consumption drops by 75%.

### Why select Vanilla GRU over LSTM or ARIMA on the Raspberry Pi 5?
Air pollution series follow physical dispersion inertia and distinct diurnal cycles. ARIMA assumes linear relationships and struggles with nonlinear microclimate spikes. Compared to LSTM, the Gated Recurrent Unit merges the forget and input gates into an update gate, eliminating the separate cell state $C_t$. This design removes 25% of parameter overhead (6,192 parameters total), completing inference in 1.6 ms with minimal CPU utilization.

---

## 7. License

This project is licensed under the MIT License. See the LICENSE file for details.
