# DiveGuard: Technical Specification & Backlog
## Propeller Detection System for Diver Safety

**Version**: 1.0 | **Status**: APPROVED | **Date**: 2026-07-24

---

## 📋 Executive Summary

**DiveGuard** is a real-time underwater propeller detection system designed to protect subsurface divers from collision risk with motorized vessels. The system combines:

1. **Hardware**: MEMS hydrophone (I2S/ALSA)
2. **Signal Processing**: LOFAR + DEMON (C++ DSP core)
3. **Machine Learning**: Quantized CNN (TFLite, 1.2MB)
4. **Bioacoustic Filtering**: Multi-species marine mammal discrimination
5. **Integration**: BlueOS extension + MAVLink alerts
6. **Deployment**: Raspberry Pi 3 / Jetson Nano

---

## 🎯 Functional Requirements

### FR-1: Real-Time Propeller Detection
- **Definition**: System must identify motorized vessel propeller acoustic signature within 100ms
- **Metrics**: 
  - Sensitivity: ≥90% @ SNR = -5dB
  - Specificity: ≥95% (false positive rate <5%)
  - Response latency: <100ms from acoustic event to MAVLink alert
- **Sources**:
  - Medwin, H. (1975). Speed of sound in water. J. Acoust. Soc. Am., 58(6), 1318-1321.
  - Weston, D. E., & Ching, P. A. (1989). Wind-generated noise modelling in assessment of underwater noise. J. Acoust. Soc. Am., 86(3), 1012-1023.

### FR-2: Marine Mammal Discrimination
- **Definition**: System must distinguish propeller noise from whale/dolphin vocalizations
- **Metrics**:
  - False negative rate (missing whale vocalization): <2%
  - False positive rate (misclassifying whale as propeller): <3%
- **Sources**:
  - Mellinger, D. K., & Clark, C. W. (2000). Recognizing transient low-frequency whale sounds by spectrogram correlation. J. Acoust. Soc. Am., 107(6), 3518-3529.
  - Gulesserian, T., et al. (2020). DeepShip: An underwater acoustic benchmark dataset. ICML 2020 Workshop.

### FR-3: Acoustic Environment Adaptation
- **Definition**: System must dynamically adjust detection threshold based on thermal profile
- **Metrics**:
  - Calibration accuracy: ±0.5% of sound velocity
  - Depth range: 0-300m
  - Temperature range: 0-35°C
- **Sources**:
  - Medwin, H., & Clay, C. S. (1998). Fundamentals of acoustical oceanography. Academic Press.

### FR-4: Low-Power Operation
- **Definition**: Total system power consumption must support ROV battery life
- **Metrics**:
  - DSP core: <20mW
  - ML inference: <30mW
  - Hydrophone: <10mW
  - Total: <60mW
- **Sources**:
  - Raj, B., et al. (2017). TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. O'Reilly.

### FR-5: Bioacoustic Library Integration
- **Definition**: Support detection of multiple marine mammal species
- **Species Coverage**:
  - Mysticetes (baleen whales): Balaenoptera musculus, Megaptera novaeangliae
  - Odontocetes (toothed whales): Physeter macrocephalus, Tursiops truncatus
  - Pinnipeds (seals): Halichoerus grypus, Zalophus californianus
- **Source Database**: Watkins Marine Mammal Sound Database (MMSD) via NOAA/NCEI

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DIVE GUARD SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HARDWARE LAYER (Subsea/Topside)                    │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────────┐   │  │
│  │  │ Delonic     │  │ DS18B20  │  │ Pressure    │   │  │
│  │  │ MEMS Hydro  │  │ Temp     │  │ Sensor      │   │  │
│  │  │ (I2S/48kHz) │  │ Sensor   │  │ (CTD)       │   │  │
│  │  └─────────────┘  └──────────┘  └──────────────┘   │  │
│  │        │                                            │  │
│  └────────┼────────────────────────────────────────────┘  │
│           │                                                 │
│  ┌────────▼────────────────────────────────────────────┐  │
│  │  SIGNAL PROCESSING LAYER (C++ DSP Core, <100ms)   │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  1. ALSA Ring Buffer (Lock-free)             │  │  │
│  │  │     ├─ 512-sample frames @ 48 kHz            │  │  │
│  │  │     └─ ~10.6ms latency per frame             │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  2. Hamming Window + KISS-FFT (LOFAR)        │  │  │
│  │  │     ├─ 1024-point FFT (21.3ms window)        │  │  │
│  │  │     ├─ 46.9 Hz frequency resolution          │  │  │
│  │  │     └─ Output: 512 frequency bins            │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  3. DEMON Algorithm (Envelope Extraction)    │  │  │
│  │  │     ├─ Hilbert transform (phase shift)       │  │  │
│  │  │     ├─ Envelope detection (magnitude)        │  │  │
│  │  │     ├─ Decimate 100:1 (480Hz → 4.8Hz)      │  │  │
│  │  │     └─ Reveals blade-pass-frequency (BPF)   │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  4. Spatial Filtering (Multi-hydrophone)     │  │  │
│  │  │     ├─ Beamforming (if N≥2 hydrophones)     │  │  │
│  │  │     ├─ Time-Difference-of-Arrival (TDOA)    │  │  │
│  │  │     └─ Directional masking (suppress vessel)│  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  5. Bioacoustic Pre-filter                   │  │  │
│  │  │     ├─ Whale vocalization detector           │  │  │
│  │  │     ├─ Critter noise suppression             │  │  │
│  │  │     └─ SNR estimation                        │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  6. TFLite Inference (INT8)                  │  │  │
│  │  │     ├─ Model: MobileNetV2-1D CNN             │  │  │
│  │  │     ├─ Input: 64×128 mel-spectrogram        │  │  │
│  │  │     ├─ Output: [propeller_prob, bg_prob]    │  │  │
│  │  │     └─ Latency: 12ms @ INT8                 │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  7. Thermal Calibration                      │  │  │
│  │  │     ├─ Sound velocity correction             │  │  │
│  │  │     ├─ Dynamic threshold adjustment          │  │  │
│  │  │     └─ Confidence weighting                  │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│           │                                                 │
│  ┌────────▼────────────────────────────────────────────┐  │
│  │  BRIDGE LAYER (Python + ZMQ IPC, 5mW)             │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • IPC socket: /tmp/dive_guard_detections         │  │
│  │  • Message format: JSON                            │  │
│  │  • Async event loop                               │  │
│  └──────────────────────────────────────────────────────┘  │
│           │                                                 │
│  ┌────────▼────────────────────────────────────────────┐  │
│  │  APPLICATION LAYER (FastAPI + BlueOS)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ Detection    │  │ MAVLink  │  │ Vue.js      │  │  │
│  │  │ History DB   │  │ Bridge   │  │ Dashboard   │  │  │
│  │  │ (JSON)       │  │          │  │             │  │  │
│  │  └──────────────┘  └──────────┘  └─────────────┘  │  │
│  │                                                    │  │
│  │  API Endpoints:                                   │  │
│  │  ├─ GET  /detection          (latest)             │  │
│  │  ├─ GET  /detections/history (50 events)          │  │
│  │  ├─ GET  /metrics            (performance)        │  │
│  │  ├─ WS   /ws/alerts          (real-time)          │  │
│  │  └─ POST /settings           (config)             │  │
│  │                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Technical Specifications

### 1. Audio Acquisition
```
Parameter               | Value                | Justification
------------------------|-----------------------|------------------
Hydrophone Type         | Delonic DT-206 MEMS | I2S digital output, low power
Frequency Response      | 10 Hz - 250 kHz     | Covers all vessel + marine mammal signatures
Sample Rate             | 48 kHz              | Nyquist: 24 kHz (covers up to 8 kHz propeller)
Bit Depth               | 16-bit              | S16_LE (signed 16-bit little-endian)
Frame Size              | 512 samples         | ~10.6ms @ 48 kHz (balance: latency vs. FFT resolution)
Ring Buffer Capacity    | 16,384 samples      | ~341ms (handles jitter, avoids drops)
```

### 2. DSP Processing Chain
```
Stage              | Algorithm          | Input/Output        | Latency
-------------------|-------------------|---------------------|----------
1. Window          | Hamming            | 1024 samples        | <1ms
2. FFT             | KISS-FFT (1024)    | 512 frequency bins  | 8ms
3. Power Spectrum  | |X[k]|² to dB      | Float [0, 80]       | 1ms
4. Bandpass        | IIR Butterworth    | 40-8000 Hz          | 2ms
5. Hilbert         | Phase shift (FIR)  | 64-tap kernel       | 3ms
6. Envelope        | Magnitude          | Float envelope      | 1ms
7. Decimate        | 100:1 downsampling | 480 Hz → 4.8 Hz     | 1ms
8. DEMON FFT       | KISS-FFT (512)     | BPF detection       | 2ms
---                                                       TOTAL: <20ms
```

### 3. Machine Learning Model
```
Layer Type          | Config                  | Parameters | Macs
--------------------|-------------------------|------------|--------
Input Conv          | 64→32, 3×1 kernel       | 192        | 24K
Inv. Res Block 1    | 32→64, expand=6, s=2    | 4.2K       | 256K
Inv. Res Block 2    | 64→64, expand=6, s=1    | 5.4K       | 128K
Inv. Res Block 3    | 64→128, expand=6, s=2   | 15.6K      | 128K
Inv. Res Block 4    | 128→128, expand=6, s=1  | 28.8K      | 64K
Inv. Res Block 5    | 128→256, expand=6, s=2  | 51.2K      | 32K
Global Pool + FC    | 256→128→2               | 32.8K      | 16.8K
--------------------|-------------------------|-----------|---------
TOTAL               |                         | 1.2M      | 650K
Quantization        | INT8 (dynamic range)    | 300KB     | N/A
Model File Size     | .tflite (TFLite)        | 1.2 MB    | N/A
Inference Time      | TFLite CPU              | 12ms      | N/A
Power (inference)   | Avg @ 1kHz detection    | 28mW      | N/A
```

### 4. Detection Performance (Target)

| Metric | Target | Source Data |
|--------|--------|------------|
| **Sensitivity (TPR)** @ SNR=-5dB | ≥90% | ShipsEar dataset |
| **Specificity (TNR)** | ≥95% | Whale vocalizations (MMSD) |
| **PPV (Precision)** | ≥92% | Mixed ocean ambient |
| **Detection Latency** | <100ms | Real-time constraint |
| **False Positive Rate** | <5% | 24h continuous monitoring |
| **False Negative Rate** (propeller miss) | <10% | Critical safety metric |
| **ROC-AUC** | ≥0.95 | Classification threshold optimization |

### 5. Environmental Calibration

```cpp
// Sound Velocity Model (Medwin, 1975)
v(T, S, P) = 1449.05 + 45.7T - 5.21T² + 0.1T³ 
           + (1.333 - 0.126T + 0.009T²)(S - 35) 
           + 16.3P + 0.2P²

where:
  T = Temperature [°C], range: 0-35
  S = Salinity [PSU], range: 30-35
  P = Depth [m], range: 0-300

// Typical values:
T=20°C, S=35PSU, P=0m   → v ≈ 1536 m/s
T=4°C,  S=35PSU, P=100m → v ≈ 1468 m/s (9% lower!)
```

**Impact on Detection**:
- Thermocline creates acoustic lens (sound ray bending)
- Propeller at 1km+ may be undetectable in cold water
- Detection threshold must increase by ~8% per 100m depth

---

## 🧬 Phase-by-Phase Implementation Backlog

### **PHASE 1: Hardware Interface (Week 1-2)**

#### Task P1.1: MEMS Hydrophone Integration
- **Subtask 1.1.1**: Procure Delonic DT-206, SPI/I2S converter board
  - Estimated cost: $45-60
  - Vendor verification: Allied Electronics, Mouser
  - Test bench setup: 3-axis hydrophone mount + aluminum housing
  
- **Subtask 1.1.2**: I2S Device Tree Overlay Configuration
  - File: `/boot/overlays/i2s-dac.dtbo`
  - Pins: GPIO 12 (BCLK), GPIO 35 (LRCLK), GPIO 40 (DOUT)
  - Reference: Raspberry Pi Forums, I2S tutorial
  - Verification: `arecord -D hw:0 -f S16_LE -r 48000 -d 5 test.wav`

- **Subtask 1.1.3**: ALSA Configuration & Calibration
  - Configure `asound.conf`: capture device, buffer parameters
  - Run: `alsamixer`, set levels to 0dB reference
  - Frequency response test: Sweep 50-10kHz, verify <±3dB deviation
  - Time sync: Synchronize audio clock with system clock (NTP)

**Dependencies**: Hardware procurement  
**Effort**: 40 hours  
**Risk**: I2S driver compatibility with older Raspbian versions

---

#### Task P1.2: Temperature & Depth Sensor Integration
- **Subtask 1.2.1**: DS18B20 1-Wire Protocol
  - Pin: GPIO 4 (1-Wire data line)
  - Library: `RPi.GPIO` or `Adafruit_CircuitPython_DS18B20`
  - Calibration: Ice bath (0°C) + hot water (60°C) verification
  - Sampling: 1Hz (update every 1 second)
  
- **Subtask 1.2.2**: I2C Pressure Sensor (CTD)
  - Device: Blue Robotics Bar30 High-Resolution
  - I2C Address: 0x76
  - Sampling rate: 10Hz
  - Conversion: Raw pressure → Depth (using salinity correction)

**Dependencies**: Task P1.1  
**Effort**: 20 hours

---

### **PHASE 2: Ring Buffer & Lock-Free Architecture (Week 2)**

#### Task P2.1: Concurrent Ring Buffer (C++)
- **Code file**: `dsp_core/ring_buffer.hpp`
- **Implementation**: 
  ```cpp
  template<size_t N> class AudioRingBuffer {
    std::atomic<uint32_t> write_pos, read_pos;
    std::array<int16_t, N> buffer;
    // push(), pop(), size() with memory_order_release/acquire
  }
  ```
- **Unit Tests**: 
  - Single-threaded correctness (write/read 1M samples)
  - Multi-threaded stress (capture + DSP threads)
  - Wrap-around boundary conditions
- **Benchmarks**: Latency distribution, cache misses (perf stat)

**Effort**: 16 hours  
**Testing Tools**: Google Test (gtest), perf counters

---

### **PHASE 3: LOFAR Spectrogram Processing (Week 3)**

#### Task P3.1: KISS-FFT Integration
- **Library**: `kiss_fft.h` (bundled, ~600 LOC)
- **Wrapper class**: `LOFARAnalyzer`
  - Input: 1024 int16_t samples
  - Output: 512 float frequency bins [0, 80 dB range]
- **Window function**: Hamming (pre-computed, not re-calculated)
- **Optimization**: 
  - Cache-friendly: sequential memory access
  - SIMD: Auto-vectorized by g++ -O3 -march=native
  - Benchmark target: <3ms per 1024-sample FFT

**Code file**: `dsp_core/lofar_analyzer.cpp`  
**Effort**: 24 hours

---

#### Task P3.2: Spectrogram Normalization
- **Algorithm**: 
  1. Linear magnitude: |X[k]|²
  2. Logarithmic scaling: 20 log₁₀(|X[k]| + ε), ε=1e-7
  3. Normalization: Subtract rolling noise floor estimate
  4. Output range: [0, 1] (for ML input)

- **Noise floor estimation**:
  ```cpp
  // Median filtering over time (last 30 frames)
  noise_floor[k] = median(spectrogram[t-30:t, k])
  normalized[t, k] = clip((spectrogram[t,k] - noise_floor[k]) / 60, 0, 1)
  ```

**Testing**: Synthetic propeller + whale sound mixtures  
**Effort**: 12 hours

---

### **PHASE 4: DEMON Algorithm (Envelope Modulation) (Week 3-4)**

#### Task P4.1: Hilbert Transform (FIR Implementation)
- **Method**: 65-tap FIR Hilbert transformer
- **Kernel generation**: 
  ```python
  h[n] = (2/π) * sin²(πn/2) / n  for n ≠ 0
  h[0] = 0
  ```
- **Window**: Hamming window to reduce side-lobes
- **Latency**: 33 samples (~0.7ms @ 48 kHz)

**Effort**: 8 hours  
**Validation**: Test on synthetic AM signal

---

#### Task P4.2: Envelope Detection & Downsampling
- **Envelope**: `sqrt(real² + imag²)` (complex magnitude)
- **Downsampling**: 100:1 decimation (48kHz → 480Hz)
- **Anti-aliasing**: Implement 4th-order Chebyshev filter before decimation

**Code file**: `dsp_core/demon_detector.cpp`  
**Effort**: 16 hours

---

#### Task P4.3: Blade Pass Frequency Extraction
- **FFT on decimated envelope**: 512-point FFT (480Hz → ~0.94Hz bins)
- **Peak detection**: Find 3 strongest peaks (BPF + 2 harmonics)
- **Frequency range**: 0.5-10 Hz (corresponds to 30-600 RPM)
- **Peak width**: ±1 Hz tolerance (vessel acceleration allowance)

**Effort**: 12 hours  
**Benchmarks**: Test on ShipsEar dataset

---

### **PHASE 5-6: Dataset Preparation & Model Training (Week 4-6)**

#### Task P5.1: ShipsEar Dataset Processing
- **Source**: stef729/ShipsEar (GitHub, 12 vessel classes)
- **Audio preprocessing** (Python/PyTorch):
  ```python
  # Load .wav, normalize to [-1, 1]
  # Resample to 48 kHz if needed
  # Segment: 2-second windows with 1-second stride
  # Total: ~90 hours audio → ~3000 training samples
  ```
- **Labeling**: Binary classification
  - Class 0 (Negative): ambient, whale, critter, wind
  - Class 1 (Positive): cargo, tanker, fishing, tugboat, passenger
- **Data augmentation**:
  - Time shift (±10 frames)
  - Frequency masking (10 bins)
  - Time masking (20 frames)
  - Gaussian noise (SNR -10 to +20 dB)
  - → 5x data expansion

**Source Paper**: Gulesserian et al. (2020), "DeepShip"  
**Effort**: 20 hours

---

#### Task P5.2: DeepShip Dataset Integration
- **Source**: Canadian dataset (500+ hours)
- **AIS Integration**: Align audio with Automatic Identification System vessel data
- **Filtering**: Retain only recordings with ground-truth vessel type
- **Training/Val/Test split**: 70/15/15

**Effort**: 16 hours

---

#### Task P6.1: CNN Model Training (PyTorch + Azure/Colab)
- **Model**: MobileNetV2-1D (1.2M parameters)
- **Training config**:
  ```
  Optimizer: Adam (lr=0.001, weight_decay=1e-5)
  Loss: CrossEntropyLoss (class weights [0.3, 0.7])
  Batch size: 32
  Epochs: 50
  Validation metric: ROC-AUC (target ≥0.95)
  ```
- **Early stopping**: Monitor validation loss, patience=5

**Code file**: `ml_training/train_model.py`  
**Effort**: 32 hours (including hyperparameter tuning)

---

#### Task P6.2: Model Quantization & Export to TFLite
- **Framework**: TensorFlow → ONNX → TFLite
- **Quantization**: INT8 (dynamic range)
- **Representative data**: 100 samples from validation set
- **Output model**: `dive_guard_quantized.tflite` (~1.2 MB)
- **Verification**: 
  - Compare FP32 vs INT8 predictions (cosine similarity >0.98)
  - Measure inference time on Raspberry Pi (target: <12ms)

**Effort**: 12 hours

---

### **PHASE 7-8: TFLite C++ Integration & Edge Deployment (Week 7-8)**

#### Task P7.1: TFLite C++ Runtime
- **Build**: Static library `libtensorflowlite.a`
- **Interpreter setup**: 
  ```cpp
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("dive_guard_quantized.tflite");
  ```
- **Tensor allocation**: `interpreter->AllocateTensors()`
- **Input quantization**: Map float spectrogram → int8 (scale, zero_point)

**Effort**: 20 hours

---

#### Task P7.2: Real-Time Inference Loop
- **Main loop**:
  ```cpp
  while (running) {
    size_t got = ring_buf.pop(frame, 1024);
    if (got == 1024) {
      lofar.analyze(frame, mel_spec);      // LOFAR
      demon.detect(frame, envelope);       // DEMON
      
      auto result = classifier.classify(mel_spec);  // TFLite
      
      if (result.propeller_confidence > dynamic_threshold) {
        send_zmq_detection(result);
      }
    }
    usleep(1000);  // 1ms poll interval
  }
  ```
- **Power consumption**: Target <50mW total DSP
- **Profiling**: Use `perf` to identify bottlenecks

**Effort**: 16 hours

---

### **PHASE 9: BlueOS Docker Container (Week 9)**

#### Task P9.1: Dockerfile & Build
- **Base image**: `blueos:latest`
- **Build stages**:
  - Stage 1: Compile C++ DSP core (cmake)
  - Stage 2: Install TFLite runtime
  - Stage 3: Python FastAPI server
- **Final size target**: <200 MB (compressed: <80 MB)

**Effort**: 12 hours

---

#### Task P9.2: BlueOS Extension Manifest
- **File**: `extension_manifest.json`
- **Properties**:
  - `docker.privileged: true` (GPIO, I2C access)
  - `docker.volumes`: Mount hydrophone device tree
  - `permissions.audio: required`
  - `permissions.i2c: required`

**Effort**: 4 hours

---

### **PHASE 10: FastAPI Bridge & ZMQ IPC (Week 9-10)**

#### Task P10.1: C++ → Python Bridge (ZMQ)
- **IPC socket**: `ipc:///tmp/dive_guard_detections`
- **Message format**: JSON
  ```json
  {
    "prop_conf": 0.92,
    "bg_conf": 0.08,
    "bpf_hz": 127.3,
    "timestamp_us": 1234567890000,
    "depth": 45.2,
    "temp": 8.5
  }
  ```
- **Latency**: <5ms (IPC overhead)

**Effort**: 12 hours

---

#### Task P10.2: FastAPI Endpoints
- **Endpoints** (as per FR):
  - `GET /detection` → Latest event
  - `GET /detections/history?limit=50` → Historical data
  - `GET /metrics` → Performance stats
  - `WS /ws/alerts` → Real-time WebSocket
  - `POST /settings` → Config updates

**Effort**: 16 hours

---

### **PHASE 11: Vue.js Dashboard (Week 10-11)**

#### Task P11.1: Real-Time Spectrogram Visualization
- **Library**: Chart.js or Plotly.js
- **Update rate**: 1Hz
- **Display**: Last 10 seconds of spectrogram (480 time frames)
- **Color map**: Viridis (0 dB = purple, 80 dB = yellow)

**Effort**: 16 hours

---

#### Task P11.2: Alert Banner & Statistics
- **Alert**: Prominently display detection with vessel type guess
- **Stats**: 24h detection count, rate/hour, false positive percentage
- **System health**: CPU, memory, inference latency

**Effort**: 12 hours

---

### **PHASE 12: MAVLink Integration & Field Testing (Week 11-12)**

#### Task P12.1: MAVLink STATUSTEXT Broadcast
- **Target**: ArduSub GCS (MAVProxy, QGroundControl)
- **Message format**:
  ```
  [DiveGuard] Propeller! cargo, 92%, 45m deep
  ```
- **Broadcast interval**: 100-500ms (avoid spam)

**Effort**: 8 hours

---

#### Task P12.2: Emergency Ascent Command (Optional)
- **Trigger**: Confidence >0.85 + medium frequency BPF (50-150 Hz)
- **Command**: `MAV_CMD_NAV_RETURN_TO_LAUNCH` (mode RTH)
- **Safety**: Operator approval before activation (GCS dialog)

**Effort**: 8 hours

---

#### Task P12.3: Field Testing Protocol
- **Test phases**:
  1. Tank tests (synthetic propeller audio playback)
  2. Controlled harbor (known vessel traffic)
  3. Open ocean (uncontrolled ambient)
  4. Whale sanctuary (validation against false positives)
- **Metrics collection**: Confusion matrix, ROC curves, latency histograms
- **Duration**: 80+ hours of dive time

**Effort**: 120 hours (field test + data analysis)

---

## 📚 Scientific Foundation (PubMed/Scholar Papers)

### Key References by Phase

| Phase | Citation | DOI | Key Contribution |
|-------|----------|-----|-----------------|
| 1-2 | Medwin, H. (1975) | 10.1121/1.380511 | Sound velocity model |
| 3 | Weston & Ching (1989) | 10.1121/1.395869 | Wind noise characterization |
| 4 | Deng et al. (2010) | JASA paper | DEMON algorithm formulation |
| 5 | Gulesserian et al. (2020) | ICML workshop | DeepShip dataset |
| 6 | Sandler et al. (2018) | CVPR | MobileNetV2 architecture |
| 7 | Bengio et al. (2015) | JMLR | Transfer learning |
| 10-12 | Mellinger & Clark (2000) | 10.1121/1.429588 | Whale sound recognition |

---

## 🎯 Success Criteria & Timeline

```
Week  1: Hardware setup (hydrophone, sensors)
Week  2: Ring buffer + LOFAR DSP core
Week  3: DEMON algorithm implementation
Week  4: Dataset preparation & CNN training begins
Week  5-6: Model training & hyperparameter tuning
Week  7: TFLite export & C++ integration
Week  8: Real-time DSP pipeline + profiling
Week  9: Docker container + FastAPI
Week  10: Dashboard + MAVLink integration
Week  11: Field testing (tank + harbor)
Week  12: Open ocean validation + final report
```

**Total Effort**: ~600-700 engineering hours  
**Team Size**: 4-5 engineers (signal processing, ML, embedded systems, integration)

---

## 🔬 Quality Assurance

### Unit Testing (C++)
- Ring buffer: capacity, wrap-around, thread safety
- FFT: known signals, energy conservation
- DEMON: synthetic AM signals, BPF extraction

### Integration Testing
- End-to-end: raw hydrophone → detection → MAVLink
- Multi-threaded: capture + DSP + API concurrency
- Performance: latency under load, memory profiling

### Field Testing
- Metric collection: Confusion matrix, ROC-AUC
- Robustness: Different water conditions (temperature, salinity, noise)
- Safety validation: False negative rate in divercontainment scenarios

---

## ⚠️ Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| I2S driver instability | Medium | High | Early prototype on Raspbian 32/64-bit |
| Acoustic masking (simultaneous whale + ship) | High | Medium | Implement spatial filtering (add 2nd hydrophone) |
| Quantization performance degradation | Low | Medium | Extensive INT8 validation vs FP32 |
| Power budget exceeded | Low | High | Profile each subsystem separately |
| False positive rate >5% | Medium | High | Expand training dataset to rare cases |

---

**Document Version**: 1.0  
**Status**: APPROVED FOR IMPLEMENTATION  
**Next Step**: Procurement of hardware + Team onboarding
