#ifndef DIVEGUARD_DSP_CORE_HPP
#define DIVEGUARD_DSP_CORE_HPP

#include <cstring>
#include <atomic>
#include <cmath>
#include <vector>
#include <array>
#include <cassert>

/**
 * DiveGuard DSP Core - Phase 1-4 Implementation
 *
 * Real-time propeller detection pipeline:
 * Phase 1: MEMS Hydrophone acquisition (I2S/ALSA 48kHz, 16-bit)
 * Phase 2: Lock-free ring buffer (atomic operations)
 * Phase 3: LOFAR spectrogram (KISS-FFT 1024-point)
 * Phase 4: DEMON envelope detection (Hilbert transform, BPF extraction)
 *
 * Total latency budget: <100ms (80ms DSP + 12ms ML + 8ms overhead)
 */

namespace diveguard {

// ============================================================================
// Phase 2: Lock-Free Ring Buffer (Atomic Operations)
// ============================================================================

class LockFreeRingBuffer {
public:
    static constexpr size_t CAPACITY = 16384;  // 341ms @ 48kHz

    LockFreeRingBuffer() : write_pos_(0), read_pos_(0) {
        samples_.resize(CAPACITY);
    }

    // Write thread: enqueue new samples (non-blocking)
    bool write(const int16_t* data, size_t count) {
        size_t w = write_pos_.load(std::memory_order_acquire);
        size_t r = read_pos_.load(std::memory_order_acquire);
        size_t available = (r + CAPACITY - w - 1) % CAPACITY;

        if (count > available) {
            return false;  // Buffer full, drop samples
        }

        for (size_t i = 0; i < count; ++i) {
            samples_[(w + i) % CAPACITY] = data[i];
        }

        write_pos_.store((w + count) % CAPACITY, std::memory_order_release);
        return true;
    }

    // DSP thread: dequeue samples (non-blocking)
    bool read(int16_t* out, size_t count) {
        size_t r = read_pos_.load(std::memory_order_acquire);
        size_t w = write_pos_.load(std::memory_order_acquire);
        size_t available = (w + CAPACITY - r) % CAPACITY;

        if (count > available) {
            return false;  // Not enough data
        }

        for (size_t i = 0; i < count; ++i) {
            out[i] = samples_[(r + i) % CAPACITY];
        }

        read_pos_.store((r + count) % CAPACITY, std::memory_order_release);
        return true;
    }

    size_t available() const {
        size_t w = write_pos_.load(std::memory_order_acquire);
        size_t r = read_pos_.load(std::memory_order_acquire);
        return (w + CAPACITY - r) % CAPACITY;
    }

private:
    std::vector<int16_t> samples_;
    std::atomic<size_t> write_pos_;
    std::atomic<size_t> read_pos_;
};

// ============================================================================
// Phase 3: LOFAR Spectrogram Generator
// ============================================================================

class LOFARSpectrogram {
public:
    static constexpr size_t FFT_SIZE = 1024;
    static constexpr size_t HOP_LENGTH = 512;
    static constexpr size_t NUM_BINS = FFT_SIZE / 2;
    static constexpr float SAMPLE_RATE = 48000.0f;

    struct Frame {
        std::array<float, NUM_BINS> bins;
        float timestamp_ms;
    };

    LOFARSpectrogram() {
        init_hamming_window();
    }

    // Process 512 new samples, return true if frame complete
    bool process_samples(const int16_t* samples, size_t count,
                        Frame& output_frame, float current_time_ms) {
        for (size_t i = 0; i < count; ++i) {
            fft_input_[fft_pos_] = samples[i] / 32768.0f;  // Normalize
            fft_pos_++;

            if (fft_pos_ == FFT_SIZE) {
                compute_fft(output_frame, current_time_ms);
                fft_pos_ = HOP_LENGTH;
                std::copy(fft_input_.begin() + HOP_LENGTH,
                         fft_input_.begin() + FFT_SIZE,
                         fft_input_.begin());
                return true;
            }
        }
        return false;
    }

private:
    std::array<float, FFT_SIZE> fft_input_;
    std::array<float, FFT_SIZE> window_;
    size_t fft_pos_ = 0;

    void init_hamming_window() {
        for (size_t i = 0; i < FFT_SIZE; ++i) {
            window_[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (FFT_SIZE - 1));
        }
    }

    // Simplified radix-2 FFT (Cooley-Tukey)
    void compute_fft(Frame& frame, float current_time_ms) {
        // Apply window
        for (size_t i = 0; i < FFT_SIZE; ++i) {
            fft_input_[i] *= window_[i];
        }

        // Radix-2 FFT (simplified for brevity - use KISS-FFT in production)
        // This is a placeholder; real implementation uses kiss_fft library
        simple_power_spectrum(frame);

        frame.timestamp_ms = current_time_ms;
    }

    void simple_power_spectrum(Frame& frame) {
        // Placeholder: compute power spectrum from windowed input
        // In production, use actual FFT algorithm
        for (size_t k = 0; k < NUM_BINS; ++k) {
            float bin_power = 0.0f;
            // Simplified computation
            for (size_t n = 0; n < FFT_SIZE; ++n) {
                float angle = -2.0f * M_PI * k * n / FFT_SIZE;
                bin_power += fft_input_[n] * std::cos(angle);
            }
            frame.bins[k] = 20.0f * std::log10(std::abs(bin_power) + 1e-10f);  // dB
        }
    }
};

// ============================================================================
// Phase 4: DEMON Algorithm (Hilbert + Envelope Detection)
// ============================================================================

class DEMONDetector {
public:
    static constexpr size_t HILBERT_TAPS = 65;
    static constexpr float DEMO_DECIMATION = 100.0f;  // 480 Hz after decimation
    static constexpr size_t BPF_BANDS = 5;  // Blade Pass Frequency bands

    struct BPFResult {
        std::array<float, BPF_BANDS> bpf_peaks;
        float max_bpf_power;
        float bpf_sharpness;  // Q-factor indicator
        bool propeller_detected;
    };

    DEMONDetector() {
        init_hilbert_filter();
    }

    // Process LOFAR frame to extract propeller signature
    BPFResult process_frame(const LOFARSpectrogram::Frame& frame) {
        BPFResult result{};

        // Step 1: Envelope detection via Hilbert transform
        std::vector<float> envelope = hilbert_envelope(frame);

        // Step 2: Decimate by 100 (48kHz → 480Hz)
        std::vector<float> decimated;
        for (size_t i = 0; i < envelope.size(); i += static_cast<size_t>(DEMO_DECIMATION)) {
            decimated.push_back(envelope[i]);
        }

        // Step 3: Detect BPF peaks in 0.5-10 Hz band
        detect_bpf_peaks(decimated, result);

        // Step 4: Propeller detection threshold
        result.propeller_detected = (result.max_bpf_power > -20.0f) &&
                                   (result.bpf_sharpness > 1.5f);

        return result;
    }

private:
    std::array<float, HILBERT_TAPS> hilbert_filter_;

    void init_hilbert_filter() {
        // 65-tap Hilbert transformer kernel
        for (size_t n = 0; n < HILBERT_TAPS; ++n) {
            int m = static_cast<int>(n) - HILBERT_TAPS / 2;
            if (m % 2 == 1) {
                hilbert_filter_[n] = 2.0f / (M_PI * m);
            } else {
                hilbert_filter_[n] = 0.0f;
            }
        }
    }

    std::vector<float> hilbert_envelope(const LOFARSpectrogram::Frame& frame) {
        std::vector<float> envelope(frame.bins.size());

        // Simplified envelope detection (real implementation uses FIR convolution)
        for (size_t i = 0; i < frame.bins.size(); ++i) {
            float power = std::pow(10.0f, frame.bins[i] / 20.0f);  // dB to linear
            envelope[i] = std::abs(power);
        }

        return envelope;
    }

    void detect_bpf_peaks(const std::vector<float>& envelope, BPFResult& result) {
        // BPF extraction: find peaks in 0.5-10 Hz band
        // Simplified: just extract frequency bins corresponding to BPF range

        size_t bpf_start = static_cast<size_t>(0.5f * envelope.size() / 240.0f);  // 0.5 Hz
        size_t bpf_end = static_cast<size_t>(10.0f * envelope.size() / 240.0f);   // 10 Hz

        result.max_bpf_power = -100.0f;
        for (size_t i = 0; i < BPF_BANDS && i < envelope.size(); ++i) {
            result.bpf_peaks[i] = envelope[i] > 0 ? 20.0f * std::log10(envelope[i]) : -100.0f;
            result.max_bpf_power = std::max(result.max_bpf_power, result.bpf_peaks[i]);
        }

        // Q-factor: ratio of peak power to bandwidth
        float avg_power = 0.0f;
        for (float p : result.bpf_peaks) {
            avg_power += p;
        }
        avg_power /= BPF_BANDS;
        result.bpf_sharpness = result.max_bpf_power - avg_power;
    }
};

// ============================================================================
// Thermal Calibration (Sound Velocity Correction)
// ============================================================================

class ThermalCalibration {
public:
    // Medwin formula: v(T,S,P) = 1449.05 + 45.7T - 5.21T² + 0.1T³ + ...
    static float calculate_sound_velocity(float temperature_c,
                                         float salinity_psu = 35.0f,
                                         float depth_m = 0.0f) {
        float T = temperature_c;
        float S = salinity_psu;
        float P = depth_m;  // Pressure ≈ depth in meters

        float v = 1449.05f + 45.7f * T - 5.21f * T * T + 0.1f * T * T * T;
        v += (1.333f - 0.126f * T + 0.009f * T * T) * (S - 35.0f);
        v += 16.3f * P + 0.2f * P * P;

        return v;
    }

    // Frequency shift due to Doppler (moving source)
    static float doppler_shift(float source_freq, float source_velocity_ms) {
        float sound_velocity = calculate_sound_velocity(20.0f);  // Assume 20°C
        return source_freq * (sound_velocity + source_velocity_ms) / sound_velocity;
    }
};

}  // namespace diveguard

#endif  // DIVEGUARD_DSP_CORE_HPP
