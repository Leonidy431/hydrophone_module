#!/usr/bin/env python3
"""
DiveGuard DSP Bridge - Phases 1-4 Integration
Real-time propeller detection pipeline via ZMQ IPC

Phases:
- Phase 1: I2S ALSA audio acquisition (48kHz, 16-bit)
- Phase 2: Lock-free ring buffer (C++ atomic operations)
- Phase 3: LOFAR spectrogram (KISS-FFT 1024-point)
- Phase 4: DEMON envelope detection (Hilbert + BPF extraction)

Total latency: <100ms (80ms DSP + 12ms ML + 8ms overhead)
"""

import zmq
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DSPBridge")


@dataclass
class AudioFrame:
    """Raw audio frame from hydrophone (48kHz, 16-bit)"""
    samples: bytes  # 512 samples × 2 bytes = 1024 bytes
    timestamp_ms: float
    sample_rate: int = 48000
    channels: int = 1
    bit_depth: int = 16

    @property
    def num_samples(self) -> int:
        return len(self.samples) // 2


@dataclass
class DSPResult:
    """Output from DSP pipeline (Phase 1-4)"""
    timestamp_ms: float
    max_bpf_power_db: float
    bpf_sharpness: float  # Q-factor
    propeller_score: float  # 0-1 confidence
    threat_level: int  # 1-10
    frame_latency_ms: float

    def to_dict(self) -> dict:
        return {
            'timestamp_ms': self.timestamp_ms,
            'max_bpf_power_db': self.max_bpf_power_db,
            'bpf_sharpness': self.bpf_sharpness,
            'propeller_score': self.propeller_score,
            'threat_level': self.threat_level,
            'frame_latency_ms': self.frame_latency_ms,
        }


class ThermalCalibrationModule:
    """
    Sound velocity correction via Medwin formula

    v(T,S,P) = 1449.05 + 45.7T - 5.21T² + 0.1T³
             + (1.333 - 0.126T + 0.009T²)(S-35)
             + 16.3P + 0.2P²
    """

    def __init__(self, calibration_interval_sec: float = 1.0):
        self.calibration_interval_sec = calibration_interval_sec
        self.last_calibration_time = 0.0
        self.current_sound_velocity = 1500.0  # Default seawater

    def update_environment(self, temperature_c: float,
                          salinity_psu: float = 35.0,
                          depth_m: float = 0.0) -> float:
        """Compute sound velocity and update threshold if needed"""
        current_time = time.time()

        if current_time - self.last_calibration_time > self.calibration_interval_sec:
            self.current_sound_velocity = self._medwin_formula(
                temperature_c, salinity_psu, depth_m
            )
            self.last_calibration_time = current_time
            logger.info(f"Thermal calibration: T={temperature_c}°C, "
                       f"S={salinity_psu}PSU, v={self.current_sound_velocity:.1f}m/s")

        return self.current_sound_velocity

    @staticmethod
    def _medwin_formula(temperature_c: float,
                        salinity_psu: float = 35.0,
                        depth_m: float = 0.0) -> float:
        """Medwin (1975) sound velocity formula"""
        T = temperature_c
        S = salinity_psu
        P = depth_m

        v = 1449.05 + 45.7*T - 5.21*T*T + 0.1*T*T*T
        v += (1.333 - 0.126*T + 0.009*T*T) * (S - 35.0)
        v += 16.3*P + 0.2*P*P

        return v


class AdaptiveThresholdModule:
    """
    Adaptive detection threshold based on ambient noise level

    Bay-specific calibration: run 24-48h baseline, measure FP rate, adjust
    Target: <5% false positive rate
    """

    def __init__(self, baseline_window_size: int = 3600):
        self.baseline_window_size = baseline_window_size  # 3600 = 1 hour
        self.background_power_samples = []
        self.fixed_threshold = 0.70
        self.adaptive_threshold = self.fixed_threshold

    def feed_noise_sample(self, max_bpf_power_db: float):
        """Accumulate background noise measurements"""
        self.background_power_samples.append(max_bpf_power_db)

        if len(self.background_power_samples) > self.baseline_window_size:
            self.background_power_samples.pop(0)
            self._recalibrate_threshold()

    def _recalibrate_threshold(self):
        """Adaptive calibration based on baseline"""
        if not self.background_power_samples:
            return

        import statistics
        mean_power = statistics.mean(self.background_power_samples)
        stdev = statistics.stdev(self.background_power_samples) if len(self.background_power_samples) > 1 else 0.0

        # Threshold = mean + 2.5×stdev (covers ~98% of background)
        self.adaptive_threshold = min(mean_power + 2.5 * stdev, 0.85)
        self.adaptive_threshold = max(self.adaptive_threshold, 0.60)

        logger.info(f"Recalibrated threshold: {self.adaptive_threshold:.3f} "
                   f"(mean={mean_power:.1f}dB, σ={stdev:.1f}dB)")

    def get_threshold(self) -> float:
        """Get current detection threshold"""
        return self.adaptive_threshold if len(self.background_power_samples) > 100 else self.fixed_threshold


class DSPPipeline:
    """
    Complete Phase 1-4 DSP pipeline orchestration

    Interface with C++ DSP core via ZMQ message queue
    """

    def __init__(self, dsp_server_endpoint: str = "ipc:///tmp/diveguard_dsp.ipc"):
        self.endpoint = dsp_server_endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout

        self.thermal_calibration = ThermalCalibrationModule()
        self.adaptive_threshold = AdaptiveThresholdModule()

        self.frame_count = 0
        self.start_time = time.time()

        logger.info(f"DSP Pipeline initialized, endpoint={dsp_server_endpoint}")

    def connect(self):
        """Connect to C++ DSP server"""
        try:
            self.socket.connect(self.endpoint)
            logger.info(f"Connected to DSP server at {self.endpoint}")
        except zmq.error.ZMQError as e:
            logger.error(f"Failed to connect to DSP server: {e}")
            raise

    def process_audio(self, audio_frame: AudioFrame,
                     temperature_c: float = 20.0,
                     salinity_psu: float = 35.0,
                     depth_m: float = 0.0) -> Optional[DSPResult]:
        """
        Process raw audio frame through DSP pipeline

        Args:
            audio_frame: Raw 16-bit audio at 48kHz
            temperature_c: Water temperature for sound velocity correction
            salinity_psu: Salinity (practical salinity units)
            depth_m: Depth for pressure correction

        Returns:
            DSP result with propeller detection score or None on timeout
        """
        process_start = time.time()

        # Update thermal calibration
        self.thermal_calibration.update_environment(temperature_c, salinity_psu, depth_m)

        # Prepare message for C++ DSP core
        message = {
            'action': 'process_frame',
            'audio_base64': self._encode_audio(audio_frame.samples),
            'timestamp_ms': audio_frame.timestamp_ms,
            'sample_rate': audio_frame.sample_rate,
            'sound_velocity_ms': self.thermal_calibration.current_sound_velocity,
        }

        try:
            # Send request to DSP server
            self.socket.send_json(message)

            # Wait for response
            response = self.socket.recv_json()

            process_time = (time.time() - process_start) * 1000  # ms

            # Parse DSP result
            result = DSPResult(
                timestamp_ms=response['timestamp_ms'],
                max_bpf_power_db=response['max_bpf_power_db'],
                bpf_sharpness=response['bpf_sharpness'],
                propeller_score=response['propeller_score'],
                threat_level=response['threat_level'],
                frame_latency_ms=process_time,
            )

            # Feed background noise for adaptive threshold
            if result.propeller_score < 0.3:  # Likely background noise
                self.adaptive_threshold.feed_noise_sample(result.max_bpf_power_db)

            self.frame_count += 1

            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                logger.info(f"Frame {self.frame_count}: propeller_score={result.propeller_score:.3f}, "
                           f"threat={result.threat_level}, latency={process_time:.1f}ms, "
                           f"fps={fps:.1f}")

            return result

        except zmq.error.Again:
            logger.warning(f"DSP server timeout on frame {self.frame_count}")
            return None
        except zmq.error.ZMQError as e:
            logger.error(f"ZMQ error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from DSP server: {e}")
            return None

    @staticmethod
    def _encode_audio(samples_bytes: bytes) -> str:
        """Base64 encode audio for JSON transmission"""
        import base64
        return base64.b64encode(samples_bytes).decode('ascii')

    def get_current_threshold(self) -> float:
        """Get adaptive detection threshold"""
        return self.adaptive_threshold.get_threshold()

    def shutdown(self):
        """Clean shutdown"""
        self.socket.close()
        self.context.term()
        logger.info("DSP Pipeline shutdown complete")


# ============================================================================
# Phase 1: I2S/ALSA Audio Acquisition
# ============================================================================

class ALSAHydrophoneReader:
    """
    ALSA PCM interface for I2S hydrophone

    Device: /dev/snd/pcmC0D0c (or similar)
    Sample rate: 48000 Hz
    Format: 16-bit signed PCM
    Channels: 1 (mono hydrophone)
    """

    def __init__(self, device_name: str = "default",
                 chunk_size: int = 512,
                 sample_rate: int = 48000):
        try:
            import alsaaudio
            self.alsaaudio = alsaaudio
        except ImportError:
            logger.warning("pyalsaaudio not installed, using stub implementation")
            self.alsaaudio = None

        self.device_name = device_name
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.pcm = None
        self.frame_timestamp_ms = 0.0

    def open(self):
        """Open ALSA device"""
        if not self.alsaaudio:
            logger.warning("ALSA audio disabled (pyalsaaudio not available)")
            return

        try:
            self.pcm = self.alsaaudio.PCM(
                self.alsaaudio.PCM_CAPTURE,
                self.alsaaudio.PCM_NONBLOCK,
                device=self.device_name
            )
            self.pcm.setchannels(1)
            self.pcm.setrate(self.sample_rate)
            self.pcm.setformat(self.alsaaudio.PCM_FORMAT_S16_LE)
            self.pcm.setperiodsize(self.chunk_size)

            logger.info(f"Opened ALSA device: {self.device_name} "
                       f"({self.sample_rate}Hz, {self.chunk_size} chunk)")
        except Exception as e:
            logger.error(f"Failed to open ALSA device: {e}")
            raise

    def read_frame(self) -> Optional[AudioFrame]:
        """Read one audio frame (512 samples)"""
        if not self.pcm:
            # Stub: return zeros for testing
            import struct
            samples = struct.pack('<512h', *([0] * 512))
            self.frame_timestamp_ms += (512 / self.sample_rate) * 1000
            return AudioFrame(samples, self.frame_timestamp_ms)

        try:
            length, data = self.pcm.read()
            if length > 0:
                self.frame_timestamp_ms += (length / self.sample_rate) * 1000
                return AudioFrame(data, self.frame_timestamp_ms, self.sample_rate)
            return None
        except Exception as e:
            logger.error(f"ALSA read error: {e}")
            return None

    def close(self):
        """Close ALSA device"""
        if self.pcm:
            self.pcm.close()


if __name__ == "__main__":
    # Example: Create and test DSP pipeline
    logger.info("DiveGuard DSP Bridge - Phase 1-4 Test")

    # Initialize ALSA reader (Phase 1)
    alsa_reader = ALSAHydrophoneReader()
    alsa_reader.open()

    # Initialize DSP pipeline
    pipeline = DSPPipeline()
    pipeline.connect()

    # Simulate audio processing
    for frame_idx in range(10):
        frame = alsa_reader.read_frame()
        if frame:
            result = pipeline.process_audio(frame, temperature_c=20.0, depth_m=10.0)
            if result:
                logger.info(f"Frame {frame_idx}: {result.to_dict()}")
        time.sleep(0.05)  # 50ms between frames

    pipeline.shutdown()
    alsa_reader.close()
