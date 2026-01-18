# sensor_fusion.py
# Модуль слияния данных SONAR и HYDROPHONE для обнаружения винтов

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorFusion")

@dataclass
class FusedState:
    """Объединённое состояние угрозы от датчиков"""
    distance: float          # Расстояние в метрах
    azimuth: float          # Азимут 0-360°
    elevation: float        # Возвышение -90 to +90°
    confidence: float       # Уверенность 0-1
    vessel_type: str        # Тип судна
    frequency_peaks: list   # Пики частот (Hz, мощность)
    closing_speed: float    # Скорость сближения м/с
    timestamp: float        # Временная метка


class ExtendedKalmanFilter:
    """Расширенный фильтр Калмана для слияния датчиков"""
    
    def __init__(self, process_noise=0.01, measurement_noise_sonar=1.0, 
                 measurement_noise_acoustic=0.5):
        """
        Инициализация EKF
        
        Args:
            process_noise: Ковариация шума процесса
            measurement_noise_sonar: Шум измерения SONAR
            measurement_noise_acoustic: Шум измерения гидрофона
        """
        # Состояние: [x, y, z, vx, vy, vz] в метрах и м/с
        self.x = np.array([100, 0, 0, -1, 0, 0], dtype=float)  # Начальное предположение
        self.P = np.eye(6) * 100  # Ковариация ошибки состояния
        
        # Шумы
        self.Q = np.eye(6) * process_noise
        self.R_sonar = measurement_noise_sonar
        self.R_acoustic = measurement_noise_acoustic
        
    def predict(self, dt=0.1):
        """Прогноз Калмана на основе физической модели"""
        # Матрица переходов (движение в пространстве)
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        logger.debug(f"EKF Predict: x={self.x[:3]}, v={self.x[3:6]}")
    
    def update_sonar(self, distance: float, azimuth_rad: float, 
                     elevation_rad: float):
        """
        Обновление на основе измерения SONAR
        
        Args:
            distance: Расстояние в метрах
            azimuth_rad: Азимут в радианах
            elevation_rad: Возвышение в радианах
        """
        # Преобразование из сферических в декартовы координаты
        x_meas = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y_meas = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z_meas = distance * np.sin(elevation_rad)
        
        z = np.array([x_meas, y_meas, z_meas])
        
        # Матрица измерения (только позиция)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R_sonar * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        
        logger.debug(f"SONAR Update: dist={distance:.1f}m, az={np.degrees(azimuth_rad):.1f}°")
    
    def update_acoustic(self, frequency_peaks: np.ndarray):
        """
        Обновление на основе акустического анализа гидрофона
        
        Args:
            frequency_peaks: Массив пиков частот [(freq, power), ...]
        """
        # Акустические пики уточняют скорость через эффект Доплера
        if len(frequency_peaks) > 0:
            # BPF (первый пик) указывает на тип винта
            bpf_freq = frequency_peaks[0][0]
            
            # Простая оценка скорости через смещение Доплера
            # Δf / f = v / c  где c = 1500 м/с в воде
            expected_bpf = 100  # Типичный BPF в Гц
            doppler_shift = (bpf_freq - expected_bpf) / expected_bpf
            estimated_velocity = doppler_shift * 1500  # м/с
            
            # Обновляем компоненту скорости
            H_v = np.array([[0, 0, 0, 1, 0, 0]])  # Только Vx
            z_v = np.array([estimated_velocity])
            
            y_v = z_v - (H_v @ self.x)
            S_v = H_v @ self.P @ H_v.T + self.R_acoustic
            K_v = self.P @ H_v.T / S_v
            
            self.x = self.x + K_v.flatten() * y_v
            
            logger.debug(f"Acoustic Update: BPF={bpf_freq:.1f}Hz, est_v={estimated_velocity:.2f}m/s")
    
    def get_state_estimate(self) -> FusedState:
        """Получить оценённое состояние"""
        x_pos, y_pos, z_pos = self.x[:3]
        vx, vy, vz = self.x[3:6]
        
        # Расстояние
        distance = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
        
        # Азимут (0-360°)
        azimuth = np.degrees(np.arctan2(y_pos, x_pos)) % 360
        
        # Возвышение (-90 to +90°)
        elevation = np.degrees(np.arcsin(z_pos / max(distance, 1)))
        
        # Скорость сближения (положительная = приближается)
        closing_speed = -(vx * x_pos + vy * y_pos + vz * z_pos) / max(distance, 1)
        
        return FusedState(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            confidence=1.0 - np.trace(self.P[:3, :3]) / 1000,
            vessel_type="unknown",
            frequency_peaks=[],
            closing_speed=closing_speed,
            timestamp=0
        )


class SensorFusionEngine:
    """Главный двигатель слияния датчиков"""
    
    def __init__(self, sonar_device=None, hydrophone_device=None, imu_device=None):
        """
        Инициализация двигателя слияния
        
        Args:
            sonar_device: Объект SONAR (Ping360)
            hydrophone_device: Объект гидрофона с буфером
            imu_device: Объект IMU (MPU9250, LSM9DS1)
        """
        self.ekf = ExtendedKalmanFilter()
        self.sonar = sonar_device
        self.hydrophone = hydrophone_device
        self.imu = imu_device
        
        self.previous_state = None
        self.state_history = []
    
    def fuse_sonar_hydrophone(self, sonar_data: Dict, acoustic_data: np.ndarray) -> FusedState:
        """
        Объединение данных SONAR и HYDROPHONE для получения оценки угрозы
        
        Args:
            sonar_data: {
                'distance': float (метры),
                'azimuth': float (градусы 0-360),
                'elevation': float (градусы -90 to +90)
            }
            acoustic_data: Буфер аудиосигнала (PCM samples)
        
        Returns:
            FusedState: Объединённое состояние угрозы
        """
        # Шаг 1: Прогноз Калмана
        self.ekf.predict(dt=0.1)
        
        # Шаг 2: Обновление SONAR
        if sonar_data:
            self.ekf.update_sonar(
                distance=sonar_data.get('distance', 100),
                azimuth_rad=np.radians(sonar_data.get('azimuth', 0)),
                elevation_rad=np.radians(sonar_data.get('elevation', 0))
            )
        
        # Шаг 3: FFT анализ гидрофона
        frequency_peaks = self._compute_fft(acoustic_data)
        
        # Шаг 4: Обновление на основе акустики
        if frequency_peaks is not None:
            self.ekf.update_acoustic(frequency_peaks)
        
        # Шаг 5: Получить оценённое состояние
        fused_state = self.ekf.get_state_estimate()
        
        # Шаг 6: Применить коррекцию по IMU компасу (если доступен)
        if self.imu:
            imu_heading = self.imu.get_magnetometer_heading()
            fused_state.azimuth = self._correct_azimuth_with_imu(
                fused_state.azimuth, imu_heading
            )
        
        # Сохранить в историю
        self.state_history.append(fused_state)
        self.previous_state = fused_state
        
        return fused_state
    
    def _compute_fft(self, audio_buffer: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Вычислить FFT и извлечь пики частот (сигнатура винта)
        
        Args:
            audio_buffer: Буфер аудиоданных
            sample_rate: Частота дискретизации (Гц)
        
        Returns:
            np.ndarray: Массив пиков [(freq, power), ...]
        """
        if audio_buffer is None or len(audio_buffer) < 512:
            return None
        
        # Применить окно Ханна для уменьшения утечки
        windowed = audio_buffer * signal.windows.hann(len(audio_buffer))
        
        # FFT
        freqs = np.fft.rfftfreq(len(windowed), 1/sample_rate)
        fft_vals = np.abs(np.fft.rfft(windowed))
        
        # Нормализация
        fft_vals = fft_vals / np.max(fft_vals) if np.max(fft_vals) > 0 else fft_vals
        
        # Найти пики (используем scipy.signal.find_peaks)
        peaks, properties = signal.find_peaks(
            fft_vals, height=0.1, distance=10, prominence=0.05
        )
        
        # Вернуть топ-5 пиков отсортированных по мощности
        if len(peaks) > 0:
            peak_freqs = freqs[peaks]
            peak_powers = fft_vals[peaks]
            
            # Отсортировать по мощности (descending)
            sorted_idx = np.argsort(peak_powers)[::-1][:5]
            
            result = np.array([
                [peak_freqs[i], peak_powers[i]] 
                for i in sorted_idx
            ])
            
            logger.debug(f"FFT Peaks: {result[:3]}")
            return result
        
        return None
    
    def _correct_azimuth_with_imu(self, sonar_azimuth: float, imu_heading: float) -> float:
        """
        Корректировать азимут SONAR с помощью компасного направления IMU
        
        Args:
            sonar_azimuth: Азимут от SONAR (относительный)
            imu_heading: Абсолютное направление робота из компаса (0-360°)
        
        Returns:
            float: Скорректированный абсолютный азимут
        """
        # Простая коррекция: добавить компасное направление
        corrected = (sonar_azimuth + imu_heading) % 360
        return corrected


if __name__ == "__main__":
    # Тест модуля
    print("=== Тест SensorFusionEngine ===")
    
    engine = SensorFusionEngine()
    
    # Имитация данных SONAR
    sonar_data = {
        'distance': 150.5,
        'azimuth': 45.2,
        'elevation': -10.0
    }
    
    # Имитация акустических данных (синусоида на 100 Hz - BPF быстрой лодки)
    sample_rate = 48000
    duration = 0.1  # 100 мс
    t = np.arange(0, duration, 1/sample_rate)
    
    # Сигнал винта (BPF 100 Hz + гармоники)
    acoustic_signal = (
        np.sin(2*np.pi*100*t) +      # BPF
        0.5*np.sin(2*np.pi*200*t) +  # 2-я гармоника
        0.3*np.sin(2*np.pi*300*t)    # 3-я гармоника
    )
    
    # Добавить шум
    acoustic_signal += 0.1 * np.random.randn(len(acoustic_signal))
    acoustic_buffer = (acoustic_signal * 32767).astype(np.int16)
    
    # Слияние
    fused = engine.fuse_sonar_hydrophone(sonar_data, acoustic_buffer)
    
    print(f"Расстояние: {fused.distance:.1f} м")
    print(f"Азимут: {fused.azimuth:.1f}°")
    print(f"Возвышение: {fused.elevation:.1f}°")
    print(f"Скорость сближения: {fused.closing_speed:.2f} м/с")
    print(f"Уверенность: {fused.confidence:.2f}")
    print(f"Обнаруженные пики частот: {fused.frequency_peaks[:3] if fused.frequency_peaks is not None else None}")
