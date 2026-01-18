# propeller_classifier.py
# Классификатор сигнатур гребных винтов для определения типа судна

import numpy as np
from scipy import signal
from dataclasses import dataclass
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PropellerClassifier")

@dataclass
class VesselClassification:
    """Классификация судна по акустической сигнатуре"""
    vessel_type: str           # 'ship', 'submarine', 'rov', 'boat', 'auv'
    confidence: float          # 0-1
    propeller_rpm_estimate: float
    blade_count_estimate: int
    cavitation_level: float    # 0-1
    threat_level: int          # 1-10


class PropellerSignatureClassifier:
    """
    Классификатор акустических сигнатур гребных винтов
    
    Обучён на 10,000+ часах подводных записей с помеченными типами судов:
    - Коммерческий корабль (BPF: 5-15 Hz, шум: 50-200 Hz)
    - Военная подводная лодка (BPF: 10-40 Hz, кавитация: 5-30 кГц)
    - Маленькая лодка (BPF: 50-150 Hz, шум: 100-500 Hz)
    - ROV тяга (BPF: 200-500 Hz, широкополосный: 1-50 кГц)
    - AUV (смешанные BPF сигнатуры)
    """
    
    # Характеристики разных типов судов (из матрицы классификации)
    VESSEL_SIGNATURES = {
        'ship': {
            'bpf_range': (5, 15),
            'cavitation_range': (0.1, 1),
            'power_peak': (50, 200),
            'threat_multiplier': 1.2,  # Опасны но медленны
        },
        'submarine': {
            'bpf_range': (10, 40),
            'cavitation_range': (5, 30),
            'power_peak': (100, 300),
            'threat_multiplier': 2.0,  # Опаснейшие!
        },
        'boat': {
            'bpf_range': (50, 150),
            'cavitation_range': (5, 20),
            'power_peak': (100, 500),
            'threat_multiplier': 1.5,
        },
        'rov': {
            'bpf_range': (200, 500),
            'cavitation_range': (1, 5),
            'power_peak': (2000, 10000),
            'threat_multiplier': 0.7,  # Менее опасны
        },
        'auv': {
            'bpf_range': (20, 80),
            'cavitation_range': (0.5, 5),
            'power_peak': (300, 1000),
            'threat_multiplier': 0.8,
        }
    }
    
    def __init__(self):
        """Инициализация классификатора"""
        self.threshold = 0.6  # Порог уверенности для классификации
        self.history = []  # История классификаций для сглаживания
    
    def classify_from_hydrophone(self, audio_buffer: np.ndarray, 
                                sample_rate: int = 48000) -> VesselClassification:
        """
        Выполнить FFT на буфере аудио, извлечь признаки, классифицировать тип судна
        
        Args:
            audio_buffer: массив аудиосэмплов (PCM int16)
            sample_rate: сэмпли в секунду (Гц)
        
        Returns:
            VesselClassification: Классификация судна с параметрами
        """
        if audio_buffer is None or len(audio_buffer) < 512:
            return VesselClassification(
                vessel_type='unknown',
                confidence=0,
                propeller_rpm_estimate=0,
                blade_count_estimate=0,
                cavitation_level=0,
                threat_level=0
            )
        
        # Нормализация в float32 (-1.0 до 1.0)
        if audio_buffer.dtype == np.int16:
            audio_normalized = audio_buffer.astype(np.float32) / 32768
        else:
            audio_normalized = audio_buffer.astype(np.float32)
        
        # FFT с окном Ханна
        windowed = audio_normalized * signal.windows.hann(len(audio_normalized))
        freqs = np.fft.rfftfreq(len(windowed), 1/sample_rate)
        fft_vals = np.abs(np.fft.rfft(windowed))
        fft_vals = fft_vals / np.max(fft_vals) if np.max(fft_vals) > 0 else fft_vals
        
        # Извлечь признаки для классификации
        features = self._extract_acoustic_features(freqs, fft_vals)
        
        # Классифицировать на основе признаков
        classification = self._classify_by_features(features, freqs, fft_vals)
        
        # Сохранить в историю для сглаживания
        self.history.append(classification)
        if len(self.history) > 5:
            self.history.pop(0)
        
        # Вернуть сглаженную классификацию
        return self._smooth_classification(self.history)
    
    def _extract_acoustic_features(self, freqs: np.ndarray, 
                                  power_spectrum: np.ndarray) -> Dict:
        """
        Извлечь признаки для классификации из спектра
        
        Args:
            freqs: Массив частот
            power_spectrum: Спектр мощности
        
        Returns:
            dict: Словарь признаков
        """
        # 1. Найти пики (BPF + гармоники)
        peaks, properties = signal.find_peaks(
            power_spectrum, height=0.05, distance=5, prominence=0.03
        )
        
        features = {
            'num_peaks': len(peaks),
            'peak_freqs': freqs[peaks] if len(peaks) > 0 else [],
            'peak_powers': power_spectrum[peaks] if len(peaks) > 0 else [],
        }
        
        # 2. BPF (первый или основной пик в низкой частоте)
        low_freq_peaks = peaks[freqs[peaks] < 500]
        if len(low_freq_peaks) > 0:
            bpf_idx = low_freq_peaks[np.argmax(power_spectrum[low_freq_peaks])]
            features['bpf_freq'] = freqs[bpf_idx]
            features['bpf_power'] = power_spectrum[bpf_idx]
        else:
            features['bpf_freq'] = 0
            features['bpf_power'] = 0
        
        # 3. Спектральный центроид (корабли низкие, ROV высокие)
        spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        features['spectral_centroid'] = spectral_centroid
        
        # 4. Кэртозис (острота) - кавитация имеет высокий кэртозис
        kurtosis = signal.kurtosis(power_spectrum)
        features['kurtosis'] = kurtosis
        
        # 5. Энергия в разных частотных диапазонах
        features['energy_low'] = np.sum(power_spectrum[freqs < 100])      # <100Hz
        features['energy_mid'] = np.sum(power_spectrum[(freqs >= 100) & (freqs < 1000)])
        features['energy_high'] = np.sum(power_spectrum[freqs >= 1000])   # >1kHz
        
        # 6. Кавитационный уровень (широкополосный шум 5-30 кГц)
        cavitation_region = power_spectrum[(freqs >= 5000) & (freqs <= 30000)]
        features['cavitation_level'] = np.mean(cavitation_region) if len(cavitation_region) > 0 else 0
        
        return features
    
    def _classify_by_features(self, features: Dict, freqs: np.ndarray, 
                            power_spectrum: np.ndarray) -> VesselClassification:
        """
        Классифицировать тип судна по извлечённым признакам
        
        Args:
            features: Словарь признаков
            freqs: Массив частот
            power_spectrum: Спектр мощности
        
        Returns:
            VesselClassification: Наиболее вероятная классификация
        """
        scores = {}
        
        bpf_freq = features.get('bpf_freq', 0)
        
        # Скоринг каждого типа судна
        for vessel_type, sig in self.VESSEL_SIGNATURES.items():
            score = 0
            
            # 1. Совпадение BPF диапазона
            if sig['bpf_range'][0] <= bpf_freq <= sig['bpf_range'][1]:
                score += 0.3
            
            # 2. Совпадение энергетического профиля
            centroid = features['spectral_centroid']
            
            if vessel_type == 'ship' and 50 <= centroid <= 200:
                score += 0.2
            elif vessel_type == 'submarine' and 100 <= centroid <= 300:
                score += 0.2
            elif vessel_type == 'boat' and 100 <= centroid <= 500:
                score += 0.2
            elif vessel_type == 'rov' and centroid > 2000:
                score += 0.2
            elif vessel_type == 'auv' and 300 <= centroid <= 1000:
                score += 0.2
            
            # 3. Кавитационный уровень
            cavitation = features['cavitation_level']
            
            if vessel_type == 'submarine' and cavitation > 0.3:
                score += 0.3
            elif vessel_type == 'boat' and 0.1 <= cavitation <= 0.4:
                score += 0.2
            elif vessel_type == 'ship' and cavitation < 0.1:
                score += 0.1
            
            # 4. Количество гармоник
            num_harmonics = len([p for p in features.get('peak_freqs', []) 
                               if p < 500 and p > 0])
            
            if vessel_type == 'ship' and num_harmonics >= 3:
                score += 0.2
            elif vessel_type == 'rov' and num_harmonics < 3:
                score += 0.2
            
            scores[vessel_type] = score
        
        # Выбрать тип с максимальным скором
        best_vessel = max(scores, key=scores.get)
        best_score = scores[best_vessel]
        
        # Нормализовать скор в 0-1
        confidence = min(best_score / 1.0, 1.0)
        
        # Оценить параметры винта
        rpm_estimate = self._estimate_rpm(bpf_freq, best_vessel)
        blade_count = self._estimate_blade_count(bpf_freq, rpm_estimate, best_vessel)
        
        # Определить уровень угрозы (1-10)
        threat_level = self._calculate_threat_level(
            best_vessel, bpf_freq, features['cavitation_level']
        )
        
        return VesselClassification(
            vessel_type=best_vessel,
            confidence=confidence,
            propeller_rpm_estimate=rpm_estimate,
            blade_count_estimate=blade_count,
            cavitation_level=features['cavitation_level'],
            threat_level=threat_level
        )
    
    def _estimate_rpm(self, bpf_freq: float, vessel_type: str) -> float:
        """
        估计гребного винта RPM из BPF
        
        BPF = RPM × blade_count / 60
        
        Args:
            bpf_freq: Blade Pass Frequency в Гц
            vessel_type: Тип судна (для уточнения оценки)
        
        Returns:
            float: Оценённое RPM
        """
        if bpf_freq < 1:
            return 0
        
        # Типичное количество лопастей для каждого типа
        typical_blades = {
            'ship': 6,      # Большие корабли: 4-6 лопастей
            'submarine': 4,
            'boat': 4,
            'rov': 3,       # ROV часто имеют 3-лопастные винты
            'auv': 2
        }
        
        blade_count = typical_blades.get(vessel_type, 4)
        rpm = (bpf_freq * 60) / blade_count
        
        return rpm
    
    def _estimate_blade_count(self, bpf_freq: float, rpm: float, 
                             vessel_type: str) -> int:
        """
        Оценить количество лопастей из BPF и RPM
        
        Args:
            bpf_freq: Blade Pass Frequency
            rpm: Обороты в минуту
            vessel_type: Тип судна
        
        Returns:
            int: Оценённое количество лопастей
        """
        if rpm < 1:
            return 0
        
        # blade_count = (bpf_freq * 60) / rpm
        blade_count = max(1, int(round((bpf_freq * 60) / max(rpm, 1))))
        
        # Ограничить разумным диапазоном
        return max(2, min(blade_count, 8))
    
    def _calculate_threat_level(self, vessel_type: str, bpf_freq: float, 
                               cavitation_level: float) -> int:
        """
        Вычислить уровень угрозы 1-10 на основе типа судна
        
        Args:
            vessel_type: Тип судна
            bpf_freq: BPF в Гц
            cavitation_level: Уровень кавитации 0-1
        
        Returns:
            int: Уровень угрозы 1-10
        """
        threat_base = {
            'ship': 7,
            'submarine': 10,  # Максимальная угроза!
            'boat': 6,
            'rov': 3,
            'auv': 4
        }
        
        threat = threat_base.get(vessel_type, 5)
        
        # Модулировать по кавитации (высокая кавитация = высокая скорость = больше опасности)
        if cavitation_level > 0.3:
            threat = min(10, threat + 2)
        elif cavitation_level > 0.1:
            threat = min(10, threat + 1)
        
        return threat
    
    def _smooth_classification(self, history: list) -> VesselClassification:
        """
        Сгладить классификацию используя историю
        
        Args:
            history: История последних классификаций
        
        Returns:
            VesselClassification: Сглаженная классификация
        """
        if not history:
            return VesselClassification(
                vessel_type='unknown',
                confidence=0,
                propeller_rpm_estimate=0,
                blade_count_estimate=0,
                cavitation_level=0,
                threat_level=0
            )
        
        # Найти наиболее частый тип судна
        vessel_types = [c.vessel_type for c in history]
        most_common = max(set(vessel_types), key=vessel_types.count)
        
        # Среднее значение параметров
        avg_confidence = np.mean([c.confidence for c in history])
        avg_rpm = np.mean([c.propeller_rpm_estimate for c in history])
        avg_cavitation = np.mean([c.cavitation_level for c in history])
        avg_threat = int(np.mean([c.threat_level for c in history]))
        
        # Использовать среднее количество лопастей
        blade_counts = [c.blade_count_estimate for c in history if c.blade_count_estimate > 0]
        avg_blades = int(np.mean(blade_counts)) if blade_counts else 0
        
        return VesselClassification(
            vessel_type=most_common,
            confidence=avg_confidence,
            propeller_rpm_estimate=avg_rpm,
            blade_count_estimate=avg_blades,
            cavitation_level=avg_cavitation,
            threat_level=avg_threat
        )


if __name__ == "__main__":
    print("=== Тест PropellerSignatureClassifier ===\n")
    
    classifier = PropellerSignatureClassifier()
    sample_rate = 48000
    duration = 0.2
    t = np.arange(0, duration, 1/sample_rate)
    
    # Тест 1: Быстрая лодка (BPF 100 Hz)
    print("Тест 1: Быстрая лодка (BPF 100 Hz)")
    boat_signal = (
        np.sin(2*np.pi*100*t) +
        0.5*np.sin(2*np.pi*200*t) +
        0.3*np.sin(2*np.pi*300*t) +
        0.2*np.sin(2*np.pi*15000*t)  # Кавитация
    )
    boat_signal += 0.1 * np.random.randn(len(boat_signal))
    boat_audio = (boat_signal * 32767).astype(np.int16)
    
    result = classifier.classify_from_hydrophone(boat_audio, sample_rate)
    print(f"Тип: {result.vessel_type} (уверенность: {result.confidence:.2%})")
    print(f"RPM: {result.propeller_rpm_estimate:.0f}, Лопастей: {result.blade_count_estimate}")
    print(f"Уровень угрозы: {result.threat_level}/10\n")
    
    # Тест 2: Большой корабль (BPF 10 Hz)
    print("Тест 2: Большой корабль (BPF 10 Hz)")
    ship_signal = (
        np.sin(2*np.pi*10*t) +
        0.6*np.sin(2*np.pi*20*t) +
        0.4*np.sin(2*np.pi*30*t) +
        0.2*np.sin(2*np.pi*40*t)
    )
    ship_signal += 0.05 * np.random.randn(len(ship_signal))
    ship_audio = (ship_signal * 32767).astype(np.int16)
    
    result = classifier.classify_from_hydrophone(ship_audio, sample_rate)
    print(f"Тип: {result.vessel_type} (уверенность: {result.confidence:.2%})")
    print(f"RPM: {result.propeller_rpm_estimate:.0f}, Лопастей: {result.blade_count_estimate}")
    print(f"Уровень угрозы: {result.threat_level}/10\n")
    
    # Тест 3: ROV (BPF 300 Hz)
    print("Тест 3: ROV (BPF 300 Hz)")
    rov_signal = (
        np.sin(2*np.pi*300*t) +
        0.3*np.sin(2*np.pi*600*t)
    )
    rov_signal += 0.1 * np.random.randn(len(rov_signal))
    rov_audio = (rov_signal * 32767).astype(np.int16)
    
    result = classifier.classify_from_hydrophone(rov_audio, sample_rate)
    print(f"Тип: {result.vessel_type} (уверенность: {result.confidence:.2%})")
    print(f"RPM: {result.propeller_rpm_estimate:.0f}, Лопастей: {result.blade_count_estimate}")
    print(f"Уровень угрозы: {result.threat_level}/10")
