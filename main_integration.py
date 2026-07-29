# main_integration.py
# Главный файл интеграции всех модулей DiveGuard Propeller Detector

import numpy as np
import time
import logging
from typing import Optional

# Импортировать все модули
from sensor_fusion import SensorFusionEngine
from propeller_classifier import PropellerSignatureClassifier
from threat_assessment import ThreatAssessmentEngine, ThreatAssessment
from diver_alert_controller import DiverAlertController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DiveGuardMain")

class DiveGuardPropellerDetector:
    """
    Главный модуль обнаружения винтов гребных винтов DiveGuard
    
    Интегрирует все компоненты:
    1. Слияние датчиков (SONAR + HYDROPHONE)
    2. Классификация сигнатур винтов
    3. Оценка угрозы столкновения
    4. Мультимодальные предупреждения дайверам
    """
    
    def __init__(self, sonar_device=None, hydrophone_device=None, 
                 imu_device=None, robot_controller=None):
        """
        Инициализация системы DiveGuard
        
        Args:
            sonar_device: Объект SONAR (Ping360) или None для имитации
            hydrophone_device: Объект гидрофона или None
            imu_device: Объект IMU (MPU9250) или None
            robot_controller: Объект контроля робота (BlueRobotics) или None
        """
        logger.info("="*60)
        logger.info("Инициализация DiveGuard Propeller Detector v1.0")
        logger.info("="*60)
        
        # Инициализировать модули
        self.sensor_fusion = SensorFusionEngine(
            sonar_device=sonar_device,
            hydrophone_device=hydrophone_device,
            imu_device=imu_device
        )
        logger.info("✓ Модуль слияния датчиков инициализирован")
        
        self.classifier = PropellerSignatureClassifier()
        logger.info("✓ Классификатор сигнатур инициализирован")
        
        self.threat_engine = ThreatAssessmentEngine()
        logger.info("✓ Двигатель оценки угроз инициализирован")
        
        self.alert_controller = DiverAlertController(
            robot_controller=robot_controller
        )
        logger.info("✓ Контроллер предупреждений инициализирован")
        
        # Статистика
        self.threats_detected = 0
        self.critical_events = 0
        self.processing_times = []
        
        logger.info("✓ DiveGuard готов к работе!\n")
    
    def process_sensor_data(self, sonar_data: dict, acoustic_data: np.ndarray) -> Optional[ThreatAssessment]:
        """
        Обработать данные датчиков и выдать предупреждение при необходимости
        
        Args:
            sonar_data: {
                'distance': float (метры),
                'azimuth': float (градусы 0-360),
                'elevation': float (градусы -90 to +90)
            }
            acoustic_data: Буфер аудиоданных (PCM int16, 48kHz)
        
        Returns:
            ThreatAssessment: Оценка угрозы или None если нет
        """
        start_time = time.time()
        
        try:
            # ШАГ 1: Слияние датчиков SONAR + HYDROPHONE
            fused_state = self.sensor_fusion.fuse_sonar_hydrophone(
                sonar_data, acoustic_data
            )
            logger.debug(f"Фузия датчиков: расстояние={fused_state.distance:.1f}м, азимут={fused_state.azimuth:.1f}°")
            
            # ШАГ 2: Классификация типа судна
            vessel_classification = self.classifier.classify_from_hydrophone(
                acoustic_data
            )
            logger.debug(f"Классификация: тип={vessel_classification.vessel_type}, уверенность={vessel_classification.confidence:.0%}")
            
            # ШАГ 3: Оценка угрозы
            threat_assessment = self.threat_engine.assess_threat(
                fused_state, vessel_classification
            )
            logger.info(f"Угроза: {threat_assessment.vessel_type} на {threat_assessment.distance_m:.0f}м, риск {threat_assessment.risk_level}/10")
            
            # ШАГ 4: Выдать предупреждение
            if threat_assessment.risk_level >= 1:
                self.threats_detected += 1
                if threat_assessment.risk_level >= 8:
                    self.critical_events += 1
                
                self.alert_controller.alert_diver(
                    threat_assessment, 
                    current_time=time.time()
                )
            
            # Записать время обработки
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            logger.debug(f"Время обработки: {processing_time*1000:.1f} мс")
            
            return threat_assessment
        
        except Exception as e:
            logger.error(f"Ошибка обработки данных датчиков: {e}", exc_info=True)
            return None
    
    def get_statistics(self) -> dict:
        """Получить статистику системы"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'threats_detected': self.threats_detected,
            'critical_events': self.critical_events,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max(self.processing_times) * 1000 if self.processing_times else 0,
            'sensor_fusion_history_len': len(self.sensor_fusion.state_history),
            'alert_mode': self.alert_controller.current_mode.name
        }
    
    def print_statistics(self):
        """Вывести статистику на экран"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("СТАТИСТИКА DiveGuard Propeller Detector")
        print("="*60)
        print(f"Обнаружено угроз: {stats['threats_detected']}")
        print(f"Критических событий: {stats['critical_events']}")
        print(f"Среднее время обработки: {stats['avg_processing_time_ms']:.1f} мс")
        print(f"Макс время обработки: {stats['max_processing_time_ms']:.1f} мс")
        print(f"История позиций: {stats['sensor_fusion_history_len']} записей")
        print(f"Текущий режим предупреждения: {stats['alert_mode']}")
        print("="*60 + "\n")


def run_simulation():
    """
    Запустить симуляцию с синтетическими данными
    """
    print("\n" + "🚀 "*30)
    print("ЗАПУСК СИМУЛЯЦИИ DiveGuard Propeller Detector")
    print("🚀 "*30 + "\n")
    
    # Инициализировать систему
    dgpd = DiveGuardPropellerDetector()
    
    # Параметры симуляции
    sample_rate = 48000
    duration_per_frame = 0.1  # 100мс на кадр
    num_frames = 20
    
    # Сценарий: Быстрая лодка приближается с азимута 45°
    print("Сценарий: Быстрая лодка приближается с расстояния 500м до 50м\n")
    
    for frame in range(num_frames):
        # Линейное уменьшение расстояния (приближение)
        distance = 500 - frame * (450 / num_frames)  # 500м -> 50м
        azimuth = 45  # Фиксированное направление
        
        # SONAR данные
        sonar_data = {
            'distance': distance,
            'azimuth': azimuth,
            'elevation': 0
        }
        
        # Синтезировать акустику лодки
        # BPF быстрой лодки около 100 Hz
        t = np.arange(0, duration_per_frame, 1/sample_rate)
        
        # Увеличивать интенсивность при приближении
        intensity = (1 - distance / 500)
        
        boat_signal = (
            np.sin(2*np.pi*100*t) * (1 + intensity) +      # BPF растёт
            0.5*np.sin(2*np.pi*200*t) * intensity +
            0.3*np.sin(2*np.pi*300*t) * intensity +
            0.2*np.sin(2*np.pi*15000*t) * intensity  # Кавитация растёт
        )
        
        # Добавить шум
        boat_signal += 0.05 * np.random.randn(len(boat_signal))
        acoustic_buffer = (boat_signal * 32767).astype(np.int16)
        
        # Обработать данные
        print(f"Кадр {frame+1}/{num_frames}: Расстояние {distance:.0f}м, Азимут {azimuth:.1f}°")
        threat = dgpd.process_sensor_data(sonar_data, acoustic_buffer)
        
        if threat:
            print(f"  → Риск: {threat.risk_level}/10, TTC: {threat.time_to_collision_s:.1f}s")
            print(f"  → {threat.recommendation}\n")
        else:
            print("  → Данные обработаны\n")
        
        time.sleep(0.05)  # Задержка для читаемости
    
    # Вывести статистику
    dgpd.print_statistics()


if __name__ == "__main__":
    # Запустить симуляцию
    run_simulation()
    
    print("\n" + "="*60)
    print("Дополнительная информация:")
    print("="*60)
    print("""
Модули DiveGuard:
1. sensor_fusion.py         - Слияние SONAR и HYDROPHONE
2. propeller_classifier.py  - Классификация типов судов по акустике
3. threat_assessment.py     - Оценка риска столкновения
4. diver_alert_controller.py - Мультимодальные предупреждения

Чтобы использовать с реальными устройствами:
- Подключите Blue Robotics Ping360 SONAR на /dev/ttyUSB0
- Подключите гидрофон SM111 PZT с буфером OPA1642 на аудиовход
- Подключите LED, динамик и моторы вибрации к GPIO
- Используйте BlueOS ROS 2 для контроля робота

Примеры интеграции:
- Для BlueRobotics: см. documentation/bluerobotic_integration.md
- Для своих систем: адаптируйте классы Device в каждом модуле
    """)
    print("="*60)
