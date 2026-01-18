# diver_alert_controller.py
# Контроллер мультимодального предупреждения дайверов (светлое, звук, вибрация, позиция)

import time
import numpy as np
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiverAlertController")

class AlertMode(Enum):
    """Режимы предупреждения"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AlertPattern:
    """Паттерн многомодального предупреждения"""
    light_color: Tuple[int, int, int]      # RGB (0-255)
    light_strobe_hz: float
    audio_freq_hz: int
    audio_pulse_rate_hz: float
    audio_volume_percent: int
    haptic_motors: List[float]  # Интенсивность для каждого мотора (0-1)
    body_yaw_deg: float
    body_pitch_deg: float

class DiverAlertController:
    """
    Мультимодальная система предупреждения дайверов
    
    Предупреждение иерархия (параллельное выполнение):
    1. СВЕТ: RGB LED стробоскоп с кодированием
    2. ЗВУК: Ультразвуковой тон + частотная модуляция
    3. ВИБРАЦИЯ: Паттерны вибрирующих моторов
    4. КОРПУС: Физическое позиционирование робота (рыскание/наклон)
    """
    
    # Паттерны предупреждений для каждого уровня риска
    ALERT_PATTERNS = {
        AlertMode.SAFE: AlertPattern(
            light_color=(0, 255, 0),          # Зелёный
            light_strobe_hz=0.5,
            audio_freq_hz=10000,
            audio_pulse_rate_hz=0.0,          # Постоянный тон
            audio_volume_percent=0,           # Без звука
            haptic_motors=[0.0, 0.0, 0.0],
            body_yaw_deg=0,
            body_pitch_deg=0
        ),
        AlertMode.LOW: AlertPattern(
            light_color=(0, 255, 0),          # Зелёный
            light_strobe_hz=1.0,
            audio_freq_hz=15000,
            audio_pulse_rate_hz=0.0,          # Постоянный
            audio_volume_percent=25,
            haptic_motors=[0.1, 0.0, 0.1],
            body_yaw_deg=0,
            body_pitch_deg=0
        ),
        AlertMode.MEDIUM: AlertPattern(
            light_color=(255, 165, 0),        # Янтарный
            light_strobe_hz=3.0,
            audio_freq_hz=25000,
            audio_pulse_rate_hz=5.0,          # 5 Hz пульс
            audio_volume_percent=60,
            haptic_motors=[0.3, 0.2, 0.3],
            body_yaw_deg=45,                  # Поворот указывает на угрозу
            body_pitch_deg=0
        ),
        AlertMode.HIGH: AlertPattern(
            light_color=(255, 100, 0),        # Красный-оранжевый
            light_strobe_hz=5.0,
            audio_freq_hz=35000,
            audio_pulse_rate_hz=8.0,          # 8 Hz пульс
            audio_volume_percent=85,
            haptic_motors=[0.6, 0.4, 0.6],
            body_yaw_deg=90,                  # Ясное указание направления
            body_pitch_deg=10
        ),
        AlertMode.CRITICAL: AlertPattern(
            light_color=(255, 0, 0),          # Ярко красный
            light_strobe_hz=10.0,
            audio_freq_hz=40000,
            audio_pulse_rate_hz=10.0,         # 10 Hz быстрый пульс
            audio_volume_percent=100,
            haptic_motors=[1.0, 1.0, 1.0],   # Максимум
            body_yaw_deg=180,                 # Агрессивный поворот
            body_pitch_deg=20
        )
    }
    
    def __init__(self, led_pins=None, speaker_pin=None, motor_pins=None, 
                 robot_controller=None):
        """
        Инициализация контроллера предупреждений
        
        Args:
            led_pins: Номера GPIO для LED
            speaker_pin: GPIO для динамика
            motor_pins: GPIO для вибромоторов [left, center, right]
            robot_controller: Объект контроля робота (для рыскания/наклона)
        """
        self.led_pins = led_pins or [17, 27, 22, 23]
        self.speaker_pin = speaker_pin or 24
        self.motor_pins = motor_pins or [5, 6, 13]
        self.robot = robot_controller
        
        self.current_mode = AlertMode.SAFE
        self.last_alert_time = 0
        self.alert_duration = 10  # секунд
        
        logger.info("DiverAlertController инициализирован")
    
    def alert_diver(self, threat_assessment, current_time: float):
        """
        Выдать скоординированное предупреждение на основе угрозы
        
        Args:
            threat_assessment: ThreatAssessment из threat_assessment.py
            current_time: Текущее время (для управления длительностью)
        """
        risk_level = threat_assessment.risk_level
        azimuth = threat_assessment.azimuth_deg
        closing_speed = threat_assessment.closing_speed_mps
        
        # Определить режим предупреждения
        if risk_level <= 1:
            alert_mode = AlertMode.SAFE
        elif risk_level <= 3:
            alert_mode = AlertMode.LOW
        elif risk_level <= 5:
            alert_mode = AlertMode.MEDIUM
        elif risk_level <= 7:
            alert_mode = AlertMode.HIGH
        else:
            alert_mode = AlertMode.CRITICAL
        
        self.current_mode = alert_mode
        self.last_alert_time = current_time
        
        # Получить паттерн для этого режима
        pattern = self.ALERT_PATTERNS[alert_mode]
        
        # Модулировать паттерн в зависимости от направления угрозы
        modulated_pattern = self._modulate_pattern_by_azimuth(pattern, azimuth)
        
        # 1. СВЕТОВОЕ ПРЕДУПРЕЖДЕНИЕ
        self._light_warning(modulated_pattern, azimuth)
        
        # 2. ЗВУКОВОЕ ПРЕДУПРЕЖДЕНИЕ
        self._audio_warning(modulated_pattern, azimuth, closing_speed)
        
        # 3. ВИБРАЦИОННОЕ ПРЕДУПРЕЖДЕНИЕ
        self._haptic_warning(modulated_pattern, azimuth)
        
        # 4. ПОЗИЦИОНИРОВАНИЕ КОРПУСА
        self._body_warning(modulated_pattern, azimuth, threat_assessment)
        
        logger.info(f"Предупреждение уровня {alert_mode.name}: риск={risk_level}, азимут={azimuth:.0f}°")
    
    def _modulate_pattern_by_azimuth(self, pattern: AlertPattern, 
                                     azimuth: float) -> AlertPattern:
        """
        Модулировать паттерн в зависимости от направления угрозы
        
        Args:
            pattern: Базовый паттерн
            azimuth: Азимут угрозы 0-360°
        
        Returns:
            AlertPattern: Модулированный паттерн
        """
        # Скопировать паттерн
        mod_pattern = AlertPattern(
            light_color=pattern.light_color,
            light_strobe_hz=pattern.light_strobe_hz,
            audio_freq_hz=pattern.audio_freq_hz,
            audio_pulse_rate_hz=pattern.audio_pulse_rate_hz,
            audio_volume_percent=pattern.audio_volume_percent,
            haptic_motors=pattern.haptic_motors.copy(),
            body_yaw_deg=azimuth,  # ← Рыскание указывает на азимут угрозы!
            body_pitch_deg=pattern.body_pitch_deg
        )
        
        return mod_pattern
    
    def _light_warning(self, pattern: AlertPattern, azimuth: float):
        """
        LED стробоскоп кодирует информацию об угрозе
        
        Кодирование:
        - Цвет: Красный (опасность) -> Янтарный (осторожность) -> Зелёный (безопасно)
        - Частота стробо: 0.1 Hz (низкий риск) -> 10 Hz (критично)
        - Позиция LED: Свет направлен на азимут угрозы
        
        Пример: Риск 8, азимут 45° (северо-восток):
        → Красный цвет, 5 Hz стробо
        → Самые яркие передне-правые LED
        """
        color = pattern.light_color
        strobe_hz = pattern.light_strobe_hz
        
        # Определить позицию LED (8 LED по окружности)
        # 0° = передний, 90° = правый (старборд), 180° = задний, 270° = левый (порт)
        led_idx = int((azimuth / 45) % 8)
        
        print(f"💡 СВЕТ: Цвет={color}, Стробо={strobe_hz}Hz, LED_направление={led_idx*45}°")
        
        # Реальная реализация:
        # self.led_strip.strobe_pattern(
        #     color=color,
        #     frequency_hz=strobe_hz,
        #     focus_led_idx=led_idx,
        #     brightness=255,
        #     duration_s=self.alert_duration
        # )
    
    def _audio_warning(self, pattern: AlertPattern, azimuth: float, 
                      closing_speed: float):
        """
        Ультразвуковой тон кодирует информацию об угрозе
        
        Кодирование:
        - Базовая частота: 15 кГц (низкий риск) -> 40 кГц (критично)
        - Модуляция:
          - Постоянный тон: Дальняя угроза
          - Пульсирующий (5 Hz): Приближается
          - Быстрый пульс (10 Hz): Критично
        - Частотный свип: Указывает направление (эффект Доплера)
        """
        base_freq = pattern.audio_freq_hz
        pulse_rate = pattern.audio_pulse_rate_hz
        volume = pattern.audio_volume_percent
        
        # Доплеровский свип для кодирования направления
        # Азимут 0° = без свипа (спереди)
        # Азимут 45° = восходящий свип (приближается спереди-справа)
        # Азимут 180° = нисходящий свип (приближается сзади)
        sweep_direction = int((azimuth - 180) / 180 * 100)  # -100 to +100%
        
        print(f"🔊 ЗВУК: Частота={base_freq}Hz, Пульс={pulse_rate}Hz, Громкость={volume}%, Доплер={sweep_direction:+d}%")
        
        # Реальная реализация:
        # self.speaker.play_tone(
        #     frequency=base_freq,
        #     pulse_rate=pulse_rate,
        #     duration_s=self.alert_duration,
        #     doppler_sweep=sweep_direction,
        #     volume=volume/100
        # )
    
    def _haptic_warning(self, pattern: AlertPattern, azimuth: float):
        """
        Вибрирующие моторы кодируют направление угрозы
        
        Расположение моторов:
        - Мотор 0: Левый борт (порт)
        - Мотор 1: Центральный
        - Мотор 2: Правый борт (старборд)
        
        Кодирование:
        - Азимут 0° (спереди): Вибрирует центр
        - Азимут 90° (справа): Вибрирует справа
        - Азимут 270° (слева): Вибрирует слева
        - Интенсивность: уровень_риска -> амплитуда вибрации (0-1)
        """
        # Преобразовать азимут в активацию моторов
        # Использовать тригонометрию для плавного распределения
        rad = np.radians(azimuth)
        
        motor_powers = [
            max(0, -np.sin(rad)) * pattern.haptic_motors[0],  # Левый
            np.cos(rad) * pattern.haptic_motors[1],           # Центральный (макс спереди)
            max(0, np.sin(rad)) * pattern.haptic_motors[2]    # Правый
        ]
        
        # Нормализовать
        max_power = max(motor_powers) if max(motor_powers) > 0 else 1
        motor_powers = [p / max_power for p in motor_powers]
        
        print(f"📳 ВИБРАЦИЯ: Левый={motor_powers[0]:.2f}, Центр={motor_powers[1]:.2f}, Правый={motor_powers[2]:.2f}")
        
        # Реальная реализация:
        # for motor_idx, power in enumerate(motor_powers):
        #     self.vibration_motors[motor_idx].set_pwm(power)
        #     time.sleep(0.05)
    
    def _body_warning(self, pattern: AlertPattern, azimuth: float, 
                     threat_assessment):
        """
        Робот физически ориентирует свой корпус к угрозе
        
        Это САМОЕ ИНТУИТИВНОЕ предупреждение для дайвера:
        Корпус робота СТАНОВИТСЯ СТРЕЛКОЙ, указывающей на опасность!
        
        Команды:
        - Рыскание: Выравнивание курса робота с азимутом угрозы
        - Наклон: Указание вверх для угроз с поверхности
        - Удержание: 5-10 секунд чтобы дайвер увидел ориентацию
        
        Пример: Угроза на азимуте 135° (юго-восток)
        → Робот рыскает 135° от своего текущего курса
        → Дайвер видит робот, указывающий на угрозу
        """
        desired_yaw = pattern.body_yaw_deg
        desired_pitch = pattern.body_pitch_deg
        
        print(f"🤖 КОРПУС: Рыскание={desired_yaw:.0f}°, Наклон={desired_pitch:.0f}°")
        print(f"   Это визуальная стрелка, указывающая на: {threat_assessment.vessel_type.upper()}")
        
        # Реальная реализация (для Blue Robotics ArduSub):
        # if self.robot:
        #     # Установить желаемый курс (рыскание)
        #     self.robot.set_desired_heading(desired_yaw, speed=0.5)
        #     
        #     # Ждать ротации (обычно 3-5 секунд для 180°)
        #     time.sleep(3)
        #     
        #     # Наклон камеры/антенны если нужно
        #     if abs(desired_pitch) > 5:
        #         self.robot.tilt_camera(angle=desired_pitch, speed=0.3)
        #         time.sleep(2)
        #     
        #     logger.info(f"Робот ориентирован на {desired_yaw}° указывая на угрозу")
    
    def clear_alert(self):
        """Очистить текущее предупреждение"""
        self.current_mode = AlertMode.SAFE
        
        # Отключить все выходы
        print("🟢 Все предупреждения отключены")


# Вспомогательный класс для имитации динамика
class MockUltrasonicSpeaker:
    """Имитация ультразвукового динамика для тестирования"""
    def play_tone(self, frequency, pulse_rate, duration_s, doppler_sweep=0, volume=1.0):
        print(f"   [SPEAKER] {frequency}Hz @ {pulse_rate}Hz пульс, свип={doppler_sweep}, громкость={volume:.0%}")

# Вспомогательный класс для имитации LED
class MockLEDStrip:
    """Имитация LED полосы для тестирования"""
    def strobe_pattern(self, color, frequency_hz, focus_led_idx, brightness=255, duration_s=10):
        color_names = {(0, 255, 0): "ЗЕЛЁНЫЙ", (255, 165, 0): "ЯНТАРНЫЙ", (255, 0, 0): "КРАСНЫЙ"}
        color_name = color_names.get(color, f"RGB{color}")
        print(f"   [LED] {color_name} @ {frequency_hz}Hz, фокус на LED#{focus_led_idx}, яркость={brightness}")


if __name__ == "__main__":
    print("=== Тест DiverAlertController ===\n")
    
    from threat_assessment import ThreatAssessment
    
    controller = DiverAlertController()
    
    # Тестовые сценарии
    test_scenarios = [
        {
            'name': 'Низкий риск: Дальний корабль',
            'distance_m': 500,
            'azimuth_deg': 45,
            'elevation_deg': 0,
            'closing_speed_mps': -1,
            'ttc_s': float('inf'),
            'risk_level': 1,
            'vessel_type': 'ship',
            'threat_prob': 0.01
        },
        {
            'name': 'Средний риск: Лодка приближается справа',
            'distance_m': 200,
            'azimuth_deg': 90,
            'elevation_deg': 0,
            'closing_speed_mps': 2.0,
            'ttc_s': 100,
            'risk_level': 4,
            'vessel_type': 'boat',
            'threat_prob': 0.3
        },
        {
            'name': 'КРИТИЧЕСКИЙ: Подводная лодка спереди!',
            'distance_m': 80,
            'azimuth_deg': 10,
            'elevation_deg': 5,
            'closing_speed_mps': 4.5,
            'ttc_s': 18,
            'risk_level': 9,
            'vessel_type': 'submarine',
            'threat_prob': 0.92
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Сценарий: {scenario['name']}")
        print(f"{'='*60}")
        
        threat = ThreatAssessment(
            distance_m=scenario['distance_m'],
            azimuth_deg=scenario['azimuth_deg'],
            elevation_deg=scenario['elevation_deg'],
            closing_speed_mps=scenario['closing_speed_mps'],
            time_to_collision_s=scenario['ttc_s'],
            risk_level=scenario['risk_level'],
            vessel_type=scenario['vessel_type'],
            threat_probability=scenario['threat_prob'],
            recommendation="Тестовое предупреждение"
        )
        
        # Выдать предупреждение
        controller.alert_diver(threat, current_time=time.time())
        print()
