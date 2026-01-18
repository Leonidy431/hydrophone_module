# threat_assessment.py
# Модуль оценки уровня угрозы и расчета параметров столкновения

import numpy as np
from dataclasses import dataclass
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThreatAssessment")

@dataclass
class ThreatAssessment:
    """Полная оценка угрозы со всеми параметрами"""
    distance_m: float
    azimuth_deg: float
    elevation_deg: float
    closing_speed_mps: float
    time_to_collision_s: float
    risk_level: int  # 1-10
    vessel_type: str
    threat_probability: float  # 0-1
    recommendation: str


class ThreatAssessmentEngine:
    """
    Двигатель оценки угрозы столкновения
    
    Вычисляет риск столкновения на основе:
    - Расстояния до винта
    - Относительной скорости (скорость сближения)
    - Размера и мощности винта (тип судна)
    - Позиции и скорости робота
    """
    
    # Пороги риска столкновения
    COLLISION_RISK_THRESHOLDS = {
        'critical': (8, 10),      # TTC < 5s
        'high': (5, 8),           # TTC 5-15s
        'medium': (2, 5),         # TTC 15-60s
        'low': (0, 2)             # TTC > 60s
    }
    
    # Опасность разных типов судов
    VESSEL_DANGER_FACTORS = {
        'ship': 0.8,           # Опасны, но медленны
        'submarine': 1.0,      # МАКСИМАЛЬНО ОПАСНЫ
        'boat': 0.9,           # Быстры и опасны
        'rov': 0.3,            # Другой ROV - предсказуем
        'auv': 0.4,            # AUV предсказуемы
        'unknown': 0.7         # По умолчанию - среднее
    }
    
    def __init__(self):
        """Инициализация двигателя оценки"""
        self.previous_assessment = None
        self.assessment_history = []
    
    def assess_threat(self, fused_state, vessel_classification) -> ThreatAssessment:
        """
        Вычислить риск столкновения на основе слитого состояния и классификации
        
        Args:
            fused_state: FusedState из sensor_fusion.py
            vessel_classification: VesselClassification из propeller_classifier.py
        
        Returns:
            ThreatAssessment: Полная оценка угрозы
        """
        distance = fused_state.distance
        azimuth = fused_state.azimuth
        elevation = fused_state.elevation
        closing_speed = fused_state.closing_speed
        vessel_type = vessel_classification.vessel_type
        
        # Прогноз времени до столкновения
        if closing_speed > 0.1:  # Движется к нам
            time_to_collision = distance / closing_speed
        else:
            time_to_collision = float('inf')
        
        # Вычислить уровень риска (1-10 шкала)
        risk_level = self._ttc_to_risk_level(time_to_collision)
        
        # Модуляция риска в зависимости от типа судна
        vessel_factor = self.VESSEL_DANGER_FACTORS.get(vessel_type, 0.7)
        adjusted_risk = self._adjust_risk_by_vessel_type(
            risk_level, vessel_type, closing_speed, 
            vessel_classification.cavitation_level
        )
        
        # Вероятность реальной угрозы
        threat_probability = self._calculate_threat_probability(
            distance, time_to_collision, closing_speed, vessel_type
        )
        
        # Рекомендация для дайвера
        recommendation = self._get_recommendation(
            adjusted_risk, vessel_type, azimuth, time_to_collision
        )
        
        assessment = ThreatAssessment(
            distance_m=distance,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            closing_speed_mps=closing_speed,
            time_to_collision_s=time_to_collision,
            risk_level=adjusted_risk,
            vessel_type=vessel_type,
            threat_probability=threat_probability,
            recommendation=recommendation
        )
        
        self.assessment_history.append(assessment)
        if len(self.assessment_history) > 10:
            self.assessment_history.pop(0)
        
        self.previous_assessment = assessment
        
        return assessment
    
    def _ttc_to_risk_level(self, ttc: float) -> int:
        """
        Преобразовать время до столкновения в уровень риска 1-10
        
        Args:
            ttc: Время до столкновения в секундах
        
        Returns:
            int: Уровень риска 1-10
        """
        if ttc < 5:
            return 10
        elif ttc < 15:
            return 8
        elif ttc < 30:
            return 6
        elif ttc < 60:
            return 4
        elif ttc < 120:
            return 2
        else:
            return 1
    
    def _adjust_risk_by_vessel_type(self, base_risk: int, vessel_type: str,
                                   closing_speed: float, 
                                   cavitation_level: float) -> int:
        """
        Модулировать базовый уровень риска в зависимости от типа судна
        
        Args:
            base_risk: Базовый уровень риска из TTC
            vessel_type: Тип судна
            closing_speed: Скорость сближения м/с
            cavitation_level: Уровень кавитации 0-1
        
        Returns:
            int: Скорректированный уровень риска 1-10
        """
        risk = base_risk
        
        # Подводная лодка: +2 (максимально опасны!)
        if vessel_type == 'submarine':
            risk = min(10, risk + 2)
            if cavitation_level > 0.3:
                risk = 10  # Критично!
        
        # Корабль: +1 (опасны но медленны)
        elif vessel_type == 'ship':
            if closing_speed > 5:  # Более 5 м/с - необычно, мог ускориться
                risk = min(10, risk + 1)
        
        # Лодка: +1 (быстры)
        elif vessel_type == 'boat':
            if closing_speed > 3:
                risk = min(10, risk + 1)
            if cavitation_level > 0.2:
                risk = min(10, risk + 1)
        
        # ROV и AUV: -1 (менее опасны, предсказуемы)
        elif vessel_type in ['rov', 'auv']:
            risk = max(1, risk - 1)
        
        return max(1, min(risk, 10))
    
    def _calculate_threat_probability(self, distance: float, ttc: float,
                                     closing_speed: float, 
                                     vessel_type: str) -> float:
        """
        Вычислить вероятность реальной угрозы столкновения (0-1)
        
        Args:
            distance: Расстояние в метрах
            ttc: Время до столкновения в секундах
            closing_speed: Скорость сближения м/с
            vessel_type: Тип судна
        
        Returns:
            float: Вероятность угрозы 0-1
        """
        # Базовая вероятность из TTC
        if ttc < 5:
            prob = 0.95  # Критично - очень вероятно столкновение
        elif ttc < 15:
            prob = 0.70  # Высокий риск
        elif ttc < 60:
            prob = 0.40  # Средний риск
        elif ttc < 300:
            prob = 0.10  # Низкий риск
        else:
            prob = 0.01  # Минимальный риск
        
        # Модулировать по скорости сближения
        if closing_speed < 0.1:
            prob *= 0.1  # Движется от нас - маловероятно
        elif closing_speed > 5:
            prob = min(0.99, prob * 1.5)  # Быстро приближается
        
        # Модулировать по типу судна
        if vessel_type == 'submarine':
            prob = min(0.99, prob * 1.3)  # Подлодки опаснее
        elif vessel_type in ['rov', 'auv']:
            prob *= 0.5  # Другие ROV менее опасны
        
        return prob
    
    def _get_recommendation(self, risk_level: int, vessel_type: str,
                           azimuth: float, ttc: float) -> str:
        """
        Получить текстовую рекомендацию для дайвера
        
        Args:
            risk_level: Уровень риска 1-10
            vessel_type: Тип судна
            azimuth: Азимут угрозы 0-360°
            ttc: Время до столкновения
        
        Returns:
            str: Рекомендация для дайвера
        """
        # Определить направление
        if azimuth < 45 or azimuth > 315:
            direction = "спереди"
        elif azimuth < 135:
            direction = "справа (старборд)"
        elif azimuth < 225:
            direction = "сзади"
        else:
            direction = "слева (порт)"
        
        # Определить вертикальное направление
        if azimuth == 0:
            vertical = "с поверхности"
        elif azimuth == 180:
            vertical = "снизу"
        else:
            vertical = ""
        
        # Сгенерировать рекомендацию в зависимости от риска
        if risk_level >= 9:
            return f"⚠️⚠️ КРИТИЧНО! {vessel_type.upper()} закрывается {direction} со скоростью! УХОДИТЕ НЕМЕДЛЕННО!"
        elif risk_level >= 7:
            if ttc < 10:
                return f"🔴 ВЫСОКИЙ РИСК! {vessel_type} приближается {direction} (TTC {ttc:.1f}s). УХОДИТЕ!"
            else:
                return f"🟡 РИСК: {vessel_type} обнаружен {direction}. Подготовьтесь к манёвру."
        elif risk_level >= 5:
            return f"🟠 СРЕДНИЙ РИСК: {vessel_type} на расстоянии, {direction}. Мониторьте."
        elif risk_level >= 3:
            return f"🟡 НИЗКИЙ РИСК: {vessel_type} обнаружен {direction}. Будьте внимательны."
        else:
            return f"🟢 Судно обнаружено {direction}, но опасности нет."
    
    def get_evasion_maneuver(self, threat_assessment: ThreatAssessment, 
                           robot_depth: float) -> Dict:
        """
        Рекомендовать манёвр уклонения для робота
        
        Args:
            threat_assessment: Оценка угрозы
            robot_depth: Текущая глубина робота в метрах
        
        Returns:
            dict: {
                'desired_yaw': float (градусы),
                'desired_pitch': float (градусы),
                'desired_depth': float (метры),
                'speed_percent': int (0-100),
                'urgency': str ('slow', 'normal', 'fast', 'emergency')
            }
        """
        azimuth = threat_assessment.azimuth_deg
        elevation = threat_assessment.elevation_deg
        risk = threat_assessment.risk_level
        distance = threat_assessment.distance_m
        
        # Определить направление манёвра - ПРОТИВОПОЛОЖНОЕ угрозе
        evasion_yaw = (azimuth + 180) % 360  # Развернуться в противоположную сторону
        
        # Если угроза с поверхности, погрузиться
        # Если угроза снизу, всплыть
        evasion_pitch = 0
        evasion_depth = robot_depth
        
        if elevation > 30:  # Угроза сверху
            evasion_depth = robot_depth + 20  # Погрузиться на 20м
            evasion_pitch = -10  # Слегка вниз
        elif elevation < -30:  # Угроза снизу
            evasion_depth = robot_depth - 10  # Всплыть на 10м
            evasion_pitch = 10  # Слегка вверх
        
        # Скорость манёвра зависит от срочности
        if risk >= 9:
            speed = 100  # На полной скорости!
            urgency = 'emergency'
        elif risk >= 7:
            speed = 80
            urgency = 'fast'
        elif risk >= 5:
            speed = 60
            urgency = 'normal'
        else:
            speed = 40
            urgency = 'slow'
        
        return {
            'desired_yaw': evasion_yaw,
            'desired_pitch': evasion_pitch,
            'desired_depth': evasion_depth,
            'speed_percent': speed,
            'urgency': urgency
        }


if __name__ == "__main__":
    print("=== Тест ThreatAssessmentEngine ===\n")
    
    from sensor_fusion import FusedState
    from propeller_classifier import VesselClassification
    
    engine = ThreatAssessmentEngine()
    
    # Тестовое состояние: быстрая лодка на расстоянии 100м, спереди, движется к нам
    fused_state = FusedState(
        distance=100.0,
        azimuth=30.0,
        elevation=0.0,
        confidence=0.85,
        vessel_type='boat',
        frequency_peaks=[[100, 0.8], [200, 0.4], [300, 0.2]],
        closing_speed=3.5,
        timestamp=0
    )
    
    vessel_class = VesselClassification(
        vessel_type='boat',
        confidence=0.88,
        propeller_rpm_estimate=1500,
        blade_count_estimate=4,
        cavitation_level=0.25,
        threat_level=6
    )
    
    # Оценить угрозу
    assessment = engine.assess_threat(fused_state, vessel_class)
    
    print(f"Тип судна: {assessment.vessel_type}")
    print(f"Расстояние: {assessment.distance_m:.1f} м")
    print(f"Азимут: {assessment.azimuth_deg:.1f}°")
    print(f"Скорость сближения: {assessment.closing_speed_mps:.2f} м/с")
    print(f"Время до столкновения: {assessment.time_to_collision_s:.1f} с")
    print(f"Уровень риска: {assessment.risk_level}/10")
    print(f"Вероятность угрозы: {assessment.threat_probability:.1%}")
    print(f"\nРекомендация: {assessment.recommendation}\n")
    
    # Манёвр уклонения
    maneuver = engine.get_evasion_maneuver(assessment, robot_depth=50)
    print(f"Манёвр уклонения:")
    print(f"  Поворот на азимут: {maneuver['desired_yaw']:.1f}°")
    print(f"  Наклон: {maneuver['desired_pitch']:.1f}°")
    print(f"  Целевая глубина: {maneuver['desired_depth']:.1f} м")
    print(f"  Скорость: {maneuver['speed_percent']}%")
    print(f"  Срочность: {maneuver['urgency']}")
