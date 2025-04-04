import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G, pi
from scipy.special import sinc
from scipy.signal.windows import hann
from typing import Dict, Tuple

class SARSystem:
    def __init__(self, config: Dict):
        """Класс для проектирования SAR-системы на малых спутниках
        
        Args:
            config (Dict): Конфигурация системы с параметрами:
                - h_sar: Высота орбиты [м]
                - f0: Рабочая частота [Гц]
                - gamma: Угол места [град]
                - grg_res: Разрешение по дальности [м]
                - az_res: Разрешение по азимуту [м]
                - swath: Ширина полосы обзора [м]
                - tau_p: Длительность импульса [с]
                - L_az: Длина антенны [м]
                - P_avg: Средняя мощность [Вт]
        """
        self.config = config
        self.R_earth = 6371e3  # Средний радиус Земли [м]

    def orbit_velocity(self, h: float) -> float:
        """Расчет орбитальной скорости для круговой орбиты
        
        Args:
            h (float): Высота орбиты над поверхностью Земли [м]
            
        Returns:
            float: Орбитальная скорость [м/с]
        """
        return np.sqrt(G * 5.9742e24 / (self.R_earth + h))
    
    def calculate_geometry(self, gamma: float) -> Tuple[float, float, float]:
        """Расчет геометрических параметров SAR
        
        Args:
            gamma (float): Угол места [град]
            
        Returns:
            Tuple[float, float, float]:
                - eta_c: Угол падения [рад]
                - R_c: Наклонная дальность [м]
                - gamma_rad: Угол места [рад]
        """
        h = self.config['h_sar']
        R_s = self.R_earth + h
        gamma_rad = np.deg2rad(gamma)
        
        # Расчет угла падения
        eta_c = np.arcsin(R_s / self.R_earth * np.sin(gamma_rad))
        
        # Расчет наклонной дальности
        R_c = R_s * np.cos(eta_c) - np.sqrt(
            self.R_earth**2 - R_s**2 + R_s**2 * np.cos(eta_c)**2
        )
        return eta_c, R_c, gamma_rad
    
    def minimum_antenna_area(self, gamma_range: np.ndarray) -> np.ndarray:
        """Расчет минимальной площади антенны (формула 2.11)
        
        Args:
            gamma_range (np.ndarray): Диапазон углов места [град]
            
        Returns:
            np.ndarray: Минимальные площади антенны [м²]
        """
        f0 = self.config['f0']
        h = self.config['h_sar']
        v = self.orbit_velocity(h)
        areas = []
        
        for gamma in gamma_range:
            eta_c, R_c, _ = self.calculate_geometry(gamma)
            area = (4 * c / f0) * R_c * v * np.tan(eta_c)
            areas.append(area)
            
        return np.array(areas)
    
    def ambiguity_analysis(self, gamma: float, PRF: float) -> Dict:
        """Анализ неоднозначностей SAR
        
        Args:
            gamma (float): Угол места [град]
            PRF (float): Частота повторения импульсов [Гц]
            
        Returns:
            Dict: Результаты анализа неоднозначностей
        """
        config = self.config
        lambda0 = c / config['f0']
        h = config['h_sar']
        v = self.orbit_velocity(h)
        
        eta_c, R_c, _ = self.calculate_geometry(gamma)
        swath_slant = config['swath'] * np.sin(eta_c)
        
        L_az = config['L_az']
        theta_az = lambda0 / L_az
        B_dop = 2 * v * theta_az / lambda0
        
        return {
            'AASR': self._calculate_AASR(PRF, B_dop),
            'RASR': self._calculate_RASR(PRF, swath_slant, eta_c),
            'PRF_min': B_dop
        }
    
    def _calculate_AASR(self, PRF: float, B_dop: float) -> float:
        """Расчет азимутальной неоднозначности"""
        return 10 * np.log10((B_dop / PRF)**2)
    
    def _calculate_RASR(self, PRF: float, swath_slant: float, eta_c: float) -> float:
        """Расчет дальностной неоднозначности"""
        tau_swath = 2 * swath_slant / c
        return 10 * np.log10((tau_swath * PRF)**2 * np.cos(eta_c))
    
    def plot_antenna_pattern(self, pattern_type: str = 'uniform'):
        """Визуализация диаграммы направленности антенны
        
        Args:
            pattern_type (str): Тип распределения:
                - 'uniform': Равномерное
                - 'hanning': С оконной функцией Хэмминга
        """
        theta = np.linspace(-np.pi/2, np.pi/2, 1000)
        L = self.config['L_az']
        lambda0 = c / self.config['f0']
        
        if pattern_type == 'uniform':
            pattern = (np.sinc(L * np.sin(theta) / lambda0))**2
        elif pattern_type == 'hanning':
            n = len(theta)
            window = hann(n)
            pattern = np.abs(np.fft.fftshift(np.fft.fft(window)))**2
            pattern /= np.max(pattern)
            
            # Масштабирование углов для физической интерпретации
            k = L / lambda0
            theta = np.arcsin(np.linspace(-1, 1, n) / k)
        else:
            raise ValueError("Неподдерживаемый тип распределения")
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.degrees(theta), 10 * np.log10(pattern))
        plt.title(f"Диаграмма направленности ({pattern_type})")
        plt.xlabel('Угол [град]')
        plt.ylabel('Нормированная мощность [дБ]')
        plt.grid(True)
        plt.ylim(-40, 0)
        plt.show()

    def design_flow(self, mode: str = 'ideal') -> Dict:
        """Процесс проектирования SAR-системы
        
        Args:
            mode (str): Режим проектирования:
                - 'ideal': Без ограничений
                - 'constrained': С ограничениями
        """
        if mode == 'ideal':
            return self._ideal_design()
        return self._constrained_design()
    
    def _ideal_design(self) -> Dict:
        """Идеальное проектирование системы"""
        h = self.config['h_sar']
        v = self.orbit_velocity(h)
        eta_c = np.deg2rad(self.config['gamma'])
        
        return {
            'Bandwidth': c / (2 * self.config['grg_res'] * np.sin(eta_c)),
            'Antenna_Length': 2 * self.config['az_res']
        }
    
    def _constrained_design(self) -> Dict:
        """Проектирование с учетом ограничений"""
        # Реализация требует дополнительных специфических расчетов
        return {'status': 'Не реализовано'}

if __name__ == "__main__":
    # Пример конфигурации для TerraSAR-X
    sar_config = {
        'h_sar': 500e3,
        'f0': 9.65e9,
        'gamma': 30,
        'grg_res': 3.0,
        'az_res': 3.0,
        'swath': 30e3,
        'tau_p': 30e-6,
        'L_az': 4.8,
        'P_avg': 300
    }
    
    sar = SARSystem(sar_config)
    
    # Пример 1: Анализ минимальной площади антенны
    angles = np.linspace(20, 45, 25)
    areas = sar.minimum_antenna_area(angles)
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, areas)
    plt.title('Зависимость минимальной площади антенны от угла места')
    plt.xlabel('Угол места [град]')
    plt.ylabel('Площадь антенны [м²]')
    plt.grid(True)
    plt.show()
    
    # Пример 2: Анализ неоднозначностей
    analysis = sar.ambiguity_analysis(35, 3000)
    print("Результаты анализа неоднозначностей:")
    print(f"AASR: {analysis['AASR']:.2f} дБ")
    print(f"RASR: {analysis['RASR']:.2f} дБ")
    print(f"Минимальная PRF: {analysis['PRF_min']:.2f} Гц")
    
    # Пример 3: Диаграммы направленности
    sar.plot_antenna_pattern('uniform')
    sar.plot_antenna_pattern('hanning')
    
    # Пример 4: Процесс проектирования
    design = sar.design_flow('ideal')
    print("\nРезультаты проектирования:")
    print(f"Требуемая полоса: {design['Bandwidth']/1e6:.2f} МГц")
    print(f"Длина антенны: {design['Antenna_Length']:.2f} м")