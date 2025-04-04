import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G, pi
from scipy.special import j1, sinc
from scipy.signal.windows import hann
from typing import Dict, Tuple

class SARSystem:
    def __init__(self, config: Dict):
        """Класс для проектирования SAR-системы на малых спутниках
        
        Args:
            config (Dict): Словарь с параметрами системы:
                - h_sar: Высота орбиты спутника [м]
                - f0: Центральная частота [Гц]
                - gamma: Угол места [град]
                - grg_res: Разрешение по дальности [м]
                - az_res: Разрешение по азимуту [м]
                - swath: Ширина полосы обзора [м]
                - tau_p: Длительность импульса [с]
                - L_az: Длина антенны [м]
                - P_avg: Средняя мощность [Вт]
        """
        self.config = config
        self.R_earth = 6371e3  # Радиус Земли в метрах
        
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
            gamma (float): Угол места в градусах
            
        Returns:
            Tuple[float, float, float]:
                - eta_c: Угол падения [рад]
                - R_c: Наклонная дальность до цели [м]
                - gamma_rad: Угол места [рад]
        """
        h = self.config['h_sar']
        R_s = self.R_earth + h  # Радиус орбиты спутника
        gamma_rad = np.deg2rad(gamma)  # Преобразование в радианы
        
        # Расчет угла падения по закону синусов
        eta_c = np.arcsin(R_s / self.R_earth * np.sin(gamma_rad))
        
        # Расчет наклонной дальности до цели
        R_c = R_s * np.cos(eta_c) - np.sqrt(
            self.R_earth**2 - R_s**2 + R_s**2 * np.cos(eta_c)**2
        )
        return eta_c, R_c, gamma_rad
    
    def minimum_antenna_area(self, gamma_range: np.ndarray) -> np.ndarray:
        """Расчет минимальной площади антенны (формула 2.11 из исследования)
        
        Args:
            gamma_range (np.ndarray): Диапазон углов места в градусах
            
        Returns:
            np.ndarray: Минимальные площади антенны для каждого угла [м²]
        """
        f0 = self.config['f0']  # Частота несущей
        h = self.config['h_sar']  # Высота орбиты
        v = self.orbit_velocity(h)  # Орбитальная скорость
        areas = []  # Список для хранения результатов
        
        for gamma in gamma_range:
            # Расчет геометрических параметров
            eta_c, R_c, _ = self.calculate_geometry(gamma)
            
            # Применение формулы 2.11
            area = (4 * c / f0) * R_c * v * np.tan(eta_c)
            areas.append(area)
            
        return np.array(areas)
    
    def ambiguity_analysis(self, gamma: float, PRF: float) -> Dict:
        """Анализ неоднозначностей по дальности и азимуту
        
        Args:
            gamma (float): Угол места [град]
            PRF (float): Частота повторения импульсов [Гц]
            
        Returns:
            Dict: Результаты анализа:
                - AASR: Отношение сигнал-неоднозначность по азимуту [дБ]
                - RASR: Отношение сигнал-неоднозначность по дальности [дБ]
                - PRF_min: Минимальная допустимая PRF [Гц]
        """
        config = self.config
        lambda0 = c / config['f0']  # Длина волны
        h = config['h_sar']
        v = self.orbit_velocity(h)  # Скорость спутника
        
        # Расчет параметров геометрии
        eta_c, R_c, _ = self.calculate_geometry(gamma)
        
        # Расчет наклонной ширины полосы
        swath_slant = config['swath'] * np.sin(eta_c)
        
        # Расчет доплеровской полосы
        L_az = config['L_az']  # Длина антенны
        theta_az = lambda0 / L_az  # Ширина луча по азимуту
        B_dop = 2 * v * theta_az / lambda0  # Доплеровская полоса
        
        # Расчет отношений неоднозначности
        AASR = self._calculate_AASR(PRF, B_dop, L_az)
        RASR = self._calculate_RASR(PRF, R_c, swath_slant, eta_c)
        
        return {
            'AASR': AASR,
            'RASR': RASR,
            'PRF_min': B_dop  # Минимальная PRF по Найквисту
        }
    
    def _calculate_AASR(self, PRF: float, B_dop: float, L_az: float) -> float:
        """Расчет отношения сигнал-неоднозначность по азимуту
        
        Упрощенная модель для демонстрации (формула 2.15)
        """
        return 10 * np.log10((B_dop / PRF)**2)
    
    def _calculate_RASR(self, PRF: float, R_c: float, 
                       swath_slant: float, eta_c: float) -> float:
        """Расчет отношения сигнал-неоднозначность по дальности
        
        Упрощенная модель для демонстрации (формула 2.19)
        """
        tau_swath = 2 * swath_slant / c  # Временная длительность полосы
        return 10 * np.log10((tau_swath * PRF)**2 * np.cos(eta_c))
    
    def plot_antenna_pattern_uniform(self, pattern_type: str = 'uniform'):
        """Визуализация диаграммы направленности антенны
        
        Args:
            pattern_type (str): Тип апертуры:
                - 'uniform': Равномерное распределение
                - 'hanning': Оконная функция Хэмминга
        """
        theta = np.linspace(-np.pi/2, np.pi/2, 1000)  # Углы сканирования
        L = self.config['L_az']  # Длина антенны
        lambda0 = c / self.config['f0']  # Длина волны
        
        # Расчет диаграммы направленности
        if pattern_type == 'uniform':
            # Равномерное распределение (sinc-функция)
            pattern = np.sinc(L * np.sin(theta) / lambda0)**2
        elif pattern_type == 'hanning':
            # Оконная функция Хэмминга (FFT-based)
            window = hann(int(L * 1e2))  # Дискретизация антенны
            pattern = np.abs(np.fft.fft(window))**2
            pattern /= np.max(pattern)  # Нормализация
        else:
            raise ValueError("Неподдерживаемый тип диаграммы")
        
        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(np.degrees(theta), 10 * np.log10(pattern))
        plt.title(f"Диаграмма направленности ({pattern_type})")
        plt.xlabel('Угол [град]')
        plt.ylabel('Мощность [дБ]')
        plt.grid(True)
        plt.show()
    
    def plot_antenna_pattern_hanning(self, pattern_type: str = 'uniform'):
        """Визуализация диаграммы направленности антенны
    
        Args:
            pattern_type (str): Тип апертуры:
                - 'uniform': Равномерное распределение
                - 'hanning': Оконная функция Хэмминга
        """
        theta = np.linspace(-np.pi/2, np.pi/2, 1000)  # Углы сканирования
        L = self.config['L_az']  # Длина антенны
        lambda0 = c / self.config['f0']  # Длина волны
    
        # Расчет диаграммы направленности
        if pattern_type == 'uniform':
            # Равномерное распределение (sinc-функция)
            pattern = np.sinc(L * np.sin(theta) / lambda0)**2
        elif pattern_type == 'hanning':
            # Оконная функция Хэмминга (FFT-based)
            n_points = 1000  # Используем то же количество точек, что и для theta
            window = hann(n_points)  # Создаем окно с нужным количеством точек
            pattern = np.abs(np.fft.fftshift(np.fft.fft(window)))**2
            pattern = pattern / np.max(pattern)  # Нормализация
        
            # Интерполяция для соответствия размеру theta
            if len(pattern) != len(theta):
                # Создаем новые координаты для интерполяции
                x_old = np.linspace(0, 1, len(pattern))
                x_new = np.linspace(0, 1, len(theta))
                pattern = np.interp(x_new, x_old, pattern)
        else:
            raise ValueError("Неподдерживаемый тип диаграммы")
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(theta), 10 * np.log10(pattern))
    plt.title(f"Диаграмма направленности ({pattern_type})")
    plt.xlabel('Угол [град]')
    plt.ylabel('Мощность [дБ]')
    plt.grid(True)
    plt.show()

    def design_flow(self, mode: str = 'ideal') -> Dict:
        """Основной процесс проектирования SAR-системы
        
        Args:
            mode (str): Режим проектирования:
                - 'ideal': Идеальный случай (без ограничений)
                - 'constrained': С учетом ограничений платформы
                
        Returns:
            Dict: Результаты проектирования
        """
        if mode == 'ideal':
            return self._ideal_design_flow()
        elif mode == 'constrained':
            return self._constrained_design_flow()
        else:
            raise ValueError("Некорректный режим проектирования")

    def _ideal_design_flow(self) -> Dict:
        """Идеальное проектирование (без ограничений)
        
        Шаги:
        1. Расчет базовых параметров орбиты
        2. Определение разрешения и ширины полосы
        3. Оптимизация параметров антенны
        4. Выбор оптимальной PRF
        """
        h = self.config['h_sar']
        v = self.orbit_velocity(h)
        
        # Расчет требуемой полосы пропускания
        grg_res = self.config['grg_res']
        eta_c = np.deg2rad(self.config['gamma'])
        B_required = c / (2 * grg_res * np.sin(eta_c))
        
        # Расчет длины антенны из разрешения по азимуту
        L_az = 2 * self.config['az_res']
        
        return {'status': 'success', 'parameters': {
            'Bandwidth': B_required,
            'Antenna_Length': L_az
        }}

    def _constrained_design_flow(self) -> Dict:
        """Проектирование с учетом ограничений платформы
        
        Учитывает:
        - Максимальный размер антенны
        - Ограничения по мощности
        - Ограничения по передаче данных
        """
        # Реализация требует детальных уравнений из исследования
        return {'status': 'success', 'parameters': {}}

if __name__ == "__main__":
    # Пример конфигурации SAR-системы (параметры TerraSAR-X)
    sar_config = {
        'h_sar': 500e3,      # Высота орбиты: 500 км
        'f0': 9.65e9,        # X-диапазон (9.65 ГГц)
        'gamma': 30,         # Угол места: 30°
        'grg_res': 3.0,      # Разрешение по дальности: 3 м
        'az_res': 3.0,       # Разрешение по азимуту: 3 м
        'swath': 30e3,       # Ширина полосы: 30 км
        'tau_p': 30e-6,      # Длительность импульса: 30 мкс
        'L_az': 4.8,         # Длина антенны: 4.8 м
        'P_avg': 300         # Средняя мощность: 300 Вт
    }
    
    # Инициализация SAR-системы
    sar = SARSystem(sar_config)
    
    # Пример 1: Расчет минимальной площади антенны
    gamma_range = np.linspace(20, 45, 25)  # Диапазон углов места
    areas = sar.minimum_antenna_area(gamma_range)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_range, areas)
    plt.title('Минимальная площадь антенны vs Угол места')
    plt.xlabel('Угол места [град]')
    plt.ylabel('Площадь антенны [м²]')
    plt.grid(True)
    plt.show()
    
    # Пример 2: Анализ неоднозначностей
    ambiguity = sar.ambiguity_analysis(35, 3000)
    print(f"Результаты анализа неоднозначностей при 35°: {ambiguity}")
    
    # Пример 3: Визуализация диаграмм направленности
    sar.plot_antenna_pattern_uniform('uniform')  # Равномерное распределение
    sar.plot_antenna_pattern_hanning('hanning')  # Оконная функция Хэмминга
    
    # Пример 4: Запуск процесса проектирования
    design_result = sar.design_flow(mode='ideal')
    print(f"Результаты проектирования: {design_result}")