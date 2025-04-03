import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import shapefile
from pyorbital.orbital import Orbital
from calc_cord import get_xyzv_from_latlon
from calc_F_L import calc_f_doplera, calc_lamda
from read_TBF import read_tle_base_file, read_tle_base_internet

# Константы
EARTH_RADIUS = 6378.140  # км
EARTH_ROTATION_RATE = 7.2292115E-5  # рад/с
WAVELENGTH = 0.000096  # длина волны зондирующего импульса (м)
EARTH_ANGULAR_VELOCITY = 1674  # Угловая скорость Земли (км/с)
ORBITAL_PERIOD = 5689  # Период обращения спутника (сек)

class ObservationParameters:
    """Класс для хранения параметров наблюдения"""
    def __init__(self):
        self.dt_start = None
        self.dt_end = None
        self.delta = None  # Шаг времени вне зоны видимости
        self.delta_obn = None  # Шаг времени в зоне видимости
        self.pos_gt = None  # Позиция наземной точки (широта, долгота, высота)

class SatelliteTracker:
    """Класс для отслеживания спутника и расчетов"""
    def __init__(self, tle_1: str, tle_2: str):
        self.orb = Orbital("N", line1=tle_1, line2=tle_2)
        self.tle_1 = tle_1
        self.tle_2 = tle_2
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        """Получить геодезические координаты спутника (долгота, широта, высота)"""
        return self.orb.get_lonlatalt(utc_time)
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Получить позицию и скорость в инерциальной системе координат"""
        return self.orb.get_position(utc_time, False)

def calculate_angles(R_s: Tuple[float, float, float], 
                    V_s: Tuple[float, float, float],
                    pos_it: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Расчет углов и расстояний между спутником и наземной точкой
    
    Параметры:
        R_s - позиция спутника в инерциальной СК (км)
        V_s - скорость спутника в инерциальной СК (км/с)
        pos_it - позиция наземной точки в инерциальной СК (км)
    
    Возвращает словарь с:
        R_s - расстояние от центра Земли до спутника
        R_e - расстояние от центра Земли до наземной точки
        R_0 - расстояние между спутником и наземной точкой
        V_s - скорость спутника
        y_grad - угол визирования (град)
        ay_grad - угол места (град)
    """
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    # Вычисление норм векторов
    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
    R_e_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
    R_0_norm = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
    V_s_norm = math.sqrt(V_s[0]**2 + V_s[1]**2 + V_s[2]**2)
    
    # Угол визирования (между направлениями на спутник и наземную точку)
    y = math.acos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    y_grad = math.degrees(y)
    
    # Угол места (возвышение спутника над горизонтом)
    ay = math.acos((R_0_norm * math.sin(y)) / R_e_norm)
    ay_grad = math.degrees(ay)
    
    return {
        'R_s': R_s_norm,
        'R_e': R_e_norm,
        'R_0': R_0_norm,
        'V_s': V_s_norm,
        'y_grad': y_grad,
        'ay_grad': ay_grad
    }

def create_orbital_track(tracker: SatelliteTracker, 
                        params: ObservationParameters,
                        track_shape: Optional[shapefile.Writer] = None,
                        angle_param: float = 0) -> List[Tuple]:
    """
    Создание орбитальной траектории и расчет параметров
    
    Параметры:
        tracker - объект для отслеживания спутника
        params - параметры наблюдения
        track_shape - объект для записи Shapefile (опционально)
        angle_param - параметр угла для расчета доплеровской частоты
    
    Возвращает список кортежей с данными о витках:
        (номер витка, время начала, позиция начала, 
         время конца, позиция конца, длительность)
    """
    lat_t, lon_t, alt_t = params.pos_gt
    t_semki = []
    current_orbit = 0
    orbit_start_time = None
    orbit_start_pos = None
    
    dt = params.dt_start
    while dt < params.dt_end:
        # Получаем данные о положении спутника
        lon_s, lat_s, alt_s = tracker.get_geodetic_position(dt)
        R_s, V_s = tracker.get_inertial_position(dt)
        pos_it, _ = get_xyzv_from_latlon(dt, lon_t, lat_t, alt_t)
        
        # Рассчитываем углы и расстояния
        angles = calculate_angles(R_s, V_s, pos_it)
        
        # Проверка условий видимости
        if 24 <= angles['y_grad'] <= 55 and angles['R_0'] < angles['R_e']:
            # Рассчет угловой скорости для подспутниковой точки
            Wp = EARTH_ANGULAR_VELOCITY * math.cos(math.radians(lat_s))
            
            # Расчет частоты доплера
            Fd = calc_f_doplera(angle_param, WAVELENGTH, math.radians(angles['ay_grad']), 
                              R_s, V_s, angles['R_0'], angles['R_e'], 
                              angles['R_s'], angles['V_s'])
            
            # Определение номера витка
            orbit_num = int((dt - params.dt_start).total_seconds() // ORBITAL_PERIOD) + 1
            print (Fd)
            # Обработка смены витка
            if orbit_num != current_orbit:
                if current_orbit != 0 and orbit_start_time:
                    contact_duration = dt - orbit_start_time
                    t_semki.append((
                        current_orbit, orbit_start_time, orbit_start_pos,
                        dt, (lon_s, lat_s, alt_s), contact_duration
                    ))
                current_orbit = orbit_num
                orbit_start_time = dt
                orbit_start_pos = (lon_s, lat_s, alt_s)
 
            # Запись в shapefile если требуется
            if track_shape:
                track_shape.point(lon_s, lat_s)
                track_shape.record(
                    len(t_semki), dt, lon_s, lat_s, 
                    angles['R_s'], angles['R_e'], angles['R_0'],
                    angles['y_grad'], angles['ay_grad'], a, Fd
                )
            dt += params.delta_obn
        else:
            dt += params.delta
    
    # Добавление последнего витка
    if orbit_start_time:
        contact_duration = dt - orbit_start_time
        t_semki.append((
            current_orbit, orbit_start_time, orbit_start_pos,
            dt, (lon_s, lat_s, alt_s), contact_duration
        ))
    
    return t_semki

def _test():
    """Тестовая функция для демонстрации работы"""
    # Инициализация параметров
    s_name, tle_1, tle_2 = read_tle_base_file(56756)  # NORAD ID спутника
    pos_gt = (59.95, 30.316667, 12)  # Координаты наземной точки (широта, долгота, высота в км)
    
    # Настройки наблюдения
    params = ObservationParameters()
    params.pos_gt = pos_gt
    params.dt_start = datetime(2024, 2, 21, 3, 0, 0)  # Начало наблюдения
    params.delta = timedelta(seconds=10)  # Шаг вне зоны видимости
    params.delta_obn = timedelta(seconds=5)  # Шаг в зоне видимости
    params.dt_end = params.dt_start + timedelta(days=2)  # Конец наблюдения
    
    # Инициализация трекера
    tracker = SatelliteTracker(tle_1, tle_2)
    
    # Создание shapefile
    filename = f"result/{s_name}"
    with shapefile.Writer(filename, shapefile.POINT) as track_shape:
        # Добавление полей
        fields = [
            ("ID", "N", 40),         # Идентификатор точки
            ("TIME", "C", 40),       # Время наблюдения (строка)
            ("LON", "F", 40),       # Долгота
            ("LAT", "F", 40),       # Широта
            ("R_s", "F", 40),       # Расстояние до спутника
            ("R_t", "F", 40),       # Расстояние до наземной точки
            ("R_n", "F", 40),       # Расстояние между спутником и точкой
            ("ϒ", "F", 40, 5),      # Угол визирования (град)
            ("φ", "F", 40, 5),      # Угол места (град)
            ("λ", "F", 40, 5),     # Длина волны (м)
            ("f", "F", 40, 5)      # Доплеровская частота (Гц)
        ]
        for field in fields:
            track_shape.field(*field)
        
        # Расчет для разных угловых параметров
        angle_values = range(85, 95, 1)  # Диапазон угловых параметров от 85 до 95 градусов
        results = {}
        
        for angle in angle_values:
            t_semki = create_orbital_track(tracker, params, track_shape, angle)
            results[angle] = t_semki
            
            # Отладочная печать
            print(f"\nУгол {angle}°: найдено {len(t_semki)} трасс")
     
            # Инициализация переменных для поиска min/max частоты
            min_freq = float('inf')
            max_freq = -float('inf')
            freq_values = []  # Для хранения всех значений частот
            
            print(f"\nРезультаты для углового параметра = {angle} град:")
            for orbit in t_semki:
                print(f"Виток {orbit[0]}: {orbit[1]} - {orbit[3]} (длительность: {orbit[5]})")
                
                # Получаем частоту Доплера из данных орбиты
                # Предполагаем, что частота хранится в orbit[6] или аналогичном поле
                # Если нет - нужно модифицировать create_orbital_track для возврата частоты
                if len(orbit) > 6:  # Проверяем наличие поля с частотой
                    freq = orbit[6]
                    freq_values.append(freq)
                    if freq < min_freq:
                        min_freq = freq
                    if freq > max_freq:
                        max_freq = freq
            
            # Выводим статистику по частотам, если есть данные
            if freq_values:
                print(f"  Статистика частот Доплера:")
                print(f"  - Минимальная частота: {min_freq:.2f} Гц")
                print(f"  - Максимальная частота: {max_freq:.2f} Гц")
                print(f"  - Средняя частота: {sum(freq_values)/len(freq_values):.2f} Гц")
            else:
                print("  Нет данных о частоте Доплера для этого угла")
        
            if not t_semki:
                print("Внимание: нет данных для записи!")

    # Сохранение проекции shapefile (WGS84)
    try:
        with open(f"{filename}.prj", "w") as prj:
            prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    except Exception as e:
        print(f"Ошибка при сохранении проекции shapefile: {e}")

if __name__ == "__main__":
    _test()