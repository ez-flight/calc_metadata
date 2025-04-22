import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import shapefile
from mpl_toolkits.mplot3d import Axes3D
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon
from calc_F_L import calc_f_doplera, calc_lamda
from read_TBF import read_tle_base_file

# Константы
EARTH_RADIUS = 6378.140  # Радиус Земли в километрах
EARTH_ROTATION_RATE = 7.2292115E-5  # Угловая скорость вращения Земли (рад/с)
WAVELENGTH = 0.095  # Длина волны в метрах (S-диапазон 9.5 см для КОНДОР ФКА)

class ObservationParameters:
    """Класс для хранения параметров наблюдения"""
    def __init__(self):
        self.dt_start = None  # Начальное время наблюдения
        self.dt_end = None    # Конечное время наблюдения
        self.delta = None     # Шаг времени для поиска
        self.delta_obn = None # Шаг времени во время наблюдения
        self.pos_gt = None    # Географические координаты цели (широта, долгота, высота)

class SatelliteTracker:
    """Класс для работы с орбитальными параметрами спутника"""
    def __init__(self, tle_1: str, tle_2: str):
        self.orb = Orbital("N", line1=tle_1, line2=tle_2)
        self.tle_1 = tle_1  # Первая строка TLE
        self.tle_2 = tle_2  # Вторая строка TLE
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        """Возвращает геодезические координаты (долгота, широта, высота)"""
        return self.orb.get_lonlatalt(utc_time)
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Возвращает позицию и скорость в инерциальной системе координат"""
        R_s, V_s = self.orb.get_position(utc_time, False)
        return R_s, V_s

def calculate_angles(R_s: Tuple[float, float, float], 
                    V_s: Tuple[float, float, float],
                    pos_it: Tuple[float, float, float]) -> Dict[str, float]:
    """Вычисляет геометрические параметры и углы"""
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    # Вычисление норм векторов
    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
    R_e_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
    R_0_norm = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
    V_s_norm = math.sqrt(V_s[0]**2 + V_s[1]**2 + V_s[2]**2)
    
    # Угол визирования (между векторами R_s и R_0)
    y = math.acos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    y_grad = math.degrees(y)
    
    # Угол места (между горизонтом и направлением на спутник)
    ay = math.acos((R_0_norm * math.sin(y)) / R_e_norm)
    ay_grad = math.degrees(ay)
    
    # Угол скоса (дополнительный к углу места)
    grazing_angle = 90 - ay_grad
    
    return {
        'R_s': R_s_norm,
        'R_e': R_e_norm,
        'R_0': R_0_norm,
        'V_s': V_s_norm,
        'y_grad': y_grad,
        'ay_grad': ay_grad,
        'grazing': grazing_angle
    }

def create_orbital_track(tracker: SatelliteTracker, 
                        params: ObservationParameters,
                        track_shape: Optional[shapefile.Writer] = None) -> Tuple[List[Tuple], List[Dict]]:
    """Рассчитывает орбитальную траекторию и доплеровские параметры"""
    lat_t, lon_t, alt_t = params.pos_gt
    t_semki = []         # Список витков
    points_data = []     # Данные для графиков
    current_orbit = 0    # Номер текущего витка
    orbit_start_time = None
    
    dt = params.dt_start
    while dt < params.dt_end:
        # Получение орбитальных параметров
        lon_s, lat_s, alt_s = tracker.get_geodetic_position(dt)
        R_s, V_s = tracker.get_inertial_position(dt)
        pos_it, _ = get_xyzv_from_latlon(dt, lon_t, lat_t, alt_t)
        
        # Расчет углов
        angles = calculate_angles(R_s, V_s, pos_it)
        
        if 24 < angles['y_grad'] < 55 and angles['R_0'] < angles['R_e']:
            # Динамический расчет угла a (88-92°)
            time_diff = (dt - params.dt_start).total_seconds()
            a = 88 + 4 * (time_diff % 15) / 15  # Пример: изменение каждые 15 сек
            
            # Расчет доплеровского смещения
            Fd = calc_f_doplera(
                a=a,
                Lam=WAVELENGTH,
                ay=math.radians(angles['ay_grad']),
                Rs=R_s,
                Vs=V_s,
                R_0=angles['R_0'],
                R_e=angles['R_e'],
                R_s=angles['R_s'],
                V_s=angles['V_s']
            )
            
            # Сохранение данных
            points_data.append({
                'y_grad': angles['y_grad'],
                'grazing': angles['grazing'],
                'Fd': Fd,
                'time': dt
            })
            
            # Обработка витков
            orbit_num = int((dt - params.dt_start).total_seconds() // 5689) + 1
            if orbit_num != current_orbit:
                # Запись завершенного витка
                if current_orbit != 0:
                    contact_duration = dt - orbit_start_time
                    t_semki.append((
                        current_orbit, orbit_start_time, 
                        (lon_s, lat_s, alt_s), dt, contact_duration
                    ))
                current_orbit = orbit_num
                orbit_start_time = dt
            
            # Запись в Shapefile
            if track_shape:
                track_shape.point(lon_s, lat_s)
                track_shape.record(
                    len(t_semki), dt, lon_s, lat_s,
                    angles['R_s'], angles['R_e'], angles['R_0'],
                    angles['y_grad'], angles['ay_grad'], Fd
                )
            
            dt += params.delta_obn
        else:
            dt += params.delta
    
    return t_semki, points_data

# Остальная часть кода без изменений...
def _test():
    # Инициализация параметров
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    pos_gt = (59.95, 30.316667, 12)
    
    # Настройки наблюдения
    params = ObservationParameters()
    params.pos_gt = pos_gt
    params.dt_start = datetime(2024, 2, 21, 3, 0, 0)
    params.delta = timedelta(seconds=10)
    params.delta_obn = timedelta(seconds=5)
    params.dt_end = params.dt_start + timedelta(days=2)
    
    # Инициализация трекера
    tracker = SatelliteTracker(tle_1, tle_2)
    
    # Создание shapefile
    filename = f"result/{s_name}"
    all_points = []
    
    with shapefile.Writer(filename, shapefile.POINT) as track_shape:
        # Добавление полей
        fields = [
            ("ID", "N", 40), ("TIME", "C", 40), ("LON", "F", 40), 
            ("LAT", "F", 40), ("R_s", "F", 40), ("R_t", "F", 40),
            ("R_n", "F", 40), ("ϒ", "F", 40, 5), ("φ", "F", 40, 5),
            ("Fd", "F", 40, 5)
        ]
        for field in fields:
            track_shape.field(*field)
        
        # Расчет траектории
        t_semki, points_data = create_orbital_track(tracker, params, track_shape)
        all_points.extend(points_data)
    
    # Построение графиков
    plt.figure(figsize=(15, 6))
    
    # График зависимости от угла визирования
    plt.subplot(121)
    y_grad = [p['y_grad'] for p in all_points]
    Fd = [p['Fd'] for p in all_points]
    plt.scatter(y_grad, Fd, c='blue', s=10, alpha=0.5)
    plt.xlabel('Угол визирования, градусы')
    plt.ylabel('Доплеровское смещение, Гц')
    plt.title('Зависимость Fd от угла визирования')
    plt.grid(True)
    
    # График зависимости от угла скоса
    plt.subplot(122)
    grazing = [p['grazing'] for p in all_points]
    plt.scatter(grazing, Fd, c='red', s=10, alpha=0.5)
    plt.xlabel('Угол скоса, градусы')
    plt.ylabel('Доплеровское смещение, Гц')
    plt.title('Зависимость Fd от угла скоса')
    plt.grid(True)
    
    # 3D график
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(
        y_grad, 
        grazing, 
        Fd, 
        c=Fd,
        cmap='viridis',
        s=20,
        alpha=0.7
    )
    ax.set_xlabel('Угол визирования, градусы')
    ax.set_ylabel('Угол скоса, градусы')
    ax.set_zlabel('Доплеровское смещение, Гц')
    plt.title('3D зависимость Fd от углов')
    fig.colorbar(surf, label='Fd, Гц')
    
    # Сохранение графиков
    plt.savefig('doppler_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохранение проекции shapefile
    try:
        with open(f"{filename}.prj", "w") as prj:
            prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    except Exception as e:
        print(f"Ошибка при сохранении shapefile: {e}")

if __name__ == "__main__":
    _test()