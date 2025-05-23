import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import shapefile
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon
from calc_F_L import calc_f_doplera, calc_lamda
from read_TBF import read_tle_base_file

# Константы
EARTH_RADIUS = 6378.140  # км
EARTH_ROTATION_RATE = 7.2292115E-5  # рад/с
WAVELENGTH = 0.000096  # м

class ObservationParameters:
    """Класс для хранения параметров наблюдения"""
    def __init__(self):
        self.dt_start = None
        self.dt_end = None
        self.delta = None
        self.delta_obn = None
        self.pos_gt = None

class SatelliteTracker:
    """Класс для отслеживания спутника и расчетов"""
    def __init__(self, tle_1: str, tle_2: str):
        self.orb = Orbital("N", line1=tle_1, line2=tle_2)
        self.tle_1 = tle_1
        self.tle_2 = tle_2
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        """Получить геодезические координаты спутника"""
        return self.orb.get_lonlatalt(utc_time)
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Получить позицию и скорость в инерциальной СК"""
        R_s, V_s = self.orb.get_position(utc_time, False)
        return R_s, V_s

def calculate_angles(R_s: Tuple[float, float, float], 
                    V_s: Tuple[float, float, float],
                    pos_it: Tuple[float, float, float]) -> Dict[str, float]:
    """Расчет углов и расстояний"""
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
    R_e_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
    R_0_norm = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
    V_s_norm = math.sqrt(V_s[0]**2 + V_s[1]**2 + V_s[2]**2)
    
    # Угол визирования
    y = math.acos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    y_grad = math.degrees(y)
    
    # Угол места
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
                        Fd: float = 0) -> List[Tuple]:
    """Создание орбитальной траектории и расчет параметров"""
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
        
        if 24 < angles['y_grad'] < 55 and angles['R_0'] < angles['R_e']:
            # Рассчет угловой скорости для подспутниковой точки
            Wp = 1674 * math.cos(math.radians(lat_s))
            
            # Рассчет угла a
            a = calc_lamda(Fd, WAVELENGTH, math.radians(angles['ay_grad']), 
                          R_s, V_s, angles['R_0'], angles['R_e'], 
                          angles['R_s'], angles['V_s'])
            
            # Определяем текущий виток
            orbit_num = int((dt - params.dt_start).total_seconds() // 5689) + 1
            
            # Обработка начала/конца витка
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
    
    # Добавляем последний виток
    if orbit_start_time:
        contact_duration = dt - orbit_start_time
        t_semki.append((
            current_orbit, orbit_start_time, orbit_start_pos,
            dt, (lon_s, lat_s, alt_s), contact_duration
        ))
    
    return t_semki

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
    with shapefile.Writer(filename, shapefile.POINT) as track_shape:
        # Добавление полей
        fields = [
            ("ID", "N", 40), ("TIME", "C", 40), ("LON", "F", 40), 
            ("LAT", "F", 40), ("R_s", "F", 40), ("R_t", "F", 40),
            ("R_n", "F", 40), ("ϒ", "F", 40, 5), ("φ", "F", 40, 5),
            ("λ", "F", 40, 5), ("f", "F", 40, 5)
        ]
        for field in fields:
            track_shape.field(*field)
        
        # Расчет для разных частот доплера
        Fd_values = range(-10000, 10001, 5000)
        results = {}
        
        for Fd in Fd_values:
            t_semki = create_orbital_track(tracker, params, track_shape, Fd)
            results[Fd] = t_semki
            print(f"\nРезультаты для Fd = {Fd} Гц:")
            for orbit in t_semki:
                print(f"Виток {orbit[0]}: {orbit[1]} - {orbit[3]} (длительность: {orbit[5]})")
    
    # Сохранение проекции shapefile
    try:
        with open(f"{filename}.prj", "w") as prj:
            prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    except Exception as e:
        print(f"Ошибка при сохранении shapefile: {e}")

if __name__ == "__main__":
    _test()