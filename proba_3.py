import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import shapefile
from pyorbital.orbital import Orbital
from tabulate import tabulate  # Для красивого табличного вывода
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
        self.delta = None
        self.delta_obn = None
        self.pos_gt = None

class SatelliteTracker:
    """Класс для отслеживания спутника и расчетов"""
    def __init__(self, tle_1: str, tle_2: str):
        self.orb = Orbital("N", line1=tle_1, line2=tle_2)
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        """Получить геодезические координаты спутника"""
        return self.orb.get_lonlatalt(utc_time)
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Получить позицию и скорость в инерциальной СК"""
        return self.orb.get_position(utc_time, False)

def calculate_angles(R_s, V_s, pos_it):
    """Расчет углов и расстояний между спутником и наземной точкой"""
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
    R_e_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
    R_0_norm = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
    V_s_norm = math.sqrt(V_s[0]**2 + V_s[1]**2 + V_s[2]**2)
    
    y = math.acos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    y_grad = math.degrees(y)
    
    ay = math.acos((R_0_norm * math.sin(y)) / R_e_norm)
    ay_grad = math.degrees(ay)
    
    return {
        'R_s': R_s_norm, 'R_e': R_e_norm, 'R_0': R_0_norm,
        'V_s': V_s_norm, 'y_grad': y_grad, 'ay_grad': ay_grad
    }

def create_orbital_track(tracker, params, track_shape=None, angle_param=0):
    """Создание орбитальной траектории и расчет параметров"""
    lat_t, lon_t, alt_t = params.pos_gt
    t_semki = []
    current_orbit = 0
    orbit_start_time = None
    orbit_points = []
    
    dt = params.dt_start
    while dt < params.dt_end:
        lon_s, lat_s, alt_s = tracker.get_geodetic_position(dt)
        R_s, V_s = tracker.get_inertial_position(dt)
        pos_it, _ = get_xyzv_from_latlon(dt, lon_t, lat_t, alt_t)
        
        angles = calculate_angles(R_s, V_s, pos_it)
        
        if 24 <= angles['y_grad'] <= 55 and angles['R_0'] < angles['R_e']:
            Fd = calc_f_doplera(angle_param, WAVELENGTH, math.radians(angles['ay_grad']), 
                              R_s, V_s, angles['R_0'], angles['R_e'], 
                              angles['R_s'], angles['V_s'])
            
            orbit_num = int((dt - params.dt_start).total_seconds() // ORBITAL_PERIOD) + 1
            
            if orbit_num != current_orbit:
                if current_orbit != 0 and orbit_start_time and orbit_points:
                    contact_duration = dt - orbit_start_time
                    t_semki.append((
                        current_orbit, orbit_points.copy(),
                        orbit_start_time, dt, contact_duration
                    ))
                current_orbit = orbit_num
                orbit_start_time = dt
                orbit_points = []
            
            orbit_points.append((dt, (lon_s, lat_s, alt_s), Fd))
            
            if track_shape:
                track_shape.point(lon_s, lat_s)
                track_shape.record(
                    current_orbit, dt.strftime("%Y-%m-%d %H:%M:%S"),
                    lon_s, lat_s, float(angles['R_s']), float(angles['R_e']),
                    float(angles['R_0']), float(angles['y_grad']),
                    float(angles['ay_grad']), float(WAVELENGTH), float(Fd)
                )
            
            dt += params.delta_obn
        else:
            dt += params.delta
    
    if orbit_start_time and orbit_points:
        contact_duration = dt - orbit_start_time
        t_semki.append((
            current_orbit, orbit_points,
            orbit_start_time, dt, contact_duration
        ))
    
    return t_semki

def print_table(data, headers):
    """Вывод данных в табличном виде"""
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))

def _test():
    """Тестовая функция"""
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    pos_gt = (59.95, 30.316667, 0)
    
    params = ObservationParameters()
    params.pos_gt = pos_gt
    params.dt_start = datetime.now()
    params.delta = timedelta(seconds=10)
    params.delta_obn = timedelta(seconds=5)
    params.dt_end = params.dt_start + timedelta(hours=6)
    
    tracker = SatelliteTracker(tle_1, tle_2)
    filename = f"result/{s_name}"
    
    # Гарантированное создание shapefile
    try:
        with shapefile.Writer(filename, shapefile.POINT) as shp:
            # Поля shapefile
            fields = [
                ("ID", "N", 10), ("TIME", "C", 40),
                ("LON", "F", 10, 10), ("LAT", "F", 10, 10),
                ("R_s", "F", 10, 5), ("R_t", "F", 10, 5),
                ("R_n", "F", 10, 5), ("ϒ", "F", 10, 5),
                ("φ", "F", 10, 5), ("λ", "F", 10, 10),
                ("f", "F", 10, 5)
            ]
            for field in fields:
                shp.field(*field if len(field) == 3 else field[:4])
            
            angle_values = range(85, 95, 1)
            for angle in angle_values:
                t_semki = create_orbital_track(tracker, params, shp, angle)
                
                # Вывод таблицы с результатами
                table_data = []
                for orbit in t_semki:
                    orbit_num, points, start, end, duration = orbit
                    for point in points[:3]:  # Первые 3 точки каждого витка
                        time, (lon, lat, alt), fd = point
                        table_data.append([
                            orbit_num, time.strftime("%H:%M:%S"),
                            f"{lat:.4f}", f"{lon:.4f}", f"{fd:.2f}"
                        ])
                
                print(f"\nРезультаты для угла {angle}°:")
                print_table(table_data, 
                          ["Виток", "Время", "Широта", "Долгота", "Fd (Гц)"])
                
                # Статистика по виткам
                stats = []
                for orbit in t_semki:
                    orbit_num, points, start, end, duration = orbit
                    fds = [p[2] for p in points]
                    stats.append([
                        orbit_num, len(points),
                        f"{min(fds):.2f}", f"{max(fds):.2f}",
                        f"{sum(fds)/len(fds):.2f}"
                    ])
                
                print("\nСтатистика по виткам:")
                print_table(stats, 
                          ["Виток", "Точек", "Fd мин", "Fd макс", "Fd средн"])
        
        # Создание PRJ-файла
        with open(f"{filename}.prj", "w") as prj:
            prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
        print(f"\nShapefile успешно создан: {filename}.shp")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    _test()