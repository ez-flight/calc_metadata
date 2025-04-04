import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import shapefile
from pyorbital.orbital import Orbital
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
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
        self.tle_1 = tle_1
        self.tle_2 = tle_2
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        return self.orb.get_lonlatalt(utc_time)
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return self.orb.get_position(utc_time, False)

def calculate_angles(R_s: Tuple[float, float, float], 
                    V_s: Tuple[float, float, float],
                    pos_it: Tuple[float, float, float]) -> Dict[str, float]:
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
    R_e_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
    R_0_norm = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
    V_s_norm = math.sqrt(V_s[0]**2 + V_s[1]**2 + V_s[2]**2)
    
    y = math.acos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    ay = math.acos((R_0_norm * math.sin(y)) / R_e_norm)
    
    return {
        'R_s': R_s_norm,
        'R_e': R_e_norm,
        'R_0': R_0_norm,
        'V_s': V_s_norm,
        'y_grad': math.degrees(y),
        'ay_grad': math.degrees(ay)
    }

def create_orbital_track(tracker: SatelliteTracker, 
                        params: ObservationParameters,
                        track_shape: Optional[shapefile.Writer] = None,
                        angle_param: float = 0) -> List[Tuple]:
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
                    t_semki.append((
                        current_orbit,
                        orbit_points.copy(),
                        orbit_start_time,
                        dt,
                        dt - orbit_start_time
                    ))
                current_orbit = orbit_num
                orbit_start_time = dt
                orbit_points = []
            
            orbit_points.append((dt, (lon_s, lat_s, alt_s), Fd))
            
            if track_shape:
                track_shape.point(lon_s, lat_s)
                track_shape.record(
                    current_orbit,
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                    lon_s, lat_s,
                    float(angles['R_s']),
                    float(angles['R_e']),
                    float(angles['R_0']),
                    float(angles['y_grad']),
                    float(angles['ay_grad']),
                    float(WAVELENGTH),
                    float(Fd)
                )
            
            dt += params.delta_obn
        else:
            dt += params.delta
    
    if orbit_start_time and orbit_points:
        t_semki.append((
            current_orbit,
            orbit_points,
            orbit_start_time,
            dt,
            dt - orbit_start_time
        ))
    
    return t_semki

def save_to_excel(data: List[Tuple], filename: str, params):
    """Сохраняет данные в Excel файл"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Satellite Data"
    
    # Заголовки
    headers = [
        "Виток", "Время", "Долгота", "Широта", "Высота",
        "Fd (Гц)", "R_s (км)", "R_e (км)", "R_0 (км)",
        "Угол визирования (°)", "Угол места (°)"
    ]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)
    
    row = 2
    for orbit in data:
        orbit_num, points, start_time, end_time, duration = orbit
        for point in points:
            dt, (lon, lat, alt), Fd = point
            angles = calculate_angles(
                tracker.get_inertial_position(dt)[0],
                tracker.get_inertial_position(dt)[1],
                get_xyzv_from_latlon(dt, *params.pos_gt)[0]
            )
            
            ws.cell(row=row, column=1, value=orbit_num)
            ws.cell(row=row, column=2, value=dt.strftime("%Y-%m-%d %H:%M:%S"))
            ws.cell(row=row, column=3, value=lon)
            ws.cell(row=row, column=4, value=lat)
            ws.cell(row=row, column=5, value=alt)
            ws.cell(row=row, column=6, value=Fd)
            ws.cell(row=row, column=7, value=angles['R_s'])
            ws.cell(row=row, column=8, value=angles['R_e'])
            ws.cell(row=row, column=9, value=angles['R_0'])
            ws.cell(row=row, column=10, value=angles['y_grad'])
            ws.cell(row=row, column=11, value=angles['ay_grad'])
            
            row += 1
    
    # Автонастройка ширины столбцов
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        ws.column_dimensions[column].width = adjusted_width
    
    wb.save(filename)
    print(f"Данные сохранены в Excel файл: {filename}")

def _test():
    # Инициализация параметров
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    pos_gt = (59.95, 30.316667, 0)
    
    # Настройки наблюдения
    params = ObservationParameters()
    params.pos_gt = pos_gt
    params.dt_start = datetime(2024, 2, 21, 3, 0, 0)
    params.delta = timedelta(seconds=10)
    params.delta_obn = timedelta(seconds=5)
    params.dt_end = params.dt_start + timedelta(days=2)
    
    # Инициализация трекера
    global tracker
    tracker = SatelliteTracker(tle_1, tle_2)
    
    # Создание shapefile
    shp_filename = f"result/{s_name}"
    track_shape = shapefile.Writer(shp_filename, shapefile.POINT)
    
    try:
        # Добавление полей
        fields = [
            ("ID", "N", 10),
            ("TIME", "C", 40),
            ("LON", "F", 10, 10),
            ("LAT", "F", 10, 10),
            ("R_s", "F", 10, 5),
            ("R_t", "F", 10, 5),
            ("R_n", "F", 10, 5),
            ("ϒ", "F", 10, 5),
            ("φ", "F", 10, 5),
            ("λ", "F", 10, 10),
            ("f", "F", 10, 5)
        ]
        
        for field in fields:
            if len(field) == 3:
                track_shape.field(*field)
            else:
                track_shape.field(field[0], field[1], field[2], field[3])
        
        # Расчет данных
        angle_values = range(85, 95, 1)
        results = {}
        
        for angle in angle_values:
            t_semki = create_orbital_track(tracker, params, track_shape, angle)
            results[angle] = t_semki
            
            # Сохранение в Excel для каждого угла
            excel_filename = f"result/{s_name}_angle_{angle}.xlsx"
            save_to_excel(t_semki, excel_filename, params)
            
            # Вывод статистики
            print(f"\nУгол {angle}°:")
            for orbit in t_semki:
                fd_values = [point[2] for point in orbit[1]]
                if fd_values:
                    print(f"Виток {orbit[0]}: точек {len(orbit[1])}, Fd от {min(fd_values):.2f} до {max(fd_values):.2f} Гц")
        
        # Гарантированное сохранение shapefile
        track_shape.save(shp_filename)
        print(f"\nShapefile сохранен: {shp_filename}.shp")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        if 'track_shape' in locals():
            try:
                track_shape.close()
            except:
                pass
    
    # Создание PRJ-файла
    try:
        with open(f"{shp_filename}.prj", "w") as prj:
            prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    except Exception as e:
        print(f"Ошибка при создании PRJ-файла: {e}")

if __name__ == "__main__":
    _test()