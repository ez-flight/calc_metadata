"""
Модуль расчета параметров съемки с математическим ядром
"""

import math
import multiprocessing
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import shapefile
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon
from calc_F_L import calc_f_doplera
from read_TBF import read_tle_base_file

# Физические константы
EARTH_RADIUS = 6378.140  # км
WAVELENGTH = 0.000096    # м 

class ObservationParameters:
    def __init__(self):
        self.dt_start: Optional[datetime] = None
        self.dt_end: Optional[datetime] = None
        self.delta: Optional[timedelta] = None
        self.delta_obn: Optional[timedelta] = None
        self.pos_gt: Optional[Tuple[float, float, float]] = None

class SatelliteTracker:
    def __init__(self, tle1: str, tle2: str):
        self.orb = Orbital("N", line1=tle1, line2=tle2)
    
    def get_geodetic_position(self, dt: datetime) -> Tuple[float, float, float]:
        return self.orb.get_lonlatalt(dt)
    
    def get_inertial_position(self, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return self.orb.get_position(dt, False)
    
    def calculate_angles(self, dt: datetime, ground_pos: Tuple[float, float, float]) -> Dict[str, float]:
        R_s, V_s = self.get_inertial_position(dt)
        pos_it, _ = get_xyzv_from_latlon(dt, *ground_pos)
        return calculate_angles(R_s, V_s, pos_it)
    
    def is_in_shooting_zone(self, angles: Dict[str, float], angle_param: float) -> bool:
        return (24 <= angles['y_grad'] <= 55 and 
                angles['R_0'] < angles['R_e'] and 
                abs(angles['ay_grad'] - angle_param) <= 5)

def calculate_angles(R_s: Tuple[float, float, float],
                    V_s: Tuple[float, float, float],
                    pos_it: Tuple[float, float, float]) -> Dict[str, float]:
    """Реализация математической модели расчета углов"""
    X_s, Y_s, Z_s = R_s
    X_t, Y_t, Z_t = pos_it
    
    R_s_norm = math.sqrt(sum(c**2 for c in R_s))
    R_e_norm = math.sqrt(sum(c**2 for c in pos_it))
    R_0_norm = math.sqrt(sum((s-t)**2 for s, t in zip(R_s, pos_it)))
    V_s_norm = math.sqrt(sum(v**2 for v in V_s))
    
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
                        angle_param: float) -> List[Tuple]:
    """Основной алгоритм расчета трека"""
    transitions = []
    current_dt = params.dt_start
    prev_lat = None
    
    # Поиск переходов между витками
    while current_dt < params.dt_end:
        _, lat = tracker.get_geodetic_position(current_dt)[1], None
        if prev_lat and prev_lat < 0 and lat >= 0:
            transitions.append(current_dt)
        prev_lat = lat
        current_dt += timedelta(seconds=60)
    
    # Обработка витков
    orbits = [(i, start, end) for i, (start, end) in enumerate(zip(
        [params.dt_start] + transitions,
        transitions + [params.dt_end]
    ))]
    
    results = []
    with multiprocessing.Pool() as pool:
        futures = []
        for orbit_num, start, end in orbits:
            dt = start
            while dt < end:
                futures.append(pool.apply_async(
                    process_time_step,
                    (tracker, dt, params.pos_gt, angle_param)
                ))
                dt += params.delta_obn
        results = [f.get() for f in futures]
    
    # Формирование выходных данных
    return process_results(results)

def process_time_step(tracker: SatelliteTracker,
                     dt: datetime,
                     ground_pos: Tuple[float, float, float],
                     angle_param: float) -> dict:
    """Обработка одного временного шага"""
    angles = tracker.calculate_angles(dt, ground_pos)
    in_zone = tracker.is_in_shooting_zone(angles, angle_param)
    
    Fd = calc_f_doplera(
        angle_param, WAVELENGTH, math.radians(angles['ay_grad']),
        tracker.get_inertial_position(dt)[0],
        tracker.get_inertial_position(dt)[1],
        angles['R_0'], angles['R_e'],
        angles['R_s'], angles['V_s']
    )
    
    return {
        'dt': dt,
        'pos': tracker.get_geodetic_position(dt),
        'Fd': Fd,
        'angles': angles,
        'in_zone': in_zone
    }

def _test():
    """Тестовый сценарий"""
    os.makedirs("result", exist_ok=True)
    s_name, tle1, tle2 = read_tle_base_file(56756)
    
    params = ObservationParameters()
    params.pos_gt = (59.95, 30.316667, 0)
    params.dt_start = datetime(2024, 2, 21, 3, 0)
    params.dt_end = params.dt_start + timedelta(days=2)
    params.delta = timedelta(seconds=10)
    params.delta_obn = timedelta(seconds=5)
    
    tracker = SatelliteTracker(tle1, tle2)
    
    for angle in range(85, 95):
        shp_filename = f"result/{s_name}_angle_{angle}"
        with shapefile.Writer(shp_filename, shapefile.POINT) as sf:
            sf.field("ORBIT", "N", 10)
            sf.field("TIME", "C", 40)
            sf.field("LON", "F", 10, 10)
            sf.field("LAT", "F", 10, 10)
            sf.field("FD", "F", 10, 5)
            
            tracks = create_orbital_track(tracker, params, angle)
            for orbit in tracks:
                for point in orbit[1]:
                    dt, pos, Fd, angles = point
                    sf.point(pos[0], pos[1])
                    sf.record(
                        orbit[0],
                        dt.isoformat(),
                        pos[0],
                        pos[1],
                        Fd
                    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    _test()