import json
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shapefile
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Константы
EARTH_RADIUS = 6378.140  # км
EARTH_ROTATION_RATE = 7.2292115E-5  # рад/с
WAVELENGTH = 0.000096  # длина волны (м)
ORBITAL_PERIOD = 5689  # Период обращения (сек)

# Исключения
class SatelliteTrackingError(Exception):
    """Базовый класс ошибок трекинга"""
    pass

class VisibilityError(SatelliteTrackingError):
    """Ошибки видимости спутника"""
    pass

# Модели Земли
class EarthModel:
    def __init__(self, model_type='WGS84'):
        self.params = self._load_model(model_type)
    
    def _load_model(self, model_type):
        models = {
            'WGS84': {'radius': 6378.137, 'flattening': 1/298.257223563},
            'SPHERE': {'radius': 6371.0, 'flattening': 0}
        }
        return models.get(model_type, models['WGS84'])

# Системы координат
class CoordinateSystem(ABC):
    @abstractmethod
    def convert(self, coordinates):
        pass

class ECEFSystem(CoordinateSystem):
    def convert(self, latlon):
        lat, lon = np.radians(latlon)
        x = self.earth_radius * np.cos(lat) * np.cos(lon)
        y = self.earth_radius * np.cos(lat) * np.sin(lon)
        z = self.earth_radius * np.sin(lat)
        return x, y, z

@dataclass
class TrackingResult:
    timestamp: datetime
    coordinates: tuple
    elevation: float
    azimuth: float
    distance: float
    
    def to_geojson(self):
        return json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.coordinates[1], self.coordinates[0]]
            },
            "properties": {
                "elevation": self.elevation,
                "azimuth": self.azimuth,
                "distance": self.distance,
                "time": self.timestamp.isoformat()
            }
        })

class ObservationParameters:
    """Параметры наблюдения с улучшенной валидацией"""
    def __init__(self):
        self._dt_start = None
        self._dt_end = None
        self._delta = None
        self._delta_obn = None
        self._pos_gt = None
        self.earth_model = EarthModel()
    
    @property
    def pos_gt(self):
        return self._pos_gt
    
    @pos_gt.setter
    def pos_gt(self, value):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError("Position must be (lat, lon, alt) tuple")
        self._pos_gt = value

class SatelliteTracker:
    """Улучшенный трекер спутников с кэшированием"""
    def __init__(self, tle_1: str, tle_2: str):
        self.orb = Orbital("N", line1=tle_1, line2=tle_2)
        self.tle = (tle_1, tle_2)
        self._position_cache = {}
    
    def get_geodetic_position(self, utc_time: datetime) -> Tuple[float, float, float]:
        """Получение позиции с кэшированием"""
        cache_key = utc_time.timestamp()
        if cache_key not in self._position_cache:
            self._position_cache[cache_key] = self.orb.get_lonlatalt(utc_time)
        return self._position_cache[cache_key]
    
    def get_inertial_position(self, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Векторизованный расчет позиции и скорости"""
        pos, vel = self.orb.get_position(utc_time, False)
        return np.array(pos), np.array(vel)

def calculate_angles_vectorized(
    R_s: np.ndarray, 
    V_s: np.ndarray,
    pos_it: np.ndarray
) -> Dict[str, float]:
    """
    Векторизованный расчет углов и расстояний
    """
    R_s_norm = np.linalg.norm(R_s)
    R_e_norm = np.linalg.norm(pos_it)
    R_0 = R_s - pos_it
    R_0_norm = np.linalg.norm(R_0)
    V_s_norm = np.linalg.norm(V_s)
    
    # Угол визирования
    y = np.arccos((R_0_norm**2 + R_s_norm**2 - R_e_norm**2) / (2 * R_0_norm * R_s_norm))
    y_grad = np.degrees(y)
    
    # Угол места с учетом рефракции
    ay = np.arccos((R_0_norm * np.sin(y)) / R_e_norm)
    ay_grad = np.degrees(ay)
    ay_grad = apply_atmospheric_refraction(ay_grad)
    
    return {
        'R_s': R_s_norm,
        'R_e': R_e_norm,
        'R_0': R_0_norm,
        'V_s': V_s_norm,
        'y_grad': y_grad,
        'ay_grad': ay_grad
    }

def apply_atmospheric_refraction(elevation: float) -> float:
    """Коррекция угла места на атмосферную рефракцию"""
    return elevation + (1.02 / np.tan(np.radians(elevation + 10.3/(elevation+5.11))))

def calculate_doppler(params: dict) -> float:
    """
    Расчет доплеровского смещения с проверкой входных данных
    
    Args:
        params: {'wavelength': float, 'velocity': float, 'angle': float}
    
    Returns:
        Доплеровская частота в Гц
    """
    required = ['wavelength', 'velocity', 'angle']
    if not all(k in params for k in required):
        raise ValueError(f"Missing required params: {required}")
    
    return (2 * params['velocity'] * 1000 * math.cos(params['angle'])) / params['wavelength']

def process_orbit_chunk(args):
    """Обработка участка орбиты для параллельных вычислений"""
    tracker, times, params = args
    results = []
    for dt in times:
        try:
            lon_s, lat_s, alt_s = tracker.get_geodetic_position(dt)
            R_s, V_s = tracker.get_inertial_position(dt)
            pos_it = get_xyzv_from_latlon(dt, *params.pos_gt)
            
            angles = calculate_angles_vectorized(R_s, V_s, pos_it)
            
            if 24 <= angles['y_grad'] <= 55:
                dopler = calculate_doppler({
                    'wavelength': WAVELENGTH,
                    'velocity': angles['V_s'],
                    'angle': math.radians(angles['ay_grad'])
                })
                
                results.append(TrackingResult(
                    timestamp=dt,
                    coordinates=(lat_s, lon_s),
                    elevation=angles['ay_grad'],
                    azimuth=0,  # Расчет азимута можно добавить
                    distance=angles['R_0']
                ))
        except Exception as e:
            print(f"Error processing {dt}: {str(e)}")
    return results

def create_orbital_track_parallel(
    tracker: SatelliteTracker,
    params: ObservationParameters,
    workers: int = 4
) -> List[TrackingResult]:
    """Параллельный расчет орбитального трека"""
    time_chunks = np.array_split(
        pd.date_range(params.dt_start, params.dt_end, freq=params.delta),
        workers
    )
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            process_orbit_chunk,
            [(tracker, chunk, params) for chunk in time_chunks]
        ))
    
    return [item for sublist in results for item in sublist]

def save_results(
    results: List[TrackingResult],
    format: str = 'geojson',
    filename: str = 'output'
):
    """Сохранение результатов в разных форматах"""
    if format == 'geojson':
        features = [json.loads(r.to_geojson()) for r in results]
        with open(f'{filename}.geojson', 'w') as f:
            json.dump({'type': 'FeatureCollection', 'features': features}, f)
    
    elif format == 'shapefile':
        with shapefile.Writer(filename) as shp:
            shp.field('time', 'C')
            shp.field('elev', 'F', decimal=2)
            for r in results:
                shp.point(r.coordinates[1], r.coordinates[0])
                shp.record(r.timestamp.isoformat(), r.elevation)

def visualize_trajectory_3d(results: List[TrackingResult]):
    """3D визуализация с использованием Plotly"""
    import plotly.graph_objects as go
    
    lons = [r.coordinates[1] for r in results]
    lats = [r.coordinates[0] for r in results]
    elevs = [r.elevation for r in results]
    
    fig = go.Figure(data=go.Scatter3d(
        x=lons, y=lats, z=elevs,
        mode='markers',
        marker=dict(
            size=3,
            color=elevs,
            colorscale='Viridis'
        )
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation'
        ),
        title='Satellite Trajectory'
    )
    fig.show()

if __name__ == "__main__":
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    # Пример использования улучшенного функционала
    params = ObservationParameters()
    params.pos_gt = (55.7558, 37.6176, 0.156)  # Москва
    params.dt_start = datetime.now()
    params.dt_end = params.dt_start + timedelta(hours=6)
    params.delta = timedelta(seconds=30)
    
    tracker = SatelliteTracker(tle_1, tle_2)  # Реальные TLE
    
    # Параллельный расчет
    results = create_orbital_track_parallel(tracker, params)
    
    # Сохранение и визуализация
    save_results(results, format='geojson')
    visualize_trajectory_3d(results[:1000])  # Первые 1000 точек для скорости