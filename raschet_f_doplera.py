import math
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import shapefile
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon
from calc_F_L import calc_f_doplera
from read_TBF import read_tle_base_file

# Константы
WAVELENGTH = 0.000096  # Длина волны в метрах

def get_lat_lon_sgp(orb, utc_time):
    return orb.get_lonlatalt(utc_time)

def get_position(orb, utc_time):
    return orb.get_position(utc_time, False)

def vectorized_calculations(R_s, V_s, pos_it):
    R_s = np.asarray(R_s)
    pos_it = np.asarray(pos_it)
    
    R_s_norm = np.linalg.norm(R_s)
    R_0 = np.linalg.norm(R_s - pos_it)
    R_e = np.linalg.norm(pos_it)
    V_s_norm = np.linalg.norm(V_s)
    
    # Проверка деления на ноль и корректности аргументов arccos
    try:
        denominator = 2 * R_0 * R_s_norm
        if denominator == 0:
            return (R_s_norm, R_0, R_e, V_s_norm, np.nan, np.nan)
        
        cos_y = (R_0**2 + R_s_norm**2 - R_e**2) / denominator
        cos_y = np.clip(cos_y, -1.0, 1.0)
        y = math.acos(cos_y)
        y_grad = np.degrees(y)
    except:
        y_grad = np.nan

    try:
        if R_e == 0:
            return (R_s_norm, R_0, R_e, V_s_norm, y_grad, np.nan)
        
        cos_ay = (R_0 * math.sin(y)) / R_e if y_grad is not np.nan else 0
        cos_ay = np.clip(cos_ay, -1.0, 1.0)
        ay = math.acos(cos_ay)
        ay_grad = np.degrees(ay)
    except:
        ay_grad = np.nan

    return (R_s_norm, R_0, R_e, V_s_norm, y_grad, ay_grad)

def create_orbital_data(orb, dt_start, dt_end, delta, pos_gt, a_values):
    timestamps = np.arange(dt_start, dt_end, delta).astype(datetime)
    n = len(timestamps)
    
    a_values_ext = np.resize(a_values, n)
    
    data = {
        'dt': timestamps,
        'lon_s': np.full(n, np.nan),
        'lat_s': np.full(n, np.nan),
        'R_s': np.full(n, np.nan),
        'R_0': np.full(n, np.nan),
        'R_e': np.full(n, np.nan),
        'y_grad': np.full(n, np.nan),
        'ay_grad': np.full(n, np.nan),
        'Fd': np.full(n, np.nan),
        'Wp': np.full(n, np.nan),
        'a_values': a_values_ext
    }

    for i, dt in enumerate(timestamps):
        try:
            # Получение позиции спутника
            R_s, V_s = get_position(orb, dt)
            pos_it, _ = get_xyzv_from_latlon(dt, *pos_gt)
            
            # Вычисление параметров
            (data['R_s'][i], 
             data['R_0'][i], 
             data['R_e'][i], 
             _, 
             data['y_grad'][i], 
             data['ay_grad'][i]) = vectorized_calculations(R_s, V_s, pos_it)
            
            # Получение геодезических координат
            lon, lat, alt = get_lat_lon_sgp(orb, dt)
            data['lon_s'][i] = lon
            data['lat_s'][i] = lat
            
            # Проверка условий для расчета Fd
            if (24 < data['y_grad'][i] < 55 and 
                data['R_0'][i] < data['R_e'][i] and 
                not np.isnan(data['ay_grad'][i])):
                
                # Расчет угловой скорости
                data['Wp'][i] = 1674 * math.cos(math.radians(lat))
                
                # Расчет доплеровской частоты
                if not np.isnan(data['a_values'][i]):
                    data['Fd'][i] = calc_f_doplera(
                        data['a_values'][i], 
                        WAVELENGTH,
                        math.radians(data['ay_grad'][i]),
                        R_s, 
                        V_s, 
                        data['R_0'][i], 
                        data['R_s'][i], 
                        data['R_e'][i], 
                        np.linalg.norm(V_s)
                    )
        except Exception as e:
            print(f"Ошибка обработки точки {i}: {str(e)}")
            continue
    
    return data

def save_to_shapefile(data, filename):
    with shapefile.Writer(filename, shapefile.POINT) as shp:
        fields = [
            ("ID", "N", 40), ("TIME", "C", 40), ("LON", "F", 40),
            ("LAT", "F", 40), ("R_s", "F", 40), ("R_t", "F", 40),
            ("R_n", "F", 40), ("ϒ", "F", 40, 5), ("φ", "F", 40, 5),
            ("λ", "F", 40, 5), ("f", "F", 40, 5)
        ]
        for field in fields:
            shp.field(*field)
        
        for i in range(len(data['dt'])):
            if not np.isnan(data['Fd'][i]):
                shp.point(data['lon_s'][i], data['lat_s'][i])
                shp.record(
                    i, data['dt'][i], data['lon_s'][i], data['lat_s'][i],
                    data['R_s'][i], data['R_e'][i], data['R_0'][i],
                    data['y_grad'][i], data['ay_grad'][i], 
                    data['a_values'][i], data['Fd'][i]
                )
        
    with open(f"{filename}.prj", "w") as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

def _test():
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    
    pos_gt = (59.95, 30.316667, 12)
    dt_start = datetime(2024, 2, 21, 3, 0, 0)
    delta = timedelta(seconds=5)
    dt_end = dt_start + timedelta(days=2)
    
    base_a_values = np.arange(88.0, 92.1, 1)
    
    data = create_orbital_data(orb, dt_start, dt_end, delta, pos_gt, base_a_values)
    
    filename = f"space/{s_name}"
    save_to_shapefile(data, filename)
    
    # Фильтрация NaN значений
    valid_mask = ~np.isnan(data['Fd'])
    
    plt.figure(figsize=(12, 6))
    if np.any(valid_mask):
        plt.scatter(data['Wp'][valid_mask], data['Fd'][valid_mask], 
                    c=data['y_grad'][valid_mask], cmap='viridis', s=10)
        plt.colorbar(label='Угол визирования (°)')
        plt.xlabel('Скорость подспутниковой точки (км/ч)')
        plt.ylabel(f'Доплеровское смещение (Гц), λ={WAVELENGTH} м')
        plt.title('Зависимость доплеровского смещения от параметров орбиты')
        plt.grid(True)
    else:
        print("Нет данных для отображения")
    plt.show()

if __name__ == "__main__":
    _test()