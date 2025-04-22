"""
Модуль для анализа зависимости расстояния до спутника от угла визирования
"""

import math
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from pyorbital.orbital import Orbital

# Локальные модули
from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Константы для расчета
EARTH_RADIUS = 6378.140  # Экваториальный радиус Земли в километрах

def get_lat_lon_sgp(tle_1: str, tle_2: str, utc_time: datetime) -> tuple:
    """
    Получает географические координаты спутника по TLE
    
    Параметры:
        tle_1 (str): Первая строка двухстрочного TLE
        tle_2 (str): Вторая строка двухстрочного TLE
        utc_time (datetime): Время в формате UTC
    
    Возвращает:
        tuple: (Долгота, Широта, Высота) в градусах и километрах
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    return orb.get_lonlatalt(utc_time)

def get_position(tle_1: str, tle_2: str, utc_time: datetime) -> tuple:
    """
    Вычисляет положение и скорость спутника в инерциальной системе координат
    
    Параметры:
        tle_1 (str): Первая строка TLE
        tle_2 (str): Вторая строка TLE
        utc_time (datetime): Временная метка
    
    Возвращает:
        tuple: (X, Y, Z, Vx, Vy, Vz) координаты (км) и скорость (км/с)
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    R_s, V_s = orb.get_position(utc_time, False)
    return (*R_s, *V_s)

def calculate_orbital_data(tle_1: str, tle_2: str, dt_start: datetime, 
                         dt_end: datetime, delta: timedelta, pos_gt: tuple) -> tuple:
    """
    Основная функция расчета орбитальных параметров
    
    Параметры:
        tle_1, tle_2 (str): Двухстрочный TLE
        dt_start (datetime): Начальное время расчета
        dt_end (datetime): Конечное время расчета
        delta (timedelta): Шаг расчета
        pos_gt (tuple): Координаты наземного объекта (широта, долгота, высота)
    
    Возвращает:
        tuple: Два списка - расстояния R0 (км) и углы визирования (градусы)
    """
    # Распаковка координат наземного объекта
    lat_t, lon_t, alt_t = pos_gt
    
    # Генерация временных меток с заданным шагом
    time_steps = np.arange(dt_start, dt_end, delta).astype(datetime)
    
    # Инициализация списков для результатов
    R_0_list = []
    y_grad_list = []
    
    # Основной цикл расчетов
    for dt in time_steps:
        # Получение координат спутника в инерциальной системе
        X_s, Y_s, Z_s, _, _, _ = get_position(tle_1, tle_2, dt)
        
        # Расчет координат наземного объекта в инерциальной системе
        pos_it, _ = get_xyzv_from_latlon(dt, lon_t, lat_t, alt_t)
        X_t, Y_t, Z_t = pos_it
        
        # Расчет расстояний:
        # R0 - расстояние между спутником и объектом
        delta_X = X_s - X_t
        delta_Y = Y_s - Y_t
        delta_Z = Z_s - Z_t
        R_0 = math.sqrt(delta_X**2 + delta_Y**2 + delta_Z**2)
        
        # Rs - расстояние от центра Земли до спутника
        R_s = math.hypot(X_s, Y_s, Z_s)
        
        # Re - расстояние от центра Земли до объекта
        R_e = math.hypot(X_t, Y_t, Z_t)
        
        # Расчет угла визирования по формуле косинусов
        try:
            # Числитель и знаменатель для формулы
            numerator = R_0**2 + R_s**2 - R_e**2
            denominator = 2 * R_0 * R_s
            
            # Защита от деления на ноль и некорректных значений
            if denominator == 0 or abs(numerator/denominator) > 1:
                continue
                
            y_rad = math.acos(numerator / denominator)
        except (ValueError, ZeroDivisionError):
            continue
        
        # Перевод радиан в градусы
        y_grad = math.degrees(y_rad)
        
        # Фильтрация данных по углу визирования и расстоянию
        if 15 < y_grad < 57 and R_0 < R_e:
            R_0_list.append(R_0)
            y_grad_list.append(y_grad)
    
    return R_0_list, y_grad_list

def plot_orbital_data(R_0: list, y_grad: list, save_path: str = None):
    """
    Визуализация данных и сохранение графика
    
    Параметры:
        R_0 (list): Список расстояний
        y_grad (list): Список углов визирования
        save_path (str): Путь для сохранения графика
    """
    # Проверка наличия данных
    if not R_0 or not y_grad:
        print("Ошибка: Нет данных для построения графика")
        return

    # Конвертация в numpy массивы для векторных операций
    y_array = np.array(y_grad)
    R_array = np.array(R_0)
    
    # Параметры рабочей зоны (углы визирования)
    y_min_zone, y_max_zone = 24, 55
    
    # Фильтрация данных в рабочей зоне
    mask = (y_array >= y_min_zone) & (y_array <= y_max_zone)
    y_filtered = y_array[mask]
    R_filtered = R_array[mask]
    
    # Проверка наличия данных в рабочей зоне
    if len(R_filtered) == 0:
        print("Ошибка: Нет данных в рабочей зоне ϒ=24°-55°")
        return
    
    # Расчет граничных значений расстояний
    R_min = np.min(R_filtered)
    R_max = np.max(R_filtered)
    
    # Определение углов при экстремальных расстояниях
    y_min_detected = y_filtered[np.argmin(R_filtered)]
    y_max_detected = y_filtered[np.argmax(R_filtered)]

    # Сортировка данных для построения графика
    sorted_indices = np.argsort(y_array)
    y_sorted = y_array[sorted_indices]
    R_sorted = R_array[sorted_indices]

    # Настройка стиля графика
    plt.style.use('seaborn' if 'seaborn' in plt.style.available else 'default')
    fig, ax = plt.subplots(
        figsize=(12, 7), 
        dpi=100, 
        facecolor='#f0f0f0'  # Цвет фона
    )

    # Построение основной линии графика
    ax.plot(
        y_sorted, 
        R_sorted, 
        linewidth=2.5,
        color='#1f77b4',  # Стандартный синий цвет matplotlib
        label='Траектория измерений'
    )

    # Вертикальные линии границ рабочей зоны
    ax.axvline(
        x=y_min_zone, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_min = {y_min_zone}°'
    )
    ax.axvline(
        x=y_max_zone, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_max = {y_max_zone}°'
    )

    # Горизонтальные линии экстремальных расстояний
    ax.axhline(
        y=R_min, 
        color='g', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_min = {R_min:.1f} км (ϒ={y_min_detected:.1f}°)'
    )
    ax.axhline(
        y=R_max, 
        color='b', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_max = {R_max:.1f} км (ϒ={y_max_detected:.1f}°)'
    )

    # Заливка рабочей зоны
    ax.fill_betweenx(
        y=[R_min, R_max],
        x1=y_min_zone,
        x2=y_max_zone,
        color='lightgreen',
        alpha=0.1,
        label='Рабочая зона'
    )

    # Настройка осей и заголовка
    ax.set_xlabel('Угол визирования ϒ, градусы', fontsize=12)
    ax.set_ylabel('Расстояние R₀, км', fontsize=12)
    # Настройка сетки
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Лимиты осей
    ax.set_xlim(max(20, y_min_zone-5), min(60, y_max_zone+5))
    ax.set_ylim(R_min*0.95, R_max*1.05)
    
    # Легенда
    ax.legend(
        loc='upper right', 
        frameon=True, 
        fontsize=10,
        title='Условные обозначения:'
    )

    # Сохранение графика
    if save_path:
        # Создание директории при необходимости
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Параметры сохранения
        fig.savefig(
            save_path,
            dpi=300,  # Высокое разрешение
            bbox_inches='tight',  # Обрезка пустых областей
            facecolor='#f0f0f0',  # Цвет фона при сохранении
            format='png'  # Формат файла
        )
        print(f"График успешно сохранен: {save_path}")
    
    # Отображение графика
    plt.tight_layout()
    plt.show()


def _test():
    """
    Тестовая функция для проверки работы модуля
    """
    # Загрузка TLE из файла для спутника с номером 56756
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    
    # Координаты наземного объекта (Санкт-Петербург)
    target_pos = (59.95, 30.316667, 12)
    
    # Настройка временного интервала
    start_time = datetime(2024, 2, 21, 3, 0, 0)
    time_delta = timedelta(seconds=10)  # Шаг расчета 10 секунд
    end_time = start_time + timedelta(days=16)  # Период 16 дней

    # Основной расчет
    R_0, y_grad = calculate_orbital_data(
        tle_1, 
        tle_2, 
        start_time, 
        end_time, 
        time_delta, 
        target_pos
    )

    # Генерация имени файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/plot_{timestamp}.png"
    
    # Визуализация и сохранение
    plot_orbital_data(R_0, y_grad, save_path)

if __name__ == "__main__":
    # Точка входа при запуске скрипта
    _test()