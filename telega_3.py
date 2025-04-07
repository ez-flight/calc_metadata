# ----------------------------
# Импорт необходимых библиотек
# ----------------------------
import math
import multiprocessing  # <-- Добавленный импорт
import os  # Добавляем импорт для работы с файловой системой
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


def _test():
    """Пример использования функционала"""
    # Создаем директорию result, если она не существует
    os.makedirs("result", exist_ok=True)
    
    # Инициализация параметров спутника
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    pos_gt = (59.95, 30.316667, 0)

    # Настройка параметров наблюдения
    params = ObservationParameters()
    params.pos_gt = pos_gt
    params.dt_start = datetime(2024, 2, 21, 3, 0, 0)
    params.delta = timedelta(seconds=10)
    params.delta_obn = timedelta(seconds=5)
    params.dt_end = params.dt_start + timedelta(days=2)

    # Инициализация трекера
    tracker = SatelliteTracker(tle_1, tle_2)
    
    # Расчет для диапазона углов
    angle_values = range(85, 95, 1)
    results = {}

    for angle in angle_values:
        shp_filename = f"result/{s_name}_angle_{angle}"
        
        # Создаем отдельный shapefile для каждого угла
        with shapefile.Writer(shp_filename, shapefile.POINT) as track_shape:
            try:
                # Определение полей
                fields = [
                    ("ID", "N", 10), ("TIME", "C", 40),
                    ("LON", "F", 10, 10), ("LAT", "F", 10, 10),
                    ("R_s", "F", 10, 5), ("R_t", "F", 10, 5),
                    ("R_n", "F", 10, 5), ("ϒ", "F", 10, 5),
                    ("φ", "F", 10, 5), ("λ", "F", 10, 10),
                    ("f", "F", 10, 5)
                ]
                for field in fields:
                    track_shape.field(*field if len(field) == 3 else field[:4])

                # Генерация трека с передачей текущего track_shape
                t_semki = create_orbital_track(tracker, params, track_shape, angle)
                results[angle] = t_semki

                # Создание PRJ-файла
                with open(f"{shp_filename}.prj", "w") as prj:
                    prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

                print(f"\nShapefile сохранен: {shp_filename}.shp")

                # Экспорт в Excel
                excel_filename = f"result/{s_name}_angle_{angle}.xlsx"
                save_to_excel(t_semki, excel_filename, params, tracker)  # Передаем tracker

                # Вывод статистики
                print(f"\nУгол {angle}°:")
                if not t_semki:
                    print("Нет витков в зоне съемки.")
                else:
                    for orbit in t_semki:
                        fd_values = [point[2] for point in orbit[1]]
                        ay_values = [point[3]['ay_grad'] for point in orbit[1]]
                        print(f"Виток {orbit[0]}: Fd {min(fd_values):.2f}-{max(fd_values):.2f} Гц, Углы {min(ay_values):.1f}°-{max(ay_values):.1f}°")

            except Exception as e:
                print(f"Ошибка при обработке угла {angle}°: {str(e)}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    _test()