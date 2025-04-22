import math
from datetime import UTC, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pyorbital.orbital import Orbital
from tqdm import tqdm

from calc_cord import get_xyzv_from_latlon


class RadarAnalyzer:
    def __init__(self, satellite_name, pos_gt, dt_start, dt_end):
        self.satellite_name = satellite_name
        self.pos_gt = pos_gt
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.tle_1, self.tle_2 = self._fetch_and_validate_tle()
        self.results = []
        self.debug_log = []

    def _fetch_and_validate_tle(self):
        """Загрузка и валидация TLE"""
        url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544"
        try:
            response = requests.get(url)
            response.raise_for_status()
            tle_lines = response.text.strip().split('\n')
            
            if len(tle_lines) < 3:
                raise ValueError("Некорректный формат TLE")

            tle_1 = tle_lines[1].strip()
            tle_2 = tle_lines[2].strip()

            if not self._validate_checksum(tle_1):
                raise ValueError(f"Неверная контрольная сумма в строке 1: {tle_1}")
            if not self._validate_checksum(tle_2):
                raise ValueError(f"Неверная контрольная сумма в строке 2: {tle_2}")

            return tle_1, tle_2

        except Exception as e:
            print(f"Ошибка загрузки TLE: {str(e)}")
            raise

    def _validate_checksum(self, line):
        """Проверка контрольной суммы"""
        checksum = 0
        for c in line[:68]:
            if c.isdigit():
                checksum += int(c)
            elif c == '-':
                checksum += 1
        return (checksum % 10) == int(line[68])

    def calculate_contacts(self, Fd_values, delta_step=10):
        """Основной цикл расчета"""
        orb = Orbital("CustomSatellite", 
                     line1=self.tle_1, 
                     line2=self.tle_2)
        
        for Fd in tqdm(Fd_values, desc="Обработка частот"):
            dt_current = self.dt_start
            current_contact = None
            
            while dt_current < self.dt_end:
                try:
                    # Расчет положения
                    R_s = orb.get_position(dt_current)[0]
                    X_s, Y_s, Z_s = R_s
                    
                    # Положение цели
                    X_t, Y_t, Z_t = get_xyzv_from_latlon(dt_current, *self.pos_gt)[0]
                    
                    # Геометрические параметры
                    R_0 = math.sqrt((X_s-X_t)**2 + (Y_s-Y_t)**2 + (Z_s-Z_t)**2)
                    R_s_norm = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
                    R_t_norm = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
                    
                    gamma = math.degrees(math.acos(
                        (R_0**2 + R_s_norm**2 - R_t_norm**2) / 
                        (2 * R_0 * R_s_norm))
                    ) if R_0 != 0 else 0

                    # Логирование параметров
                    self.debug_log.append({
                        'time': dt_current,
                        'gamma': gamma,
                        'R_0': R_0,
                        'visible': 24 < gamma < 55 and R_0 < 6378.140
                    })

                    # Проверка условий видимости
                    if 24 < gamma < 55 and R_0 < 6378.140:
                        if not current_contact:
                            current_contact = self._init_contact(dt_current, Fd)
                        current_contact = self._update_contact(current_contact, dt_current, gamma)
                    else:
                        if current_contact:
                            self._save_contact(current_contact)
                            current_contact = None
                            
                    dt_current += timedelta(seconds=delta_step)
                    
                except Exception as e:
                    print(f"Ошибка в {dt_current}: {str(e)}")
                    dt_current += timedelta(seconds=delta_step)

            if current_contact:
                self._save_contact(current_contact)

    def _init_contact(self, dt, Fd):
        return {
            'start': dt,
            'end': dt,
            'Fd': Fd,
            'gamma_sum': 0.0,
            'count': 0
        }

    def _update_contact(self, contact, dt, gamma):
        contact['end'] = dt
        contact['gamma_sum'] += gamma
        contact['count'] += 1
        return contact

    def _save_contact(self, contact):
        duration = contact['end'] - contact['start']
        self.results.append({
            'Fd': contact['Fd'],
            'start': contact['start'].isoformat(),
            'end': contact['end'].isoformat(),
            'duration_sec': duration.total_seconds(),
            'mean_gamma': contact['gamma_sum'] / contact['count']
        })

    def analyze_results(self):
        """Анализ и диагностика"""
        if not self.results:
            print("\nНет данных для анализа. Причины:")
            self._print_debug_stats()
            return

        df = pd.DataFrame(self.results)
        self._plot_results(df)
        self._print_statistics(df)

    def _print_debug_stats(self):
        """Диагностика отсутствия контактов"""
        debug_df = pd.DataFrame(self.debug_log)
        
        if debug_df.empty:
            print(" - Нет данных для диагностики")
            return
            
        print(f" - Всего записей: {len(debug_df)}")
        print(f" - Временной диапазон: {debug_df['time'].min()} - {debug_df['time'].max()}")
        print(f" - Диапазон углов: {debug_df['gamma'].min():.1f}° - {debug_df['gamma'].max():.1f}°")
        print(f" - Процент времени в зоне видимости: {debug_df['visible'].mean()*100:.1f}%")

    def _plot_results(self, df):
        """Визуализация результатов"""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['Fd'], df['duration_sec'], c=df['mean_gamma'], cmap='viridis')
        plt.colorbar(label='Средний угол визирования (°)')
        plt.xlabel('Доплеровская частота (Гц)')
        plt.ylabel('Длительность контакта (сек)')
        plt.title('Зависимость параметров съемки')
        
        plt.subplot(1, 2, 2)
        plt.hist(df['duration_sec'], bins=20, edgecolor='black')
        plt.xlabel('Длительность контакта (сек)')
        plt.ylabel('Количество')
        plt.title('Распределение длительности')
        
        plt.tight_layout()
        plt.show()

    def _print_statistics(self, df):
        """Вывод статистики"""
        print("\nСтатистика:")
        print(f" - Всего контактов: {len(df)}")
        print(f" - Максимальная длительность: {df['duration_sec'].max():.1f} сек")
        print(f" - Средняя длительность: {df['duration_sec'].mean():.1f} сек")
        print(f" - Диапазон углов: {df['mean_gamma'].min():.1f}° - {df['mean_gamma'].max():.1f}°")

if __name__ == "__main__":
    # Конфигурация с примером для МКС
    config = {
        'satellite': 'ISS',
        'target_pos': (55.75583, 37.6173, 140),  # Москва
        'start_time': datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
        'end_time': datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=3),
        'Fd_range': np.linspace(-10000, 10000, 5)
    }

    try:
        analyzer = RadarAnalyzer(
            config['satellite'],
            config['target_pos'],
            config['start_time'],
            config['end_time']
        )
        analyzer.calculate_contacts(config['Fd_range'])
        analyzer.analyze_results()
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")