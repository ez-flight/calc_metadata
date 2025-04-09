import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Геодезические параметры
R_earth = 6378.1  # Экваториальный радиус Земли (км)
e_earth = 0.0818   # Эксцентриситет Земли

# Параметры РСА Кондор (S-диапазон)
f0 = 3.0e9         # Центральная частота (Гц)
lambda_ = 0.1      # Длина волны (м)
h_orbit = 550      # Высота орбиты (км)
inc = 97.4         # Наклонение орбиты (град)

def earth_ellipsoid(lat):
    """Модель эллипсоида Земли"""
    return R_earth / np.sqrt(1 - (e_earth**2 * np.sin(np.deg2rad(lat))**2)

def sar_geometry(lat, look_angle):
    """
    Геометрия съемки для РСА
    Возвращает:
        slant_range - наклонная дальность
        ground_range - наземная дальность
        incidence_angle - угол падения
    """
    Re = earth_ellipsoid(lat)
    h = h_orbit
    theta = np.deg2rad(look_angle)
    
    # Решение геодезических уравнений
    slant_range = h / np.cos(theta)
    ground_range = Re * np.arcsin((slant_range * np.sin(theta)) / Re)
    incidence_angle = np.arcsin((Re/(Re + h)) * np.sin(theta))
    
    return slant_range, ground_range, np.rad2deg(incidence_angle)

def resolution_parameters(incidence_angle, lambda_, V, h):
    """
    Расчет разрешающей способности (исправленная версия)
    h - высота орбиты в метрах
    """
    # Пересчитываем наклонную дальность для каждого угла
    Re = earth_ellipsoid(0) * 1e3  # Переводим в метры
    theta = np.arcsin((Re/(Re + h)) * np.sin(np.deg2rad(incidence_angle)))
    slant_range = h / np.cos(theta)
    
    # Разрешение по азимуту
    T_synth = slant_range * lambda_ / (2 * V * h)
    delta_az = lambda_ / (2 * T_synth * V)
    
    # Разрешение по дальности
    B = 50e6  # Полоса сигнала (Гц)
    delta_rg = 3e8 / (2 * B * np.sin(np.deg2rad(incidence_angle)))
    
    return delta_az, delta_rg

# Общие параметры для расчетов
lat_range = np.linspace(-70, 70, 50)
look_angles = np.linspace(20, 60, 5)
V = 7.5 * 1e3  # Орбитальная скорость (м/с)
h = h_orbit * 1e3  # Высота в метрах

# 1. 3D модель геометрии съемки
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
for angle in look_angles:
    slant, ground, inc_ang = sar_geometry(lat_range, angle)
    ax1.plot(lat_range, ground, slant, lw=2, label=f'{angle}°')
ax1.set_xlabel('Широта (°)')
ax1.set_ylabel('Наземная дальность (км)')
ax1.set_zlabel('Наклонная дальность (км)')
ax1.set_title('3D геометрия съемки')
ax1.legend()
plt.tight_layout()

# 2. Карта покрытия
fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(111)
ground_swath = []
for angle in look_angles:
    _, gr, _ = sar_geometry(0, angle)
    ground_swath.append(2*gr)
ax2.plot(look_angles, ground_swath, 'ro-')
ax2.set_xlabel('Угол визирования (°)')
ax2.set_ylabel('Ширина полосы (км)')
ax2.grid(True)
ax2.set_title('Зависимость ширины полосы от угла визирования')
plt.tight_layout()

# 3. Разрешающая способность
fig3 = plt.figure(figsize=(10, 5))
ax3 = fig3.add_subplot(111)
incidence = np.linspace(20, 70, 100)
delta_az, delta_rg = resolution_parameters(incidence, lambda_, V, h)
ax3.semilogy(incidence, delta_az, label='По азимуту')
ax3.semilogy(incidence, delta_rg, label='По дальности')
ax3.set_xlabel('Угол падения (°)')
ax3.set_ylabel('Разрешение (м)')
ax3.legend()
ax3.grid(True)
ax3.set_title('Разрешающая способность')
plt.tight_layout()

# 4. Доплеровская характеристика
fig4 = plt.figure(figsize=(10, 5))
ax4 = fig4.add_subplot(111)
theta = np.linspace(20, 60, 100)
f_d = (2 * V / lambda_) * np.sin(np.deg2rad(theta))
ax4.plot(theta, f_d/1e3)
ax4.set_xlabel('Угол визирования (°)')
ax4.set_ylabel('Доплеровская частота (кГц)')
ax4.grid(True)
ax4.set_title('Доплеровский профиль')
plt.tight_layout()

# 5. Диаграмма рабочей зоны
fig5 = plt.figure(figsize=(8, 8))
ax5 = fig5.add_subplot(111, polar=True)
theta = np.deg2rad(np.linspace(20, 160, 100))
r = np.linspace(400, 800, 100)
T, R = np.meshgrid(theta, r)
ax5.pcolormesh(T, R, np.log(R*T), cmap='viridis')
ax5.set_title('Рабочая зона в полярных координатах', pad=20)
plt.tight_layout()

# 6. Гистограмма параметров
fig6 = plt.figure(figsize=(10, 5))
ax6 = fig6.add_subplot(111)
params = ['Разрешение', 'Ширина полосы', 'Частота повт.', 'Энергия сигнала']
values = [2.5, 120, 3000, 85]
ax6.bar(params, values, color=['cyan', 'magenta', 'yellow', 'lime'])
ax6.set_title('Сравнение режимов работы')
plt.tight_layout()

# Показать все окна
plt.show()