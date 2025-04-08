import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Физические константы
R_earth = 6378.140  # Экваториальный радиус Земли в км
GM = 3.986e5        # Гравитационный параметр Земли в км³/с²
c = 299792458e-3    # Скорость света в км/с

def calculate_gamma(h, alpha_deg):
    """
    Расчет угла визирования γ с учетом сферичности Земли.
    
    Параметры:
        h - высота орбиты спутника (км)
        alpha_deg - угол места в градусах
    
    Возвращает:
        Угол визирования в радианах
    """
    R_s = R_earth + h  # Радиус орбиты спутника
    alpha = np.deg2rad(alpha_deg)
    return np.arcsin((R_earth / R_s) * np.cos(alpha))

def calculate_Fd(alpha_deg, h, lambda_m):
    """
    Расчет доплеровской частоты для РСА Кондор в S-диапазоне.
    
    Параметры:
        alpha_deg - угол места в градусах
        h - высота орбиты (км)
        lambda_m - длина волны (м)
    
    Возвращает:
        Доплеровскую частоту в Гц
    """
    alpha = np.deg2rad(alpha_deg)
    R_s = R_earth + h
    V_s = np.sqrt(GM / R_s)  # Орбитальная скорость спутника
    
    gamma = calculate_gamma(h, alpha_deg)  # Угол визирования
    
    # Проверка корректности длины волны
    lambda_m = np.asarray(lambda_m)
    if np.any(lambda_m <= 0):
        raise ValueError("Длина волны должна быть положительной")
        
    # Расчет доплеровской частоты (перевод скорости в м/с)
    Fd = (2 * V_s * 1e3) / lambda_m * np.cos(alpha) * np.sin(gamma)
    return Fd

# Диапазоны параметров для S-диапазона (λ=10 см)
alpha_range = np.linspace(0, 90, 100)      # Угол места: 0-90°
h_range = np.linspace(500, 1000, 100)      # Высота орбиты: 500-1000 км
lambda_range = np.linspace(0.075, 0.15, 5) # S-диапазон (7.5-15 см)
max_Fd = 800  # Максимальная доплеровская частота 800 кГц

# Создание 3D графиков
fig = plt.figure(figsize=(20, 7))

# График 1: Зависимость Fd от угла места и высоты (λ=0.1 м)
ax1 = fig.add_subplot(131, projection='3d')
alpha_grid, h_grid = np.meshgrid(alpha_range, h_range)
Fd_grid = np.vectorize(calculate_Fd)(alpha_grid, h_grid, 0.10)  # Для λ=10 см
surf = ax1.plot_surface(alpha_grid, h_grid, Fd_grid/1e3, 
                       cmap='viridis', edgecolor='none')
ax1.set_xlabel('Угол места (°)', fontsize=10)
ax1.set_ylabel('Высота (км)', fontsize=10)
ax1.set_zlabel('Fd (кГц)', fontsize=10)
ax1.set_title('Допплеровская частота для λ=10 см (S-диапазон)', pad=15)
fig.colorbar(surf, ax=ax1, shrink=0.5)

# График 2: Зависимость Fd от угла места при h=700 км
ax2 = fig.add_subplot(132)
h_fixed = 700  # Типичная высота для Кондор-ФКА
for lambda_val in lambda_range:
    Fd = calculate_Fd(alpha_range, h_fixed, lambda_val)
    ax2.plot(alpha_range, Fd/1e3, 
            linewidth=2, 
            label=f'λ={lambda_val:.3f} м')
ax2.set_xlabel('Угол места (°)', fontsize=10)
ax2.set_ylabel('Fd (кГц)', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Зависимость Fd от угла места (h={h_fixed} км)', pad=15)

# График 3: Зависимость Fd от высоты при α=30°
ax3 = fig.add_subplot(133)
alpha_fixed = 30  # Средний угол места
for lambda_val in lambda_range:
    Fd = calculate_Fd(alpha_fixed, h_range, lambda_val)
    ax3.plot(h_range, Fd/1e3, 
            linewidth=2, 
            label=f'λ={lambda_val:.3f} м')
ax3.set_xlabel('Высота (км)', fontsize=10)
ax3.set_ylabel('Fd (кГц)', fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title(f'Зависимость Fd от высоты (α={alpha_fixed}°)', pad=15)

plt.tight_layout()
plt.show()

# Анализ рабочих режимов РСА
def print_valid_ranges(param_range, condition, unit):
    """Форматирование допустимых диапазонов параметров"""
    if np.any(condition):
        valid = param_range[condition]
        return f"{valid[0]:.1f}{unit}-{valid[-1]:.1f}{unit}"
    return "Нет допустимых значений"

# Проверка ограничений для типовых сценариев
configurations = [
    {'h': 700, 'lambda': 0.10, 'alpha': None},  # Номинальный режим
    {'alpha': 30, 'lambda': 0.10, 'h': None},   # Типичный угол места
    {'h': 700, 'alpha': 30, 'lambda': None}     # Выбор длины волны
]

print("\nАнализ рабочих режимов РСА Кондор (S-диапазон):")
for config in configurations:
    if config['h'] and config['lambda']:
        condition = calculate_Fd(alpha_range, config['h'], config['lambda'])/1e3 < max_Fd
        print(f"Допустимые углы при h={config['h']}км, λ={config['lambda']}м: "
              f"{print_valid_ranges(alpha_range, condition, '°')}")
    
    if config['alpha'] and config['lambda']:
        condition = calculate_Fd(config['alpha'], h_range, config['lambda'])/1e3 < max_Fd
        print(f"Допустимые высоты при α={config['alpha']}°, λ={config['lambda']}м: "
              f"{print_valid_ranges(h_range, condition, 'км')}")
    
    if config['h'] and config['alpha']:
        condition = calculate_Fd(config['alpha'], config['h'], lambda_range)/1e3 < max_Fd
        print(f"Допустимые длины волн при h={config['h']}км, α={config['alpha']}°: "
              f"{print_valid_ranges(lambda_range, condition, 'м')}")