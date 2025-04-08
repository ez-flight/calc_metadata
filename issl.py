import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Физические константы
R_earth = 6378.140  # км
GM = 3.986e5        # км³/с²
c = 299792458e-3    # км/с

def calculate_gamma(h, alpha_deg):
    """Уточненный расчет угла визирования с учетом угла места"""
    R_s = R_earth + h
    alpha = np.deg2rad(alpha_deg)
    return np.arcsin((R_earth / R_s) * np.cos(alpha))

def calculate_Fd(alpha_deg, h, lambda_m):
    """Расчет доплеровской частоты с обновленной моделью"""
    alpha = np.deg2rad(alpha_deg)
    R_s = R_earth + h
    V_s = np.sqrt(GM / R_s)  # Орбитальная скорость
    
    gamma = calculate_gamma(h, alpha_deg)  # Используем уточненный угол
    
    # Преобразование в массив для универсальной обработки
    lambda_m = np.asarray(lambda_m)
    if np.any(lambda_m <= 0):
        raise ValueError("Длина волны должна быть положительной")
        
    Fd = (2 * V_s * 1e3) / lambda_m * np.cos(alpha) * np.sin(gamma)
    return Fd

# Обновленные диапазоны параметров
alpha_range = np.linspace(0, 90, 100)      # Угол места, градусы
h_range = np.linspace(600, 1000, 100)      # Высота орбиты, км (600-1000 км)
lambda_range = np.linspace(0.15, 0.3, 5)   # L-диапазон (0.15-0.30 м)
max_Fd = 350  # кГц (обновленное значение)

# Визуализация 3D
fig = plt.figure(figsize=(20, 7))

# График 1: Fd(alpha, h) для lambda=0.23 м (L-диапазон)
ax1 = fig.add_subplot(131, projection='3d')
alpha_grid, h_grid = np.meshgrid(alpha_range, h_range)
Fd_grid = np.vectorize(calculate_Fd)(alpha_grid, h_grid, 0.23)
surf = ax1.plot_surface(alpha_grid, h_grid, Fd_grid/1e3, 
                       cmap='viridis', edgecolor='none')
ax1.set_xlabel('Угол места (°)', fontsize=10)
ax1.set_ylabel('Высота (км)', fontsize=10)
ax1.set_zlabel('Fd (кГц)', fontsize=10)
ax1.set_title('Fd(α, h) при λ=0.23 м (L-диапазон)', pad=15)
fig.colorbar(surf, ax=ax1, shrink=0.5)

# График 2: Fd(alpha) для h=700 км
ax2 = fig.add_subplot(132)
h_fixed = 700
for lambda_val in lambda_range:
    Fd = calculate_Fd(alpha_range, h_fixed, lambda_val)
    ax2.plot(alpha_range, Fd/1e3, 
            linewidth=2, 
            label=f'λ={lambda_val:.2f} м')
ax2.set_xlabel('Угол места (°)', fontsize=10)
ax2.set_ylabel('Fd (кГц)', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Fd(α) при h={h_fixed} км', pad=15)

# График 3: Fd(h) для alpha=30°
ax3 = fig.add_subplot(133)
alpha_fixed = 30
for lambda_val in lambda_range:
    Fd = calculate_Fd(alpha_fixed, h_range, lambda_val)
    ax3.plot(h_range, Fd/1e3, 
            linewidth=2, 
            label=f'λ={lambda_val:.2f} м')
ax3.set_xlabel('Высота (км)', fontsize=10)
ax3.set_ylabel('Fd (кГц)', fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title(f'Fd(h) при α={alpha_fixed}°', pad=15)

plt.tight_layout()
plt.show()

# Определение рабочих диапазонов с проверками
def print_valid_ranges(param_range, condition, unit):
    if np.any(condition):
        valid = param_range[condition]
        return f"{valid[0]:.1f}{unit}-{valid[-1]:.1f}{unit}"
    return "Нет допустимых значений"

# Анализ для разных конфигураций
configurations = [
    {'h': 700, 'lambda': 0.23, 'alpha': None},
    {'alpha': 30, 'lambda': 0.23, 'h': None},
    {'h': 700, 'alpha': 30, 'lambda': None}
]

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
