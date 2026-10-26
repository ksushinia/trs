"""
start_point_generator.py
Модуль генерации начальных условий Slave:
- Окрестность динамической точки Master(t_start)
- Окрестность центра фазового портрета (аттрактора)
- Выбор распределения: Равномерное в 3D шаре / Нормальное (Гауссово)
- Жесткая защита от нулевых координат (|x| >= 10^-8).
"""
import numpy as np
from fast_trs import pack_system_params, integrate_master

# Порог отсечения нуля (чтобы система не застряла в тривиальном равновесии)
ZERO_EPS = 1e-8


def avoid_zero(val: float, eps: float = ZERO_EPS) -> float:
    """Если координата по модулю меньше eps, подтягивает ее к знаку * eps."""
    if abs(val) < eps:
        return eps if val >= 0.0 else -eps
    return float(val)


def avoid_zero_coords(x: float, y: float, z: float, eps: float = ZERO_EPS) -> tuple:
    """Применяет защиту от нуля ко всем трем координатам."""
    return (
        avoid_zero(x, eps),
        avoid_zero(y, eps),
        avoid_zero(z, eps)
    )


def generate_offsets(n_points: int, radius: float, dist_type: str = "uniform", rng=None) -> np.ndarray:
    """
    Генерирует n_points 3D-векторов смещения вокруг нуля:
    - "uniform": равномерное распределение внутри шара радиуса R.
    - "normal": изотропное нормальное (гауссово) распределение с sigma = R / 3
                (по правилу 3-сигм 99.7% точек лягут в пределы сферы радиуса R).
    """
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if radius <= 0.0:
        return np.zeros((n_points, 3), dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    if dist_type == "normal":
        # 3-сигма правило: разброс задается радиусом R
        sigma = radius / 3.0
        offsets = rng.normal(loc=0.0, scale=sigma, size=(n_points, 3))
        return offsets

    # По умолчанию: равномерное распределение по объему шара
    phi = rng.uniform(0.0, 2.0 * np.pi, n_points)
    costheta = rng.uniform(-1.0, 1.0, n_points)
    u = rng.uniform(0.0, 1.0, n_points)

    theta = np.arccos(costheta)
    r = radius * (u ** (1.0 / 3.0))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def get_attractor_center_from_trajectory(xa, ya, za, valid_len: int = None) -> tuple:
    """Вычисляет центр аттрактора по готовой траектории после переходного процесса."""
    if valid_len is None:
        valid_len = len(xa)
    if valid_len <= 0:
        return 0.1, 0.1, 0.1

    # Отсекаем начальный переходной процесс (первые 20% или 500 шагов)
    start_idx = min(valid_len // 5, 500)
    if start_idx >= valid_len:
        start_idx = 0

    xc = float(np.mean(xa[start_idx:valid_len]))
    yc = float(np.mean(ya[start_idx:valid_len]))
    zc = float(np.mean(za[start_idx:valid_len]))
    return avoid_zero_coords(xc, yc, zc)


def compute_attractor_center(sys_name: str, params_dict: dict, master_0: tuple,
                             h: float = 0.01, t_total: float = 150.0) -> tuple:
    """Быстрый Numba-расчет центра аттрактора."""
    sys_id, p_arr = pack_system_params(sys_name, params_dict)
    n_steps = max(100, int(round(t_total / h)))
    xa, ya, za, valid = integrate_master(sys_id, p_arr, master_0[0], master_0[1], master_0[2], h, n_steps)
    return get_attractor_center_from_trajectory(xa, ya, za, valid)


def get_master_point_at_time(sys_name: str, params_dict: dict, master_0: tuple,
                             h: float, t_target: float) -> tuple:
    """Находит точку траектории Master в момент времени t_target."""
    sys_id, p_arr = pack_system_params(sys_name, params_dict)
    n_steps = max(1, int(round(t_target / h)))
    xa, ya, za, valid = integrate_master(sys_id, p_arr, master_0[0], master_0[1], master_0[2], h, n_steps)
    idx = min(n_steps, max(0, valid - 1))
    return avoid_zero_coords(float(xa[idx]), float(ya[idx]), float(za[idx]))


def sample_slave_initial_condition(mode: str, sys_name: str, params_dict: dict,
                                   master_0: tuple, h: float, t_start: float,
                                   radius: float, dist_type: str = "uniform", rng=None) -> tuple:
    """
    Генерирует 1 точку Slave (для кнопки в основном интерфейсе).
    Возвращает ((sx, sy, sz), (base_x, base_y, base_z))
    """
    if mode == "center":
        bx, by, bz = compute_attractor_center(sys_name, params_dict, master_0, h=h)
    else:  # "near_master"
        bx, by, bz = get_master_point_at_time(sys_name, params_dict, master_0, h=h, t_target=t_start)

    if radius > 0.0:
        offset = generate_offsets(1, radius, dist_type=dist_type, rng=rng)[0]
        sx = bx + offset[0]
        sy = by + offset[1]
        sz = bz + offset[2]
    else:
        sx, sy, sz = bx, by, bz

    return avoid_zero_coords(sx, sy, sz), (bx, by, bz)