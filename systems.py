"""
systems.py
Конечно-разностные схемы систем с JIT-компиляцией Numba (скорость C++).
Старые сигнатуры функций сохранены для 100% совместимости с GUI.
"""
from numba import njit
import numpy as np


# =====================================================================
# 1. СИСТЕМА SPROTT B (Симметричный CD-интегратор)
# =====================================================================

@njit(fastmath=True, nogil=True)
def _sprott_b_fwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, a4, a5):
    ex = K_x * (xm - x)
    ey = K_y * (ym - y)
    ez = K_z * (zm - z)

    h2 = h * 0.5

    # Полушаг вперед
    x_half = x + h2 * (a1 * y * z)
    y_half = y + h2 * (a2 * x_half + a3 * y)
    z_half = z + h2 * (a4 + a5 * x_half * y_half)

    # Полный шаг
    z_next = z_half + h2 * (a4 + a5 * x_half * y_half)
    y_next = (y_half + h2 * (a2 * x_half)) / (1.0 - h2 * a3)
    x_next = x_half + h2 * (a1 * y_next * z_next)

    # Добавка связи
    x_next += ex * h
    y_next += ey * h
    z_next += ez * h

    return x_next, y_next, z_next


@njit(fastmath=True, nogil=True)
def _sprott_b_bwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, a4, a5):
    ex = -K_x * (xm - x)
    ey = -K_y * (ym - y)
    ez = -K_z * (zm - z)

    h_back = -h
    h2 = h_back * 0.5

    # Полушаг назад
    x_half = x + h2 * (a1 * y * z)
    y_half = y + h2 * (a2 * x_half + a3 * y)
    z_half = z + h2 * (a4 + a5 * x_half * y_half)

    # Полный шаг назад
    z_next = z_half + h2 * (a4 + a5 * x_half * y_half)
    y_next = (y_half + h2 * (a2 * x_half)) / (1.0 - h2 * a3)
    x_next = x_half + h2 * (a1 * y_next * z_next)

    # Добавка связи
    x_next += ex * h_back
    y_next += ey * h_back
    z_next += ez * h_back

    return x_next, y_next, z_next


def sprott_b_forward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _sprott_b_fwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"], params["a4"], params["a5"]
    )


def sprott_b_backward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _sprott_b_bwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"], params["a4"], params["a5"]
    )


# =====================================================================
# 2. СИСТЕМА DADRAS-MOMENI (VSCD-интегратор)
# =====================================================================

@njit(fastmath=True, nogil=True)
def _dadras_momeni_fwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, a4, a5, s):
    ex = K_x * (xm - x)
    ey = K_y * (ym - y)
    ez = K_z * (zm - z)

    h1 = s * h
    h2 = (1.0 - s) * h

    # Стадия A (явная часть полушага)
    x_s = x + h1 * (y - a1 * x + a2 * y * z)
    y_s = y + h1 * (a3 * y - x_s * z + z)
    z_s = z + h1 * (a4 * x_s * y_s - a5 * z)

    # Стадия B (неявное замыкание)
    z_next = (z_s + h2 * a4 * x_s * y_s) / (1.0 + h2 * a5)
    y_next = (y_s + h2 * (z_next - x_s * z_next)) / (1.0 - h2 * a3)
    x_next = (x_s + h2 * (y_next + a2 * y_next * z_next)) / (1.0 + h2 * a1)

    # Добавка связи TRS
    x_next += ex * h
    y_next += ey * h
    z_next += ez * h

    return x_next, y_next, z_next


@njit(fastmath=True, nogil=True)
def _dadras_momeni_bwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, a4, a5, s):
    ex = -K_x * (xm - x)
    ey = -K_y * (ym - y)
    ez = -K_z * (zm - z)

    h_back = -h
    h1 = s * h_back
    h2 = (1.0 - s) * h_back

    # Стадия A назад
    x_s = x + h1 * (y - a1 * x + a2 * y * z)
    y_s = y + h1 * (a3 * y - x_s * z + z)
    z_s = z + h1 * (a4 * x_s * y_s - a5 * z)

    # Стадия B назад
    z_next = (z_s + h2 * a4 * x_s * y_s) / (1.0 + h2 * a5)
    y_next = (y_s + h2 * (z_next - x_s * z_next)) / (1.0 - h2 * a3)
    x_next = (x_s + h2 * (y_next + a2 * y_next * z_next)) / (1.0 + h2 * a1)

    # Добавка связи TRS
    x_next += ex * h_back
    y_next += ey * h_back
    z_next += ez * h_back

    return x_next, y_next, z_next


def dadras_momeni_forward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _dadras_momeni_fwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"], params["a4"], params["a5"],
        params.get("s", 0.5)
    )


def dadras_momeni_backward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _dadras_momeni_bwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"], params["a4"], params["a5"],
        params.get("s", 0.5)
    )


# =====================================================================
# 3. СИСТЕМА LORENZ (Симметричный CD-интегратор)
# =====================================================================

@njit(fastmath=True, nogil=True)
def _lorenz_fwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, s):
    ex = K_x * (xm - x)
    ey = K_y * (ym - y)
    ez = K_z * (zm - z)

    h1 = s * h
    h2 = (1.0 - s) * h

    x_s = x + h1 * (a1 * (y - x))
    y_s = y + h1 * (x_s * (a2 - z) - y)
    z_s = z + h1 * (x_s * y_s - a3 * z)

    z_next = (z_s + h2 * x_s * y_s) / (1.0 + h2 * a3)
    y_next = (y_s + h2 * x_s * (a2 - z_next)) / (1.0 + h2)
    x_next = (x_s + h2 * y_next * a1) / (1.0 + h2 * a1)

    x_next += ex * h
    y_next += ey * h
    z_next += ez * h

    return x_next, y_next, z_next


@njit(fastmath=True, nogil=True)
def _lorenz_bwd_core(x, y, z, h, xm, ym, zm, K_x, K_y, K_z, a1, a2, a3, s):
    ex = -K_x * (xm - x)
    ey = -K_y * (ym - y)
    ez = -K_z * (zm - z)

    h_back = -h
    h1 = s * h_back
    h2 = (1.0 - s) * h_back

    x_s = x + h1 * (a1 * (y - x))
    y_s = y + h1 * (x_s * (a2 - z) - y)
    z_s = z + h1 * (x_s * y_s - a3 * z)

    z_next = (z_s + h2 * x_s * y_s) / (1.0 + h2 * a3)
    y_next = (y_s + h2 * x_s * (a2 - z_next)) / (1.0 + h2)
    x_next = (x_s + h2 * y_next * a1) / (1.0 + h2 * a1)

    x_next += ex * h_back
    y_next += ey * h_back
    z_next += ez * h_back

    return x_next, y_next, z_next


def lorenz_forward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _lorenz_fwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"],
        params.get("s", 0.5)
    )


def lorenz_backward(x, y, z, h, xm=0.0, ym=0.0, zm=0.0, K_x=0.0, K_y=0.0, K_z=0.0, params=None):
    return _lorenz_bwd_core(
        x, y, z, h, xm, ym, zm, K_x, K_y, K_z,
        params["a1"], params["a2"], params["a3"],
        params.get("s", 0.5)
    )


# =====================================================================
# РЕЕСТР СИСТЕМ ДЛЯ GUI
# =====================================================================
SYSTEMS_REGISTRY = {
    "Sprott B": {
        "forward": sprott_b_forward,
        "backward": sprott_b_backward,
    },
    "Dadras Momeni": {
        "forward": dadras_momeni_forward,
        "backward": dadras_momeni_backward,
    },
    "Lorenz": {
        "forward": lorenz_forward,
        "backward": lorenz_backward,
    }
}