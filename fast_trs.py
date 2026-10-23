"""
fast_trs.py
Сверхбыстрый скомпилированный движок TRS на базе Numba.
Выполняет расчет на чистой скорости машинного кода (C++) без накладных расходов Python.
"""
from numba import njit
import numpy as np

from systems import (
    _sprott_b_fwd_core, _sprott_b_bwd_core,
    _dadras_momeni_fwd_core, _dadras_momeni_bwd_core,
    _lorenz_fwd_core, _lorenz_bwd_core
)

# Идентификаторы систем для Numba (чтобы не использовать медленные строки)
SYS_SPROTT_B = 0
SYS_DADRAS_MOMENI = 1
SYS_LORENZ = 2

SYSTEM_NAME_TO_ID = {
    "Sprott B": SYS_SPROTT_B,
    "Dadras Momeni": SYS_DADRAS_MOMENI,
    "Lorenz": SYS_LORENZ
}

@njit(nogil=True, fastmath=True)
def integrate_master(sys_id, p_arr, x0, y0, z0, h, n_total):
    """Одна длинная траектория Master. Возвращает массивы и число валидных шагов."""
    xa = np.empty(n_total + 1); ya = np.empty(n_total + 1); za = np.empty(n_total + 1)
    x, y, z = x0, y0, z0
    xa[0] = x; ya[0] = y; za[0] = z
    for i in range(n_total):
        x, y, z = _step_fwd_dispatch(sys_id, x, y, z, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_arr)
        if np.isnan(x) or np.isnan(y) or np.isnan(z) or abs(x) > 1e4:
            return xa, ya, za, i
        xa[i + 1] = x; ya[i + 1] = y; za[i + 1] = z
    return xa, ya, za, n_total


@njit(nogil=True, fastmath=True)
def run_fast_trs_window_score(sys_id, p_arr, xm_arr, ym_arr, zm_arr, i0, n_steps,
                              xs0, ys0, zs0, h, max_iters,
                              kxf, kyf, kzf, kxb, kyb, kzb, target_err):
    """
    TRS по окну [i0, i0+n_steps] готовой траектории Master.
    Возвращает (ok, iters, score), score = log10(ошибки) — непрерывная целевая функция.
    При расходимости score = штраф 4..5, зависящий от номера итерации (чем позже, тем лучше).
    """
    xs, ys, zs = xs0, ys0, zs0
    best = 1e10
    for it in range(max_iters):
        if it % 2 == 0:
            for k in range(n_steps):
                j = i0 + k
                xs, ys, zs = _step_fwd_dispatch(sys_id, xs, ys, zs, h,
                                                xm_arr[j], ym_arr[j], zm_arr[j], kxf, kyf, kzf, p_arr)
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 5.0 - it / max_iters
            ex = xm_arr[i0 + n_steps] - xs
            ey = ym_arr[i0 + n_steps] - ys
            ez = zm_arr[i0 + n_steps] - zs
        else:
            for k in range(n_steps, 0, -1):
                j = i0 + k
                xs, ys, zs = _step_bwd_dispatch(sys_id, xs, ys, zs, h,
                                                xm_arr[j], ym_arr[j], zm_arr[j], kxb, kyb, kzb, p_arr)
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 5.0 - it / max_iters
            ex = xm_arr[i0] - xs
            ey = ym_arr[i0] - ys
            ez = zm_arr[i0] - zs

        err = np.sqrt((ex * ex + ey * ey + ez * ez) / 3.0)
        if np.isnan(err) or err > 1e4:
            return False, it + 1, 5.0 - it / max_iters
        if err < best:
            best = err
        if err <= target_err:
            return True, it + 1, np.log10(max(err, 1e-300))
    return False, max_iters, np.log10(max(best, 1e-300))

@njit(fastmath=True, inline='always')
def _step_fwd_dispatch(sys_id, x, y, z, h, xm, ym, zm, kx, ky, kz, p):
    if sys_id == SYS_SPROTT_B:
        return _sprott_b_fwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3], p[4])
    elif sys_id == SYS_DADRAS_MOMENI:
        return _dadras_momeni_fwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3], p[4], p[5])
    else:  # SYS_LORENZ
        return _lorenz_fwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3])


@njit(fastmath=True, inline='always')
def _step_bwd_dispatch(sys_id, x, y, z, h, xm, ym, zm, kx, ky, kz, p):
    if sys_id == SYS_SPROTT_B:
        return _sprott_b_bwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3], p[4])
    elif sys_id == SYS_DADRAS_MOMENI:
        return _dadras_momeni_bwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3], p[4], p[5])
    else:  # SYS_LORENZ
        return _lorenz_bwd_core(x, y, z, h, xm, ym, zm, kx, ky, kz, p[0], p[1], p[2], p[3])


@njit(nogil=True)
def run_fast_candidate_trs(sys_id, p_arr,
                           mx0, my0, mz0,
                           sx0, sy0, sz0,
                           near_master,
                           h, t_start, dt,
                           max_iters,
                           kxf, kyf, kzf,
                           kxb, kyb, kzb,
                           target_err):
    n_steps = max(1, int(round(dt / h)))
    steps_to_start = int(round(t_start / h))

    xm = mx0
    ym = my0
    zm = mz0
    for _ in range(steps_to_start):
        xm, ym, zm = _step_fwd_dispatch(sys_id, xm, ym, zm, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_arr)
        if np.isnan(xm) or np.isnan(ym) or np.isnan(zm) or abs(xm) > 1e4:
            return False, 1, 1000.0

    if near_master:
        xs = xm + 0.01
        ys = ym + 0.01
        zs = zm + 0.01
    else:
        xs = sx0
        ys = sy0
        zs = sz0

    xm_arr = np.empty(n_steps + 1, dtype=np.float64)
    ym_arr = np.empty(n_steps + 1, dtype=np.float64)
    zm_arr = np.empty(n_steps + 1, dtype=np.float64)

    xm_arr[0] = xm
    ym_arr[0] = ym
    zm_arr[0] = zm

    for i in range(n_steps):
        xm, ym, zm = _step_fwd_dispatch(sys_id, xm, ym, zm, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_arr)
        if np.isnan(xm) or abs(xm) > 1e4:
            return False, 1, 1000.0
        xm_arr[i + 1] = xm
        ym_arr[i + 1] = ym
        zm_arr[i + 1] = zm

    final_err = 1e10

    for it in range(max_iters):
        if it % 2 == 0:
            for k in range(n_steps):
                xs, ys, zs = _step_fwd_dispatch(sys_id, xs, ys, zs, h, xm_arr[k], ym_arr[k], zm_arr[k], kxf, kyf, kzf, p_arr)
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 1000.0
            ex = xm_arr[n_steps] - xs
            ey = ym_arr[n_steps] - ys
            ez = zm_arr[n_steps] - zs
        else:
            for k in range(n_steps, 0, -1):
                xs, ys, zs = _step_bwd_dispatch(sys_id, xs, ys, zs, h, xm_arr[k], ym_arr[k], zm_arr[k], kxb, kyb, kzb, p_arr)
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 1000.0
            ex = xm_arr[0] - xs
            ey = ym_arr[0] - ys
            ez = zm_arr[0] - zs

        err = np.sqrt((ex * ex + ey * ey + ez * ez) / 3.0)
        if np.isnan(err) or err > 1e4:
            return False, it + 1, 1000.0

        final_err = err
        if err <= target_err:
            return True, it + 1, err

    return (final_err <= target_err), max_iters, final_err

def pack_system_params(sys_name, params_dict):
    """Упаковывает словарь параметров в плоский numpy массив для Numba."""
    sys_id = SYSTEM_NAME_TO_ID[sys_name]
    if sys_id == SYS_SPROTT_B:
        p_arr = np.array([
            params_dict["a1"], params_dict["a2"], params_dict["a3"],
            params_dict["a4"], params_dict["a5"], 0.0
        ], dtype=np.float64)
    elif sys_id == SYS_DADRAS_MOMENI:
        p_arr = np.array([
            params_dict["a1"], params_dict["a2"], params_dict["a3"],
            params_dict["a4"], params_dict["a5"], params_dict.get("s", 0.5)
        ], dtype=np.float64)
    else:  # Lorenz
        p_arr = np.array([
            params_dict["a1"], params_dict["a2"], params_dict["a3"],
            params_dict.get("s", 0.5), 0.0, 0.0
        ], dtype=np.float64)
    return sys_id, p_arr

# =====================================================================
# БЫСТРЫЙ TRS ДЛЯ СЛЕЙВА ПО ГОТОВОЙ ТРАЕКТОРИИ МАСТЕРА
# =====================================================================

@njit(nogil=True, fastmath=True)
def run_fast_trs_slave_window(sys_id, p_arr,
                              xm_arr, ym_arr, zm_arr,
                              xs0, ys0, zs0,
                              h, n_steps, max_iters,
                              kxf, kyf, kzf,
                              kxb, kyb, kzb,
                              target_err):
    """
    Сверхбыстрый расчет сходимости Slave по уже рассчитанной траектории Master.
    Не тратит время на повторное интегрирование Master от 0 до t_start.
    """
    xs, ys, zs = xs0, ys0, zs0
    final_err = 1e10

    for it in range(max_iters):
        if it % 2 == 0:
            # FORWARD
            for k in range(n_steps):
                xs, ys, zs = _step_fwd_dispatch(
                    sys_id, xs, ys, zs, h,
                    xm_arr[k], ym_arr[k], zm_arr[k],
                    kxf, kyf, kzf, p_arr
                )
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 1000.0
            ex = xm_arr[n_steps] - xs
            ey = ym_arr[n_steps] - ys
            ez = zm_arr[n_steps] - zs
        else:
            # BACKWARD
            for k in range(n_steps, 0, -1):
                xs, ys, zs = _step_bwd_dispatch(
                    sys_id, xs, ys, zs, h,
                    xm_arr[k], ym_arr[k], zm_arr[k],
                    kxb, kyb, kzb, p_arr
                )
                if np.isnan(xs) or abs(xs) > 1e4:
                    return False, it + 1, 1000.0
            ex = xm_arr[0] - xs
            ey = ym_arr[0] - ys
            ez = zm_arr[0] - zs

        err = np.sqrt((ex * ex + ey * ey + ez * ez) / 3.0)
        if np.isnan(err) or err > 1e4:
            return False, it + 1, 1000.0

        final_err = err
        if err <= target_err:
            return True, it + 1, err

    return (final_err <= target_err), max_iters, final_err
