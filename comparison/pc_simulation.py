"""
comparison/pc_simulation.py
Движок моделирования: совместный шаг Pecora-Carroll и сравнение с TRS.
"""
import numpy as np
from numba import njit, prange
from fast_trs import pack_system_params, integrate_master, run_fast_trs_slave_window
from systems import (
    _sprott_b_fwd_core,
    _lorenz_fwd_core
)
from trs_solver import solve_trs
from systems import SYSTEMS_REGISTRY

# -------------------------------------------------------------
# 1. ШАГ PECORA-CARROLL ДЛЯ DADRAS-MOMENI (код пользователя)
# -------------------------------------------------------------
@njit(fastmath=True, nogil=True)
def step_dadras_pc(xm, ym, zm, xs, ys, zs, h1, h2, a1, a2, a3, a4, a5, Kx, Ky, Kz):
    ex = xm - xs
    ey = ym - ys
    ez = zm - zs

    # Master
    xm_s = xm + h1 * (ym - a1 * xm + a2 * ym * zm)
    ym_s = ym + h1 * (a3 * ym - xm_s * zm + zm)
    zm_s = zm + h1 * (a4 * xm_s * ym_s - a5 * zm)

    zm_next = (zm_s + h2 * a4 * xm_s * ym_s) / (1.0 + h2 * a5)
    ym_next = (ym_s + h2 * (zm_next - xm_s * zm_next)) / (1.0 - h2 * a3)
    xm_next = (xm_s + h2 * (ym_next + a2 * ym_next * zm_next)) / (1.0 + h2 * a1)

    # Slave
    xs_s = xs + h1 * (ys - a1 * xs + a2 * ys * zs + Kx * ex)
    ys_s = ys + h1 * (a3 * ys - xs_s * zs + zs + Ky * ey)
    zs_s = zs + h1 * (a4 * xs_s * ys_s - a5 * zs + Kz * ez)

    zs_next = (zs_s + h2 * (a4 * xs_s * ys_s + Kz * ez)) / (1.0 + h2 * a5)
    ys_next = (ys_s + h2 * (zs_next - xs_s * zs_next + Ky * ey)) / (1.0 - h2 * a3)
    xs_next = (xs_s + h2 * (ys_next + a2 * ys_next * zs_next + Kx * ex)) / (1.0 + h2 * a1)

    return xm_next, ym_next, zm_next, xs_next, ys_next, zs_next


def run_comparison_simulation(sys_name, params, master_0, slave_0,
                              K, h, t_max, t_start, dt, trs_iters):
    """
    Рассчитывает траектории Master, PC Slave и TRS Slave от 0 до t_max.
    Синхронизация включается в момент t_start.
    """
    Kx, Ky, Kz = K
    n_total = max(10, int(round(t_max / h)))
    idx_sync = int(round(t_start / h))
    t_arr = np.linspace(0.0, t_max, n_total + 1)

    # 1. Траектория Master
    sys_id, p_arr = pack_system_params(sys_name, params)
    xm_arr, ym_arr, zm_arr, valid = integrate_master(
        sys_id, p_arr, master_0[0], master_0[1], master_0[2], h, n_total
    )

    # 2. Моделирование Pecora-Carroll
    xs_pc = np.empty(n_total + 1)
    ys_pc = np.empty(n_total + 1)
    zs_pc = np.empty(n_total + 1)
    err_pc = np.empty(n_total + 1)

    xs, ys, zs = slave_0
    xs_pc[0], ys_pc[0], zs_pc[0] = xs, ys, zs
    err_pc[0] = np.sqrt(((xm_arr[0] - xs)**2 + (ym_arr[0] - ys)**2 + (zm_arr[0] - zs)**2) / 3.0)

    s_param = params.get("s", 0.5)
    h1 = s_param * h
    h2 = (1.0 - s_param) * h

    for i in range(n_total):
        # До момента t_start связь выключена (K=0)
        cur_kx = Kx if i >= idx_sync else 0.0
        cur_ky = Ky if i >= idx_sync else 0.0
        cur_kz = Kz if i >= idx_sync else 0.0

        if sys_name == "Dadras Momeni":
            _, _, _, xs, ys, zs = step_dadras_pc(
                xm_arr[i], ym_arr[i], zm_arr[i], xs, ys, zs,
                h1, h2, params["a1"], params["a2"], params["a3"], params["a4"], params["a5"],
                cur_kx, cur_ky, cur_kz
            )
        elif sys_name == "Sprott B":
            xs, ys, zs = _sprott_b_fwd_core(
                xs, ys, zs, h, xm_arr[i], ym_arr[i], zm_arr[i],
                cur_kx, cur_ky, cur_kz,
                params["a1"], params["a2"], params["a3"], params["a4"], params["a5"]
            )
        else:  # Lorenz
            xs, ys, zs = _lorenz_fwd_core(
                xs, ys, zs, h, xm_arr[i], ym_arr[i], zm_arr[i],
                cur_kx, cur_ky, cur_kz,
                params["a1"], params["a2"], params["a3"], s_param
            )

        xs_pc[i + 1] = xs
        ys_pc[i + 1] = ys
        zs_pc[i + 1] = zs
        e = np.sqrt(((xm_arr[i + 1] - xs)**2 + (ym_arr[i + 1] - ys)**2 + (zm_arr[i + 1] - zs)**2) / 3.0)
        err_pc[i + 1] = max(1e-16, e)

    # 3. Моделирование TRS
    xs_trs = np.empty(n_total + 1)
    ys_trs = np.empty(n_total + 1)
    zs_trs = np.empty(n_total + 1)
    xs_trs[:idx_sync + 1] = xs_pc[:idx_sync + 1]
    ys_trs[:idx_sync + 1] = ys_pc[:idx_sync + 1]
    zs_trs[:idx_sync + 1] = zs_pc[:idx_sync + 1]

    # Челночный расчет на окне dt
    sys_funcs = SYSTEMS_REGISTRY[sys_name]
    s0_trs = (xs_trs[idx_sync], ys_trs[idx_sync], zs_trs[idx_sync])

    t_win, win_xyz, runs_zigzag, iter_rms = solve_trs(
        step_forward=sys_funcs["forward"],
        step_backward=sys_funcs["backward"],
        params=params,
        master_0=master_0,
        slave_0=s0_trs,
        h=h,
        t_start=t_start,
        t_end=t_start + dt,
        N_iterations=trs_iters,
        K_x_fwd=Kx, K_y_fwd=Ky, K_z_fwd=Kz,
        K_x_bwd=Kx, K_y_bwd=Ky, K_z_bwd=Kz
    )

    # Заполняем само окно траекторией последней четной итерации челнока (движение вперед)
    last_fwd_run = [run for run in runs_zigzag if "Fwd" in run[0]][-1]
    # Финальная точка Слейва в конце окна TRS (при t = t_start + dt)
    n_win_steps = len(win_xyz[0]) - 1
    idx_win_end = min(n_total, idx_sync + n_win_steps)

    # Если TRS сошелся (ошибка < 1e-6), Слейв дальше идет синхронно с Мастером
    # Если не сошелся — летит дальше автономно от своей реальной точки разлета
    if iter_rms and iter_rms[-1] < 1e-4:
        xs_trs[idx_sync:] = xm_arr[idx_sync:]
        ys_trs[idx_sync:] = ym_arr[idx_sync:]
        zs_trs[idx_sync:] = zm_arr[idx_sync:]
    else:
        # Честный полет несконвергировавшего Слейва вперед от точки выхода из TRS
        cur_xs, cur_ys, cur_zs = xs_trs[idx_sync], ys_trs[idx_sync], zs_trs[idx_sync]
        for j in range(idx_sync, n_total):
            cur_xs, cur_ys, cur_zs = sys_funcs["forward"](cur_xs, cur_ys, cur_zs, h, params=params)
            xs_trs[j + 1] = cur_xs
            ys_trs[j + 1] = cur_ys
            zs_trs[j + 1] = cur_zs

    return {
        "t": t_arr,
        "master": (xm_arr, ym_arr, zm_arr),
        "pc_slave": (xs_pc, ys_pc, zs_pc),
        "pc_error": err_pc,
        "trs_slave": (xs_trs, ys_trs, zs_trs),
        "t_start": t_start,
        "dt": dt,
        "runs_zigzag": runs_zigzag,
        "iter_rms": iter_rms
    }

# -------------------------------------------------------------
# 2. БЫСТРАЯ КАРТА ОШИБОК ДЛЯ PECORA-CARROLL (Numba parallel)
# -------------------------------------------------------------
@njit(nogil=True, fastmath=True, parallel=True)
def run_fast_pc_portrait_chunk(sys_id, p_arr, xm, ym, zm, start_indices, ns,
                               kx, ky, kz, h, use_matlab_shift, sx0, sy0, sz0):
    n_pts = len(start_indices)
    err10_out = np.empty(n_pts, dtype=np.float64)
    s_param = p_arr[5] if len(p_arr) > 5 else 0.5
    h1 = s_param * h
    h2 = (1.0 - s_param) * h

    for idx in prange(n_pts):
        start_idx = start_indices[idx]
        if use_matlab_shift:
            xs = xm[start_idx] + 0.01
            ys = ym[start_idx] + 0.01
            zs = zm[start_idx] + 0.01
        else:
            xs = sx0
            ys = sy0
            zs = sz0

        diverged = False
        for i in range(ns - 1):
            cur_m = start_idx + i
            if sys_id == 1:  # Dadras Momeni
                _, _, _, xs, ys, zs = step_dadras_pc(
                    xm[cur_m], ym[cur_m], zm[cur_m], xs, ys, zs,
                    h1, h2, p_arr[0], p_arr[1], p_arr[2], p_arr[3], p_arr[4],
                    kx, ky, kz
                )
            elif sys_id == 0:  # Sprott B
                xs, ys, zs = _sprott_b_fwd_core(
                    xs, ys, zs, h, xm[cur_m], ym[cur_m], zm[cur_m],
                    kx, ky, kz, p_arr[0], p_arr[1], p_arr[2], p_arr[3], p_arr[4]
                )
            else:  # Lorenz
                xs, ys, zs = _lorenz_fwd_core(
                    xs, ys, zs, h, xm[cur_m], ym[cur_m], zm[cur_m],
                    kx, ky, kz, p_arr[0], p_arr[1], p_arr[2], s_param
                )

            if np.isnan(xs) or abs(xs) > 1e4:
                diverged = True
                break

        end_m = start_idx + ns - 1
        ex = xm[end_m] - xs
        ey = ym[end_m] - ys
        ez = zm[end_m] - zs
        final_err = np.sqrt((ex * ex + ey * ey + ez * ez) / 3.0)

        if diverged or np.isnan(final_err) or final_err > 100.0:
            err10_out[idx] = 3.0
        else:
            err10_out[idx] = np.log10(max(final_err, 1e-16))

    return err10_out