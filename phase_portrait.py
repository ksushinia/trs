"""
phase_portrait.py
Расчет 3D карты ошибок фазового портрета через функции выбранной системы.
"""
import math
import numpy as np
from fast_trs import pack_system_params, integrate_master, run_fast_portrait_chunk
from trs_solver import make_sparse_mask

def add_awgn_arrays(x, y, z, noise, seed=None):
    rng = np.random.default_rng(seed)
    traj = np.column_stack([x, y, z])
    noisy = traj + noise * rng.standard_normal(traj.shape)

    return (np.ascontiguousarray(noisy[:, 0]),
            np.ascontiguousarray(noisy[:, 1]),
            np.ascontiguousarray(noisy[:, 2]))

def run_trs_window(y0, h, ym_window, ns, nglobal,
                   K_x_fwd, K_y_fwd, K_z_fwd,
                   K_x_bwd, K_y_bwd, K_z_bwd,
                   step_forward, step_backward, params):
    xs, ys, zs = y0[:3]
    reached_error = 1e-16

    for j in range(1, nglobal + 1):
        diverged = False

        # --- FORWARD ---
        for i in range(ns - 1):
            if j == nglobal and i == ns - 2:
                ex = ym_window[0][i] - xs
                ey = ym_window[1][i] - ys
                ez = ym_window[2][i] - zs
                reached_error = math.sqrt(ex * ex + ey * ey + ez * ez)

            xs, ys, zs = step_forward(
                xs, ys, zs, h,
                xm=ym_window[0][i], ym=ym_window[1][i], zm=ym_window[2][i],
                K_x=K_x_fwd, K_y=K_y_fwd, K_z=K_z_fwd,
                params=params
            )
            if abs(xs) > 1e4 or abs(ys) > 1e4 or abs(zs) > 1e4 or math.isnan(xs):
                diverged = True
                break

        if diverged:
            return 1000.0

        # --- BACKWARD ---
        if j < nglobal:
            for i in range(ns - 1, 0, -1):
                xs, ys, zs = step_backward(
                    xs, ys, zs, h,
                    xm=ym_window[0][i], ym=ym_window[1][i], zm=ym_window[2][i],
                    K_x=K_x_bwd, K_y=K_y_bwd, K_z=K_z_bwd,
                    params=params
                )
                if abs(xs) > 1e4 or abs(ys) > 1e4 or abs(zs) > 1e4 or math.isnan(xs):
                    diverged = True
                    break

        if diverged:
            return 1000.0

    if math.isnan(reached_error) or reached_error > 100.0:
        return 1000.0

    return max(reached_error, 1e-16)


def compute_error_mapped_attractor(step_forward, step_backward, params,
                                   master_0, slave_0, h,
                                   K_x_fwd, K_y_fwd, K_z_fwd,
                                   K_x_bwd, K_y_bwd, K_z_bwd,
                                   t_transient=40.0, t_total=140.0,
                                   window_dt=1.0, N_iterations=50, decimator=1,
                                   use_matlab_shift=True,
                                   progress_callback=None,
                                   sys_name="Dadras Momeni", noise=None, sparse = None):
    TT_iter = int(round(t_transient / h))
    total_steps = int(round(t_total / h))
    ns = max(2, int(round(window_dt / h)))

    keep = sparse.get("keep", 1.0) if sparse else 1.0
    mask = make_sparse_mask(ns + 1, keep=keep, seed=(sparse or {}).get("seed")).astype(np.float64)

    # 1. Сверхбыстрая генерация Master-траектории через Numba
    sys_id, p_arr = pack_system_params(sys_name, params)
    n_total_steps = total_steps + ns
    xm_all, ym_all, zm_all, valid = integrate_master(
        sys_id, p_arr, master_0[0], master_0[1], master_0[2], h, n_total_steps
    )

    # Отсекаем переходный процесс
    xm = xm_all[TT_iter:valid]
    ym = ym_all[TT_iter:valid]
    zm = zm_all[TT_iter:valid]

    if noise and noise.get("enabled"):
        xo, yo, zo = add_awgn_arrays(xm, ym, zm, noise["noise"], noise.get("seed"))
    else:
        xo, yo, zo = xm, ym, zm

    N = len(xm) - ns
    if N <= 0:
        return [], [], [], []

    start_indices = np.arange(0, N, max(1, decimator), dtype=np.int64)
    num_evals = len(start_indices)

    # Точки траектории для построения аттрактора
    xs_starts = xm[start_indices]
    ys_starts = ym[start_indices]
    zs_starts = zm[start_indices]

    # 2. Многопоточный расчет окон порциями для живого прогресс-бара
    err10_list = np.empty(num_evals, dtype=np.float64)
    chunk_size = max(500, num_evals // 20)

    for i in range(0, num_evals, chunk_size):
        chunk_idx = start_indices[i : i + chunk_size]
        err_chunk = run_fast_portrait_chunk(
            sys_id, p_arr, xm, ym, zm, xo, yo, zo, chunk_idx, ns, N_iterations,
            K_x_fwd, K_y_fwd, K_z_fwd, K_x_bwd, K_y_bwd, K_z_bwd,
            h, use_matlab_shift, slave_0[0], slave_0[1], slave_0[2], mask
        )
        err10_list[i : i + chunk_size] = err_chunk
        if progress_callback:
            progress_callback(int((min(i + chunk_size, num_evals) / num_evals) * 100))

    if progress_callback:
        progress_callback(100)

    return xs_starts, ys_starts, zs_starts, err10_list