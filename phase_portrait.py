"""
phase_portrait.py
Расчет 3D карты ошибок фазового портрета через функции выбранной системы.
"""
import math
import numpy as np

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
                                   progress_callback=None):
    """
    Строит аттрактор системы и картографирует финальную ошибку TRS в каждой точке.
    """
    TT_iter = int(round(t_transient / h))
    total_steps = int(round(t_total / h))
    ns = max(2, int(round(window_dt / h)))

    xm, ym, zm = master_0

    # 1. Генерация Master-траектории с учетом переходного процесса
    all_x, all_y, all_z = [], [], []
    for _ in range(total_steps + ns):
        xm, ym, zm = step_forward(xm, ym, zm, h, params=params)
        all_x.append(xm)
        all_y.append(ym)
        all_z.append(zm)

    # Отсекаем переходный процесс
    traj_x = all_x[TT_iter:]
    traj_y = all_y[TT_iter:]
    traj_z = all_z[TT_iter:]

    N = len(traj_x) - ns
    if N <= 0:
        return [], [], [], []

    start_indices = list(range(0, N, max(1, decimator)))
    num_evals = len(start_indices)

    xs_starts, ys_starts, zs_starts = [], [], []
    err10_list = []

    # Начальные условия y0 для TRS
    if use_matlab_shift:
        y0_target = (traj_x[0] + 0.01, traj_y[0] + 0.01, traj_z[0] + 0.01)
    else:
        y0_target = slave_0

    step_notify = max(1, num_evals // 100)

    for idx_num, start_idx in enumerate(start_indices):
        if progress_callback and (idx_num % step_notify == 0 or idx_num == num_evals - 1):
            progress_callback(int((idx_num / num_evals) * 100))

        sub_ym = [
            traj_x[start_idx: start_idx + ns],
            traj_y[start_idx: start_idx + ns],
            traj_z[start_idx: start_idx + ns]
        ]

        xs_starts.append(sub_ym[0][0])
        ys_starts.append(sub_ym[1][0])
        zs_starts.append(sub_ym[2][0])

        # Слейв стартует строго в окрестности (+0.01) НАЧАЛА ТЕКУЩЕГО ОКНА!
        if use_matlab_shift:
            y0_current = (sub_ym[0][0] + 0.01, sub_ym[1][0] + 0.01, sub_ym[2][0] + 0.01)
        else:
            y0_current = slave_0

        err = run_trs_window(
            y0=y0_current,
            h=h,
            ym_window=sub_ym,
            ns=ns,
            nglobal=N_iterations,
            K_x_fwd=K_x_fwd,
            K_y_fwd=K_y_fwd,
            K_z_fwd=K_z_fwd,
            K_x_bwd=K_x_bwd,
            K_y_bwd=K_y_bwd,
            K_z_bwd=K_z_bwd,
            step_forward=step_forward,
            step_backward=step_backward,
            params=params
        )

        if math.isnan(err) or err > 100.0:
            err = 1000.0

        err10_list.append(math.log10(max(err, 1e-16)))

    if progress_callback:
        progress_callback(100)

    return xs_starts, ys_starts, zs_starts, err10_list