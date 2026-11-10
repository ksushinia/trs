"""
trs_solver.py
Челночный алгоритм TRS (Time-Reversed Synchronization) с защитой от переполнения.
"""
import math
import numpy as np

def add_awgn_xyz(xm, ym, zm, snr_db, seed=None):
    #добавление шума
    rng = np.random.default_rng(seed)
    traj = np.column_stack([xm, ym, zm])
    p = np.mean(traj ** 2, axis=0)
    sigma = np.sqrt(p / 10 ** (snr_db / 10))
    noisy = traj + sigma * rng.standard_normal(traj.shape)
    return noisy[:, 0].tolist(), noisy[:, 1].tolist(), noisy[:, 2].tolist()

def solve_trs(step_forward, step_backward, params, master_0, slave_0, h,
              t_start, t_end, N_iterations,
              K_x_fwd, K_y_fwd, K_z_fwd,
              K_x_bwd, K_y_bwd, K_z_bwd, noise=None):

    N_steps = max(1, int(round((t_end - t_start) / h)))

    # -------------------------------------------------------------
    # 1. ТРАЕКТОРИЯ MASTER
    # -------------------------------------------------------------
    xm, ym, zm = master_0

    # Докручиваем Master от 0 до t_start
    steps_to_start = int(round(t_start / h))
    for _ in range(steps_to_start):
        xm, ym, zm = step_forward(xm, ym, zm, h, params=params)

    # Записываем точки Master внутри окна [t_start, t_end]
    xm_arr = [xm]
    ym_arr = [ym]
    zm_arr = [zm]

    for _ in range(N_steps):
        xm, ym, zm = step_forward(xm, ym, zm, h, params=params)
        xm_arr.append(xm)
        ym_arr.append(ym)
        zm_arr.append(zm)

    t_arr = [t_start + i * h for i in range(N_steps + 1)]

    if noise and noise.get("enabled"):# проверка и добавление шума на сигнал
        xo_arr, yo_arr, zo_arr = add_awgn_xyz(
            xm_arr, ym_arr, zm_arr, noise["snr_db"], noise.get("seed")
        )
    else:
        xo_arr, yo_arr, zo_arr = xm_arr, ym_arr, zm_arr

    # -------------------------------------------------------------
    # 2. ЧЕЛНОЧНЫЙ TRS (SLAVE)
    # -------------------------------------------------------------
    xs, ys, zs = slave_0
    runs_zigzag = []
    iter_rms = []

    def safe_error(xm_val, ym_val, zm_val, xs_val, ys_val, zs_val):
        """Безопасный расчет RMS-ошибки без вылета при переполнении."""
        try:
            ex = xm_val - xs_val
            ey = ym_val - ys_val
            ez = zm_val - zs_val
            val = (ex * ex + ey * ey + ez * ez) / 3.0
            err = math.sqrt(val)
            if math.isnan(err) or math.isinf(err) or err > 1e10:
                return 1e10
            return max(err, 1e-16)
        except (OverflowError, ValueError):
            return 1e10

    def clamp_state(val):
        """Предотвращает NaN/Inf внутри шага интегрирования при разлете."""
        if math.isnan(val) or math.isinf(val):
            return 1e6
        return max(-1e6, min(1e6, val))

    for it in range(N_iterations):
        run_t = []
        run_err = []

        if it % 2 == 0:
            # FORWARD: от левого края к правому (используем K_*_fwd)
            for k in range(N_steps):
                err = safe_error(xm_arr[k], ym_arr[k], zm_arr[k], xs, ys, zs)
                run_t.append(t_arr[k])
                run_err.append(err)

                xs, ys, zs = step_forward(
                    xs, ys, zs, h,
                    xm=xo_arr[k], ym=yo_arr[k], zm=zo_arr[k],
                    K_x=K_x_fwd, K_y=K_y_fwd, K_z=K_z_fwd,
                    params=params
                )
                xs, ys, zs = clamp_state(xs), clamp_state(ys), clamp_state(zs)

            # Точка на правом краю
            final_err = safe_error(xm_arr[-1], ym_arr[-1], zm_arr[-1], xs, ys, zs)
            run_t.append(t_arr[-1])
            run_err.append(final_err)
            runs_zigzag.append((f"Iter {it + 1}: Fwd", run_t, run_err))

        else:
            # BACKWARD: от правого края к левому (используем K_*_bwd)
            for k in range(N_steps, 0, -1):
                err = safe_error(xm_arr[k], ym_arr[k], zm_arr[k], xs, ys, zs)
                run_t.append(t_arr[k])
                run_err.append(err)

                xs, ys, zs = step_backward(
                    xs, ys, zs, h,
                    xm=xo_arr[k], ym=yo_arr[k], zm=zo_arr[k],
                    K_x=K_x_bwd, K_y=K_y_bwd, K_z=K_z_bwd,
                    params=params
                )
                xs, ys, zs = clamp_state(xs), clamp_state(ys), clamp_state(zs)

            # Точка на левом краю
            final_err = safe_error(xm_arr[0], ym_arr[0], zm_arr[0], xs, ys, zs)
            run_t.append(t_arr[0])
            run_err.append(final_err)
            runs_zigzag.append((f"Iter {it + 1}: Bwd", run_t, run_err))

        iter_rms.append(final_err)

    return t_arr, (xm_arr, ym_arr, zm_arr), runs_zigzag, iter_rms