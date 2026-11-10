"""
param_search.py
Модуль перебора параметров TRS с поддержкой чекбоксов, экспорта в пресеты,
мгновенного построения 3D портрета на паузе и числовой фильтрации.
"""
import time
import math
import itertools
import os
import csv
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSpinBox, QCheckBox, QPushButton, QProgressBar,
    QTableView, QHeaderView, QGroupBox, QScrollArea, QMessageBox,
    QSplitter, QFileDialog,  QInputDialog, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel

from systems import SYSTEMS_REGISTRY
from config import SYSTEMS_CONFIG
from fast_trs import (
    run_fast_candidate_trs, pack_system_params,
    integrate_master, run_fast_trs_window_score
)
from spinboxes import QDoubleSpinBox, QSpinBox
from start_point_generator import (
    generate_offsets, avoid_zero_coords, get_attractor_center_from_trajectory
)

def hooke_jeeves(f, x0, steps, lo, hi, shrink=0.5, min_step=1e-3, max_evals=400):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    clip = lambda x: np.minimum(np.maximum(x, lo), hi)
    steps = np.asarray(steps, float).copy()
    evals = 0

    def F(x):
        nonlocal evals
        evals += 1
        return f(x)

    def explore(x, fx):
        x = x.copy()
        for i in range(len(x)):
            if steps[i] == 0: continue
            for sgn in (1.0, -1.0):
                t = x.copy(); t[i] += sgn * steps[i]; t = clip(t)
                ft = F(t)
                if ft < fx:
                    x, fx = t, ft
                    break
        return x, fx

    base = clip(np.asarray(x0, float)); fb = F(base)
    while steps.max() > min_step and evals < max_evals:
        new, fn = explore(base, fb)
        if fn < fb:
            while fn < fb and evals < max_evals:
                pattern = clip(2 * new - base)
                base, fb = new, fn
                new, fn = explore(pattern, F(pattern))
        else:
            steps *= shrink
    return base, fb
# =====================================================================
# 1. МОДЕЛЬ ДАННЫХ С ИНТЕРАКТИВНЫМИ ЧЕКБОКСАМИ [✓]
# =====================================================================
class SolutionsTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.results = []
        self.headers = ["✓", "№", "Ошибка", "Итераций", "Δt", "h", "t_start", "K (Fwd)", "K (Bwd)", "Slave 0"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.results)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        res = self.results[row]

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return Qt.CheckState.Checked if res.get("checked", True) else Qt.CheckState.Unchecked

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return ""
            elif col == 1: return str(row + 1)
            elif col == 2: return f"{res['error']:.2e}"
            elif col == 3: return str(res['iters'])
            elif col == 4: return f"{res['dt']:.3f}"
            elif col == 5: return f"{res['h']:.4f}"
            elif col == 6: return f"{res['t_start']:.2f}"
            elif col == 7: return str(res['k_fwd'])
            elif col == 8: return str(res['k_bwd'])
            elif col == 9: return str(res['s0'])

        elif role == Qt.ItemDataRole.UserRole:
            if col == 0: return 1 if res.get("checked", True) else 0
            elif col == 1: return row + 1
            elif col == 2: return float(res['error'])
            elif col == 3: return int(res['iters'])
            elif col == 4: return float(res['dt'])
            elif col == 5: return float(res['h'])
            elif col == 6: return float(res['t_start'])
            elif col == 7: return float(res['k_fwd'][0])
            elif col == 8: return float(res['k_bwd'][0])
            elif col == 9: return float(res['s0'][0])

        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if index.isValid() and index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            self.results[index.row()]["checked"] = (value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        return False

    def flags(self, index):
        f = super().flags(index)
        if index.column() == 0:
            f |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
        return f

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None

    def add_results_batch(self, batch):
        if not batch: return
        start_row = len(self.results)
        end_row = start_row + len(batch) - 1
        for item in batch:
            if "checked" not in item:
                item["checked"] = True
        self.beginInsertRows(QModelIndex(), start_row, end_row)
        self.results.extend(batch)
        self.endInsertRows()

    def set_all_checked(self, checked_indices, state=True):
        if not checked_indices or not self.results:
            return
        for idx in checked_indices:
            if 0 <= idx < len(self.results):
                self.results[idx]["checked"] = state

        # Обновляем только состояние чекбоксов в первом столбце
        top_idx = self.index(min(checked_indices), 0)
        bot_idx = self.index(max(checked_indices), 0)
        self.dataChanged.emit(top_idx, bot_idx, [Qt.ItemDataRole.CheckStateRole, Qt.ItemDataRole.UserRole])

    def clear(self):
        self.beginResetModel()
        self.results.clear()
        self.endResetModel()


class SolutionsFilterProxyModel(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self.filter_error_enabled = False
        self.max_error = 1e-12
        self.filter_iters_enabled = False
        self.max_iters = 150
        self.filter_dt_enabled = False
        self.min_dt = 0.0
        self.max_dt = 100.0

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if model is None or source_row >= len(model.results):
            return False
        res = model.results[source_row]

        if self.filter_error_enabled and res['error'] > self.max_error:
            return False
        if self.filter_iters_enabled and res['iters'] > self.max_iters:
            return False
        if self.filter_dt_enabled and not (self.min_dt <= res['dt'] <= self.max_dt):
            return False

        return True


# =====================================================================
# 2. ПОТОК ВЫЧИСЛЕНИЙ
# =====================================================================
class ParamSearchWorker(QThread):
    progress_signal = pyqtSignal(int, int, int, str, float, float)
    batch_found = pyqtSignal(list)
    finished_search = pyqtSignal(int)

    def __init__(self, sys_name, param_grid, max_iters, target_err):
        super().__init__()
        self.sys_name = sys_name
        self.param_grid = param_grid
        self.max_iters = max_iters
        self.target_err = target_err
        self._is_cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def pause(self): self._pause_event.clear()
    def resume(self): self._pause_event.set()
    def is_paused(self): return not self._pause_event.is_set()
    def cancel(self):
        self._is_cancelled = True
        self._pause_event.set()

    def run(self):
        sys_conf = SYSTEMS_CONFIG[self.sys_name]
        params_dict = sys_conf["params"]
        step_fwd = SYSTEMS_REGISTRY[self.sys_name]["forward"]
        sys_id, p_arr = pack_system_params(self.sys_name, params_dict)

        try:
            run_fast_candidate_trs(
                sys_id, p_arr, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, False,
                0.01, 0.0, 0.1, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-6
            )
        except Exception:
            pass

        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        total_combos = 1
        for lst in value_lists: total_combos *= len(lst)

        start_time = time.time()
        success_count = 0
        last_update_time = 0.0
        num_workers = max(1, os.cpu_count() or 4)

        def eval_single_cfg(values):
            if self._is_cancelled: return None, None
            cfg = dict(zip(keys, values))
            if cfg.get("link_k", True):
                kxf, kyf, kzf = cfg["kxf"], cfg["kyf"], cfg["kzf"]
                kxb, kyb, kzb = kxf, kyf, kzf
            else:
                kxf, kyf, kzf = cfg["kxf"], cfg["kyf"], cfg["kzf"]
                kxb, kyb, kzb = cfg["kxb"], cfg["kyb"], cfg["kzb"]

            mx, my, mz = cfg["mx"], cfg["my"], cfg["mz"]
            sx, sy, sz = cfg["sx"], cfg["sy"], cfg["sz"]
            near_master = cfg.get("near_master", False)

            h = cfg["h"]
            t_start = cfg["t_start"]
            # Привязываем dt строго к реальному целому числу шагов
            n_steps = max(1, int(round(cfg["dt"] / h)))
            dt = round(n_steps * h, 5)

            ok, iters_used, err = run_fast_candidate_trs(
                sys_id=sys_id, p_arr=p_arr,
                mx0=mx, my0=my, mz0=mz, sx0=sx, sy0=sy, sz0=sz,
                near_master=near_master, h=h, t_start=t_start, dt=dt,
                max_iters=self.max_iters,
                kxf=kxf, kyf=kyf, kzf=kzf, kxb=kxb, kyb=kyb, kzb=kzb,
                target_err=self.target_err
            )

            info = (kxf, dt, h, t_start, mx, my, mz, sx, sy, sz)
            if ok:
                m0 = (mx, my, mz)
                if near_master:
                    xm, ym, zm = m0
                    steps_to_start = int(round(t_start / h))
                    for _ in range(steps_to_start):
                        xm, ym, zm = step_fwd(xm, ym, zm, h, params=params_dict)
                    s0 = (round(xm + 0.01, 5), round(ym + 0.01, 5), round(zm + 0.01, 5))
                else:
                    s0 = (sx, sy, sz)

                res_dict = {
                    "m0": m0, "s0": s0,
                    "k_fwd": (kxf, kyf, kzf), "k_bwd": (kxb, kyb, kzb),
                    "h": h, "t_start": t_start, "dt": dt,
                    "iters": iters_used, "error": err, "checked": True
                }
                return res_dict, info
            return None, info

        chunk_size = 250
        combos_iter = itertools.product(*value_lists)
        evaluated_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while not self._is_cancelled and evaluated_count < total_combos:
                self._pause_event.wait()
                if self._is_cancelled: break

                chunk = list(itertools.islice(combos_iter, chunk_size))
                if not chunk: break

                results = executor.map(eval_single_cfg, chunk)
                found_in_chunk = []
                last_info_str = ""

                for res_dict, info in results:
                    evaluated_count += 1
                    if info:
                        kxf, dt, h, t_st, mx, my, mz, sx, sy, sz = info
                        last_info_str = f"Kx={kxf:.2f} | dt={dt:.2f} | h={h:.4f} | t_st={t_st:.1f} | S0=({sx:.1f},{sy:.1f},{sz:.1f})"
                    if res_dict:
                        found_in_chunk.append(res_dict)

                if found_in_chunk:
                    success_count += len(found_in_chunk)
                    self.batch_found.emit(found_in_chunk)

                now = time.time()
                if now - last_update_time > 0.08 or evaluated_count == total_combos:
                    last_update_time = now
                    elapsed = now - start_time
                    speed = evaluated_count / elapsed if elapsed > 0 else 1.0
                    eta = (total_combos - evaluated_count) / speed if speed > 0 else 0.0
                    pct = int((evaluated_count / total_combos) * 100)
                    self.progress_signal.emit(pct, evaluated_count, total_combos, last_info_str, elapsed, eta)

        self.finished_search.emit(success_count)

class CachedGridWorker(ParamSearchWorker):
    def _setup(self):
        conf = SYSTEMS_CONFIG[self.sys_name]
        self.sys_id, self.p_arr = pack_system_params(self.sys_name, conf["params"])
        # прогрев компиляции новых функций
        xa, ya, za, _ = integrate_master(self.sys_id, self.p_arr, 0.1, 0.1, 0.1, 0.01, 50)
        run_fast_trs_window_score(self.sys_id, self.p_arr, xa, ya, za, 0, 10,
                                  0.1, 0.1, 0.1, 0.01, 2, 0., 0., 0., 0., 0., 0., 1e-6)

    def run(self):
        self._setup()
        g = self.param_grid
        link_k = g["link_k"][0]
        slave_mode = g["slave_mode"]
        slave_radius = g["slave_radius"]
        slave_count = g["slave_count"]
        slave_dist = g["slave_dist"]
        kb_lists = [[None]] * 3 if link_k else [g["kxb"], g["kyb"], g["kzb"]]

        if slave_mode != "manual":
            n_slave = slave_count
            offsets = generate_offsets(slave_count, slave_radius, dist_type=slave_dist)
        else:
            n_slave = len(g["sx"]) * len(g["sy"]) * len(g["sz"])
            offsets = None

        n_rest = n_slave
        for lst in (g["t_start"], g["dt"], g["kxf"], g["kyf"], g["kzf"], *kb_lists):
            n_rest *= len(lst)
        n_master = len(g["mx"]) * len(g["my"]) * len(g["mz"]) * len(g["h"])
        total = n_master * n_rest

        start = time.time(); done = 0; found_total = 0; last_emit = 0.0
        workers = max(1, os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mx, my, mz, h in itertools.product(g["mx"], g["my"], g["mz"], g["h"]):
                if self._is_cancelled: break
                n_total = int(round((max(g["t_start"]) + max(g["dt"])) / h)) + 1
                xm, ym, zm, valid = integrate_master(self.sys_id, self.p_arr, mx, my, mz, h, n_total)

                if slave_mode == "center":
                    xc, yc, zc = get_attractor_center_from_trajectory(xm, ym, zm, valid)
                else:
                    xc, yc, zc = 0.0, 0.0, 0.0

                if slave_mode != "manual":
                    combos = itertools.product(range(slave_count), g["t_start"], g["dt"],
                                               g["kxf"], g["kyf"], g["kzf"], *kb_lists)

                    def ev(v, xm=xm, ym=ym, zm=zm, valid=valid, h=h, m0=(mx, my, mz)):
                        if self._is_cancelled: return None
                        off_idx, ts, dt0, kxf, kyf, kzf, kxb, kyb, kzb = v
                        if kxb is None: kxb, kyb, kzb = kxf, kyf, kzf
                        i0 = int(round(ts / h));
                        n = max(1, int(round(dt0 / h)))
                        if i0 + n > valid: return None
                        off = offsets[off_idx]
                        if slave_mode == "near_master":
                            s0 = avoid_zero_coords(xm[i0] + off[0], ym[i0] + off[1], zm[i0] + off[2])
                        else:  # "center"
                            s0 = avoid_zero_coords(xc + off[0], yc + off[1], zc + off[2])
                        ok, it, sc = run_fast_trs_window_score(
                            self.sys_id, self.p_arr, xm, ym, zm, i0, n, s0[0], s0[1], s0[2], h,
                            self.max_iters, kxf, kyf, kzf, kxb, kyb, kzb, self.target_err)
                        if not ok: return None
                        return {"m0": m0, "s0": s0, "k_fwd": (kxf, kyf, kzf), "k_bwd": (kxb, kyb, kzb),
                                "h": h, "t_start": ts, "dt": round(n * h, 5),
                                "iters": it, "error": 10 ** sc, "checked": True}
                else:
                    combos = itertools.product(g["sx"], g["sy"], g["sz"], g["t_start"], g["dt"],
                                               g["kxf"], g["kyf"], g["kzf"], *kb_lists)

                    def ev(v, xm=xm, ym=ym, zm=zm, valid=valid, h=h, m0=(mx, my, mz)):
                        if self._is_cancelled: return None
                        sx, sy, sz, ts, dt0, kxf, kyf, kzf, kxb, kyb, kzb = v
                        if kxb is None: kxb, kyb, kzb = kxf, kyf, kzf
                        i0 = int(round(ts / h));
                        n = max(1, int(round(dt0 / h)))
                        if i0 + n > valid: return None
                        s0 = avoid_zero_coords(sx, sy, sz)
                        ok, it, sc = run_fast_trs_window_score(
                            self.sys_id, self.p_arr, xm, ym, zm, i0, n, s0[0], s0[1], s0[2], h,
                            self.max_iters, kxf, kyf, kzf, kxb, kyb, kzb, self.target_err)
                        if not ok: return None
                        return {"m0": m0, "s0": s0, "k_fwd": (kxf, kyf, kzf), "k_bwd": (kxb, kyb, kzb),
                                "h": h, "t_start": ts, "dt": round(n * h, 5),
                                "iters": it, "error": 10 ** sc, "checked": True}
                while not self._is_cancelled:
                    self._pause_event.wait()
                    chunk = list(itertools.islice(combos, 500))
                    if not chunk: break
                    batch = [r for r in ex.map(ev, chunk) if r]
                    done += len(chunk)
                    if batch:
                        found_total += len(batch)
                        self.batch_found.emit(batch)
                    now = time.time()
                    if now - last_emit > 0.08 or done >= total:
                        last_emit = now
                        el = now - start
                        speed = done / el if el > 0 else 1.0
                        self.progress_signal.emit(int(100 * done / total), done, total,
                                                  f"h={h:.4f}", el, (total - done) / speed)
        self.finished_search.emit(found_total)

# =====================================================================
# 3. ВСПОМОГАТЕЛЬНАЯ СТРОКА НАСТРОЙКИ ПАРАМЕТРА
# =====================================================================
class ParamRangeRow(QWidget):
    def __init__(self, label_text, default_val=0.0, step=0.1, min_limit=-1000.0, max_limit=1000.0, decimals=3):
        super().__init__()
        self.decimals = decimals
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self.chk_vary = QCheckBox()
        self.lbl_name = QLabel(label_text)
        self.lbl_name.setFixedWidth(80)

        self.sp_val = QDoubleSpinBox()
        self.sp_val.setRange(min_limit, max_limit)
        self.sp_val.setDecimals(decimals)
        self.sp_val.setSingleStep(step)
        self.sp_val.setValue(default_val)

        self.lbl_to = QLabel("до:")
        self.sp_to = QDoubleSpinBox()
        self.sp_to.setRange(min_limit, max_limit)
        self.sp_to.setDecimals(decimals)
        self.sp_to.setSingleStep(step)
        self.sp_to.setValue(default_val)
        self.sp_to.setEnabled(False)

        self.lbl_step = QLabel("шаг:")
        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(1e-4, 500.0)
        self.sp_step.setDecimals(decimals)
        self.sp_step.setValue(step)
        self.sp_step.setEnabled(False)

        self.chk_vary.toggled.connect(self.on_vary_toggled)

        layout.addWidget(self.chk_vary)
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.sp_val)
        layout.addWidget(self.lbl_to)
        layout.addWidget(self.sp_to)
        layout.addWidget(self.lbl_step)
        layout.addWidget(self.sp_step)

    def on_vary_toggled(self, checked):
        self.sp_to.setEnabled(checked)
        self.sp_step.setEnabled(checked)
        if checked and self.sp_to.value() <= self.sp_val.value():
            self.sp_to.setValue(self.sp_val.value() + self.sp_step.value() * 3)

    def get_values(self):
        if not self.chk_vary.isChecked():
            return [round(self.sp_val.value(), self.decimals)]

        v_from = self.sp_val.value()
        v_to = self.sp_to.value()
        step = max(1e-5, self.sp_step.value())

        if v_to < v_from: v_from, v_to = v_to, v_from
        vals = []
        cur = v_from
        while cur <= v_to + 1e-9:
            vals.append(round(cur, self.decimals))
            cur += step
        return vals if len(vals) > 0 else [v_from]


# =====================================================================
# 4. МОДАЛЬНОЕ / НЕМОДАЛЬНОЕ ОКНО ПЕРЕБОРА
# =====================================================================
class ParamSearchDialog(QDialog):
    solution_preview_requested = pyqtSignal(dict)
    build_portrait_requested = pyqtSignal(dict)
    save_presets_requested = pyqtSignal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.current_settings = current_settings
        self.sys_name = current_settings.get("system", "Sprott B")

        self.setWindowTitle(f"Поиск параметров TRS — [{self.sys_name}]")
        self.resize(1200, 840)

        self.worker = None
        self.selected_solution = None
        self.search_t_max = None

        self.init_ui()
        self.load_from_settings(current_settings)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)

        # Компактная плашка системы
        self.lbl_sys_banner = QLabel(f"Исследуемая динамическая система: <b>{self.sys_name}</b>")
        self.lbl_sys_banner.setFixedHeight(30)
        self.lbl_sys_banner.setStyleSheet("""
            QLabel {
                font-size: 12px; color: #0f172a; background-color: #e2e8f0;
                padding: 4px 10px; border-radius: 4px; border: 1px solid #cbd5e1;
            }
        """)
        main_layout.addWidget(self.lbl_sys_banner, stretch=0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- ЛЕВАЯ КОЛОНКА (СЕТКА) ---
        scroll_widget = QWidget()
        left_layout = QVBoxLayout(scroll_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        grp_stop = QGroupBox("Критерии успешной сходимости")
        grid_stop = QGridLayout(grp_stop)
        grid_stop.addWidget(QLabel("Требуемая ошибка:"), 0, 0)
        self.sp_target_err = QDoubleSpinBox()
        self.sp_target_err.setDecimals(14)
        self.sp_target_err.setRange(1e-16, 1.0)
        self.sp_target_err.setValue(1e-12)
        grid_stop.addWidget(self.sp_target_err, 0, 1)

        grid_stop.addWidget(QLabel("Макс. итераций:"), 1, 0)
        self.sp_max_iters = QSpinBox()
        self.sp_max_iters.setRange(2, 2000)
        self.sp_max_iters.setValue(150)
        grid_stop.addWidget(self.sp_max_iters, 1, 1)
        left_layout.addWidget(grp_stop)

        grp_init = QGroupBox("Начальные условия (Master и Slave)")
        v_init = QVBoxLayout(grp_init)

        h_smode = QHBoxLayout()
        h_smode.addWidget(QLabel("<b>Режим Slave:</b>"))
        self.combo_slave_mode = QComboBox()
        self.combo_slave_mode.addItem("В окрестности Master (t_start)", "near_master")
        self.combo_slave_mode.addItem("В окрестности центра аттрактора", "center")
        self.combo_slave_mode.addItem("Ручная сетка Slave (X, Y, Z)", "manual")
        self.combo_slave_mode.currentIndexChanged.connect(self.on_slave_mode_changed)
        h_smode.addWidget(self.combo_slave_mode)
        v_init.addLayout(h_smode)

        self.box_slave_opts = QWidget()
        grid_sopts = QGridLayout(self.box_slave_opts)
        grid_sopts.setContentsMargins(0, 2, 0, 2)

        grid_sopts.addWidget(QLabel("Радиус (R):"), 0, 0)
        self.sp_slave_radius = QDoubleSpinBox()
        self.sp_slave_radius.setRange(0.0001, 100.0);
        self.sp_slave_radius.setDecimals(4)
        self.sp_slave_radius.setSingleStep(0.01);
        self.sp_slave_radius.setValue(0.01)
        grid_sopts.addWidget(self.sp_slave_radius, 0, 1)

        grid_sopts.addWidget(QLabel("Точек Slave:"), 0, 2)
        self.sp_slave_count = QSpinBox()
        self.sp_slave_count.setRange(1, 5000);
        self.sp_slave_count.setValue(100);
        self.sp_slave_count.setSingleStep(10)
        grid_sopts.addWidget(self.sp_slave_count, 0, 3)

        grid_sopts.addWidget(QLabel("Распределение:"), 1, 0)
        self.combo_slave_dist = QComboBox()
        self.combo_slave_dist.addItem("Равномерное", "uniform")
        self.combo_slave_dist.addItem("Нормальное", "normal")
        grid_sopts.addWidget(self.combo_slave_dist, 1, 1, 1, 3)

        v_init.addWidget(self.box_slave_opts)

        self.r_mx = ParamRangeRow("Master X:", 0.1, step=0.5)
        self.r_my = ParamRangeRow("Master Y:", 0.1, step=0.5)
        self.r_mz = ParamRangeRow("Master Z:", 0.1, step=0.5)
        self.r_sx = ParamRangeRow("Slave X:", 5.0, step=1.0)
        self.r_sy = ParamRangeRow("Slave Y:", 2.0, step=1.0)
        self.r_sz = ParamRangeRow("Slave Z:", 0.0, step=1.0)
        for r in (self.r_mx, self.r_my, self.r_mz, self.r_sx, self.r_sy, self.r_sz):
            v_init.addWidget(r)

        self.on_slave_mode_changed()
        left_layout.addWidget(grp_init)

        grp_k = QGroupBox("Коэффициенты связи (K)")
        v_k = QVBoxLayout(grp_k)
        self.chk_link_k = QCheckBox("Связать коэффициенты: K_bwd = K_fwd (симметричный режим)")
        self.chk_link_k.setChecked(True)
        self.chk_link_k.toggled.connect(self.on_link_k_toggled)
        v_k.addWidget(self.chk_link_k)

        self.r_kxf = ParamRangeRow("K_x (fwd):", 5.0, step=0.5)
        self.r_kyf = ParamRangeRow("K_y (fwd):", 0.0, step=0.5)
        self.r_kzf = ParamRangeRow("K_z (fwd):", 0.0, step=0.5)
        self.r_kxb = ParamRangeRow("K_x (bwd):", 5.0, step=0.5)
        self.r_kyb = ParamRangeRow("K_y (bwd):", 0.0, step=0.5)
        self.r_kzb = ParamRangeRow("K_z (bwd):", 0.0, step=0.5)
        for r in (self.r_kxf, self.r_kyf, self.r_kzf, self.r_kxb, self.r_kyb, self.r_kzb):
            v_k.addWidget(r)
        self.on_link_k_toggled(True)
        left_layout.addWidget(grp_k)

        grp_time = QGroupBox("Временные параметры и шаг h")
        v_time = QVBoxLayout(grp_time)
        self.r_tstart = ParamRangeRow("t_start:", 245.0, step=1.0)
        self.r_dt = ParamRangeRow("Длина Δt:", 1.0, step=0.2)
        self.r_h = ParamRangeRow("Шаг h:", 0.01, step=0.005, decimals=4)
        v_time.addWidget(self.r_tstart)
        v_time.addWidget(self.r_dt)
        v_time.addWidget(self.r_h)
        left_layout.addWidget(grp_time)

        left_scroll = QScrollArea()
        left_scroll.setWidget(scroll_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(480)
        splitter.addWidget(left_scroll)

        # --- ПРАВАЯ КОЛОНКА (ФИЛЬТРЫ + ТАБЛИЦА) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 4, 0)
        right_layout.setSpacing(6)

        # Блок фильтров над таблицей
        grp_filter = QGroupBox("Фильтрация найденных решений")
        filter_grid = QGridLayout(grp_filter)
        filter_grid.setContentsMargins(6, 6, 6, 6)
        filter_grid.setSpacing(6)

        self.chk_f_err = QCheckBox("Ошибка ≤")
        self.sp_f_err = QDoubleSpinBox()
        self.sp_f_err.setDecimals(14); self.sp_f_err.setRange(1e-16, 1.0); self.sp_f_err.setValue(1e-13)
        self.sp_f_err.setEnabled(False)
        self.chk_f_err.toggled.connect(self.sp_f_err.setEnabled)
        self.chk_f_err.toggled.connect(self.on_filter_changed)
        self.sp_f_err.valueChanged.connect(self.on_filter_changed)
        filter_grid.addWidget(self.chk_f_err, 0, 0)
        filter_grid.addWidget(self.sp_f_err, 0, 1)

        self.chk_f_iters = QCheckBox("Итераций ≤")
        self.sp_f_iters = QSpinBox(); self.sp_f_iters.setRange(1, 2000); self.sp_f_iters.setValue(25)
        self.sp_f_iters.setEnabled(False)
        self.chk_f_iters.toggled.connect(self.sp_f_iters.setEnabled)
        self.chk_f_iters.toggled.connect(self.on_filter_changed)
        self.sp_f_iters.valueChanged.connect(self.on_filter_changed)
        filter_grid.addWidget(self.chk_f_iters, 0, 2)
        filter_grid.addWidget(self.sp_f_iters, 0, 3)

        self.chk_f_dt = QCheckBox("Δt от:")
        self.sp_f_dt_min = QDoubleSpinBox(); self.sp_f_dt_min.setRange(0.01, 100.0); self.sp_f_dt_min.setValue(0.5)
        self.sp_f_dt_max = QDoubleSpinBox(); self.sp_f_dt_max.setRange(0.01, 100.0); self.sp_f_dt_max.setValue(2.0)
        self.sp_f_dt_min.setEnabled(False); self.sp_f_dt_max.setEnabled(False)
        self.chk_f_dt.toggled.connect(self.sp_f_dt_min.setEnabled)
        self.chk_f_dt.toggled.connect(self.sp_f_dt_max.setEnabled)
        self.chk_f_dt.toggled.connect(self.on_filter_changed)
        self.sp_f_dt_min.valueChanged.connect(self.on_filter_changed)
        self.sp_f_dt_max.valueChanged.connect(self.on_filter_changed)

        h_dt_filter = QHBoxLayout()
        h_dt_filter.addWidget(self.sp_f_dt_min)
        h_dt_filter.addWidget(QLabel("до:"))
        h_dt_filter.addWidget(self.sp_f_dt_max)
        filter_grid.addWidget(self.chk_f_dt, 1, 0)
        filter_grid.addLayout(h_dt_filter, 1, 1)

        self.btn_reset_filters = QPushButton("Сбросить фильтры")
        self.btn_reset_filters.clicked.connect(self.reset_filters)
        filter_grid.addWidget(self.btn_reset_filters, 1, 2, 1, 2)
        right_layout.addWidget(grp_filter)

        # Строка управления чекбоксами [✓]
        h_check_ctrl = QHBoxLayout()
        self.lbl_table_header = QLabel("<b>Найденные решения: 0 шт.</b>")
        h_check_ctrl.addWidget(self.lbl_table_header)
        h_check_ctrl.addStretch()

        self.btn_check_all = QPushButton("✔ Выбрать все отфильтрованные")
        self.btn_check_all.clicked.connect(lambda: self.set_filtered_checks(True))
        self.btn_uncheck_all = QPushButton("✖ Снять выбор")
        self.btn_uncheck_all.clicked.connect(lambda: self.set_filtered_checks(False))
        h_check_ctrl.addWidget(self.btn_check_all)
        h_check_ctrl.addWidget(self.btn_uncheck_all)
        right_layout.addLayout(h_check_ctrl)

        # Таблица
        self.table_model = SolutionsTableModel()
        self.proxy_model = SolutionsFilterProxyModel()
        self.proxy_model.setSourceModel(self.table_model)

        self.table_view = QTableView()
        self.table_view.setModel(self.proxy_model)
        self.table_view.setSortingEnabled(False)  # Включаем только после перебора
        # Фиксированные / интерактивные размеры — никакого замера 100 000 строк!
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        right_layout.addWidget(self.table_view)

        # Панель быстрых действий над решением
        btn_row = QHBoxLayout()

        self.btn_preview_live = QPushButton("Применить на графики")
        self.btn_preview_live.setEnabled(False)
        self.btn_preview_live.setStyleSheet("font-weight: bold; padding: 6px; background-color: #e0f2fe;")
        self.btn_preview_live.clicked.connect(self.apply_live_preview)

        self.btn_save_favorites = QPushButton("Сохранить отмеченные в пресеты")
        self.btn_save_favorites.setEnabled(False)
        self.btn_save_favorites.setStyleSheet("font-weight: bold; padding: 6px; background-color: #dcfce7;")
        self.btn_save_favorites.clicked.connect(self.save_checked_to_presets)

        self.btn_export_csv = QPushButton("Экспорт в CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setStyleSheet("padding: 6px;")
        self.btn_export_csv.clicked.connect(self.export_solutions_to_csv)

        btn_row.addWidget(self.btn_preview_live)
        btn_row.addWidget(self.btn_save_favorites)
        btn_row.addWidget(self.btn_export_csv)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        # Нижняя панель управления
        bot_box = QGroupBox()
        bot_layout = QVBoxLayout(bot_box)
        bot_layout.setContentsMargins(6, 6, 6, 6)
        bot_layout.setSpacing(4)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100); self.pbar.setValue(0); self.pbar.setTextVisible(True)
        self.pbar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #b0b0b0; border-radius: 4px; text-align: center;
                font-size: 11px; font-weight: bold; color: #0f172a;
                background-color: #f1f5f9; height: 18px;
            }
            QProgressBar::chunk { background-color: #22c55e; border-radius: 3px; }
        """)
        bot_layout.addWidget(self.pbar)

        self.lbl_stats = QLabel("Готов к запуску перебора. Настройте сетку и нажмите «Запустить».")
        self.lbl_stats.setStyleSheet("font-size: 11px; color: #334155;")
        bot_layout.addWidget(self.lbl_stats)

        ctrl_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ Запустить перебор")
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 12px; padding: 7px; background-color: #e0f2fe;")
        self.btn_start.clicked.connect(self.start_search)

        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("padding: 7px; font-weight: bold;")
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("padding: 7px;")
        self.btn_stop.clicked.connect(self.stop_search)

        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.close)

        ctrl_btns.addWidget(self.btn_start)
        ctrl_btns.addWidget(self.btn_pause)
        ctrl_btns.addWidget(self.btn_stop)
        ctrl_btns.addStretch()
        ctrl_btns.addWidget(self.btn_close)
        bot_layout.addLayout(ctrl_btns)

        main_layout.addWidget(bot_box, stretch=0)
        self.table_view.selectionModel().selectionChanged.connect(self.on_table_selection)

    def closeEvent(self, event):
        self.stop_and_wait_worker()
        event.accept()

    def stop_and_wait_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)

    def on_filter_changed(self):
        self.proxy_model.filter_error_enabled = self.chk_f_err.isChecked()
        self.proxy_model.max_error = self.sp_f_err.value()
        self.proxy_model.filter_iters_enabled = self.chk_f_iters.isChecked()
        self.proxy_model.max_iters = self.sp_f_iters.value()
        self.proxy_model.filter_dt_enabled = self.chk_f_dt.isChecked()
        self.proxy_model.min_dt = self.sp_f_dt_min.value()
        self.proxy_model.max_dt = self.sp_f_dt_max.value()

        self.proxy_model.invalidateFilter()
        visible_count = self.proxy_model.rowCount()
        total_count = len(self.table_model.results)
        self.lbl_table_header.setText(
            f"<b>Найденные решения: показано {visible_count:,} из {total_count:,} шт.</b>"
        )

    def reset_filters(self):
        # Временно глушим сигналы, чтобы не дергать таблицу 4 раза подряд
        for chk in (self.chk_f_err, self.chk_f_iters, self.chk_f_dt):
            chk.blockSignals(True)

        self.chk_f_err.setChecked(False)
        self.chk_f_iters.setChecked(False)
        self.chk_f_dt.setChecked(False)

        for chk in (self.chk_f_err, self.chk_f_iters, self.chk_f_dt):
            chk.blockSignals(False)

        self.on_filter_changed()

    def set_filtered_checks(self, state=True):
        """Отмечает или снимает галочки только с отображаемых (отфильтрованных) строк."""
        source_indices = []
        for row in range(self.proxy_model.rowCount()):
            proxy_idx = self.proxy_model.index(row, 0)
            source_idx = self.proxy_model.mapToSource(proxy_idx)
            source_indices.append(source_idx.row())
        self.table_model.set_all_checked(source_indices, state)

    def on_link_k_toggled(self, checked):
        self.r_kxb.setVisible(not checked)
        self.r_kyb.setVisible(not checked)
        self.r_kzb.setVisible(not checked)

    def load_from_settings(self, p):
        m0 = p.get("master_0", [0.1, 0.1, 0.1])
        s0 = p.get("slave_0", [5.0, 2.0, 0.0])
        kf = p.get("K_fwd", [5.0, 0.0, 0.0])
        kb = p.get("K_bwd", [5.0, 0.0, 0.0])
        tm = p.get("time", {})
        sim = p.get("sim", {})

        self.r_mx.sp_val.setValue(m0[0]); self.r_my.sp_val.setValue(m0[1]); self.r_mz.sp_val.setValue(m0[2])
        self.r_sx.sp_val.setValue(s0[0]); self.r_sy.sp_val.setValue(s0[1]); self.r_sz.sp_val.setValue(s0[2])
        self.r_kxf.sp_val.setValue(kf[0]); self.r_kyf.sp_val.setValue(kf[1]); self.r_kzf.sp_val.setValue(kf[2])
        self.r_kxb.sp_val.setValue(kb[0]); self.r_kyb.sp_val.setValue(kb[1]); self.r_kzb.sp_val.setValue(kb[2])
        self.r_tstart.sp_val.setValue(tm.get("t_start", 245.0))
        self.r_dt.sp_val.setValue(tm.get("dt", 1.0))
        self.r_h.sp_val.setValue(sim.get("h", 0.01))

    def on_slave_mode_changed(self):
        mode = self.combo_slave_mode.currentData()
        is_manual = (mode == "manual")
        self.box_slave_opts.setVisible(not is_manual)
        for r in (self.r_sx, self.r_sy, self.r_sz):
            r.setVisible(is_manual)

    def build_grid_dict(self):
        slave_mode = self.combo_slave_mode.currentData()
        grid = {
            "mx": self.r_mx.get_values(), "my": self.r_my.get_values(), "mz": self.r_mz.get_values(),
            "kxf": self.r_kxf.get_values(), "kyf": self.r_kyf.get_values(), "kzf": self.r_kzf.get_values(),
            "t_start": self.r_tstart.get_values(), "dt": self.r_dt.get_values(), "h": self.r_h.get_values(),
            "link_k": [self.chk_link_k.isChecked()],
            "slave_mode": slave_mode,
            "slave_radius": self.sp_slave_radius.value(),
            "slave_count": self.sp_slave_count.value() if slave_mode != "manual" else 1,
            "slave_dist": self.combo_slave_dist.currentData(),
            "sx": self.r_sx.get_values() if slave_mode == "manual" else [0.0],
            "sy": self.r_sy.get_values() if slave_mode == "manual" else [0.0],
            "sz": self.r_sz.get_values() if slave_mode == "manual" else [0.0],
        }
        if not self.chk_link_k.isChecked():
            grid["kxb"] = self.r_kxb.get_values()
            grid["kyb"] = self.r_kyb.get_values()
            grid["kzb"] = self.r_kzb.get_values()
        return grid

    def start_search(self):
        self.table_view.setSortingEnabled(False)  # Отключаем пересортировку во время поиска
        grid = self.build_grid_dict()
        self.search_t_max = max(grid["t_start"])
        self.table_model.clear()
        self.on_filter_changed()
        self.btn_preview_live.setEnabled(False)
        self.btn_save_favorites.setEnabled(False)
        self.btn_export_csv.setEnabled(False)
        self.pbar.setValue(0)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Пауза")
        self.btn_stop.setEnabled(True)

        self.worker = CachedGridWorker(
            sys_name=self.sys_name,
            param_grid=grid,
            max_iters=self.sp_max_iters.value(),
            target_err=self.sp_target_err.value()
        )
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.batch_found.connect(self.on_solutions_batch)
        self.worker.finished_search.connect(self.on_worker_finished)
        self.worker.start()

    def solution_t_max(self, res):
        """Время моделирования для решения: 'до' по t_start, но не меньше конца окна."""
        base = self.search_t_max
        if base is None:
            base = self.current_settings.get("time", {}).get("t_max", 300.0)
        t_end = res["t_start"] + res["dt"]
        return round(max(base, t_end), 5)

    def toggle_pause(self):
        if not self.worker: return
        if self.worker.is_paused():
            self.worker.resume()
            self.btn_pause.setText("⏸ Пауза")
            self.btn_pause.setStyleSheet("padding: 7px; font-weight: bold;")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶ Возобновить")
            self.btn_pause.setStyleSheet("padding: 7px; font-weight: bold; background-color: #fef08a;")
            self.lbl_stats.setText("[ПАУЗА] Вычисления приостановлены. Главное окно активно.")

    def stop_search(self):
        if self.worker:
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.lbl_stats.setText("Остановка перебора...")
            self.worker.cancel()
            self.worker.wait(2000)
            self.btn_start.setEnabled(True)
            self.lbl_stats.setText("Перебор остановлен пользователем.")

    def on_worker_progress(self, pct, cur, total, cur_info, elapsed, eta):
        self.pbar.setValue(pct)
        def fmt_time(sec):
            m, s = divmod(int(sec), 60)
            h_part, m = divmod(m, 60)
            if h_part > 0: return f"{h_part:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        self.lbl_stats.setText(
            f"Прогресс: {cur:,} / {total:,} ({pct}%) | Прошло: {fmt_time(elapsed)} | Осталось ~ {fmt_time(eta)}<br>"
            f"<b>Текущие:</b> {cur_info}"
        )

    def on_solutions_batch(self, batch):
        # Тихий бэкап на диск: если что-то случится, данные останутся в файле
        try:
            with open("autosave_cache.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                for r in batch:
                    writer.writerow([r['error'], r['iters'], r['dt'], r['h'], r['t_start'], r['k_fwd'], r['k_bwd'], r['m0'], r['s0']])
        except Exception:
            pass

        self.table_model.add_results_batch(batch)
        total_count = len(self.table_model.results)
        vis_count = self.proxy_model.rowCount()
        self.lbl_table_header.setText(f"<b>Найденные решения: показано {vis_count:,} из {total_count:,} шт.</b>")
        has_items = total_count > 0
        self.btn_export_csv.setEnabled(has_items)
        self.btn_save_favorites.setEnabled(has_items)

    def on_worker_finished(self, found_count):
        if self.sender() is not self.worker:
            return
        self.table_view.setSortingEnabled(True)   # Включаем, когда всё готово
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.pbar.setValue(100)
        self.lbl_stats.setText(f"Перебор завершен! Всего найдено подходящих конфигураций: {found_count:,}.")
        QMessageBox.information(self, "Завершено", f"Поиск завершен!\nНайдено подходящих конфигураций: {found_count:,}")

    def on_table_selection(self):
        selected_rows = self.table_view.selectionModel().selectedRows()
        has_sel = len(selected_rows) > 0
        self.btn_preview_live.setEnabled(has_sel)

    def get_selected_solution_dict(self):
        selected_rows = self.table_view.selectionModel().selectedRows()
        if not selected_rows: return None
        proxy_idx = selected_rows[0]
        source_idx = self.proxy_model.mapToSource(proxy_idx)
        res = self.table_model.results[source_idx.row()]

        return {
            "system": self.sys_name,
            "master_0": list(res["m0"]),
            "slave_0": list(res["s0"]),
            "K_fwd": list(res["k_fwd"]),
            "K_bwd": list(res["k_bwd"]),
            "time": {
                "t_max": self.solution_t_max(res),
                "t_start": res["t_start"],
                "dt": res["dt"]
            },
            "sim": {
                "h": res["h"],
                "iterations": self.sp_max_iters.value()
            }
        }

    def apply_live_preview(self):
        sol = self.get_selected_solution_dict()
        if sol: self.solution_preview_requested.emit(sol)

    def save_checked_to_presets(self):
        """Сохраняет все отмеченные чекбоксом [✓] решения в пресеты главного окна."""
        checked_solutions = [r for r in self.table_model.results if r.get("checked", True)]
        if not checked_solutions:
            QMessageBox.warning(self, "Внимание", "Не выбрано ни одного решения чекбоксом [✓]!")
            return

        code, ok = QInputDialog.getText(
            self, "Подтверждение",
            f"Будет добавлено пресетов: {len(checked_solutions):,}.\nВведите 12345 для подтверждения:",
            QLineEdit.EchoMode.Normal
        )
        if not ok:
            return
        if code.strip() != "12345":
            QMessageBox.warning(self, "Неверный код", "Код введён неверно. Сохранение отменено.")
            return

        presets_to_save = {}
        for r in checked_solutions:
            preset_name = f"{self.sys_name} (Kx={r['k_fwd'][0]:.1f}, dt={r['dt']:.2f}, h={r['h']:.4f}, it={r['iters']})"
            presets_to_save[preset_name] = {
                "system": self.sys_name,
                "master_0": list(r["m0"]),
                "slave_0": list(r["s0"]),
                "K_fwd": list(r["k_fwd"]),
                "K_bwd": list(r["k_bwd"]),
                "time": {
                    "t_max": self.solution_t_max(r),
                    "t_start": r["t_start"],
                    "dt": r["dt"]
                },
                "sim": {
                    "h": r["h"],
                    "iterations": self.sp_max_iters.value()
                }
            }

        self.save_presets_requested.emit(presets_to_save)
        QMessageBox.information(
            self, "Успех", f"Успешно добавлено {len(presets_to_save)} решений в пресеты главного окна!"
        )

    def export_solutions_to_csv(self):
        if not self.table_model.results: return
        exports_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports_csv")
        os.makedirs(exports_csv_dir, exist_ok=True)

        default_filename = f"trs_solutions_{self.sys_name.replace(' ', '_')}.csv"
        default_path = os.path.join(exports_csv_dir, default_filename)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить решения", default_path, "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path: return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["ID", "Error", "Iterations", "dt", "h", "t_start", "K_fwd", "K_bwd", "Master_0", "Slave_0"])
                for idx, r in enumerate(self.table_model.results, 1):
                    writer.writerow([
                        idx, f"{r['error']:.6e}", r['iters'], f"{r['dt']:.4f}", f"{r['h']:.5f}",
                        f"{r['t_start']:.3f}", str(r['k_fwd']), str(r['k_bwd']), str(r['m0']), str(r['s0'])
                    ])
            QMessageBox.information(
                self, "Успех", f"Успешно экспортировано {len(self.table_model.results):,} решений в файл:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось сохранить CSV: {e}")

    def update_system(self, current_settings):
        """Переключает диалог на другую систему (вызывается из главного окна)."""
        # подтверждение, если есть что терять
        has_data = bool(self.table_model.results)
        running = self.worker is not None and self.worker.isRunning()
        if has_data or running:
            answer = QMessageBox.question(
                self, "Смена системы",
                "Система в главном окне изменилась.\n"
                "Текущий перебор будет остановлен, а таблица найденных решений очищена. Продолжить?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False

        self.stop_and_wait_worker()

        self.current_settings = current_settings
        self.sys_name = current_settings.get("system", "Sprott B")

        self.setWindowTitle(f"Поиск параметров TRS — [{self.sys_name}]")
        self.lbl_sys_banner.setText(f"Исследуемая динамическая система: <b>{self.sys_name}</b>")

        # сброс результатов и состояния кнопок
        self.table_model.clear()
        self.on_filter_changed()
        self.pbar.setValue(0)
        self.lbl_stats.setText("Система изменена. Настройте сетку и нажмите «Запустить».")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Пауза")
        self.btn_stop.setEnabled(False)
        self.btn_preview_live.setEnabled(False)
        self.btn_save_favorites.setEnabled(False)
        self.btn_export_csv.setEnabled(False)

        # подтянуть начальные значения новой системы
        self.load_from_settings(current_settings)
        return True