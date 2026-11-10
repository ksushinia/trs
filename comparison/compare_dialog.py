"""
comparison/compare_dialog.py
Окно сравнения TRS и Pecora-Carroll с временными графиками и двумя 3D картами ошибок.
"""
import numpy as np
import matplotlib
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QPushButton, QComboBox, QRadioButton, QButtonGroup, QScrollArea,
    QSplitter, QTabWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from config import SYSTEMS_CONFIG
from spinboxes import QDoubleSpinBox, QSpinBox
from fast_trs import pack_system_params, integrate_master, run_fast_portrait_chunk
from comparison.pc_simulation import run_comparison_simulation, run_fast_pc_portrait_chunk
from matplotlib.colors import Normalize, ListedColormap  # добавь ListedColormap


def get_matlab_custom_cmap():
    base_turbo = matplotlib.colormaps['turbo'](np.linspace(0, 1, 450))
    r_over = np.linspace(0.4796, 0.8, 100)
    g_over = np.linspace(0.01583, 0.8, 100)
    b_over = np.linspace(0.01055, 0.8, 100)
    over_colors = np.column_stack([r_over, g_over, b_over, np.ones(100)])
    full_colors = np.vstack([base_turbo, over_colors])
    return ListedColormap(full_colors)


def create_smooth_segments(xs, ys, zs, errs, subs=4):
    n = len(xs)
    if n < 2:
        return np.empty((0, 2, 3)), np.empty(0)
    subs = max(2, int(subs or 4))
    pts = np.column_stack([xs, ys, zs])
    errs = np.clip(np.asarray(errs, dtype=np.float64), -12.0, 2.0)
    kernel = np.array([0.2, 0.6, 0.2])
    errs_smooth = np.convolve(errs, kernel, mode='same')
    d_err = np.abs(errs_smooth[1:] - errs_smooth[:-1])
    is_trans = d_err > 0.03
    n_segs = n - 1
    changes = np.where(is_trans[1:] != is_trans[:-1])[0] + 1
    splits = np.concatenate([[0], changes, [n_segs]])
    segments, colors = [], []
    alphas = np.linspace(0.0, 1.0, subs + 1)
    al = alphas[np.newaxis, :, np.newaxis]
    for s_idx in range(len(splits) - 1):
        start, end = splits[s_idx], splits[s_idx + 1]
        if not is_trans[start]:
            segments.append(pts[start : end + 1])
            colors.append(np.mean(errs_smooth[start : end + 1]))
        else:
            p0 = pts[start : end, np.newaxis, :]
            p1 = pts[start + 1 : end + 1, np.newaxis, :]
            sub_pts = p0 * (1.0 - al) + p1 * al
            e0 = errs_smooth[start : end, np.newaxis]
            e1 = errs_smooth[start + 1 : end + 1, np.newaxis]
            sub_errs = e0 * (1.0 - alphas[np.newaxis, :]) + e1 * alphas[np.newaxis, :]
            segs = np.stack([sub_pts[:, :-1, :], sub_pts[:, 1:, :]], axis=2).reshape(-1, 2, 3)
            cols = 0.5 * (sub_errs[:, :-1] + sub_errs[:, 1:]).reshape(-1)
            for seg, col in zip(segs, cols):
                segments.append(seg)
                colors.append(col)
    return segments, np.asarray(colors, dtype=np.float64)

class BothPortraitsWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)

    def __init__(self, sys_name, params, master_0, slave_0, K, h, dt, iters,
                 t_trans, t_total, decimator, use_shift=True):
        super().__init__()
        self.sys_name = sys_name
        self.params = params
        self.master_0 = master_0
        self.slave_0 = slave_0
        self.K = K
        self.h = h
        self.dt = dt
        self.iters = iters
        self.t_trans = t_trans
        self.t_total = t_total
        self.decimator = max(1, decimator)
        self.use_shift = use_shift

    def run(self):
        sys_id, p_arr = pack_system_params(self.sys_name, self.params)
        TT_iter = int(round(self.t_trans / self.h))
        total_steps = int(round(self.t_total / self.h))
        ns = max(2, int(round(self.dt / self.h)))

        # 1. Траектория Master
        xm_all, ym_all, zm_all, valid = integrate_master(
            sys_id, p_arr, self.master_0[0], self.master_0[1], self.master_0[2],
            self.h, total_steps + ns
        )
        xm = xm_all[TT_iter:valid]
        ym = ym_all[TT_iter:valid]
        zm = zm_all[TT_iter:valid]

        N = len(xm) - ns
        if N <= 0:
            self.finished.emit(None, None)
            return

        start_indices = np.arange(0, N, self.decimator, dtype=np.int64)
        num_evals = len(start_indices)

        xs_starts = xm[start_indices]
        ys_starts = ym[start_indices]
        zs_starts = zm[start_indices]

        kx, ky, kz = self.K
        chunk_size = max(300, num_evals // 15)

        # 2. Расчет TRS (0% -> 50%)
        err10_trs = np.empty(num_evals, dtype=np.float64)
        for i in range(0, num_evals, chunk_size):
            c_idx = start_indices[i : i + chunk_size]
            err10_trs[i : i + chunk_size] = run_fast_portrait_chunk(
                sys_id, p_arr, xm, ym, zm, c_idx, ns, self.iters,
                kx, ky, kz, kx, ky, kz, self.h, self.use_shift,
                self.slave_0[0], self.slave_0[1], self.slave_0[2]
            )
            pct = int((min(i + chunk_size, num_evals) / num_evals) * 50)
            self.progress.emit(pct)

        # 3. Расчет PC (50% -> 100%)
        err10_pc = np.empty(num_evals, dtype=np.float64)
        for i in range(0, num_evals, chunk_size):
            c_idx = start_indices[i : i + chunk_size]
            err10_pc[i : i + chunk_size] = run_fast_pc_portrait_chunk(
                sys_id, p_arr, xm, ym, zm, c_idx, ns,
                kx, ky, kz, self.h, self.use_shift,
                self.slave_0[0], self.slave_0[1], self.slave_0[2]
            )
            pct = 50 + int((min(i + chunk_size, num_evals) / num_evals) * 50)
            self.progress.emit(pct)

        self.progress.emit(100)
        trs_data = (xs_starts, ys_starts, zs_starts, err10_trs)
        pc_data = (xs_starts, ys_starts, zs_starts, err10_pc)
        self.finished.emit(trs_data, pc_data)


class ComparisonDialog(QDialog):
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение: TRS vs Pecora-Carroll")
        self.resize(1450, 880)
        self.current_settings = current_settings
        self.sys_name = "Dadras Momeni"
        self.sim_data = None
        self.portrait_trs_data = None
        self.portrait_pc_data = None
        self._initial_view_3d = (30, -60)

        self.init_ui()
        self.load_defaults_from_settings()
        self.run_simulation()

    def init_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(self.main_splitter)

        # ---------------- ЛЕВАЯ ПАНЕЛЬ: НАСТРОЙКИ ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)

        grp_sys = QGroupBox("Динамическая система")
        sys_l = QVBoxLayout(grp_sys)
        self.combo_sys = QComboBox()
        self.combo_sys.addItems(["Dadras Momeni", "Sprott B", "Lorenz"])
        self.combo_sys.setCurrentText(self.sys_name)
        self.combo_sys.currentIndexChanged.connect(self.on_system_changed)
        sys_l.addWidget(self.combo_sys)
        left_layout.addWidget(grp_sys)

        grp_init = QGroupBox("Начальные условия")
        grid_i = QGridLayout(grp_init)
        grid_i.addWidget(QLabel("Master (x,y,z):"), 0, 0)
        self.sp_mx = QDoubleSpinBox(); self.sp_mx.setRange(-50, 50); self.sp_mx.setValue(0.1)
        self.sp_my = QDoubleSpinBox(); self.sp_my.setRange(-50, 50); self.sp_my.setValue(0.1)
        self.sp_mz = QDoubleSpinBox(); self.sp_mz.setRange(-50, 50); self.sp_mz.setValue(0.1)


        hm = QHBoxLayout();
        hm.setSpacing(3)
        hm.addWidget(self.sp_mx);
        hm.addWidget(self.sp_my);
        hm.addWidget(self.sp_mz)
        grid_i.addLayout(hm, 0, 1)

        grid_i.addWidget(QLabel("Slave (x,y,z):"), 1, 0)
        self.sp_sx = QDoubleSpinBox(); self.sp_sx.setRange(-50, 50); self.sp_sx.setValue(15.0)
        self.sp_sy = QDoubleSpinBox(); self.sp_sy.setRange(-50, 50); self.sp_sy.setValue(10.0)
        self.sp_sz = QDoubleSpinBox(); self.sp_sz.setRange(-50, 50); self.sp_sz.setValue(6.0)
        hs = QHBoxLayout();
        hs.setSpacing(3)
        hs.addWidget(self.sp_sx);
        hs.addWidget(self.sp_sy);
        hs.addWidget(self.sp_sz)
        grid_i.addLayout(hs, 1, 1)

        for sp in (self.sp_mx, self.sp_my, self.sp_mz, self.sp_sx, self.sp_sy, self.sp_sz):
            sp.setMaximumWidth(84)

        left_layout.addWidget(grp_init)

        grp_k = QGroupBox("Коэффициенты связи (K)")
        grid_k = QGridLayout(grp_k)
        grid_k.addWidget(QLabel("K_x:"), 0, 0)
        self.sp_kx = QDoubleSpinBox(); self.sp_kx.setRange(-100, 100); self.sp_kx.setValue(0.0)
        grid_k.addWidget(self.sp_kx, 0, 1)

        grid_k.addWidget(QLabel("K_y:"), 1, 0)
        self.sp_ky = QDoubleSpinBox(); self.sp_ky.setRange(-100, 100); self.sp_ky.setValue(20.0)
        grid_k.addWidget(self.sp_ky, 1, 1)

        grid_k.addWidget(QLabel("K_z:"), 2, 0)
        self.sp_kz = QDoubleSpinBox(); self.sp_kz.setRange(-100, 100); self.sp_kz.setValue(0.0)
        grid_k.addWidget(self.sp_kz, 2, 1)
        left_layout.addWidget(grp_k)

        grp_time = QGroupBox("Время моделирования")
        grid_t = QGridLayout(grp_time)
        grid_t.addWidget(QLabel("T общее:"), 0, 0)
        self.sp_tmax = QDoubleSpinBox(); self.sp_tmax.setRange(10, 1000); self.sp_tmax.setValue(30.0)
        grid_t.addWidget(self.sp_tmax, 0, 1)

        grid_t.addWidget(QLabel("Старт синхр. (t_start):"), 1, 0)
        self.sp_tstart = QDoubleSpinBox(); self.sp_tstart.setRange(0, 1000); self.sp_tstart.setValue(10.5)
        grid_t.addWidget(self.sp_tstart, 1, 1)

        grid_t.addWidget(QLabel("Окно TRS (Δt):"), 2, 0)
        self.sp_dt = QDoubleSpinBox(); self.sp_dt.setRange(0.01, 50); self.sp_dt.setValue(0.30)
        grid_t.addWidget(self.sp_dt, 2, 1)

        grid_t.addWidget(QLabel("Шаг h:"), 3, 0)
        self.sp_h = QDoubleSpinBox(); self.sp_h.setRange(0.001, 0.1); self.sp_h.setValue(0.01); self.sp_h.setDecimals(3)
        grid_t.addWidget(self.sp_h, 3, 1)

        grid_t.addWidget(QLabel("Итерации TRS:"), 4, 0)
        self.sp_iter = QSpinBox(); self.sp_iter.setRange(2, 2000); self.sp_iter.setValue(120)
        grid_t.addWidget(self.sp_iter, 4, 1)
        left_layout.addWidget(grp_time)

        grp_coord = QGroupBox("Отображаемая координата")
        h_coord = QHBoxLayout(grp_coord)
        self.rb_x = QRadioButton("X")
        self.rb_y = QRadioButton("Y"); self.rb_y.setChecked(True)
        self.rb_z = QRadioButton("Z")
        self.coord_group = QButtonGroup(self)
        for rb in (self.rb_x, self.rb_y, self.rb_z):
            self.coord_group.addButton(rb)
            h_coord.addWidget(rb)
            rb.toggled.connect(self.redraw_plots)
        left_layout.addWidget(grp_coord)

        self.btn_run = QPushButton("Пересчитать графики")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 7px; background-color: #e0f2fe;")
        self.btn_run.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.btn_run)

        # Блок параметров для 3D карт ошибок
        grp_3d = QGroupBox("Фазовые портреты")
        grid_3d = QGridLayout(grp_3d)
        grid_3d.addWidget(QLabel("Шаг (каждая N-я):"), 0, 0)
        self.sp_decimator = QSpinBox(); self.sp_decimator.setRange(1, 100); self.sp_decimator.setValue(5)
        grid_3d.addWidget(self.sp_decimator, 0, 1)

        grid_3d.addWidget(QLabel("T общее (3D):"), 1, 0)
        self.sp_t_attr = QDoubleSpinBox(); self.sp_t_attr.setRange(10, 1000); self.sp_t_attr.setValue(100.0)
        grid_3d.addWidget(self.sp_t_attr, 1, 1)

        self.btn_build_3d = QPushButton("Построить оба 3D портрета")
        self.btn_build_3d.setStyleSheet("font-weight: bold; background-color: #dcfce7; padding: 7px;")
        self.btn_build_3d.clicked.connect(self.build_both_portraits)
        grid_3d.addWidget(self.btn_build_3d, 2, 0, 1, 2)

        self.btn_reset_3d = QPushButton("Исходный ракурс 3D")
        self.btn_reset_3d.clicked.connect(self.reset_3d_views)
        grid_3d.addWidget(self.btn_reset_3d, 3, 0, 1, 2)

        self.pbar_3d = QProgressBar()
        self.pbar_3d.setRange(0, 100); self.pbar_3d.setValue(0); self.pbar_3d.setTextVisible(True)
        self.pbar_3d.setStyleSheet("""
            QProgressBar { border: 1px solid #b0b0b0; border-radius: 4px; text-align: center; height: 16px; font-size: 10px; font-weight: bold; }
            QProgressBar::chunk { background-color: #22c55e; }
        """)
        grid_3d.addWidget(self.pbar_3d, 4, 0, 1, 2)
        left_layout.addWidget(grp_3d)

        for sp in (self.sp_kx, self.sp_ky, self.sp_kz, self.sp_tmax, self.sp_tstart,
                   self.sp_dt, self.sp_h, self.sp_iter, self.sp_decimator, self.sp_t_attr):
            sp.setMaximumWidth(100)

        grid_k.setColumnStretch(0, 1)
        grid_t.setColumnStretch(0, 1)
        grid_3d.setColumnStretch(0, 1)

        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(255)
        self.main_splitter.addWidget(scroll)

        # ---------------- ПРАВАЯ ОБЛАСТЬ: ВКЛАДКИ ----------------
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { font-weight: bold; padding: 6px 14px; }")

        # --- ВКЛАДКА 1: ВРЕМЕННЫЕ РЯДЫ (5 ГРАФИКОВ СО СПЛИТТЕРАМИ) ---
        tab_time = QWidget()
        l_time = QHBoxLayout(tab_time)
        l_time.setContentsMargins(0, 0, 0, 0)

        self.h_plots_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.v_left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.fig_pc = Figure(figsize=(5, 3))
        self.canvas_pc = FigureCanvas(self.fig_pc)
        self.v_left_splitter.addWidget(self.canvas_pc)

        self.fig_trs = Figure(figsize=(5, 3))
        self.canvas_trs = FigureCanvas(self.fig_trs)
        self.v_left_splitter.addWidget(self.canvas_trs)

        self.fig_pc_err = Figure(figsize=(5, 3))
        self.canvas_pc_err = FigureCanvas(self.fig_pc_err)
        self.v_left_splitter.addWidget(self.canvas_pc_err)

        self.fig_zigzag = Figure(figsize=(4.5, 4))
        self.canvas_zigzag = FigureCanvas(self.fig_zigzag)
        self.v_right_splitter.addWidget(self.canvas_zigzag)

        self.fig_conv = Figure(figsize=(4.5, 4))
        self.canvas_conv = FigureCanvas(self.fig_conv)
        self.v_right_splitter.addWidget(self.canvas_conv)

        self.h_plots_splitter.addWidget(self.v_left_splitter)
        self.h_plots_splitter.addWidget(self.v_right_splitter)
        l_time.addWidget(self.h_plots_splitter)
        self.tabs.addTab(tab_time, "Временные ряды")

        # --- ВКЛАДКА 2: ДВА 3D ФАЗОВЫХ ПОРТРЕТА ---
        tab_3d = QWidget()
        l_3d = QHBoxLayout(tab_3d)
        l_3d.setContentsMargins(0, 0, 0, 0)

        self.splitter_3d = QSplitter(Qt.Orientation.Horizontal)

        # 3D Холст для PC
        self.fig_3d_pc = Figure(figsize=(5, 5))
        self.canvas_3d_pc = FigureCanvas(self.fig_3d_pc)
        self.splitter_3d.addWidget(self.canvas_3d_pc)

        # 3D Холст для TRS
        self.fig_3d_trs = Figure(figsize=(5, 5))
        self.canvas_3d_trs = FigureCanvas(self.fig_3d_trs)
        self.splitter_3d.addWidget(self.canvas_3d_trs)

        l_3d.addWidget(self.splitter_3d)
        self.tabs.addTab(tab_3d, "3D Фазовые портреты (Карты ошибок)")

        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setSizes([285, 1050])

        # Стилизация всех сплиттеров
        splitter_style = """
            QSplitter::handle { background-color: #cbd5e1; }
            QSplitter::handle:hover { background-color: #3b82f6; }
        """
        for sp in (self.main_splitter, self.h_plots_splitter, self.v_left_splitter, self.v_right_splitter, self.splitter_3d):
            sp.setStyleSheet(splitter_style)
            sp.setHandleWidth(6)
            sp.setChildrenCollapsible(False)
            sp.setOpaqueResize(False)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.h_plots_splitter.setStretchFactor(0, 6)
        self.h_plots_splitter.setStretchFactor(1, 4)

    def load_defaults_from_settings(self):
        self.combo_sys.setCurrentText("Dadras Momeni")
        self.sp_mx.setValue(0.10); self.sp_my.setValue(0.10); self.sp_mz.setValue(0.10)
        self.sp_sx.setValue(15.00); self.sp_sy.setValue(10.00); self.sp_sz.setValue(6.00)
        self.sp_kx.setValue(0.00); self.sp_ky.setValue(20.00); self.sp_kz.setValue(0.00)
        self.sp_tmax.setValue(30.00)
        self.sp_tstart.setValue(10.50)
        self.sp_dt.setValue(0.30)
        self.sp_h.setValue(0.010)
        self.sp_iter.setValue(120)
        self.rb_y.setChecked(True)

    def on_system_changed(self):
        self.sys_name = self.combo_sys.currentText()
        if self.sys_name in SYSTEMS_CONFIG:
            conf = SYSTEMS_CONFIG[self.sys_name]
            m0 = conf.get("master_0", (0.1, 0.1, 0.1))
            self.sp_mx.setValue(m0[0]); self.sp_my.setValue(m0[1]); self.sp_mz.setValue(m0[2])

    def run_simulation(self):
        sys_name = self.combo_sys.currentText()
        params = SYSTEMS_CONFIG[sys_name]["params"]

        self.sim_data = run_comparison_simulation(
            sys_name=sys_name,
            params=params,
            master_0=(self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value()),
            slave_0=(self.sp_sx.value(), self.sp_sy.value(), self.sp_sz.value()),
            K=(self.sp_kx.value(), self.sp_ky.value(), self.sp_kz.value()),
            h=self.sp_h.value(),
            t_max=self.sp_tmax.value(),
            t_start=self.sp_tstart.value(),
            dt=self.sp_dt.value(),
            trs_iters=self.sp_iter.value()
        )
        self.redraw_plots()

    def redraw_plots(self):
        if not self.sim_data:
            return

        c_idx = 0 if self.rb_x.isChecked() else (1 if self.rb_y.isChecked() else 2)
        c_name = "x" if c_idx == 0 else ("y" if c_idx == 1 else "z")

        t = self.sim_data["t"]
        t_sync = self.sim_data["t_start"]
        dt = self.sim_data["dt"]
        m_c = self.sim_data["master"][c_idx]
        pc_c = self.sim_data["pc_slave"][c_idx]
        trs_c = self.sim_data["trs_slave"][c_idx]
        err_pc = self.sim_data["pc_error"]
        runs_zigzag = self.sim_data.get("runs_zigzag", [])
        iter_rms = self.sim_data.get("iter_rms", [])

        # 1. PC
        self.fig_pc.clear()
        ax_pc = self.fig_pc.add_subplot(111)
        ax_pc.plot(t, m_c, label=f"Master {c_name}_m", color="#1f77b4", lw=1.3)
        ax_pc.plot(t, pc_c, label=f"Slave {c_name}_s (PC)", color="#e11d48", lw=1.1, linestyle="--")
        ax_pc.axvline(t_sync, color="black", linestyle=":", lw=1.5, label="Старт синхр.")
        ax_pc.set_ylabel(f"PC ({c_name})", fontsize=8)
        ax_pc.set_title(f"Сравнение методов синхронизации [{self.sys_name}]", fontsize=9, fontweight="bold")
        ax_pc.grid(True, linestyle=":")
        ax_pc.legend(loc="upper right", fontsize=7)
        self.fig_pc.tight_layout()
        self.canvas_pc.draw_idle()

        # 2. TRS
        self.fig_trs.clear()
        ax_trs = self.fig_trs.add_subplot(111)
        ax_trs.set_xlim(0, t[-1])
        ax_trs.plot(t, m_c, label=f"Master {c_name}_m", color="#1f77b4", lw=1.3)
        ax_trs.plot(t, trs_c, label=f"Slave {c_name}_s (TRS)", color="#16a34a", lw=1.2, linestyle="-.")
        ax_trs.axvline(t_sync, color="black", linestyle=":", lw=1.5)
        ax_trs.set_ylabel(f"TRS ({c_name})", fontsize=8)
        ax_trs.grid(True, linestyle=":")
        ax_trs.legend(loc="upper right", fontsize=7)
        self.fig_trs.tight_layout()
        self.canvas_trs.draw_idle()

        # 3. PC Error
        self.fig_pc_err.clear()
        ax_err = self.fig_pc_err.add_subplot(111)
        ax_err.set_xlim(0, t[-1])
        ax_err.plot(t, err_pc, color="#6b21a8", lw=1.3, label="Ошибка PC ||e||")
        ax_err.axvline(t_sync, color="black", linestyle=":", lw=1.5)
        ax_err.set_yscale("log")
        ax_err.set_ylabel("PC Error (log)", fontsize=8)
        ax_err.set_xlabel("Время t, с", fontsize=8)
        ax_err.grid(True, which="both", linestyle=":")
        ax_err.legend(loc="upper right", fontsize=7)
        self.fig_pc_err.tight_layout()
        self.canvas_pc_err.draw_idle()

        # 4. Zigzag
        self.fig_zigzag.clear()
        ax_z = self.fig_zigzag.add_subplot(111)
        cmap = matplotlib.colormaps['turbo']
        total = len(runs_zigzag)
        for i, (name, r_t, r_err) in enumerate(runs_zigzag):
            color = cmap(i / max(1, total - 1))
            ls = '-' if (i % 2 == 0) else '--'
            lbl = name if (i == 0 or i == total - 1) else None
            step = max(1, len(r_t) // 600)
            ax_z.plot(r_t[::step], r_err[::step], color=color, linestyle=ls, lw=1.1, label=lbl)

        ax_z.set_yscale('log')
        ax_z.set_title("Зигзаг сходимости во времени (Turbo)", fontsize=9)
        ax_z.set_xlabel("Время t, с", fontsize=8)
        ax_z.set_ylabel("RMS ||e||", fontsize=8)
        ax_z.set_ylim(1e-16, 1e2)
        ax_z.set_xlim(t_sync, t_sync + dt)
        ax_z.ticklabel_format(axis='x', useOffset=False)
        ax_z.grid(True, which='both', linestyle=':')
        if total > 0:
            ax_z.legend(loc="upper right", fontsize=7)
        self.fig_zigzag.tight_layout()
        self.canvas_zigzag.draw_idle()

        # 5. Iteration RMS
        self.fig_conv.clear()
        ax_c = self.fig_conv.add_subplot(111)
        if iter_rms:
            ax_c.plot(range(1, len(iter_rms) + 1), iter_rms, color='crimson', lw=1.5)
        ax_c.set_yscale('log')
        ax_c.set_title("Сходимость ошибки по итерациям", fontsize=9)
        ax_c.set_xlabel("Номер итерации", fontsize=8)
        ax_c.set_ylabel("Финальная RMS", fontsize=8)
        ax_c.set_ylim(1e-16, 1e2)
        ax_c.grid(True, which='both', linestyle=':')
        self.fig_conv.tight_layout()
        self.canvas_conv.draw_idle()

    # ---------------- МЕТОДЫ ДЛЯ 3D КАРТ ОШИБОК ----------------
    def build_both_portraits(self):
        sys_name = self.combo_sys.currentText()
        params = SYSTEMS_CONFIG[sys_name]["params"]

        self.btn_build_3d.setEnabled(False)
        self.pbar_3d.setValue(0)
        self.tabs.setCurrentIndex(1)  # Авто-переключение на 3D вкладку

        self.worker = BothPortraitsWorker(
            sys_name=sys_name,
            params=params,
            master_0=(self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value()),
            slave_0=(self.sp_sx.value(), self.sp_sy.value(), self.sp_sz.value()),
            K=(self.sp_kx.value(), self.sp_ky.value(), self.sp_kz.value()),
            h=self.sp_h.value(),
            dt=self.sp_dt.value(),
            iters=self.sp_iter.value(),
            t_trans=25.0,
            t_total=self.sp_t_attr.value(),
            decimator=self.sp_decimator.value(),
            use_shift=True
        )
        self.worker.progress.connect(self.pbar_3d.setValue)
        self.worker.finished.connect(self.on_portraits_finished)
        self.worker.start()

    def on_portraits_finished(self, trs_data, pc_data):
        self.portrait_trs_data = trs_data
        self.portrait_pc_data = pc_data
        self.btn_build_3d.setEnabled(True)
        self.pbar_3d.setValue(100)
        self.redraw_3d_portraits()

    def redraw_3d_portraits(self):
        norm = Normalize(vmin=-12.0, vmax=2.0)
        matlab_cmap = get_matlab_custom_cmap()

        # 1. 3D PC Портрет
        self.fig_3d_pc.clear()
        ax_pc = self.fig_3d_pc.add_subplot(111, projection='3d')
        ax_pc.view_init(elev=self._initial_view_3d[0], azim=self._initial_view_3d[1])
        if hasattr(ax_pc, 'computed_zorder'): ax_pc.computed_zorder = False

        if self.portrait_pc_data:
            xs, ys, zs, err10 = self.portrait_pc_data
            segs, cols = create_smooth_segments(xs, ys, zs, err10, subs=4)
            lc = Line3DCollection(segs, cmap=matlab_cmap, norm=norm, linewidth=1.2)
            lc.set_array(cols)
            lc.set_sort_zpos(0)
            ax_pc.add_collection(lc)
            ax_pc.set_xlim(min(xs), max(xs)); ax_pc.set_ylim(min(ys), max(ys)); ax_pc.set_zlim(min(zs), max(zs))
            cb = self.fig_3d_pc.colorbar(lc, ax=ax_pc, shrink=0.7, pad=0.08)
            cb.set_label("$\\log_{10}(||Error||)$", fontsize=8)

        ax_pc.set_title(f"Pecora-Carroll (Карта ошибок)\nΔt = {self.sp_dt.value():.2f} с", fontsize=9, fontweight="bold")
        ax_pc.tick_params(labelsize=7)
        self.fig_3d_pc.tight_layout()
        self.canvas_3d_pc.draw_idle()

        # 2. 3D TRS Портрет
        self.fig_3d_trs.clear()
        ax_trs = self.fig_3d_trs.add_subplot(111, projection='3d')
        ax_trs.view_init(elev=self._initial_view_3d[0], azim=self._initial_view_3d[1])
        if hasattr(ax_trs, 'computed_zorder'): ax_trs.computed_zorder = False

        if self.portrait_trs_data:
            xs, ys, zs, err10 = self.portrait_trs_data
            segs, cols = create_smooth_segments(xs, ys, zs, err10, subs=4)
            lc = Line3DCollection(segs, cmap=matlab_cmap, norm=norm, linewidth=1.2)
            lc.set_array(cols)
            lc.set_sort_zpos(0)
            ax_trs.add_collection(lc)
            ax_trs.set_xlim(min(xs), max(xs)); ax_trs.set_ylim(min(ys), max(ys)); ax_trs.set_zlim(min(zs), max(zs))
            cb = self.fig_3d_trs.colorbar(lc, ax=ax_trs, shrink=0.7, pad=0.08)
            cb.set_label("$\\log_{10}(||Error||)$", fontsize=8)

        ax_trs.set_title(f"TRS (Карта ошибок)\nΔt = {self.sp_dt.value():.2f} с ({self.sp_iter.value()} итераций)", fontsize=9, fontweight="bold")
        ax_trs.tick_params(labelsize=7)
        self.fig_3d_trs.tight_layout()
        self.canvas_3d_trs.draw_idle()

    def reset_3d_views(self):
        for fig, canvas in ((self.fig_3d_pc, self.canvas_3d_pc), (self.fig_3d_trs, self.canvas_3d_trs)):
            if fig.axes:
                fig.axes[0].view_init(elev=self._initial_view_3d[0], azim=self._initial_view_3d[1])
                canvas.draw_idle()