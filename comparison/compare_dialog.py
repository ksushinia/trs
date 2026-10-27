"""
comparison/compare_dialog.py
Окно сравнения TRS и Pecora-Carroll с 5 графиками и интерактивными сплиттерами.
"""
import matplotlib
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QPushButton, QComboBox, QRadioButton, QButtonGroup, QScrollArea,
    QSplitter
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from config import SYSTEMS_CONFIG
from spinboxes import QDoubleSpinBox, QSpinBox
from comparison.pc_simulation import run_comparison_simulation


class ComparisonDialog(QDialog):
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение: TRS vs Pecora-Carroll")
        self.resize(1450, 880)
        self.current_settings = current_settings
        self.sys_name = current_settings.get("system", "Dadras Momeni")
        self.sim_data = None

        self.init_ui()
        self.load_defaults_from_settings()
        self.run_simulation()

    def init_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)

        # Главный сплиттер (панель параметров слева | область графиков справа)
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
        hm = QHBoxLayout(); hm.addWidget(self.sp_mx); hm.addWidget(self.sp_my); hm.addWidget(self.sp_mz)
        grid_i.addLayout(hm, 0, 1)

        grid_i.addWidget(QLabel("Slave (x,y,z):"), 1, 0)
        self.sp_sx = QDoubleSpinBox(); self.sp_sx.setRange(-50, 50); self.sp_sx.setValue(5.0)
        self.sp_sy = QDoubleSpinBox(); self.sp_sy.setRange(-50, 50); self.sp_sy.setValue(5.0)
        self.sp_sz = QDoubleSpinBox(); self.sp_sz.setRange(-50, 50); self.sp_sz.setValue(5.0)
        hs = QHBoxLayout(); hs.addWidget(self.sp_sx); hs.addWidget(self.sp_sy); hs.addWidget(self.sp_sz)
        grid_i.addLayout(hs, 1, 1)
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
        self.sp_tmax = QDoubleSpinBox(); self.sp_tmax.setRange(10, 1000); self.sp_tmax.setValue(25.0)
        grid_t.addWidget(self.sp_tmax, 0, 1)

        grid_t.addWidget(QLabel("Старт синхр. (t_start):"), 1, 0)
        self.sp_tstart = QDoubleSpinBox(); self.sp_tstart.setRange(0, 1000); self.sp_tstart.setValue(5.0)
        grid_t.addWidget(self.sp_tstart, 1, 1)

        grid_t.addWidget(QLabel("Окно TRS (Δt):"), 2, 0)
        self.sp_dt = QDoubleSpinBox(); self.sp_dt.setRange(0.01, 50); self.sp_dt.setValue(0.30)
        grid_t.addWidget(self.sp_dt, 2, 1)

        grid_t.addWidget(QLabel("Шаг h:"), 3, 0)
        self.sp_h = QDoubleSpinBox(); self.sp_h.setRange(0.001, 0.1); self.sp_h.setValue(0.01); self.sp_h.setDecimals(3)
        grid_t.addWidget(self.sp_h, 3, 1)

        grid_t.addWidget(QLabel("Итерации TRS:"), 4, 0)
        self.sp_iter = QSpinBox(); self.sp_iter.setRange(2, 2000); self.sp_iter.setValue(150)
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

        self.btn_run = QPushButton("Пересчитать сравнение")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px; background-color: #e0f2fe;")
        self.btn_run.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.btn_run)

        left_layout.addStretch()

        # Панель параметров в ScrollArea без жестко зафиксированной ширины (можно тянуть)
        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        self.main_splitter.addWidget(scroll)

        # ---------------- ПРАВАЯ ОБЛАСТЬ: 5 ГРАФИКОВ СО СПЛИТТЕРАМИ ----------------
        self.h_plots_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.v_left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Левая колонка (3 графика): PC, TRS, PC Error
        self.fig_pc = Figure(figsize=(5, 3))
        self.canvas_pc = FigureCanvas(self.fig_pc)
        self.v_left_splitter.addWidget(self.canvas_pc)

        self.fig_trs = Figure(figsize=(5, 3))
        self.canvas_trs = FigureCanvas(self.fig_trs)
        self.v_left_splitter.addWidget(self.canvas_trs)

        self.fig_pc_err = Figure(figsize=(5, 3))
        self.canvas_pc_err = FigureCanvas(self.fig_pc_err)
        self.v_left_splitter.addWidget(self.canvas_pc_err)

        # Правая колонка (2 графика): Зигзаг TRS во времени, Сходимость по итерациям
        self.fig_zigzag = Figure(figsize=(4.5, 4))
        self.canvas_zigzag = FigureCanvas(self.fig_zigzag)
        self.v_right_splitter.addWidget(self.canvas_zigzag)

        self.fig_conv = Figure(figsize=(4.5, 4))
        self.canvas_conv = FigureCanvas(self.fig_conv)
        self.v_right_splitter.addWidget(self.canvas_conv)

        self.h_plots_splitter.addWidget(self.v_left_splitter)
        self.h_plots_splitter.addWidget(self.v_right_splitter)
        self.main_splitter.addWidget(self.h_plots_splitter)

        # Стилизация ручек разделения (как в основном интерфейсе)
        splitter_style = """
            QSplitter::handle {
                background-color: #cbd5e1;
            }
            QSplitter::handle:hover {
                background-color: #3b82f6;
            }
        """
        for sp in (self.main_splitter, self.h_plots_splitter, self.v_left_splitter, self.v_right_splitter):
            sp.setStyleSheet(splitter_style)
            sp.setHandleWidth(6)
            sp.setChildrenCollapsible(False)
            sp.setOpaqueResize(False)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.h_plots_splitter.setStretchFactor(0, 6)
        self.h_plots_splitter.setStretchFactor(1, 4)

    def load_defaults_from_settings(self):
        # Твои идеальные подобранные параметры для Dadras Momeni
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

        # ---------------- 1. ЛЕВАЯ КОЛОНКА ----------------
        # 1.1 График PC
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

        # 1.2 График TRS
        self.fig_trs.clear()
        ax_trs = self.fig_trs.add_subplot(111, sharex=ax_pc)
        ax_trs.plot(t, m_c, label=f"Master {c_name}_m", color="#1f77b4", lw=1.3)
        ax_trs.plot(t, trs_c, label=f"Slave {c_name}_s (TRS)", color="#16a34a", lw=1.2, linestyle="-.")
        ax_trs.axvline(t_sync, color="black", linestyle=":", lw=1.5)
        ax_trs.set_ylabel(f"TRS ({c_name})", fontsize=8)
        ax_trs.grid(True, linestyle=":")
        ax_trs.legend(loc="upper right", fontsize=7)
        self.fig_trs.tight_layout()
        self.canvas_trs.draw_idle()

        # 1.3 График ошибки Pecora-Carroll
        self.fig_pc_err.clear()
        ax_err = self.fig_pc_err.add_subplot(111, sharex=ax_pc)
        ax_err.plot(t, err_pc, color="#6b21a8", lw=1.3, label="Ошибка PC ||e||")
        ax_err.axvline(t_sync, color="black", linestyle=":", lw=1.5)
        ax_err.set_yscale("log")
        ax_err.set_ylabel("PC Error (log)", fontsize=8)
        ax_err.set_xlabel("Время t, с", fontsize=8)
        ax_err.grid(True, which="both", linestyle=":")
        ax_err.legend(loc="upper right", fontsize=7)
        self.fig_pc_err.tight_layout()
        self.canvas_pc_err.draw_idle()

        # ---------------- 2. ПРАВАЯ КОЛОНКА ----------------
        # 2.1 Зигзаг TRS
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

        # 2.2 Сходимость по итерациям
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