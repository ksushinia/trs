"""
gui.py
Главное окно интерфейса TRS с автоматическим экспортом графиков без перезаписи.
Запуск: python gui.py
"""
import os
import sys
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QSlider, QPushButton, QCheckBox,
    QGroupBox, QSplitter, QComboBox, QTabWidget, QScrollArea, QProgressBar,
    QMenu, QMessageBox, QAbstractSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QThread, QObject, QEvent
from PyQt6.QtGui import QDesktopServices

from config import DEFAULT_SIM_CONFIG, SYSTEMS_CONFIG
from systems import SYSTEMS_REGISTRY
from trs_solver import solve_trs
from phase_portrait import compute_error_mapped_attractor
from presets import PresetManagerWidget
from param_search import ParamSearchDialog
from spinboxes import QDoubleSpinBox, QSpinBox
from start_point_generator import sample_slave_initial_condition

# Директория для автоматического экспорта графиков
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_PLOTS_DIR = os.path.join(BASE_DIR, "exports_plots")
EXPORTS_CSV_DIR = os.path.join(BASE_DIR, "exports_csv")

def get_matlab_custom_cmap():
    """Палитра из статьи: 450 цветов Turbo + 100 цветов для разлета (0..2)."""
    base_turbo = matplotlib.colormaps['turbo'](np.linspace(0, 1, 450))
    r_over = np.linspace(0.4796, 0.8, 100)
    g_over = np.linspace(0.01583, 0.8, 100)
    b_over = np.linspace(0.01055, 0.8, 100)
    over_colors = np.column_stack([r_over, g_over, b_over, np.ones(100)])
    full_colors = np.vstack([base_turbo, over_colors])
    return ListedColormap(full_colors)


def create_smooth_segments(xs, ys, zs, errs, subs=4):
    """
    Гибридный рендер:
    - Градиентные переходы: точное микро-дробление (subs=4) с мягким ядром [0.2, 0.6, 0.2].
    - Однородные участки: сжаты в длинные непрерывные линии.
    Дает ту самую красивую интерполяцию при легком вращении 3D-сцены.
    """
    n = len(xs)
    if n < 2:
        return np.empty((0, 2, 3)), np.empty(0)

    subs = max(2, int(subs or 4))
    pts = np.column_stack([xs, ys, zs])
    errs = np.clip(np.asarray(errs, dtype=np.float64), -12.0, 2.0)

    # 1. Тот самый мягкий 3-точечный фильтр из удачного варианта
    kernel = np.array([0.2, 0.6, 0.2])
    errs_smooth = np.convolve(errs, kernel, mode='same')

    # 2. Определяем участки изменения цвета
    d_err = np.abs(errs_smooth[1:] - errs_smooth[:-1])
    is_trans = d_err > 0.03
    n_segs = n - 1

    # 3. Находим границы между переходами и плоскими зонами
    changes = np.where(is_trans[1:] != is_trans[:-1])[0] + 1
    splits = np.concatenate([[0], changes, [n_segs]])

    segments = []
    colors = []

    alphas = np.linspace(0.0, 1.0, subs + 1)
    al = alphas[np.newaxis, :, np.newaxis]

    for s_idx in range(len(splits) - 1):
        start = splits[s_idx]
        end = splits[s_idx + 1]

        if not is_trans[start]:
            # Однородная зона: сливаем в 1 непрерывную полилинию (колоссальная экономия FPS)
            # Конец предыдущего участка и начало следующего строго совпадают в pts[end]
            polyline = pts[start : end + 1]
            segments.append(polyline)
            colors.append(np.mean(errs_smooth[start : end + 1]))
        else:
            # Та самая детальная интерполяция: микро-сегменты с непрерывным градиентом
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

class SmoothPlotZoomFilter(QObject):
    """
    Плавный зум тачпадом (нативный пинч двумя пальцами + плавный скролл).
    Масштабирует 2D-графики строго вокруг курсора (включая log-шкалы) и приближает камеру в 3D.
    """
    def __init__(self, canvas, fig, parent=None):
        super().__init__(parent or canvas)
        self.canvas = canvas
        self.fig = fig

    def eventFilter(self, obj, event):
        # 1. Нативный жест сведения/разведения пальцев (Pinch to Zoom)
        if event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                scale = 1.0 - event.value() * 1.5
                self._apply_zoom(scale, event.position())
                return True

        # 2. Плавный двухпальцевый скролл тачпадом или колесо мыши
        elif event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if abs(delta) > 0:
                scale = 0.88 ** (delta / 120.0)
                self._apply_zoom(scale, event.position())
                return True

        return super().eventFilter(obj, event)

    def _apply_zoom(self, scale, pos):
        if not self.fig.axes:
            return

        ax = self.fig.axes[0]

        # Для 3D графика: меняем дистанцию камеры (zoom in/out)
        if hasattr(ax, 'dist') and hasattr(ax, 'elev'):
            ax.dist = max(2.0, min(80.0, ax.dist * scale))
            self.canvas.draw_idle()
            return

        # Для 2D графиков: масштабирование строго в точку под курсором
        dpr = getattr(self.canvas, 'device_pixel_ratio', self.canvas.devicePixelRatio())
        px = pos.x() * dpr
        py = (self.canvas.height() - pos.y()) * dpr

        target_ax = None
        for a in self.fig.axes:
            if a.bbox.contains(px, py):
                target_ax = a
                break
        if target_ax is None:
            target_ax = ax

        try:
            x_cur, y_cur = target_ax.transData.inverted().transform((px, py))
        except Exception:
            x_cur, y_cur = None, None

        # Зумируем ось X
        xlim = target_ax.get_xlim()
        if x_cur is not None and min(xlim) <= x_cur <= max(xlim):
            new_x0 = x_cur - (x_cur - xlim[0]) * scale
            new_x1 = x_cur + (xlim[1] - x_cur) * scale
        else:
            x_mid = 0.5 * (xlim[0] + xlim[1])
            new_x0 = x_mid - (x_mid - xlim[0]) * scale
            new_x1 = x_mid + (xlim[1] - x_mid) * scale
        target_ax.set_xlim(new_x0, new_x1)

        # Зумируем ось Y (с поддержкой логарифмической оси у зигзага и RMS)
        ylim = target_ax.get_ylim()
        if target_ax.get_yscale() == 'log' and ylim[0] > 0 and ylim[1] > 0:
            log_y0 = np.log10(ylim[0])
            log_y1 = np.log10(ylim[1])
            log_yc = np.log10(y_cur) if (y_cur is not None and y_cur > 0) else 0.5 * (log_y0 + log_y1)
            target_ax.set_ylim(10 ** (log_yc - (log_yc - log_y0) * scale),
                               10 ** (log_yc + (log_y1 - log_yc) * scale))
        else:
            if y_cur is not None and min(ylim) <= y_cur <= max(ylim):
                new_y0 = y_cur - (y_cur - ylim[0]) * scale
                new_y1 = y_cur + (ylim[1] - y_cur) * scale
            else:
                y_mid = 0.5 * (ylim[0] + ylim[1])
                new_y0 = y_mid - (y_mid - ylim[0]) * scale
                new_y1 = y_mid + (ylim[1] - y_mid) * scale
            target_ax.set_ylim(new_y0, new_y1)

        self.canvas.draw_idle()

class PortraitWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        # Передаем наш Qt-сигнал как колбэк прогресса
        self.kwargs["progress_callback"] = self.progress.emit
        data = compute_error_mapped_attractor(**self.kwargs)
        self.finished.emit(data)

class NoWheelFilter(QObject): # чтобы дебилка колесико не прокручивало значения
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(
            obj, (QAbstractSpinBox, QComboBox, QSlider)
        ):
            parent = obj.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, event)   # пусть прокрутится ScrollArea
            return True
        return super().eventFilter(obj, event)

class TRSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ax_3d = None
        self.setWindowTitle("Time-Reversed Synchronization Interface")
        self.resize(1340, 880)

        self.calc_timer = QTimer()
        self.calc_timer.setSingleShot(True)
        self.calc_timer.setInterval(250)
        self.calc_timer.timeout.connect(self.calculate_trs)

        self._block_signals = False

        # Базовый начальный ракурс камеры 3D (elev, azim)
        self._initial_view_3d = (30, -60)

        # Кэш расчетов
        self.portrait_data = None
        self.portrait_segments = None
        self.window_xyz = None
        self.t_arr = None
        self.runs_zigzag = []
        self.iter_rms = []

        self.init_ui()
        self.load_system_defaults()
        self.calculate_trs()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # ==============================================================
        # ЛЕВАЯ ПАНЕЛЬ: БЛОКИ УПРАВЛЕНИЯ
        # ==============================================================

        # 0. Блок пресетов
        self.preset_widget = PresetManagerWidget(get_settings_fn=self.get_current_settings_dict)
        self.preset_widget.preset_applied.connect(self.apply_settings_dict)

        # 1. Выбор системы
        grp_sys = QGroupBox("Динамическая система")
        sys_layout = QVBoxLayout(grp_sys)
        self.combo_system = QComboBox()
        available = [name for name in SYSTEMS_REGISTRY if name in SYSTEMS_CONFIG]
        self.combo_system.addItems(available)
        self.combo_system.currentIndexChanged.connect(self.on_system_changed)
        sys_layout.addWidget(self.combo_system)

        # 2. Начальные условия
        grp_init = QGroupBox("Начальные условия (x, y, z)")
        grid_init = QGridLayout(grp_init)
        grid_init.addWidget(QLabel("Master:"), 0, 0)
        self.sp_mx = QDoubleSpinBox();
        self.sp_mx.setRange(-50, 50);
        self.sp_mx.setDecimals(5)
        self.sp_my = QDoubleSpinBox();
        self.sp_my.setRange(-50, 50);
        self.sp_my.setDecimals(5)
        self.sp_mz = QDoubleSpinBox();
        self.sp_mz.setRange(-50, 50);
        self.sp_mz.setDecimals(5)
        h_m = QHBoxLayout()
        h_m.addWidget(self.sp_mx); h_m.addWidget(self.sp_my); h_m.addWidget(self.sp_mz)
        grid_init.addLayout(h_m, 0, 1)

        grid_init.addWidget(QLabel("Slave:"), 1, 0)
        self.sp_sx = QDoubleSpinBox();
        self.sp_sx.setRange(-50, 50);
        self.sp_sx.setDecimals(5)
        self.sp_sy = QDoubleSpinBox();
        self.sp_sy.setRange(-50, 50);
        self.sp_sy.setDecimals(5)
        self.sp_sz = QDoubleSpinBox();
        self.sp_sz.setRange(-50, 50);
        self.sp_sz.setDecimals(5)
        h_s = QHBoxLayout()
        h_s.addWidget(self.sp_sx); h_s.addWidget(self.sp_sy); h_s.addWidget(self.sp_sz)
        grid_init.addLayout(h_s, 1, 1)

        # 3. Коэффициенты связи K (раздельные для прямого и обратного прохода)
        grp_k = QGroupBox("Коэффициенты связи (K)")
        grid_k = QGridLayout(grp_k)
        grid_k.addWidget(QLabel("<b>Вперед</b>"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        grid_k.addWidget(QLabel("<b>Назад</b>"), 0, 2, Qt.AlignmentFlag.AlignCenter)

        grid_k.addWidget(QLabel("K_x:"), 1, 0)
        self.sp_kx_fwd = QDoubleSpinBox();
        self.sp_kx_fwd.setRange(-50.0, 50.0);
        self.sp_kx_fwd.setSingleStep(0.2)
        self.sp_kx_fwd.setValue(DEFAULT_SIM_CONFIG["K_x_fwd"])
        grid_k.addWidget(self.sp_kx_fwd, 1, 1)
        self.sp_kx_bwd = QDoubleSpinBox();
        self.sp_kx_bwd.setRange(-50.0, 50.0);
        self.sp_kx_bwd.setSingleStep(0.2)
        self.sp_kx_bwd.setValue(DEFAULT_SIM_CONFIG["K_x_bwd"])
        grid_k.addWidget(self.sp_kx_bwd, 1, 2)

        grid_k.addWidget(QLabel("K_y:"), 2, 0)
        self.sp_ky_fwd = QDoubleSpinBox();
        self.sp_ky_fwd.setRange(-50.0, 50.0);
        self.sp_ky_fwd.setSingleStep(0.2)
        self.sp_ky_fwd.setValue(DEFAULT_SIM_CONFIG["K_y_fwd"])
        grid_k.addWidget(self.sp_ky_fwd, 2, 1)
        self.sp_ky_bwd = QDoubleSpinBox();
        self.sp_ky_bwd.setRange(-50.0, 50.0);
        self.sp_ky_bwd.setSingleStep(0.2)
        self.sp_ky_bwd.setValue(DEFAULT_SIM_CONFIG["K_y_bwd"])
        grid_k.addWidget(self.sp_ky_bwd, 2, 2)

        grid_k.addWidget(QLabel("K_z:"), 3, 0)
        self.sp_kz_fwd = QDoubleSpinBox();
        self.sp_kz_fwd.setRange(-50.0, 50.0);
        self.sp_kz_fwd.setSingleStep(0.2)
        self.sp_kz_fwd.setValue(DEFAULT_SIM_CONFIG["K_z_fwd"])
        grid_k.addWidget(self.sp_kz_fwd, 3, 1)
        self.sp_kz_bwd = QDoubleSpinBox();
        self.sp_kz_bwd.setRange(-50.0, 50.0);
        self.sp_kz_bwd.setSingleStep(0.2)
        self.sp_kz_bwd.setValue(DEFAULT_SIM_CONFIG["K_z_bwd"])
        grid_k.addWidget(self.sp_kz_bwd, 3, 2)

        # 4. Временная ось и окно TRS
        grp_time = QGroupBox("Окно наблюдения TRS")
        grid_time = QGridLayout(grp_time)

        grid_time.addWidget(QLabel("Время моделирования:"), 0, 0)
        self.sp_tmax = QDoubleSpinBox()
        self.sp_tmax.setRange(10.0, 2000.0)
        self.sp_tmax.setValue(DEFAULT_SIM_CONFIG["t_max"])
        grid_time.addWidget(self.sp_tmax, 0, 1)

        grid_time.addWidget(QLabel("t_start:"), 1, 0)
        self.sp_tstart = QDoubleSpinBox()
        self.sp_tstart.setRange(0.0, DEFAULT_SIM_CONFIG["t_max"])
        self.sp_tstart.setSingleStep(1.0)
        self.sp_tstart.setValue(DEFAULT_SIM_CONFIG["t_start"])
        grid_time.addWidget(self.sp_tstart, 1, 1)

        self.slider_tstart = QSlider(Qt.Orientation.Horizontal)
        self.slider_tstart.setRange(0, 1000)
        grid_time.addWidget(self.slider_tstart, 2, 0, 1, 2)

        grid_time.addWidget(QLabel("Длина (Δt):"), 3, 0)
        self.sp_dt = QDoubleSpinBox()
        self.sp_dt.setRange(0.01, 50.0)
        self.sp_dt.setSingleStep(0.1)
        self.sp_dt.setValue(DEFAULT_SIM_CONFIG["dt"])
        grid_time.addWidget(self.sp_dt, 3, 1)

        self.slider_dt = QSlider(Qt.Orientation.Horizontal)
        self.slider_dt.setRange(1, 500)
        grid_time.addWidget(self.slider_dt, 4, 0, 1, 2)

        grid_time.addWidget(QLabel("t_end:"), 5, 0)
        self.sp_tend = QDoubleSpinBox()
        self.sp_tend.setRange(0.0, DEFAULT_SIM_CONFIG["t_max"] + 100.0)
        self.sp_tend.setValue(DEFAULT_SIM_CONFIG["t_start"] + DEFAULT_SIM_CONFIG["dt"])
        grid_time.addWidget(self.sp_tend, 5, 1)

        # 5. Параметры интегратора
        grp_sim = QGroupBox("Параметры интегрирования")
        grid_sim = QGridLayout(grp_sim)
        grid_sim.addWidget(QLabel("Шаг h:"), 0, 0)
        self.sp_h = QDoubleSpinBox(); self.sp_h.setRange(0.001, 0.1); self.sp_h.setSingleStep(0.005); self.sp_h.setDecimals(3)
        self.sp_h.setValue(DEFAULT_SIM_CONFIG["h"])
        grid_sim.addWidget(self.sp_h, 0, 1)
        grid_sim.addWidget(QLabel("Итерации TRS:"), 1, 0)
        self.sp_iter = QSpinBox(); self.sp_iter.setRange(2, 2000); self.sp_iter.setSingleStep(10)
        self.sp_iter.setValue(DEFAULT_SIM_CONFIG["N_iterations"])
        grid_sim.addWidget(self.sp_iter, 1, 1)

        self.chk_auto = QCheckBox("Авто-пересчет при изменениях")
        self.chk_auto.setChecked(True)
        self.btn_run = QPushButton("Рассчитать TRS")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 7px;")
        self.btn_run.clicked.connect(self.calculate_trs)

        self.btn_search = QPushButton("Подбор параметров")
        self.btn_search.setStyleSheet("font-weight: bold; padding: 7px; background-color: #fef08a;")
        self.btn_search.clicked.connect(self.open_param_search_dialog)

        self.btn_compare = QPushButton("Сравнение с Pecora-Carroll")
        self.btn_compare.setStyleSheet("font-weight: bold; padding: 7px; background-color: #e0e7ff;")
        self.btn_compare.clicked.connect(self.open_comparison_dialog)

        # 6. Блок экспорта графиков (без диалога выбора папки и без перезаписи)
        grp_export = QGroupBox("Экспорт графиков")
        grid_exp = QGridLayout(grp_export)
        grid_exp.setSpacing(5)

        self.btn_save_all = QPushButton("Сохранить все графики")
        self.btn_save_all.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_save_all.clicked.connect(self.save_all_figures)

        self.combo_export_fig = QComboBox()
        self.combo_export_fig.addItems([
            "3D Фазовый портрет",
            "Окно Master x_m(t)",
            "Зигзаг сходимости во времени",
            "Сходимость по итерациям"
        ])

        self.btn_save_one = QPushButton("Сохранить выбранный")
        self.btn_save_one.clicked.connect(self.save_selected_figure)

        self.btn_open_folder = QPushButton("Открыть папку с графиками")
        self.btn_open_folder.clicked.connect(self.open_exports_folder)

        grid_exp.addWidget(self.btn_save_all, 0, 0, 1, 2)
        grid_exp.addWidget(self.combo_export_fig, 1, 0)
        grid_exp.addWidget(self.btn_save_one, 1, 1)
        grid_exp.addWidget(self.btn_open_folder, 2, 0, 1, 2)

        self.lbl_status = QLabel("Готов к расчету")
        self.lbl_status.setStyleSheet("color: gray;")

        # 7. Блок для 3D фазового портрета
        grp_portrait = QGroupBox("Карта ошибок 3D (Фазовый портрет)")
        grid_p = QGridLayout(grp_portrait)
        grid_p.addWidget(QLabel("Каждая N-ная точка:"), 0, 0)
        self.sp_decimator = QSpinBox();
        self.sp_decimator.setRange(1, 200)
        self.sp_decimator.setValue(DEFAULT_SIM_CONFIG.get("decimator", 10))
        grid_p.addWidget(self.sp_decimator, 0, 1)

        grid_p.addWidget(QLabel("T переходный (с):"), 1, 0)
        self.sp_t_trans = QDoubleSpinBox();
        self.sp_t_trans.setRange(0.0, 500.0)
        self.sp_t_trans.setValue(DEFAULT_SIM_CONFIG.get("t_transient", 50.0))
        grid_p.addWidget(self.sp_t_trans, 1, 1)

        grid_p.addWidget(QLabel("T общее (с):"), 2, 0)
        self.sp_t_attr = QDoubleSpinBox();
        self.sp_t_attr.setRange(20.0, 2000.0)
        self.sp_t_attr.setValue(DEFAULT_SIM_CONFIG.get("t_attractor", 250.0))
        grid_p.addWidget(self.sp_t_attr, 2, 1)

        self.chk_matlab_shift = QCheckBox("Для карты 3D: старт Slave = Master + 0.01")
        self.chk_matlab_shift.setChecked(True)
        grid_p.addWidget(self.chk_matlab_shift, 3, 0, 1, 2)

        self.chk_show_portrait = QCheckBox("Отображать 3D карту ошибок")
        self.chk_show_portrait.setChecked(True)
        self.chk_show_portrait.stateChanged.connect(self.redraw_3d_only)
        grid_p.addWidget(self.chk_show_portrait, 4, 0, 1, 2)

        self.chk_show_window = QCheckBox("Подсвечивать окно на аттракторе")
        self.chk_show_window.setChecked(True)
        self.chk_show_window.stateChanged.connect(self.redraw_3d_only)
        grid_p.addWidget(self.chk_show_window, 5, 0, 1, 2)

        self.chk_show_start_marker = QCheckBox("Маркер начала окна (точка)")
        self.chk_show_start_marker.setChecked(False)
        self.chk_show_start_marker.stateChanged.connect(self.redraw_3d_only)
        grid_p.addWidget(self.chk_show_start_marker, 6, 0, 1, 2)

        # Кнопки построения и удаления
        btn_portrait_layout = QHBoxLayout()
        self.btn_build_portrait = QPushButton("Построить портрет")
        self.btn_build_portrait.setStyleSheet("font-weight: bold; background-color: #e0f2fe; padding: 6px;")
        self.btn_build_portrait.clicked.connect(self.build_portrait)

        self.btn_clear_portrait = QPushButton("Убрать портрет")
        self.btn_clear_portrait.setStyleSheet("padding: 6px;")
        self.btn_clear_portrait.clicked.connect(self.clear_portrait)

        self.btn_reset_view = QPushButton("Исходный ракурс")
        self.btn_reset_view.setStyleSheet("padding: 6px;")
        self.btn_reset_view.clicked.connect(self.reset_3d_view)

        btn_portrait_layout.addWidget(self.btn_build_portrait)
        btn_portrait_layout.addWidget(self.btn_clear_portrait)
        btn_portrait_layout.addWidget(self.btn_reset_view)
        grid_p.addLayout(btn_portrait_layout, 7, 0, 1, 2)

        # Зеленый индикатор прогресса (как MATLAB waitbar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #b0b0b0;
                        border-radius: 4px;
                        text-align: center;
                        font-size: 11px;
                        font-weight: bold;
                        color: #0f172a;
                        background-color: #f1f5f9;
                        height: 18px;
                    }
                    QProgressBar::chunk {
                        background-color: #22c55e;
                        border-radius: 3px;
                    }
                """)
        grid_p.addWidget(self.progress_bar, 8, 0, 1, 2)

        # Вкладки
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { font-weight: bold; padding: 6px 14px; }")

        tab_trs = QWidget()
        layout_trs = QVBoxLayout(tab_trs)
        layout_trs.setSpacing(6)
        layout_trs.addWidget(self.preset_widget)
        layout_trs.addWidget(grp_sys)
        layout_trs.addWidget(grp_init)
        layout_trs.addWidget(grp_k)
        layout_trs.addWidget(grp_time)
        layout_trs.addWidget(grp_sim)
        layout_trs.addWidget(self.chk_auto)
        layout_trs.addWidget(self.btn_run)
        layout_trs.addWidget(self.btn_search)
        layout_trs.addWidget(self.btn_compare)
        layout_trs.addWidget(grp_export)
        layout_trs.addWidget(self.lbl_status)
        layout_trs.addStretch()
        tabs.addTab(tab_trs, "TRS Анализ")

        tab_3d = QWidget()
        layout_3d = QVBoxLayout(tab_3d)
        layout_3d.setSpacing(10)
        layout_3d.addWidget(grp_portrait)
        layout_3d.addStretch()
        tabs.addTab(tab_3d, "3D Аттрактор")

        scroll_area = QScrollArea()
        scroll_area.setWidget(tabs)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setMinimumWidth(370)

        main_splitter.addWidget(scroll_area)

        # ==============================================================
        # ПРАВАЯ ПАНЕЛЬ: ГРАФИКИ
        # ==============================================================
        self.v_grid_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_grid_splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        self.h_top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_top_splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        # 1. 3D Фазовый портрет
        self.fig_3d = Figure(figsize=(4.5, 4))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.h_top_splitter.addWidget(self.canvas_3d)

        # 2. Окно Master x_m(t)
        self.fig_win = Figure(figsize=(4.5, 4))
        self.canvas_win = FigureCanvas(self.fig_win)
        self.h_top_splitter.addWidget(self.canvas_win)

        self.h_bot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_bot_splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        # 3. Зигзаг сходимости TRS
        self.fig_zigzag = Figure(figsize=(4.5, 4))
        self.canvas_zigzag = FigureCanvas(self.fig_zigzag)
        self.h_bot_splitter.addWidget(self.canvas_zigzag)

        # 4. Сходимость по итерациям
        self.fig_conv = Figure(figsize=(4.5, 4))
        self.canvas_conv = FigureCanvas(self.fig_conv)
        self.h_bot_splitter.addWidget(self.canvas_conv)

        self.v_grid_splitter.addWidget(self.h_top_splitter)
        self.v_grid_splitter.addWidget(self.h_bot_splitter)

        # СТАЛО:
        main_splitter.addWidget(self.v_grid_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        # 1. НАСТРОЙКА РАЗДЕЛИТЕЛЕЙ (вставляем сюда)
        splitter_style = """
                    QSplitter::handle {
                        background-color: #cbd5e1;
                    }
                    QSplitter::handle:hover {
                        background-color: #3b82f6;
                    }
                """
        for sp in (main_splitter, self.v_grid_splitter, self.h_top_splitter, self.h_bot_splitter):
            sp.setStyleSheet(splitter_style)
            sp.setHandleWidth(6)  # Удобнее ухватиться мышкой
            sp.setChildrenCollapsible(False)  # Окна не схлопываются в ноль
            sp.setOpaqueResize(False)  # Мгновенное и плавное перетаскивание

        # Настройка контекстного меню (ПКМ)
        self.setup_canvas_context_menu(self.canvas_3d, self.fig_3d, "3D Фазовый портрет")
        self.setup_canvas_context_menu(self.canvas_win, self.fig_win, "Окно Master x_m(t)")
        self.setup_canvas_context_menu(self.canvas_zigzag, self.fig_zigzag, "Зигзаг сходимости во времени")
        self.setup_canvas_context_menu(self.canvas_conv, self.fig_conv, "Сходимость по итерациям")

        self.connect_signals()
        self.sync_sliders()

        # 2. ПОДКЛЮЧЕНИЕ ЗУМА ТАЧПАДОМ (вставляем в самый конец init_ui)
        for cvs, fig in ((self.canvas_3d, self.fig_3d),
                         (self.canvas_win, self.fig_win),
                         (self.canvas_zigzag, self.fig_zigzag),
                         (self.canvas_conv, self.fig_conv)):
            cvs.installEventFilter(SmoothPlotZoomFilter(cvs, fig, self))

    def get_plot_mapping(self):
        """Словарь привязки графиков и их названий."""
        return {
            "3D Фазовый портрет": (self.fig_3d, "phase_portrait_3d"),
            "Окно Master x_m(t)": (self.fig_win, "master_window"),
            "Зигзаг сходимости во времени": (self.fig_zigzag, "zigzag_convergence"),
            "Сходимость по итерациям": (self.fig_conv, "rms_convergence")
        }

    def on_generate_random_slave(self):
        """Генерирует 1 случайную точку Slave в окрестности выбранной базы."""
        sys_name = self.combo_system.currentText()
        if sys_name not in SYSTEMS_CONFIG: return
        params = SYSTEMS_CONFIG[sys_name]["params"]
        m0 = (self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value())
        h = self.sp_h.value()
        t_start = self.sp_tstart.value()
        radius = self.sp_slave_gen_radius.value()
        mode = self.combo_slave_gen_mode.currentData()

        (sx, sy, sz), (bx, by, bz) = sample_slave_initial_condition(
            mode=mode, sys_name=sys_name, params_dict=params,
            master_0=m0, h=h, t_start=t_start, radius=radius
        )

        self._block_signals = True
        self.sp_sx.setValue(sx);
        self.sp_sy.setValue(sy);
        self.sp_sz.setValue(sz)
        self._block_signals = False

        m_name = "Master(t_start)" if mode == "near_master" else "Центр"
        self.lbl_slave_gen_info.setText(
            f"База ({m_name}): ({bx:.4f}, {by:.4f}, {bz:.4f})<br>"
            f"<b>Slave:</b> ({sx:.5f}, {sy:.5f}, {sz:.5f})"
        )
        self.on_param_changed()

    def on_set_exact_base_slave(self):
        """Устанавливает Slave строго в координаты базы (радиус = 0)."""
        sys_name = self.combo_system.currentText()
        if sys_name not in SYSTEMS_CONFIG: return
        params = SYSTEMS_CONFIG[sys_name]["params"]
        m0 = (self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value())
        h = self.sp_h.value()
        t_start = self.sp_tstart.value()
        mode = self.combo_slave_gen_mode.currentData()

        (sx, sy, sz), (bx, by, bz) = sample_slave_initial_condition(
            mode=mode, sys_name=sys_name, params_dict=params,
            master_0=m0, h=h, t_start=t_start, radius=0.0
        )

        self._block_signals = True
        self.sp_sx.setValue(sx);
        self.sp_sy.setValue(sy);
        self.sp_sz.setValue(sz)
        self._block_signals = False

        m_name = "Master(t_start)" if mode == "near_master" else "Центр"
        self.lbl_slave_gen_info.setText(
            f"Точная база ({m_name}): ({bx:.5f}, {by:.5f}, {bz:.5f})"
        )
        self.on_param_changed()

    def save_all_figures(self):
        """Сохраняет все 4 графика в отдельную подпапку с датой и временем."""
        os.makedirs(EXPORTS_PLOTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = os.path.join(EXPORTS_PLOTS_DIR, f"run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)

        plots = self.get_plot_mapping()
        for idx, (title, (fig, filename)) in enumerate(plots.items(), 1):
            file_path = os.path.join(run_folder, f"{idx}_{filename}.png")
            fig.savefig(file_path, dpi=300, bbox_inches="tight")

        self.lbl_status.setText(f"Все 4 графика сохранены в run_{timestamp}")
        QMessageBox.information(
            self, "Успех",
            f"Все 4 графика сохранены в:\nexports_plots/run_{timestamp}\n(DPI = 300, без перезаписи)"
        )

    def save_selected_figure(self):
        """Сохраняет конкретный выбранный график с меткой времени."""
        os.makedirs(EXPORTS_PLOTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        title = self.combo_export_fig.currentText()
        plots = self.get_plot_mapping()

        if title in plots:
            fig, filename = plots[title]
            file_name = f"{filename}_{timestamp}.png"
            full_path = os.path.join(EXPORTS_PLOTS_DIR, file_name)
            fig.savefig(full_path, dpi=300, bbox_inches="tight")

            self.lbl_status.setText(f"График сохранен: exports_plots/{file_name}")
            QMessageBox.information(
                self, "Успех",
                f"График «{title}» сохранен:\nexports_plots/{file_name}\n(DPI = 300)"
            )

    def open_exports_folder(self):
        """Открывает папку с сохраненными графиками в Проводнике."""
        os.makedirs(EXPORTS_PLOTS_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(EXPORTS_PLOTS_DIR))

    def _quick_save_figure(self, fig, plot_title):
        os.makedirs(EXPORTS_PLOTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots = self.get_plot_mapping()
        filename = plots.get(plot_title, (None, "plot"))[1]
        file_name = f"{filename}_{timestamp}.png"
        full_path = os.path.join(EXPORTS_PLOTS_DIR, file_name)
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        self.lbl_status.setText(f"График сохранен: exports_plots/{file_name}")
        QMessageBox.information(self, "Успех", f"График «{plot_title}» сохранен в:\nexports_plots/{file_name}")

    def setup_canvas_context_menu(self, canvas, fig, plot_title):
        """Включает контекстное меню по правому клику мыши на холст."""
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(
            lambda pos: self.show_canvas_context_menu(canvas, fig, plot_title, pos)
        )

    def show_canvas_context_menu(self, canvas, fig, plot_title, pos):
        """Меню ПКМ: быстрый экспорт графика или данных."""
        menu = QMenu(self)

        action_save_img = menu.addAction("Сохранить этот график (PNG)")
        action_save_img.triggered.connect(lambda: self._quick_save_figure(fig, plot_title))

        # Пункты экспорта в CSV в зависимости от выбранного графика
        if plot_title == "3D Фазовый портрет":
            action_save_csv = menu.addAction("Экспортировать точки в CSV")
            action_save_csv.triggered.connect(self.export_portrait_csv)
        elif plot_title == "Сходимость по итерациям":
            action_save_csv = menu.addAction("Экспортировать сходимость в CSV")
            action_save_csv.triggered.connect(self.export_iter_rms_csv)
        elif plot_title == "Зигзаг сходимости во времени":
            action_save_csv = menu.addAction("Экспортировать зигзаг в CSV")
            action_save_csv.triggered.connect(self.export_zigzag_csv)

        menu.exec(canvas.mapToGlobal(pos))

    def export_portrait_csv(self):
        """Экспорт точек 3D фазового портрета (x, y, z, error) в CSV-файл."""
        if self.portrait_data is None:
            QMessageBox.warning(
                self, "Внимание",
                "Фазовый портрет еще не построен!\nСначала нажмите «Построить портрет» во вкладке «3D Аттрактор»."
            )
            return

        os.makedirs(EXPORTS_CSV_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sys_name = self.combo_system.currentText().replace(" ", "_")
        file_name = f"phase_portrait_{sys_name}_{timestamp}.csv"
        full_path = os.path.join(EXPORTS_CSV_DIR, file_name)

        xs, ys, zs, err10 = self.portrait_data
        # Восстанавливаем реальное значение ошибки из log10
        errors = 10.0 ** np.asarray(err10, dtype=np.float64)

        try:
            with open(full_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                # Заголовок из ровно 4 столбцов
                writer.writerow(["x", "y", "z", "error"])
                for x, y, z, err in zip(xs, ys, zs, errors):
                    writer.writerow([f"{x:.8e}", f"{y:.8e}", f"{z:.8e}", f"{err:.8e}"])

            self.lbl_status.setText(f"Данные ФП сохранены: exports_csv/{file_name}")
            QMessageBox.information(
                self, "Успех",
                f"Точки фазового портрета ({len(xs):,} шт.) успешно сохранены в:\nexports_csv/{file_name}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось записать CSV: {e}")

    def connect_signals(self):
        for sp in (self.sp_mx, self.sp_my, self.sp_mz, self.sp_sx, self.sp_sy, self.sp_sz,
                   self.sp_kx_fwd, self.sp_ky_fwd, self.sp_kz_fwd,
                   self.sp_kx_bwd, self.sp_ky_bwd, self.sp_kz_bwd,
                   self.sp_h, self.sp_iter):
            sp.valueChanged.connect(self.on_param_changed)

        self.chk_matlab_shift.stateChanged.connect(self.on_param_changed)

        self.sp_tmax.valueChanged.connect(self.on_tmax_changed)
        self.sp_tstart.valueChanged.connect(self.on_tstart_changed)
        self.sp_dt.valueChanged.connect(self.on_dt_changed)
        self.sp_tend.valueChanged.connect(self.on_tend_changed)

        self.slider_tstart.valueChanged.connect(self.on_slider_tstart_moved)
        self.slider_dt.valueChanged.connect(self.on_slider_dt_moved)

    def export_iter_rms_csv(self):
        """Экспорт сходимости ошибки по итерациям (iteration; error)."""
        if not self.iter_rms:
            QMessageBox.warning(self, "Внимание", "Расчет еще не выполнен!\nСначала нажмите «Рассчитать TRS».")
            return

        os.makedirs(EXPORTS_CSV_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sys_name = self.combo_system.currentText().replace(" ", "_")
        file_name = f"convergence_rms_{sys_name}_{timestamp}.csv"
        full_path = os.path.join(EXPORTS_CSV_DIR, file_name)
        try:
            with open(full_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["iteration", "error"])
                for it_idx, err in enumerate(self.iter_rms, 1):
                    writer.writerow([it_idx, f"{err:.8e}"])

            self.lbl_status.setText(f"Сходимость сохранена: exports_csv/{file_name}")
            QMessageBox.information(
                self, "Успех",
                f"Сходимость по итерациям ({len(self.iter_rms):,} шт.) сохранена в:\nexports_csv/{file_name}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось записать CSV: {e}")

    def export_zigzag_csv(self):
        """Полный экспорт зигзага во времени (iteration; direction; t; error)."""
        if not self.runs_zigzag:
            QMessageBox.warning(self, "Внимание", "Расчет еще не выполнен!\nСначала нажмите «Рассчитать TRS».")
            return

        os.makedirs(EXPORTS_CSV_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sys_name = self.combo_system.currentText().replace(" ", "_")
        file_name = f"zigzag_{sys_name}_{timestamp}.csv"
        full_path = os.path.join(EXPORTS_CSV_DIR, file_name)
        total_pts = 0
        try:
            with open(full_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["iteration", "direction", "t", "error"])
                for it_idx, (name, r_t, r_err) in enumerate(self.runs_zigzag, 1):
                    # name имеет вид "Iter 1: Fwd" или "Iter 2: Bwd"
                    direction = "Fwd" if "Fwd" in name else "Bwd"
                    for t_val, err_val in zip(r_t, r_err):
                        writer.writerow([it_idx, direction, f"{t_val:.6f}", f"{err_val:.8e}"])
                        total_pts += 1

            self.lbl_status.setText(f"Зигзаг сохранен: exports_csv/{file_name}")
            QMessageBox.information(
                self, "Успех",
                f"Зигзаг ({len(self.runs_zigzag):,} итераций, {total_pts:,} точек) сохранен в:\nexports_csv/{file_name}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось записать CSV: {e}")

    def get_current_settings_dict(self):
        """Сохраняет полное состояние всех параметров симуляции."""
        return {
            "system": self.combo_system.currentText(),
            "master_0": [self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value()],
            "slave_0": [self.sp_sx.value(), self.sp_sy.value(), self.sp_sz.value()],
            "K": [self.sp_kx_fwd.value(), self.sp_ky_fwd.value(), self.sp_kz_fwd.value()],
            "K_fwd": [self.sp_kx_fwd.value(), self.sp_ky_fwd.value(), self.sp_kz_fwd.value()],
            "K_bwd": [self.sp_kx_bwd.value(), self.sp_ky_bwd.value(), self.sp_kz_bwd.value()],
            "time": {
                "t_max": self.sp_tmax.value(),
                "t_start": self.sp_tstart.value(),
                "dt": self.sp_dt.value(),
            },
            "sim": {
                "h": self.sp_h.value(),
                "iterations": self.sp_iter.value(),
            },
            "portrait": {
                "decimator": self.sp_decimator.value(),
                "t_transient": self.sp_t_trans.value(),
                "t_attractor": self.sp_t_attr.value(),
                "matlab_shift": self.chk_matlab_shift.isChecked()
            }
        }

    def apply_settings_dict(self, p):
        """Гарантированно восстанавливает ВСЕ параметры пресета без сброса."""
        self._block_signals = True

        # 1. Восстанавливаем систему
        sys_name = p.get("system", "Sprott B")
        sys_idx = self.combo_system.findText(sys_name)
        if sys_idx >= 0:
            self.combo_system.blockSignals(True)
            self.combo_system.setCurrentIndex(sys_idx)
            self.combo_system.blockSignals(False)

        # 2. Восстанавливаем начальные условия
        m0 = p.get("master_0", [0.1, 0.1, 0.1])
        s0 = p.get("slave_0", [1.0, 1.0, 1.0])
        self.sp_mx.setValue(m0[0]); self.sp_my.setValue(m0[1]); self.sp_mz.setValue(m0[2])
        self.sp_sx.setValue(s0[0]); self.sp_sy.setValue(s0[1]); self.sp_sz.setValue(s0[2])

        # 3. Восстанавливаем K (поддержка старых и новых пресетов)
        k_legacy = p.get("K", [0.0, 0.0, 0.0])
        kf = p.get("K_fwd", k_legacy)
        kb = p.get("K_bwd", k_legacy)
        self.sp_kx_fwd.setValue(kf[0]); self.sp_ky_fwd.setValue(kf[1]); self.sp_kz_fwd.setValue(kf[2])
        self.sp_kx_bwd.setValue(kb[0]); self.sp_ky_bwd.setValue(kb[1]); self.sp_kz_bwd.setValue(kb[2])

        # 4. Восстанавливаем время и длину окна ДТ (то, что потерялось!)
        tm = p.get("time", {})
        dt_val = tm.get("dt", 1.0)
        tstart_val = tm.get("t_start", 245.0)
        self.sp_tmax.setValue(tm.get("t_max", 300.0))
        self.sp_tstart.setValue(tstart_val)
        self.sp_dt.setValue(dt_val)
        self.sp_tend.setValue(tstart_val + dt_val)

        # 5. Восстанавливаем шаг h и итерации
        sim = p.get("sim", {})
        self.sp_h.setValue(sim.get("h", 0.01))
        self.sp_iter.setValue(sim.get("iterations", 300))

        # 6. Портрет
        port = p.get("portrait", {})
        self.sp_decimator.setValue(port.get("decimator", 1))
        self.sp_t_trans.setValue(port.get("t_transient", 40.0))
        self.sp_t_attr.setValue(port.get("t_attractor", 140.0))
        self.chk_matlab_shift.setChecked(port.get("matlab_shift", True))

        self._block_signals = False

        # Синхронизируем ползунки и запускаем пересчет
        self.sync_sliders()
        self.calculate_trs()
        self.lbl_status.setText("Параметры пресета успешно применены!")
        self._sync_search_dialog()

    def load_system_defaults(self):
        sys_name = self.combo_system.currentText()
        if sys_name not in SYSTEMS_CONFIG:
            return

        conf = SYSTEMS_CONFIG[sys_name]
        m0 = conf.get("master_0", (0.1, 0.1, 0.1))
        s0 = conf.get("slave_0", (5.0, 2.0, 0.0))

        self._block_signals = True
        self.sp_mx.setValue(m0[0]); self.sp_my.setValue(m0[1]); self.sp_mz.setValue(m0[2])
        self.sp_sx.setValue(s0[0]); self.sp_sy.setValue(s0[1]); self.sp_sz.setValue(s0[2])
        self._block_signals = False

    def on_system_changed(self):
        self.portrait_data = None
        self.portrait_segments = None
        self.load_system_defaults()
        self.calculate_trs()
        self._sync_search_dialog()

    def on_tmax_changed(self, val):
        self.sp_tstart.setMaximum(val)
        self.sp_tend.setMaximum(val + self.sp_dt.value())
        self.sync_sliders()
        self.on_param_changed()

    def on_tstart_changed(self, val):
        if self._block_signals: return
        self._block_signals = True
        self.sp_tend.setValue(val + self.sp_dt.value())
        self._block_signals = False
        self.sync_sliders()
        self.on_param_changed()

    def on_dt_changed(self, val):
        if self._block_signals: return
        self._block_signals = True
        self.sp_tend.setValue(self.sp_tstart.value() + val)
        self._block_signals = False
        self.sync_sliders()
        self.on_param_changed()

    def on_tend_changed(self, val):
        if self._block_signals: return
        self._block_signals = True
        t_start = self.sp_tstart.value()
        if val > t_start:
            self.sp_dt.setValue(val - t_start)
        else:
            self.sp_tstart.setValue(max(0.0, val - 0.1))
            self.sp_dt.setValue(0.1)
        self._block_signals = False
        self.sync_sliders()
        self.on_param_changed()

    def on_slider_tstart_moved(self, slider_val):
        if self._block_signals: return
        t_max = max(1.0, self.sp_tmax.value())
        val = (slider_val / 1000.0) * t_max
        self._block_signals = True
        self.sp_tstart.setValue(val)
        self.sp_tend.setValue(val + self.sp_dt.value())
        self._block_signals = False
        self.on_param_changed()

    def on_slider_dt_moved(self, slider_val):
        if self._block_signals: return
        dt = max(0.01, slider_val / 10.0)
        self._block_signals = True
        self.sp_dt.setValue(dt)
        self.sp_tend.setValue(self.sp_tstart.value() + dt)
        self._block_signals = False
        self.on_param_changed()

    def sync_sliders(self):
        was_blocked = self._block_signals
        self._block_signals = True
        t_max = max(1.0, self.sp_tmax.value())
        t_start = self.sp_tstart.value()
        dt = self.sp_dt.value()

        self.slider_tstart.setValue(int(min(1000, max(0, (t_start / t_max) * 1000))))
        self.slider_dt.setValue(int(min(500, max(1, round(dt * 10.0)))))
        self._block_signals = was_blocked

    def on_param_changed(self):
        if not self._block_signals and self.chk_auto.isChecked():
            self.calc_timer.start()

    def clear_portrait(self):
        """Полная очистка фазового портрета из памяти и с экрана."""
        self.portrait_data = None
        self.portrait_segments = None
        self.progress_bar.setValue(0)
        self.redraw_3d_only()
        self.lbl_status.setText("Фазовый портрет убран.")

    def reset_3d_view(self):
        """Возвращает ракурс (30, -60) и дистанцию камеры 3D к исходным значениям."""
        if hasattr(self, 'ax_3d') and self.ax_3d is not None:
            init_v = getattr(self, '_initial_view_3d', (30, -60))
            self.ax_3d.view_init(elev=init_v[0], azim=init_v[1])
            if hasattr(self.ax_3d, 'dist'):
                self.ax_3d.dist = 10  # сброс приближения камеры
            self.canvas_3d.draw_idle()

    def build_portrait(self):
        sys_name = self.combo_system.currentText()
        if sys_name not in SYSTEMS_CONFIG or sys_name not in SYSTEMS_REGISTRY:
            return

        sys_funcs = SYSTEMS_REGISTRY[sys_name]
        sys_conf = SYSTEMS_CONFIG[sys_name]

        self.btn_build_portrait.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Расчет карты ошибок 3D в фоне...")

        worker_kwargs = dict(
            sys_name=sys_name,
            step_forward=sys_funcs["forward"],
            step_backward=sys_funcs["backward"],
            params=sys_conf["params"],
            master_0=(self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value()),
            slave_0=(self.sp_sx.value(), self.sp_sy.value(), self.sp_sz.value()),
            use_matlab_shift=self.chk_matlab_shift.isChecked(),
            h=self.sp_h.value(),
            K_x_fwd=self.sp_kx_fwd.value(),
            K_y_fwd=self.sp_ky_fwd.value(),
            K_z_fwd=self.sp_kz_fwd.value(),
            K_x_bwd=self.sp_kx_bwd.value(),
            K_y_bwd=self.sp_ky_bwd.value(),
            K_z_bwd=self.sp_kz_bwd.value(),
            t_transient=self.sp_t_trans.value(),
            t_total=self.sp_t_attr.value(),
            window_dt=self.sp_dt.value(),
            N_iterations=self.sp_iter.value(),
            decimator=self.sp_decimator.value()
        )

        self._portrait_thread = PortraitWorker(worker_kwargs)
        self._portrait_thread.progress.connect(self.progress_bar.setValue)
        self._portrait_thread.finished.connect(self._on_portrait_finished)
        self._portrait_thread.start()

    def _on_portrait_finished(self, portrait_data):
        self.portrait_data = portrait_data
        if self.portrait_data is not None:
            xs, ys, zs, err10 = self.portrait_data
            # Передаем subs=6 (вместо subs=2)
            self.portrait_segments = create_smooth_segments(xs, ys, zs, err10, subs=6) if len(xs) > 1 else None
        else:
            self.portrait_segments = None

        self.chk_show_portrait.setChecked(True)
        self.redraw_3d_only()
        self.progress_bar.setValue(100)
        self.btn_build_portrait.setEnabled(True)
        self.lbl_status.setText("Карта ошибок построена!")

    def calculate_trs(self):
        sys_name = self.combo_system.currentText()
        if sys_name not in SYSTEMS_CONFIG or sys_name not in SYSTEMS_REGISTRY:
            return

        sys_funcs = SYSTEMS_REGISTRY[sys_name]
        sys_conf = SYSTEMS_CONFIG[sys_name]

        t_start = self.sp_tstart.value()
        t_end = self.sp_tend.value()
        h = self.sp_h.value()
        n_iter = self.sp_iter.value()

        m0 = (self.sp_mx.value(), self.sp_my.value(), self.sp_mz.value())
        s0 = (self.sp_sx.value(), self.sp_sy.value(), self.sp_sz.value())

        self.t_arr, self.window_xyz, self.runs_zigzag, self.iter_rms = solve_trs(
            step_forward=sys_funcs["forward"],
            step_backward=sys_funcs["backward"],
            params=sys_conf["params"],
            master_0=m0,
            slave_0=s0,
            h=h,
            t_start=t_start,
            t_end=t_end,
            N_iterations=n_iter,
            K_x_fwd=self.sp_kx_fwd.value(),
            K_y_fwd=self.sp_ky_fwd.value(),
            K_z_fwd=self.sp_kz_fwd.value(),
            K_x_bwd=self.sp_kx_bwd.value(),
            K_y_bwd=self.sp_ky_bwd.value(),
            K_z_bwd=self.sp_kz_bwd.value()
        )

        self.redraw_all()
        final_err = self.iter_rms[-1] if self.iter_rms else 0.0
        self.lbl_status.setText(f"Готово. Финальная ошибка: {final_err:.2e}")

    def redraw_3d_only(self):
        # 1. Запоминаем текущий ракурс камеры перед перерисовкой (если сцена уже была)
        saved_view = None
        if hasattr(self, 'ax_3d') and self.ax_3d is not None:
            saved_view = (self.ax_3d.elev, self.ax_3d.azim)

        self.fig_3d.clear()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        ax_3d = self.ax_3d

        # 2. Восстанавливаем ракурс, чтобы камера не прыгала при переключении галочек
        if saved_view is not None:
            ax_3d.view_init(elev=saved_view[0], azim=saved_view[1])
        else:
            ax_3d.view_init(elev=self._initial_view_3d[0], azim=self._initial_view_3d[1])

        # Отключаем глобальный пересчет z-order сцены при вращении (дает прирост FPS)
        if hasattr(ax_3d, 'computed_zorder'):
            ax_3d.computed_zorder = False

        if self.chk_show_portrait.isChecked() and self.portrait_segments is not None:
            smooth_segs, smooth_errs = self.portrait_segments

            norm = Normalize(vmin=-12.0, vmax=2.0)
            matlab_cmap = get_matlab_custom_cmap()

            # linewidth=1.2 для быстрой растризации
            lc = Line3DCollection(smooth_segs, cmap=matlab_cmap, norm=norm, linewidth=1.2)
            lc.set_array(smooth_errs)

            # Отключаем поштучную сортировку 16000 отрезков по Z при движении мыши:
            lc.set_sort_zpos(0)

            ax_3d.add_collection(lc)

            xs, ys, zs, _ = self.portrait_data
            ax_3d.set_xlim(min(xs), max(xs))
            ax_3d.set_ylim(min(ys), max(ys))
            ax_3d.set_zlim(min(zs), max(zs))

            cbar = self.fig_3d.colorbar(lc, ax=ax_3d, shrink=0.7, pad=0.1)
            cbar.set_label("$\\log_{10}(||Error||)$", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

        # Окно TRS (желтая линия)
        if self.chk_show_window.isChecked() and self.window_xyz is not None:
            xm, ym, zm = self.window_xyz
            ax_3d.plot(xm, ym, zm, color='black', lw=5.0, zorder=10)
            ax_3d.plot(xm, ym, zm, color='#ffee00', lw=2.5, zorder=11, label="Окно TRS")

            if self.chk_show_start_marker.isChecked():
                ax_3d.scatter([xm[0]], [ym[0]], [zm[0]], color='#00ff44', s=35, edgecolors='black', zorder=12, label="Старт окна")

            ax_3d.legend(loc="upper right", fontsize=7)

        ax_3d.set_title(self.combo_system.currentText(), fontsize=10, fontweight="bold")
        ax_3d.set_xlabel("x", labelpad=-8, fontsize=8)
        ax_3d.set_ylabel("y", labelpad=-8, fontsize=8)
        ax_3d.set_zlabel("z", labelpad=-8, fontsize=8)
        ax_3d.tick_params(labelsize=7)

        self.fig_3d.tight_layout()
        self.canvas_3d.draw_idle()

    def redraw_all(self):
        self.redraw_3d_only()

        if self.t_arr is None or self.window_xyz is None:
            return

        xm_arr, ym_arr, zm_arr = self.window_xyz
        t_start = self.t_arr[0]
        t_real_end = self.t_arr[-1]
        dt_real = t_real_end - t_start

        # 2. Окно Master x_m(t)
        self.fig_win.clear()
        ax_win = self.fig_win.add_subplot(111)
        ax_win.plot(self.t_arr, xm_arr, color="#1f77b4", lw=1.5, label="Master $x_m(t)$")
        ax_win.set_title(f"Окно TRS: [{t_start:.3f}, {t_real_end:.3f}] c (Δt = {dt_real:.3f} c)", fontsize=9)
        ax_win.set_xlabel("Время $t$, с", fontsize=8)
        ax_win.set_ylabel("$x_m$", fontsize=8)
        ax_win.set_xlim(t_start, t_real_end)
        ax_win.ticklabel_format(axis='x', useOffset=False)
        ax_win.grid(True, linestyle=":")
        ax_win.legend(loc="upper right", fontsize=8)
        self.fig_win.tight_layout()
        self.canvas_win.draw_idle()

        # 3. Зигзаг TRS
        self.fig_zigzag.clear()
        ax_zigzag = self.fig_zigzag.add_subplot(111)
        cmap = matplotlib.colormaps['turbo']
        total = len(self.runs_zigzag)

        for i, (name, r_t, r_err) in enumerate(self.runs_zigzag):
            color = cmap(i / max(1, total - 1))
            ls = '-' if (i % 2 == 0) else '--'
            lbl = name if (i == 0 or i == total - 1) else None

            step = max(1, len(r_t) // 600)
            ax_zigzag.plot(r_t[::step], r_err[::step], color=color, linestyle=ls, lw=1.1, label=lbl)

        ax_zigzag.set_yscale('log')
        ax_zigzag.set_title("Зигзаг сходимости во времени (Turbo)", fontsize=9)
        ax_zigzag.set_xlabel("Время $t$, с", fontsize=8)
        ax_zigzag.set_ylabel("RMS $||e||$", fontsize=8)
        ax_zigzag.set_ylim(1e-16, 1e2)
        ax_zigzag.set_xlim(t_start, t_real_end)
        ax_zigzag.ticklabel_format(axis='x', useOffset=False)
        ax_zigzag.grid(True, which='both', linestyle=':')
        if total > 0:
            ax_zigzag.legend(loc="upper right", fontsize=7)
        self.fig_zigzag.tight_layout()
        self.canvas_zigzag.draw_idle()

        # 4. Сходимость по итерациям
        self.fig_conv.clear()
        ax_conv = self.fig_conv.add_subplot(111)
        ax_conv.plot(range(1, len(self.iter_rms) + 1), self.iter_rms, color='crimson', lw=1.5)
        ax_conv.set_yscale('log')
        ax_conv.set_title("Сходимость ошибки по итерациям", fontsize=9)
        ax_conv.set_xlabel("Номер итерации", fontsize=8)
        ax_conv.set_ylabel("Финальная RMS", fontsize=8)
        ax_conv.set_ylim(1e-16, 1e2)
        ax_conv.grid(True, which='both', linestyle=':')
        self.fig_conv.tight_layout()
        self.canvas_conv.draw_idle()

    def open_param_search_dialog(self):
        """Открывает независимое окно перебора параметров (главное окно полностью доступно)."""
        settings = self.get_current_settings_dict()

        if getattr(self, 'search_dlg', None) is None:
            self.search_dlg = ParamSearchDialog(current_settings=settings, parent=self)
            self.search_dlg.setWindowModality(Qt.WindowModality.NonModal)
            self.search_dlg.solution_preview_requested.connect(self.apply_settings_dict)
            self.search_dlg.save_presets_requested.connect(self.preset_widget.add_presets_batch)
        else:
            new_sys = settings.get("system")
            if new_sys != self.search_dlg.sys_name:
                # система сменилась: переключаем диалог (остановка перебора, очистка таблицы, новые значения)
                self.search_dlg.update_system(settings)
            else:
                # та же система: просто обновляем настройки, не трогая результаты
                self.search_dlg.current_settings = settings

        self.search_dlg.show()
        self.search_dlg.raise_()
        self.search_dlg.activateWindow()

    def _sync_search_dialog(self):
        dlg = getattr(self, 'search_dlg', None)
        if dlg is not None and dlg.isVisible():
            settings = self.get_current_settings_dict()
            if settings["system"] != dlg.sys_name:
                dlg.update_system(settings)

    def on_build_portrait_from_search(self, solution):
        """Строит 3D фазовый портрет для выбранного в поиске решения."""
        self.apply_settings_dict(solution)
        self.build_portrait()

    def open_comparison_dialog(self):
        """Открывает окно сравнительного анализа TRS и Pecora-Carroll."""
        from comparison.compare_dialog import ComparisonDialog
        settings = self.get_current_settings_dict()
        dlg = ComparisonDialog(current_settings=settings, parent=self)
        dlg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    no_wheel = NoWheelFilter(app)
    app.installEventFilter(no_wheel)
    window = TRSApp()
    window.show()
    sys.exit(app.exec())