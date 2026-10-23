"""
presets.py
Модуль управления пользовательскими пресетами настроек TRS.
"""
import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QComboBox, QPushButton,
    QInputDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PRESETS_FILE = os.path.join(BASE_DIR, "user_presets.json")


class PresetManagerWidget(QGroupBox):
    """Виджет выбора, создания, переименования, перезаписи и удаления пресетов."""
    preset_applied = pyqtSignal(dict)

    def __init__(self, get_settings_fn, parent=None):
        super().__init__("Управление пресетами", parent)
        self.get_settings_fn = get_settings_fn
        self.user_presets = {}
        self.init_ui()
        self.load_user_presets()
        self.refresh_combo()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(self.combo)

        # Сетка кнопок управления
        grid_btns = QGridLayout()
        grid_btns.setSpacing(5)

        self.btn_apply = QPushButton("Применить")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_apply.clicked.connect(self.apply_current)

        self.btn_save_new = QPushButton("Сохранить как новый...")
        self.btn_save_new.clicked.connect(self.save_new_preset_dialog)

        self.btn_update = QPushButton("Перезаписать")
        self.btn_update.setToolTip("Обновить выбранный пресет текущими настройками")
        self.btn_update.clicked.connect(self.overwrite_current)

        self.btn_rename = QPushButton("Переименовать")
        self.btn_rename.setToolTip("Изменить название выбранного пресета")
        self.btn_rename.clicked.connect(self.rename_preset_dialog)

        self.btn_delete = QPushButton("Удалить")
        self.btn_delete.setToolTip("Удалить выбранный пресет")
        self.btn_delete.clicked.connect(self.delete_current)

        # Расположение: 2 колонки
        grid_btns.addWidget(self.btn_apply, 0, 0)
        grid_btns.addWidget(self.btn_save_new, 0, 1)
        grid_btns.addWidget(self.btn_update, 1, 0)
        grid_btns.addWidget(self.btn_rename, 1, 1)
        grid_btns.addWidget(self.btn_delete, 2, 0, 1, 2)

        layout.addLayout(grid_btns)

    def load_user_presets(self):
        if os.path.exists(USER_PRESETS_FILE):
            try:
                with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                    self.user_presets = json.load(f)
            except Exception:
                self.user_presets = {}
        else:
            self.user_presets = {}

    def save_user_presets_to_disk(self):
        try:
            with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.user_presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить пресеты: {e}")

    def refresh_combo(self, select_name=None):
        self.combo.blockSignals(True)
        self.combo.clear()

        for name in sorted(self.user_presets.keys()):
            self.combo.addItem(name)

        self.combo.blockSignals(False)

        if select_name and select_name in self.user_presets:
            idx = self.combo.findText(select_name)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        elif self.combo.count() > 0:
            self.combo.setCurrentIndex(0)

        self.on_selection_changed()

    def on_selection_changed(self):
        has_items = self.combo.count() > 0
        self.btn_apply.setEnabled(has_items)
        self.btn_update.setEnabled(has_items)
        self.btn_rename.setEnabled(has_items)
        self.btn_delete.setEnabled(has_items)

    def add_presets_batch(self, new_presets_dict):
        """Пакетное добавление группы пресетов из окна перебора."""
        for name, data in new_presets_dict.items():
            self.user_presets[name] = data
        self.save_user_presets_to_disk()
        self.refresh_combo()
        if new_presets_dict:
            first_name = list(new_presets_dict.keys())[0]
            self.combo.setCurrentText(first_name)

    def apply_current(self):
        name = self.combo.currentText()
        if not name or name not in self.user_presets:
            return

        data = self.user_presets.get(name)
        if data:
            self.preset_applied.emit(data)

    def save_new_preset_dialog(self):
        current_data = self.get_settings_fn()
        sys_name = current_data.get("system", "Система")
        kf = current_data.get("K_fwd", [0, 0, 0])
        kb = current_data.get("K_bwd", [0, 0, 0])
        dt = current_data.get("time", {}).get("dt", 1.0)
        default_name = f"{sys_name} (Kxf={kf[0]:.1f}, Kxb={kb[0]:.1f}, dt={dt:.2f})"
        name, ok = QInputDialog.getText(
            self, "Новый пресет", "Введите название нового пресета:", text=default_name
        )
        if ok and name.strip():
            clean_name = name.strip()
            if clean_name in self.user_presets:
                res = QMessageBox.question(
                    self, "Предупреждение", f"Пресет «{clean_name}» уже существует. Перезаписать?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if res != QMessageBox.StandardButton.Yes:
                    return

            self.user_presets[clean_name] = current_data
            self.save_user_presets_to_disk()
            self.refresh_combo(select_name=clean_name)
            QMessageBox.information(self, "Успех", f"Пресет «{clean_name}» сохранен.")

    def rename_preset_dialog(self):
        old_name = self.combo.currentText()
        if not old_name or old_name not in self.user_presets:
            return

        new_name, ok = QInputDialog.getText(
            self, "Переименование", "Введите новое название пресета:", text=old_name
        )
        if ok and new_name.strip():
            clean_name = new_name.strip()
            if clean_name == old_name:
                return

            if clean_name in self.user_presets:
                QMessageBox.warning(self, "Ошибка", f"Пресет с именем «{clean_name}» уже существует.")
                return

            # Переименовываем ключ в словаре с сохранением данных
            self.user_presets[clean_name] = self.user_presets.pop(old_name)
            self.save_user_presets_to_disk()
            self.refresh_combo(select_name=clean_name)
            QMessageBox.information(self, "Успех", f"Пресет переименован в «{clean_name}».")

    def overwrite_current(self):
        name = self.combo.currentText()
        if not name or name not in self.user_presets:
            return

        res = QMessageBox.question(
            self, "Подтверждение", f"Перезаписать пресет «{name}» текущими настройками?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if res == QMessageBox.StandardButton.Yes:
            self.user_presets[name] = self.get_settings_fn()
            self.save_user_presets_to_disk()
            QMessageBox.information(self, "Успех", f"Пресет «{name}» успешно обновлен.")

    def delete_current(self):
        name = self.combo.currentText()
        if not name or name not in self.user_presets:
            return

        res = QMessageBox.question(
            self, "Подтверждение", f"Удалить пресет «{name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if res == QMessageBox.StandardButton.Yes:
            del self.user_presets[name]
            self.save_user_presets_to_disk()
            self.refresh_combo()