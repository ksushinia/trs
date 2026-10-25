from PyQt6.QtWidgets import QDoubleSpinBox as _QDoubleSpinBox
from PyQt6.QtCore import QLocale


class QDoubleSpinBox(_QDoubleSpinBox):
    #Принимает 0.5 и 0,5, пробелы игнорирует, лишние знаки после точки обрезает по decimals()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(QLocale(QLocale.Language.C))
        self.setGroupSeparatorShown(False)

    def _clean(self, text):
        t = text.replace(",", ".").replace("\u00a0", "").replace(" ", "")
        if "." in t:
            head, tail = t.split(".", 1)
            d = self.decimals()
            if len(tail) > d:
                try:
                    t = f"{float(t):.{d}f}"      # 50.000 -> 50.00, 0.126 -> 0.13
                except ValueError:
                    pass
        return t

    def validate(self, text, pos):
        return super().validate(self._clean(text), pos)

    def valueFromText(self, text):
        return super().valueFromText(self._clean(text))

    def fixup(self, text):
        return super().fixup(self._clean(text))

from PyQt6.QtWidgets import QSpinBox as _QSpinBox


class QSpinBox(_QSpinBox):
    #Принимает 50, 50.0, 50,000, 1 234: дробная часть отбрасывается через округление

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)       # значение применяется по Enter / потере фокуса

    @staticmethod
    def _clean(text):
        t = text.replace(",", ".").replace("\u00a0", "").replace(" ", "")
        if "." in t:
            try:
                t = str(int(round(float(t))))
            except ValueError:
                pass
        return t

    def validate(self, text, pos):
        return super().validate(self._clean(text), pos)

    def valueFromText(self, text):
        return super().valueFromText(self._clean(text))

    def fixup(self, text):
        return super().fixup(self._clean(text))