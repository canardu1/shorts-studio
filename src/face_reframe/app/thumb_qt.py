"""Thumbnail picker: 4 variants, editable words, 16:9 / 9:16, save as JPG."""

from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QToolButton, QVBoxLayout)

from . import thumbnail as T
from .preview import to_qimage
from ..i18n import tr

PREVIEW = {"16:9": (640, 360), "9:16": (270, 480)}


class ThumbDialog(QDialog):
    def __init__(self, parent, variants, text, hl, renderer, out_dir, stem):
        super().__init__(parent)
        self.setWindowTitle(tr("Thumbnails"))
        self.variants, self.renderer, self.out_dir, self.stem = variants, renderer, out_dir, stem
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Text")))
        self.text = QLineEdit(text)
        self.text.setPlaceholderText(tr("2-4 strong words"))
        row.addWidget(self.text, 1)
        row.addWidget(QLabel(tr("Highlighted word")))
        self.hl = QSpinBox()
        self.hl.setRange(0, 12)
        self.hl.setSpecialValueText(tr("none"))
        self.hl.setValue((hl + 1) if hl is not None else 0)
        row.addWidget(self.hl)
        self.aspect = QComboBox()
        self.aspect.addItem("YouTube 16:9", "16:9")
        self.aspect.addItem("Shorts 9:16", "9:16")
        row.addWidget(self.aspect)
        lay.addLayout(row)
        grid = QGridLayout()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        names = dict(T.STYLES)
        for i, v in enumerate(variants):
            b = QToolButton()
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setText(tr(names.get(v.style, v.style)))
            self.group.addButton(b, i)
            grid.addWidget(b, i // 2, i % 2)
            self.buttons.append(b)
        if self.buttons:
            self.buttons[0].setChecked(True)
        lay.addLayout(grid)
        bot = QHBoxLayout()
        bot.addWidget(QLabel(tr("Pick a thumbnail and press “Save”")))
        bot.addStretch(1)
        save_all = QPushButton(tr("Save all"))
        save_all.clicked.connect(self._save_all)
        save = QPushButton(tr("Save"))
        save.setDefault(True)
        save.clicked.connect(self._save_selected)
        bot.addWidget(save_all)
        bot.addWidget(save)
        lay.addLayout(bot)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.text.textChanged.connect(self.timer.start)
        self.hl.valueChanged.connect(self.timer.start)
        self.aspect.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def _hl(self):
        return self.hl.value() - 1 if self.hl.value() > 0 else None

    def _render(self, v, size=None):
        if size is None:
            return T.render(v, self.text.text(), self._hl(), self.aspect.currentData(), self.renderer)
        return T.compose(v.shot, v.matte, self.text.text(), self._hl(), v.style, size, self.renderer)

    def refresh(self):
        size = PREVIEW[self.aspect.currentData()]
        for b, v in zip(self.buttons, self.variants):
            img = self._render(v, size)
            b.setIcon(QIcon(QPixmap.fromImage(to_qimage(img))))
            b.setIconSize(QSize(*size))

    def _name(self, v, i):
        tag = "youtube" if self.aspect.currentData() == "16:9" else "shorts"
        return os.path.join(self.out_dir, "%s_thumb_%s_%d_%s.jpg" % (self.stem, tag, i + 1, v.style))

    def save_variant(self, i) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        v = self.variants[i]
        return T.save(self._render(v), self._name(v, i))

    def _save_selected(self):
        i = self.group.checkedId()
        if i < 0:
            return
        p = self.save_variant(i)
        QMessageBox.information(self, tr("Thumbnails"), tr("Saved: %s") % p)

    def _save_all(self):
        ps = [self.save_variant(i) for i in range(len(self.variants))]
        QMessageBox.information(self, tr("Thumbnails"),
                                tr("Saved %d thumbnails to %s") % (len(ps), self.out_dir))
