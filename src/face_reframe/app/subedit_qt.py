"""Subtitle editor window: fix what Whisper misheard before burning captions."""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from . import subedit
from ..i18n import tr


def _ts(t: float) -> str:
    return "%d:%05.2f" % (int(t) // 60, t - 60 * (int(t) // 60))


class SubtitleEditor(QDialog):
    """Non-modal: edit a line (double-click or start typing), Enter to apply.
    Clicking a line jumps the preview there; the playing line is highlighted.
    Every change is saved at once (the preview and the export use it)."""

    def __init__(self, parent, title: str, get_words: Callable[[], list], set_words: Callable[[list], None],
                 settings: Callable[[], tuple], seek: Callable[[float], None],
                 now: Callable[[], Optional[float]], restore: Callable[[], Optional[list]]):
        super().__init__(parent)
        self.setWindowTitle(tr("Captions — ") + title)
        self.resize(640, 640)
        self.get_words, self.set_words = get_words, set_words
        self.settings, self.seek, self.now, self.restore = settings, seek, now, restore
        self._rows: List[tuple] = []
        self._filling = False

        lay = QVBoxLayout(self)
        tip = QLabel(tr("Double-click a row to fix it (Enter to confirm, empty text = deletes "
                        "the row). Click a row to see it in the preview."))
        tip.setWordWrap(True)
        tip.setObjectName("hint")
        lay.addWidget(tip)

        fr = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText(tr("Find (e.g. a misspelled name)"))
        self.repl = QLineEdit()
        self.repl.setPlaceholderText(tr("Replace with"))
        self.whole = QCheckBox(tr("Whole word"))
        self.whole.setChecked(True)
        go = QPushButton(tr("Replace all"))
        go.clicked.connect(self._replace_all)
        fr.addWidget(self.find, 1)
        fr.addWidget(self.repl, 1)
        fr.addWidget(self.whole)
        fr.addWidget(go)
        lay.addLayout(fr)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("Start"), tr("End"), tr("Text")])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        self.table.itemChanged.connect(self._edited)
        self.table.cellClicked.connect(self._clicked)
        lay.addWidget(self.table, 1)

        bot = QHBoxLayout()
        self.count = QLabel("")
        self.count.setObjectName("hint")
        rest = QPushButton(tr("Restore original transcript"))
        rest.clicked.connect(self._restore)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        bot.addWidget(self.count, 1)
        bot.addWidget(rest)
        bot.addWidget(close)
        lay.addLayout(bot)

        self.follow = QTimer(self)
        self.follow.timeout.connect(self._highlight_now)
        self.follow.start(200)
        self.refresh()

    # ------------------------------------------------------------ data
    def refresh(self, keep_row: Optional[int] = None):
        words = self.get_words() or []
        mw, md = self.settings()
        self._rows = subedit.lines(words, mw, md)
        self._filling = True
        self.table.setRowCount(len(self._rows))
        for r, (a, b, text, _f, _n) in enumerate(self._rows):
            for c, val in enumerate((_ts(a), _ts(b), text)):
                it = QTableWidgetItem(val)
                if c < 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setForeground(QBrush(QColor("#8d8d8d")))
                self.table.setItem(r, c, it)
        self._filling = False
        self.count.setText(tr("%d lines · %d words") % (len(self._rows), len(words)))
        if keep_row is not None and self._rows:
            self.table.selectRow(min(keep_row, len(self._rows) - 1))

    def _edited(self, item):
        if self._filling or item.column() != 2:
            return
        r = item.row()
        if r >= len(self._rows):
            return
        a, b, old, first, n = self._rows[r]
        new = item.text().strip()
        if new == old:
            return
        words = subedit.edit_line(self.get_words() or [], first, n, new)
        self.set_words(words)
        QTimer.singleShot(0, lambda: self.refresh(keep_row=r))

    def _replace_all(self):
        mw, md = self.settings()
        words, n = subedit.replace_all(self.get_words() or [], self.find.text(), self.repl.text(),
                                       mw, md, self.whole.isChecked())
        if n:
            self.set_words(words)
            self.refresh()
        QMessageBox.information(self, tr("Captions"), tr("%d lines edited.") % n)

    def _restore(self):
        if QMessageBox.question(self, tr("Captions"), tr("Discard all corrections and go back to the "
                                "Whisper transcript?")) != QMessageBox.Yes:
            return
        orig = self.restore()
        if orig is not None:
            self.refresh()

    def _clicked(self, r, _c):
        if r < len(self._rows):
            self.seek(self._rows[r][0] + 0.01)

    def _highlight_now(self):
        t = self.now()
        if t is None or not self._rows or self.table.state() == QAbstractItemView.EditingState:
            return
        for r, (a, b, *_rest) in enumerate(self._rows):
            if a <= t < b:
                if self.table.currentRow() != r:
                    self.table.selectRow(r)
                    self.table.scrollToItem(self.table.item(r, 2), QAbstractItemView.PositionAtCenter)
                return
