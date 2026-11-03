"""Shorts Studio — standalone desktop app on top of the face_reframe core.

    python -m face_reframe.app        (or: face-reframe-app)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
import threading
import traceback

import numpy as np
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap, QPolygonF, QShortcut)
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout,
    QWidget,
)

from ..cinematic import CinematicParams
from ..enhance import EnhanceOptions
from .. import i18n
from ..i18n import tr
from ..pipeline import Options
from ..silence import SpeechParams
from ..track import ShotSettings, TrackSettings
from . import diskcache, presets
from .jobs import AnalysisCache, Cancelled, JobSettings, output_path, output_size, run_job
from .preview import FrameView, VideoReader, to_qimage
from .timeline import Timeline

APP_NAME = "Shorts Studio"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mxf", ".webm", ".mts"}
AUDIO_EXT = "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"
ASPECTS = ["9:16", "4:5", "1:1", "16:9"]
LANGS = [("Automatic", None), ("Italiano", "it"), ("English", "en"), ("Español", "es"),
         ("Français", "fr"), ("Deutsch", "de"), ("Português", "pt")]
ENHANCE = [("Off", "off"), ("Fast (sharpening + face)", "fast"), ("AI upscale (Real-ESRGAN)", "ai")]
RESTORE = [("Off", "off"), ("GPEN 256 — fast (~0.15 s/frame)", "gpen256"),
           ("GPEN 512 — detailed (~1.8 s/frame)", "gpen512"),
           ("GFPGAN 1.4 — commercial use ok (~2 s/frame)", "gfpgan"),
           ("CodeFormer — max quality (~5 s/frame)", "codeformer")]
SUBS = [("None", "off"), (".srt file only", "srt"), ("Burned into video + .srt", "burn")]
X264 = [("Fast", "veryfast"), ("Balanced", "fast"), ("Quality", "medium")]
RES = [("1080 × 1920", 1920), ("720 × 1280 (slow PC)", 1280)]
COLORS = [("White", "#FFFFFF"), ("Yellow", "#FFD400"), ("Green", "#7CFF6B"), ("Cyan", "#4FD8FF"),
          ("Pink", "#FF5DA2")]
DUCK = [("Light", 3.0), ("Medium", 6.0), ("Strong", 12.0)]
# areas covered by each app's buttons/captions on a 9:16 frame (x, y, w, h fractions)
SAFE_ZONES = {
    "tiktok": [(0.0, 0.0, 1.0, 0.09), (0.85, 0.38, 0.15, 0.48), (0.0, 0.80, 0.85, 0.20)],
    "reels": [(0.0, 0.0, 1.0, 0.08), (0.86, 0.45, 0.14, 0.43), (0.0, 0.82, 0.86, 0.18)],
    "shorts": [(0.0, 0.0, 1.0, 0.07), (0.86, 0.42, 0.14, 0.44), (0.0, 0.80, 0.86, 0.20)],
}
TRANSITIONS = [("None", "off"), ("Whip pan", "whip"), ("Zoom", "zoom"), ("Glitch", "glitch"),
               ("Random", "random")]


# ------------------------------------------------------------------ workers

CANCELLED = "__cancelled__"


class Worker(QObject):
    """Runs ``fn(progress_cb)`` in a QThread; result/failed are queued to the GUI.

    ``cancel`` makes the next progress call raise, so long jobs (face
    analysis, Whisper) stop within a moment — needed to quit cleanly: a
    QThread destroyed while its job still runs aborts the whole app."""
    progress = Signal(float)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = threading.Event()

    def _progress(self, f):
        if self.cancel.is_set():
            raise Cancelled()
        self.progress.emit(f)

    def run(self):
        try:
            self.result.emit(self.fn(self._progress))
        except Cancelled:
            self.failed.emit(CANCELLED)
        except Exception as exc:  # surfaced in the log
            traceback.print_exc()
            self.failed.emit(str(exc))


class Relay(QObject):
    """Lives in the GUI thread: worker signals connected to ``deliver`` are
    queued here, so callbacks (closures included) never run in the worker."""

    def __init__(self, cb, parent=None):
        super().__init__(parent)
        self.cb = cb

    @Slot(object)
    def deliver(self, value):
        self.cb(value)


class BatchWorker(QObject):
    file_started = Signal(int)
    progress = Signal(int, str, float)
    log = Signal(str)
    file_done = Signal(int, bool, object)
    finished = Signal()

    def __init__(self, items: List[dict], settings: JobSettings, cache: AnalysisCache, renderer):
        super().__init__()
        self.items, self.settings, self.cache, self.renderer = items, settings, cache, renderer
        self.title_renderer = None
        self.thumb_renderer = None
        self.cancel = threading.Event()

    def run(self):
        for i, it in enumerate(self.items):
            if self.cancel.is_set():
                break
            self.file_started.emit(i)
            self.log.emit("━━ %s" % it["label"])
            try:
                outs = run_job(it["path"], self.settings,
                               lambda stage, f, i=i: self.progress.emit(i, stage, f),
                               self.log.emit, self.cancel, self.cache, self.renderer,
                               clip=it.get("clip"), stem=it.get("stem"), title_renderer=self.title_renderer,
                               thumb_renderer=self.thumb_renderer)
                self.file_done.emit(i, True, outs)
            except Cancelled:
                self.file_done.emit(i, False, tr("cancelled"))
                break
            except Exception as exc:
                self.log.emit(traceback.format_exc())
                self.file_done.emit(i, False, str(exc))
        self.finished.emit()


# ------------------------------------------------------------------ helpers

def dspin(lo, hi, step, val, dec=2, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    w.setButtonSymbols(QAbstractSpinBox.PlusMinus)
    w.setMinimumWidth(58)
    if suffix:
        w.setSuffix(suffix)
    return w


def ispin(lo, hi, val, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setButtonSymbols(QAbstractSpinBox.PlusMinus)
    w.setMinimumWidth(58)
    if suffix:
        w.setSuffix(suffix)
    return w


def combo(items):
    w = QComboBox()
    # long entries must not widen the settings column (no horizontal scroll)
    w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    w.setMinimumContentsLength(14)
    for it in items:
        if isinstance(it, tuple):
            w.addItem(tr(it[0]), it[1])
        else:
            w.addItem(tr(it), it)
    return w


class SegControl(QWidget):
    """iOS-style segmented row for 2-4 mutually exclusive options: reads at a
    glance where a combo box would hide the choices. Combo-like API
    (``value`` / ``setValue`` / ``changed``) so it drops into the same
    settings machinery."""

    changed = Signal()

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setObjectName("segbox")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._btns = []
        n = len(items)
        for i, it in enumerate(items):
            label, data = it if isinstance(it, tuple) else (it, it)
            b = QPushButton(tr(label))
            b.setCheckable(True)
            b.setObjectName("segFirst" if i == 0 else "segLast" if i == n - 1 else "segMid")
            b.setFlat(True)
            lay.addWidget(b, 1)
            self.group.addButton(b)
            self._btns.append((b, data))
        self._btns[0][0].setChecked(True)
        self.group.buttonClicked.connect(lambda _b: self.changed.emit())

    def value(self):
        for b, d in self._btns:
            if b.isChecked():
                return d
        return self._btns[0][1]

    def setValue(self, v):
        for i, (b, d) in enumerate(self._btns):
            if d == v or str(d) == str(v):
                b.setChecked(True)
                return
        self._btns[0][0].setChecked(True)


def seg(items):
    return SegControl(items)


def hint(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("hint")
    return lab


def section(text):
    lab = QLabel(text.upper())
    lab.setObjectName("section")
    return lab


def _fmt(sec: float) -> str:
    sec = int(sec)
    return "%02d:%02d" % (sec // 60, sec % 60)


# ------------------------------------------------------------------ window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1480, 900)
        self.setAcceptDrops(True)
        self.cache = AnalysisCache()
        self.result = None
        self.reader: Optional[VideoReader] = None
        self.current: Optional[str] = None
        self.clip_range = None            # (first, last) frame of the selected clip item
        self.w: Dict[str, QWidget] = {}
        self._workers = []
        self.batch: Optional[BatchWorker] = None
        self._enh = None
        self._enh_key = None
        self.person_pick: Dict[str, list] = {}   # abs path -> identity vectors
        self.manual: Dict[str, list] = {}        # abs path -> [(frame, x, y)]
        self._people = []
        self._people_for = None
        self.pcm = {}                            # abs path -> 16 kHz audio (current file only)
        self.removed = []                        # preview: cut ranges (s), source timeline
        self._split = None
        self._punch = None
        self._burner = None
        self._renderer = None
        self.user_presets: Dict[str, dict] = {}
        self.music_path = ""
        self.logo_path = ""
        self.hook_text: Dict[str, str] = {}     # abs path -> custom hook title
        self._undo: List[tuple] = []
        self._redo: List[tuple] = []
        self.player = None          # QMediaPlayer: plays the source audio, master clock
        self.audio_out = None
        self._player_src = None
        self._muted = False
        self._pa_windows = []        # treated preview audio windows [{key, path, start, end}]
        self._pa_offset = 0.0        # source seconds at player position 0
        self._pa_pending = None      # time to (re)build the processed window from
        self._pending_seek = None
        self._seek_guard = None      # (target source time, wall clock) of the last seek
        self._pa_dir = None
        self.pa_timer = None
        self.proxy: Optional[VideoReader] = None   # light copy for smooth playback
        self._proxy_retries = set()                # files already given one proxy rebuild
        self._skin = {}              # abs path -> measured SkinCorrection
        self._skin_view = None       # (key, correction at the current strength)
        self._voice = {}             # abs path -> voice-tagged speech spans (ECAPA)
        self._voice_on = False       # a voice-id job is running
        self._restorer = None        # preview face restorer (built in the background)
        self._restorer_key = None
        self._matter = None          # preview person matte for background blur
        self._restore_pending = None
        self._frame_at = 0.0         # wall clock of the last previewed frame

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self.replan_timer = QTimer(self)
        self.replan_timer.setSingleShot(True)
        self.replan_timer.setInterval(150)
        self.replan_timer.timeout.connect(self._replan)
        self.pa_timer = QTimer(self)      # debounce: rebuild the treated audio after edits
        self.pa_timer.setSingleShot(True)
        self.pa_timer.setInterval(450)
        self.pa_timer.timeout.connect(self._pa_refresh)

        root = QSplitter(Qt.Horizontal)
        root.setHandleWidth(1)
        root.addWidget(self._build_queue())
        root.addWidget(self._build_preview())
        root.addWidget(self._build_settings())
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 0)
        root.setSizes([270, 800, 410])

        vs = QSplitter(Qt.Vertical)
        vs.setHandleWidth(1)
        vs.addWidget(root)
        vs.addWidget(self._build_bottom())
        vs.setStretchFactor(0, 1)
        vs.setSizes([740, 150])
        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)
        vlay.addWidget(self._build_header())
        vlay.addWidget(vs, 1)
        self.setCentralWidget(central)

        self._load_settings()
        self._sync_undo()
        QTimer.singleShot(400, self._offer_restore)
        for wd in self.w.values():
            sig = None
            if isinstance(wd, (QDoubleSpinBox, QSpinBox)):
                sig = wd.valueChanged
            elif isinstance(wd, QComboBox):
                sig = wd.currentIndexChanged
            elif isinstance(wd, SegControl):
                sig = wd.changed
            elif isinstance(wd, QCheckBox):
                sig = wd.toggled
            if sig is not None:
                sig.connect(self._settings_changed)
        self._sync_enabled()
        self._refresh_cache_label()

    # ================================================================ layout
    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("header")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 10, 0)
        lay.setSpacing(0)
        a = QLabel("SHORTS")
        a.setObjectName("brand")
        b = QLabel("STUDIO")
        b.setObjectName("brandAlt")
        lay.addWidget(a)
        lay.addWidget(b)
        lay.addStretch(1)
        po = QPushButton(tr("Open project…"))
        po.clicked.connect(self._open_project)
        ps = QPushButton(tr("Save project…"))
        ps.clicked.connect(self._save_project)
        lay.addWidget(po)
        lay.addWidget(ps)
        return bar

    def _build_queue(self):
        box = QWidget()
        box.setObjectName("panel")
        box.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 4, 10)
        title = QLabel("Video")
        title.setObjectName("h1")
        lay.addWidget(title)
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue.setTextElideMode(Qt.ElideRight)
        self.queue.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.queue.currentItemChanged.connect(self._select)
        self.queue.itemDoubleClicked.connect(self._open_output)
        lay.addWidget(self.queue, 1)
        self.drop_hint = hint(tr("Drop videos or folders here, or use “Add”."))
        lay.addWidget(self.drop_hint)
        row = QHBoxLayout()
        add = QPushButton(tr("Add…"))
        add.clicked.connect(self._add_dialog)
        addd = QPushButton(tr("Folder…"))
        addd.clicked.connect(self._add_folder)
        rem = QPushButton(tr("Remove"))
        rem.clicked.connect(self._remove_selected)
        row.addWidget(add)
        row.addWidget(addd)
        row.addWidget(rem)
        lay.addLayout(row)
        self.hl_btn = QPushButton(tr("Find the best moments"))
        self.hl_btn.setToolTip(tr("Suggests the strongest 30–60 s clips of the selected video and adds them to the list as separate shorts"))
        self.hl_btn.clicked.connect(self._find_highlights)
        hl_row = QHBoxLayout()
        self.hl_count = ispin(1, 10, 4)
        self.hl_count.setToolTip(tr("How many clips to suggest"))
        hl_row.addWidget(self.hl_btn, 1)
        hl_row.addWidget(self.hl_count)
        lay.addLayout(hl_row)
        srow = QHBoxLayout()
        soc = QPushButton(tr("Title, description & hashtags…"))
        soc.clicked.connect(self._show_social)
        srow.addWidget(soc)
        thb = QPushButton(tr("Thumbnails…"))
        thb.setToolTip(tr("Click-worthy thumbnails for YouTube (16:9) and Shorts (9:16) from the best frames"))
        thb.clicked.connect(self._open_thumbs)
        srow.addWidget(thb)
        lay.addLayout(srow)
        return box

    def _build_preview(self):
        box = QWidget()
        box.setObjectName("previewPane")
        box.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 10, 4, 6)
        views = QSplitter(Qt.Horizontal)
        views.setHandleWidth(6)
        self.src_view = FrameView(tr("Select a video to preview"), draggable=True)
        self.src_view.crop_dragged.connect(self._crop_dragged)
        self.src_view.object_picked.connect(self._object_picked)
        self.out_view = FrameView(tr("Vertical preview\n\nPress “Analyze faces”"))
        views.addWidget(self.src_view)
        views.addWidget(self.out_view)
        views.setSizes([560, 280])
        lay.addWidget(views, 1)

        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek)
        lay.addWidget(self.timeline)

        card = QFrame()
        card.setObjectName("ctlcard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(10, 7, 10, 8)
        card_lay.setSpacing(7)
        ctl = QHBoxLayout()
        self.play_btn = QToolButton()
        self.play_btn.setIcon(_icon("play"))
        self.play_btn.setIconSize(QSize(20, 20))
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._show_frame)
        self.slider.sliderMoved.connect(self._user_seek)
        self.slider.sliderReleased.connect(self._refresh_frame)
        self.mute_btn = QToolButton()
        self.mute_btn.setIcon(_icon("vol"))
        self.mute_btn.setIconSize(QSize(18, 18))
        self.mute_btn.setToolTip(tr("Preview audio on/off"))
        self.mute_btn.clicked.connect(self._toggle_mute)
        self.ab_btn = QToolButton()
        self.ab_btn.setText("A/B")
        self.ab_btn.setCheckable(True)
        self.ab_btn.setToolTip(tr("Held: you hear the original audio. Released: the audio as it will be exported (cinematic audio and music, if on)"))
        self.ab_btn.toggled.connect(self._ab_toggled)
        self.time_lab = QLabel("00:00 / 00:00")
        self.time_lab.setMinimumWidth(110)
        self.skip_cuts = QCheckBox(tr("Skip cuts"))
        self.skip_cuts.setToolTip(tr("While playing, skip the parts that will be cut (in red)"))
        self.skip_cuts.setChecked(True)
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.time_lab)
        ctl.addWidget(self.mute_btn)
        ctl.addWidget(self.ab_btn)
        ctl.addWidget(self.skip_cuts)
        card_lay.addLayout(ctl)

        row = QHBoxLayout()
        self.analyze_btn = QPushButton(tr("Analyze faces"))
        self.analyze_btn.setObjectName("accent")
        self.analyze_btn.clicked.connect(self._analyze)
        self.tr_btn = QPushButton(tr("Transcribe"))
        self.tr_btn.setToolTip(tr("Whisper: needed to see fillers, retakes and captions in the preview (the result is reused by the render)"))
        self.tr_btn.clicked.connect(self._transcribe)
        self.sub_btn = QPushButton(tr("Captions…"))
        self.sub_btn.setToolTip(tr("Fix the caption text before burning it into the video"))
        self.sub_btn.clicked.connect(self._open_subeditor)
        self.enh_preview = QCheckBox(tr("Enhancement"))
        self.enh_preview.setToolTip(tr("Show the image enhancement in the vertical preview"))
        self.enh_preview.toggled.connect(self._refresh_frame)
        self.auto_btn = QPushButton(tr("Auto from here"))
        self.auto_btn.setToolTip(tr("From this frame on, the crop follows the face by itself again"))
        self.auto_btn.clicked.connect(self._manual_auto_here)
        self.clear_man_btn = QPushButton(tr("Clear corrections"))
        self.clear_man_btn.clicked.connect(self._manual_clear)
        row.addWidget(self.analyze_btn)
        row.addWidget(self.tr_btn)
        row.addWidget(self.sub_btn)
        row.addWidget(self.enh_preview)
        self.zones_box = combo([(tr("Safe zones: off"), None), ("TikTok", "tiktok"), ("Reels", "reels"),
                                ("Shorts", "shorts")])
        self.zones_box.setToolTip(tr("Show the areas covered by the platform's interface"))
        self.zones_box.currentIndexChanged.connect(self._refresh_frame)
        row.addWidget(self.zones_box)
        row.addStretch(1)
        row.addWidget(self.auto_btn)
        row.addWidget(self.clear_man_btn)
        self.undo_btn = QToolButton()
        self.undo_btn.setText("↶")
        self.undo_btn.setToolTip(tr("Undo (⌘Z): crop corrections, people, captions, title"))
        self.undo_btn.clicked.connect(self._undo_action)
        self.redo_btn = QToolButton()
        self.redo_btn.setText("↷")
        self.redo_btn.setToolTip(tr("Redo (⇧⌘Z)"))
        self.redo_btn.clicked.connect(self._redo_action)
        row.addWidget(self.undo_btn)
        row.addWidget(self.redo_btn)
        QShortcut(QKeySequence.Undo, self, self._shortcut_undo)
        QShortcut(QKeySequence.Redo, self, self._shortcut_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._shortcut_redo)
        card_lay.addLayout(row)
        lay.addWidget(card)
        self.info_lab = hint("")
        lay.addWidget(self.info_lab)

        self.people_bar = QWidget()
        pl = QHBoxLayout(self.people_bar)
        pl.setContentsMargins(0, 2, 0, 0)
        pl.setSpacing(6)
        lab = QLabel(tr("Who to follow:"))
        lab.setObjectName("hint")
        pl.addWidget(lab)
        self.people_group = QButtonGroup(self)
        self.people_group.setExclusive(False)
        self.people_group.idClicked.connect(self._pick_person)
        self.people_layout = pl
        pl.addStretch(1)
        self.people_bar.setVisible(False)
        lay.addWidget(self.people_bar)
        return box

    def _build_settings(self):
        col = QWidget()
        col.setObjectName("panel")
        col.setAttribute(Qt.WA_StyledBackground, True)
        cl = QVBoxLayout(col)
        cl.setContentsMargins(0, 8, 6, 0)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Preset"))
        self.preset_box = QComboBox()
        self.preset_box.setMinimumWidth(150)
        self.preset_box.activated.connect(self._apply_preset)
        save = QPushButton(tr("Save…"))
        save.clicked.connect(self._save_preset)
        dele = QPushButton(tr("Delete"))
        dele.clicked.connect(self._delete_preset)
        prow.addWidget(self.preset_box, 1)
        prow.addWidget(save)
        prow.addWidget(dele)
        cl.addLayout(prow)

        tabs = QTabWidget()
        tabs.setMinimumWidth(330)
        tabs.tabBar().setExpanding(False)
        tabs.tabBar().hide()
        self.tabs = tabs
        srow2 = QHBoxLayout()
        srow2.setContentsMargins(0, 0, 0, 0)
        srow2.setSpacing(0)
        srow2.addWidget(self._build_rail(tabs))
        srow2.addWidget(tabs, 1)
        cl.addLayout(srow2, 1)
        W = self.w

        # --- framing
        f = QFormLayout()
        W["aspect"] = seg(ASPECTS)
        W["subject"] = seg([(tr("Main person"), "main"), (tr("Whoever speaks"), "speaker"),
                            (tr("Picked object"), "object")])
        W["noface"] = seg([(tr("Keep centred"), "center"), (tr("Follow object (neural)"), "object")])
        W["speaker_engine"] = seg([(tr("Mouth motion"), "mouth"), (tr("Voice ID (neural)"), "voice")])
        W["layout"] = combo([(tr("One person"), "single"), (tr("Split 2 people (top/bottom)"), "split")])
        W["smoothing"] = dspin(0, 3, 0.1, 0.5, 1, " s")
        W["dead_zone"] = dspin(0, 0.5, 0.01, 0.05)
        W["max_speed"] = dspin(0, 5, 0.1, 0.6, 1)
        W["zoom"] = dspin(1, 3, 0.05, 1.0, 2, "×")
        W["follow_vertical"] = QCheckBox(tr("Follow the face vertically too"))
        W["face_pos"] = dspin(0.2, 0.5, 0.01, 0.4)
        W["look_room"] = QCheckBox(tr("Leave room toward the gaze"))
        W["look_room_amount"] = dspin(0.05, 0.4, 0.05, 0.15)
        W["punch"] = QCheckBox(tr("Auto punch-in on phrases"))
        W["punch_zoom"] = dspin(1.02, 1.5, 0.01, 1.15, 2, "×")
        W["wide_mode"] = seg([(tr("Zoom on subject"), "zoom"), (tr("Full height"), "keep")])
        W["wide_threshold"] = dspin(0.02, 0.5, 0.01, 0.12)
        W["max_zoom"] = dspin(1, 4, 0.1, 2.0, 1, "×")
        W["stride"] = combo([(tr("Every frame (slow)"), 1), (tr("Every 2 frames"), 2), (tr("Every 3 frames (fast)"), 3)])
        W["stride"].setCurrentIndex(1)
        f.addRow(tr("Aspect"), W["aspect"])
        f.addRow(tr("Follow"), W["subject"])
        self.pick_obj_btn = QPushButton(tr("Pick object on this frame…"))
        self.pick_obj_btn.clicked.connect(self._pick_object)
        f.addRow(self.pick_obj_btn)
        f.addRow(tr("No face in shot"), W["noface"])
        f.addRow(tr("Speaker engine"), W["speaker_engine"])
        f.addRow(tr("Layout"), W["layout"])
        f.addRow(tr("Smoothing"), W["smoothing"])
        f.addRow(tr("Dead zone"), W["dead_zone"])
        f.addRow(tr("Max pan speed"), W["max_speed"])
        f.addRow("Zoom", W["zoom"])
        f.addRow(W["follow_vertical"])
        f.addRow(tr("Face height"), W["face_pos"])
        f.addRow(W["look_room"])
        f.addRow(tr("Gaze room"), W["look_room_amount"])
        f.addRow(W["punch"])
        f.addRow(tr("Punch-in zoom"), W["punch_zoom"])
        f.addRow(tr("Wide shots"), W["wide_mode"])
        f.addRow(tr("Wide when face <"), W["wide_threshold"])
        f.addRow(tr("Max wide zoom"), W["max_zoom"])
        f.addRow(tr("Detection"), W["stride"])
        f.addRow(hint(tr("“Follow object” tracks the most salient thing in shots without faces — products, pets, B-roll — with the same U2-Net-p model used for thumbnail cut-outs (downloaded once, ~5 MB). "
                         "Drag the yellow box in the preview to correct the crop from that point on. Below the preview, pick who to follow (select every thumbnail of the same person). “Whoever speaks” cuts to who is talking — Mouth motion judges by the lips, Voice ID (ECAPA, 84 MB once) listens to who is actually speaking and marks voice changes on the timeline.")))
        tabs.addTab(self._page(f), "Crop")

        # --- cuts
        f = QFormLayout()
        W["cut_pauses"] = QCheckBox(tr("Cut pauses"))
        W["speech_engine"] = seg([(tr("Neural (Silero VAD)"), "vad"), (tr("Levels (dB)"), "db")])
        W["min_pause"] = dspin(0.1, 3, 0.05, 0.35, 2, " s")
        W["padding"] = dspin(0, 0.5, 0.01, 0.16, 2, " s")
        W["margin_db"] = dspin(3, 20, 0.5, 8.0, 1, " dB")
        W["cut_breaths"] = QCheckBox(tr("Cut breaths"))
        W["cut_fillers"] = QCheckBox(tr("Cut ums / uhs  (Whisper)"))
        W["cut_retakes"] = QCheckBox(tr("Keep only the last retake  (Whisper)"))
        f.addRow(W["cut_pauses"])
        f.addRow(tr("Speech detection"), W["speech_engine"])
        f.addRow(tr("Min pause"), W["min_pause"])
        f.addRow(tr("Padding"), W["padding"])
        f.addRow(tr("Voice threshold"), W["margin_db"])
        f.addRow(W["cut_breaths"])
        f.addRow(W["cut_fillers"])
        f.addRow(W["cut_retakes"])
        f.addRow(hint(tr("Cut parts are red on the timeline. Fillers and retakes appear after “Transcribe”. Whisper runs once per video and the result stays cached.")))
        tabs.addTab(self._page(f), tr("Cuts"))

        # --- audio
        f = QFormLayout()
        W["cinematic"] = QCheckBox(tr("Cinematic audio"))
        W["denoise"] = dspin(0, 40, 1, 12, 0, " dB")
        W["denoise_engine"] = seg([(tr("RNNoise"), "rnn"), (tr("Spectral (FFT)"), "fft"),
                                   (tr("DeepFilterNet"), "dfn")])
        W["dereverb"] = dspin(0, 30, 1, 12, 0, " dB")
        W["warmth"] = dspin(-6, 6, 0.1, 1.6, 1, " dB")
        W["presence"] = dspin(-6, 6, 0.1, 2.2, 1, " dB")
        W["air"] = dspin(-6, 6, 0.1, 1.2, 1, " dB")
        self.music_edit = QLineEdit()
        self.music_edit.setPlaceholderText(tr("No music"))
        self.music_edit.setReadOnly(True)
        mb = QPushButton("…")
        mb.setFixedWidth(32)
        mb.clicked.connect(self._pick_music)
        mc = QPushButton("✕")
        mc.setFixedWidth(32)
        mc.clicked.connect(lambda: self._set_music(""))
        mrow = QHBoxLayout()
        mrow.addWidget(self.music_edit, 1)
        mrow.addWidget(mb)
        mrow.addWidget(mc)
        W["music_db"] = dspin(-40, 0, 1, -18, 0, " dB")
        W["duck"] = combo(DUCK)
        W["duck"].setCurrentIndex(1)
        f.addRow(W["cinematic"])
        f.addRow(tr("Noise reduction"), W["denoise"])
        f.addRow(tr("Denoise engine"), W["denoise_engine"])
        f.addRow(hint(tr("RNNoise is light and safe, Spectral adapts to the measured noise floor, DeepFilterNet (15 MB binary, downloaded once) is strongest on changing noise like traffic or crowds. The dB depth sets DeepFilterNet's attenuation limit.")))
        f.addRow(tr("De-reverb"), W["dereverb"])
        f.addRow(tr("Warmth"), W["warmth"])
        f.addRow(tr("Presence"), W["presence"])
        f.addRow(tr("Air"), W["air"])
        f.addRow(hint(tr("Denoise, de-esser, compression, EQ and loudness at −14 LUFS.")))
        f.addRow(tr("Background music"), mrow)
        f.addRow(tr("Music volume"), W["music_db"])
        f.addRow(tr("Duck under voice"), W["duck"])
        f.addRow(hint(tr("The music loops for the whole duration and ducks by itself when someone talks.")))
        W["audio_events"] = QCheckBox(tr("Sound events for the best moments"))
        f.addRow(W["audio_events"])
        f.addRow(hint(tr("YAMNet (16 MB, downloaded once) listens for laughter, applause and music: clips that have them score higher when you press “Find the best moments”.")))
        tabs.addTab(self._page(f), "Audio")

        # --- image
        f = QFormLayout()
        W["restore"] = combo(RESTORE)
        W["restore_blend"] = dspin(0.1, 1.0, 0.05, 0.8)
        W["restore_fidelity"] = dspin(0.0, 1.0, 0.05, 0.7)
        W["enhance"] = combo(ENHANCE)
        W["sharpen"] = dspin(0, 2, 0.05, 0.5)
        W["face_boost"] = dspin(0, 2, 0.05, 0.35)
        f.addRow(section(tr("Face restoration")))
        f.addRow(tr("Model"), W["restore"])
        f.addRow(tr("Strength"), W["restore_blend"])
        f.addRow(tr("Fidelity (CodeFormer)"), W["restore_fidelity"])
        f.addRow(hint(tr("Rebuilds the face (eyes, skin, teeth) and blends it back into the frame with feathered edges. Time per face per frame on an M1: 1 minute of short ≈ 4 min with GPEN 256, ≈ 2 hours with CodeFormer. High fidelity = closer to the original. The first run downloads the model (75–380 MB). CodeFormer and GPEN are free for non-commercial use only; GFPGAN is fine for work. Preview it with “Enhancement” on, on a paused frame.")))
        W["skin"] = QCheckBox(tr("Normalize skin color"))
        W["skin_strength"] = dspin(0.0, 1.0, 0.05, 0.7)
        self.skin_lab = hint("")
        cube = QPushButton(tr("Save as .cube LUT…"))
        cube.setToolTip(tr("The same correction as a 3D LUT, for DaVinci Resolve"))
        cube.clicked.connect(self._save_skin_cube)
        W["beauty"] = QCheckBox(tr("Face beauty"))
        W["beauty_smooth"] = dspin(0, 1, 0.05, 0.5)
        W["beauty_blemish"] = dspin(0, 1, 0.05, 0.5)
        W["beauty_wrinkles"] = dspin(0, 1, 0.05, 0.4)
        f.addRow(section(tr("Beauty")))
        f.addRow(W["beauty"])
        f.addRow(tr("Skin smoothing"), W["beauty_smooth"])
        f.addRow(tr("Spots and blemishes"), W["beauty_blemish"])
        f.addRow(tr("Wrinkles and eye bags"), W["beauty_wrinkles"])
        f.addRow(hint(tr("Only on face skin: eyes, eyebrows, lips, beard and hair stay sharp. Smoothing that keeps some grain (no plastic look), spots replaced by the skin around them, wrinkles softened and eye bags lightened. ~10–25 ms per face. The preview shows it on paused frames.")))
        f.addRow(section(tr("Skin")))
        f.addRow(W["skin"])
        f.addRow(tr("Strength"), W["skin_strength"])
        f.addRow(self.skin_lab)
        f.addRow(cube)
        f.addRow(hint(tr("Measures cheeks and forehead, then pulls the tint back to the natural skin line and removes excess saturation, only on skin colours: red clothes and background stay as they were. Useful for cameras that make skin too red.")))
        W["stabilize"] = QCheckBox(tr("Stabilize shaky footage"))
        W["stab_amount"] = dspin(0.1, 1.0, 0.05, 0.8)
        f.addRow(section(tr("Stabilization")))
        f.addRow(W["stabilize"])
        f.addRow(tr("Strength"), W["stab_amount"])
        f.addRow(hint(tr("Measures camera shake and compensates by shifting the crop (plus a slight rotation). The first run analyses the whole video.")))
        f.addRow(section(tr("Background")))
        W["bg_blur"] = QCheckBox(tr("Blur background"))
        W["bg_blur_amount"] = dspin(0.1, 1.5, 0.05, 0.6)
        W["bg_engine"] = seg([(tr("Fast (RVM)"), "rvm"), (tr("Quality (MODNet)"), "modnet")])
        f.addRow(W["bg_blur"])
        f.addRow(tr("Strength"), W["bg_blur_amount"])
        f.addRow(tr("Matte engine"), W["bg_engine"])
        f.addRow(hint(tr("MODNet (24 MB once) cuts much finer edges — hair, glasses, hands. The preview always shows the fast engine; the export uses the one you pick. Edges are snapped to real pixels either way.")))
        f.addRow(hint(tr("Neural person matte (RVM): keeps the subject sharp and blurs everything else, like a portrait lens. The 14 MB model downloads on first export.")))
        f.addRow(section(tr("Sharpness & upscale")))
        f.addRow(tr("Enhancement"), W["enhance"])
        f.addRow(tr("Sharpening"), W["sharpen"])
        f.addRow(tr("Extra on face"), W["face_boost"])
        f.addRow(hint(tr("Works only on the vertical crop. “Fast” costs almost nothing.")))
        tabs.addTab(self._page(f), tr("Quality"))

        # --- social moves
        f = QFormLayout()
        W["mv_handheld"] = QCheckBox(tr("Handheld camera"))
        W["mv_handheld_amount"] = dspin(0, 1, 0.05, 0.4)
        W["mv_push"] = QCheckBox(tr("Slow push on every phrase"))
        W["mv_push_amount"] = dspin(0, 1, 0.05, 0.5)
        W["mv_crash"] = QCheckBox(tr("Crash zoom on strong words"))
        W["mv_crash_amount"] = dspin(0, 1, 0.05, 0.5)
        W["mv_shake"] = QCheckBox(tr("Shake on voice peaks"))
        W["mv_shake_amount"] = dspin(0, 1, 0.05, 0.5)
        W["mv_transition"] = combo(TRANSITIONS)
        W["mv_every"] = seg([(tr("Every cut"), "all"), (tr("Every other one"), "alternate"),
                             (tr("Only shot changes"), "shots")])
        W["mv_gap"] = dspin(0.5, 10, 0.5, 1.5, 1, " s")
        f.addRow(section(tr("Camera moves")))
        f.addRow(W["mv_handheld"])
        f.addRow(tr("Strength"), W["mv_handheld_amount"])
        f.addRow(W["mv_push"])
        f.addRow(tr("Strength"), W["mv_push_amount"])
        f.addRow(W["mv_crash"])
        f.addRow(tr("Strength"), W["mv_crash_amount"])
        f.addRow(W["mv_shake"])
        f.addRow(tr("Strength"), W["mv_shake_amount"])
        f.addRow(section(tr("Cut transitions")))
        f.addRow(tr("Effect"), W["mv_transition"])
        f.addRow(tr("When"), W["mv_every"])
        f.addRow(tr("Min distance"), W["mv_gap"])
        f.addRow(hint(tr("Crash zoom uses the transcript's strong words (questions, exclamations, numbers, “never”, “secret”…) or, without a transcript, voice peaks. Cuts are the shot changes and the pause cuts. Everything is previewed in play; export cost is nearly zero.")))
        tabs.addTab(self._page(f), tr("Effects"))

        # --- colour
        f = QFormLayout()
        from .look import LOOKS
        W["look"] = combo(LOOKS)
        W["look_strength"] = dspin(0, 1, 0.05, 1.0)
        W["auto_wb"] = QCheckBox(tr("Auto white balance and exposure"))
        f.addRow(section(tr("Look")))
        f.addRow(tr("Preset"), W["look"])
        f.addRow(tr("Strength"), W["look_strength"])
        f.addRow(W["auto_wb"])
        f.addRow(hint(tr("Auto balance evens out colour and exposure: handy to give shorts of the same series the same look.")))
        W["lowlight"] = seg([(tr("Off"), "off"), (tr("Fast"), "sg_llie"), (tr("Quality"), "uretinex")])
        W["lowlight_strength"] = dspin(0, 1, 0.05, 0.6)
        f.addRow(section(tr("Low light (neural)")))
        f.addRow(tr("Lift dark shots"), W["lowlight"])
        f.addRow(tr("Strength"), W["lowlight_strength"])
        f.addRow(hint(tr("Tiny enhancement nets that brighten underexposed shots: SG-LLIE (~17 ms/frame, good in motion) or URetinex-Net (~0.6 s/frame, cleaner). Models (0.05–1.6 MB) download on first export; the preview shows it on paused frames once downloaded.")))
        W["color_temp"] = dspin(-1, 1, 0.05, 0.0)
        W["color_tint"] = dspin(-1, 1, 0.05, 0.0)
        W["color_exp"] = dspin(-1, 1, 0.05, 0.0)
        W["color_con"] = dspin(0, 2, 0.05, 1.0)
        W["color_sat"] = dspin(0, 2, 0.05, 1.0)
        W["color_shad"] = dspin(-1, 1, 0.05, 0.0)
        W["color_high"] = dspin(-1, 1, 0.05, 0.0)
        W["color_vig"] = dspin(0, 1, 0.05, 0.0)
        f.addRow(section(tr("Grade")))
        f.addRow(tr("Temperature"), W["color_temp"])
        f.addRow(tr("Tint"), W["color_tint"])
        f.addRow(tr("Exposure"), W["color_exp"])
        f.addRow(tr("Contrast"), W["color_con"])
        f.addRow(tr("Saturation"), W["color_sat"])
        f.addRow(tr("Shadows"), W["color_shad"])
        f.addRow(tr("Highlights"), W["color_high"])
        f.addRow(tr("Vignette"), W["color_vig"])
        res = QPushButton(tr("Reset grade"))
        res.clicked.connect(self._reset_grade)
        f.addRow(res)
        f.addRow(hint(tr("Fine-tune the picture on top of the preset: the grade is baked into the same fast transform — export cost is nearly zero.")))
        tabs.addTab(self._page(f), tr("Color"))

        # --- subtitles
        f = QFormLayout()
        W["subtitles"] = combo(SUBS)
        W["caption_style"] = seg([(tr("Word by word"), "word"), (tr("Whole line"), "line")])
        W["language"] = combo(LANGS)
        W["caption_words"] = ispin(1, 8, 3)
        W["caption_max_dur"] = dspin(0.5, 5, 0.1, 1.6, 1, " s")
        W["caption_pos"] = dspin(0.1, 0.95, 0.01, 0.7)
        W["caption_color"] = combo(COLORS)
        W["caption_hl"] = combo(COLORS)
        W["caption_hl"].setCurrentIndex(1)
        W["caption_upper"] = QCheckBox(tr("UPPERCASE"))
        W["caption_upper"].setChecked(True)
        f.addRow(tr("Captions"), W["subtitles"])
        f.addRow(tr("Style"), W["caption_style"])
        f.addRow(tr("Language"), W["language"])
        f.addRow(tr("Words per line"), W["caption_words"])
        f.addRow(tr("Max line duration"), W["caption_max_dur"])
        f.addRow(tr("Vertical position"), W["caption_pos"])
        f.addRow(tr("Text colour"), W["caption_color"])
        f.addRow(tr("Active word colour"), W["caption_hl"])
        f.addRow(W["caption_upper"])
        W["sub_lang"] = combo([(tr("Original language"), "orig"), (tr("English (translation)"), "en"),
                               (tr("Original + English below"), "both")])
        f.addRow(tr("Caption language"), W["sub_lang"])
        f.addRow(hint(tr("Whisper does the English translation, offline, from any language.")))
        W["hook"] = combo([(tr("None"), "off"), (tr("Auto (first sentence)"), "auto"),
                           (tr("Written by me (per video)"), "custom")])
        W["hook_dur"] = dspin(1, 8, 0.5, 2.8, 1, " s")
        self.hook_edit = QLineEdit()
        self.hook_edit.setPlaceholderText(tr("Opening title for the selected video"))
        self.hook_edit.textEdited.connect(self._hook_edited)
        self.cta_edit = QLineEdit()
        self.cta_edit.setPlaceholderText(tr("e.g. Follow me for more videos!  (empty = none)"))
        self.cta_edit.textEdited.connect(lambda *_: self._settings_changed())
        W["cta_dur"] = dspin(1, 8, 0.5, 2.5, 1, " s")
        self.logo_edit = QLineEdit()
        self.logo_edit.setReadOnly(True)
        self.logo_edit.setPlaceholderText(tr("No logo"))
        lb = QPushButton("…")
        lb.setFixedWidth(32)
        lb.clicked.connect(self._pick_logo)
        lc = QPushButton("✕")
        lc.setFixedWidth(32)
        lc.clicked.connect(lambda: self._set_logo(""))
        lrow = QHBoxLayout()
        lrow.addWidget(self.logo_edit, 1)
        lrow.addWidget(lb)
        lrow.addWidget(lc)
        W["logo_corner"] = combo([(tr("Top right"), "tr"), (tr("Top left"), "tl"),
                                  (tr("Bottom right"), "br"), (tr("Bottom left"), "bl")])
        W["logo_size"] = dspin(0.05, 0.4, 0.01, 0.16)
        W["logo_opacity"] = dspin(0.1, 1.0, 0.05, 0.85)
        f.addRow(section(tr("Opening title (hook)")))
        f.addRow(tr("Title"), W["hook"])
        f.addRow(tr("Text"), self.hook_edit)
        f.addRow(tr("Duration"), W["hook_dur"])
        f.addRow(section(tr("Closing")))
        f.addRow(tr("Closing CTA"), self.cta_edit)
        f.addRow(tr("Duration"), W["cta_dur"])
        f.addRow(section(tr("Logo")))
        f.addRow(tr("Image (PNG)"), lrow)
        f.addRow(tr("Position"), W["logo_corner"])
        f.addRow(tr("Size"), W["logo_size"])
        f.addRow(tr("Opacity"), W["logo_opacity"])
        f.addRow(hint(tr("After “Transcribe”, the captions show in the vertical preview.")))
        tabs.addTab(self._page(f), tr("Texts"))

        # --- export
        f = QFormLayout()
        W["resolution"] = seg(RES)
        W["preset"] = combo(X264)
        W["crf"] = ispin(14, 30, 18)
        W["fmt_45"] = QCheckBox(tr("4:5 (Instagram feed)"))
        W["fmt_11"] = QCheckBox(tr("1:1 (square)"))
        W["fmt_169"] = QCheckBox(tr("16:9 (YouTube landscape)"))
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText(tr("Next to the source video"))
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._pick_out_dir)
        row = QHBoxLayout()
        row.addWidget(self.out_dir, 1)
        row.addWidget(browse)
        self.suffix = QLineEdit("_short")
        self.cache_lab = hint("")
        clear = QPushButton(tr("Clear cache"))
        clear.clicked.connect(self._clear_cache)
        crow = QHBoxLayout()
        crow.addWidget(self.cache_lab, 1)
        crow.addWidget(clear)
        f.addRow(tr("Resolution"), W["resolution"])
        f.addRow(tr("Encoding"), W["preset"])
        f.addRow(tr("Quality (CRF)"), W["crf"])
        f.addRow(QLabel(tr("Also export as:")))
        f.addRow(W["fmt_45"])
        f.addRow(W["fmt_11"])
        f.addRow(W["fmt_169"])
        f.addRow(tr("Output folder"), row)
        f.addRow(tr("File suffix"), self.suffix)
        W["encoder"] = combo([(tr("Auto (hardware on Mac)"), "auto"), ("Hardware (VideoToolbox)", "hardware"),
                              ("Software (x264)", "software")])
        W["slowmo"] = seg([(tr("Off"), 1), ("2× slower", 2), ("4× slower", 4)])
        W["slowmo_engine"] = seg([(tr("Neural (RIFE)"), "rife"), (tr("FFmpeg blend"), "ffmpeg")])
        W["cover"] = QCheckBox(tr("Also create the thumbnail (JPG)"))
        W["social"] = QCheckBox(tr("Create title, description and hashtags (.txt)"))
        f.addRow(tr("Encoder"), W["encoder"])
        f.addRow(tr("Slow motion"), W["slowmo"])
        f.addRow(tr("Interp engine"), W["slowmo_engine"])
        f.addRow(hint(tr("RIFE invents the in-between frames (21 MB, once): real slow motion, export only. FFmpeg's blend is instant but smears fast action.")))
        f.addRow(W["cover"])
        f.addRow(W["social"])
        f.addRow(tr("Analysis cache"), crow)
        f.addRow(hint(tr("Extra formats reuse the same analysis: they only cost the render. The Mac hardware encoder is 3–5× faster. Lower CRF = higher quality and bigger files (with hardware it sets the bitrate).")))
        self.ui_lang = combo([("English", "en"), ("Italiano", "it")])
        self.ui_lang.setCurrentIndex(max(0, self.ui_lang.findData(i18n.lang())))
        self.ui_lang.currentIndexChanged.connect(
            lambda _i: self._settings().setValue("ui/lang", self.ui_lang.currentData()))
        f.addRow(tr("UI language"), self.ui_lang)
        f.addRow(hint(tr("Restart the app to apply the new language.")))
        tabs.addTab(self._page(f), "Export")
        return col

    RAIL = [("crop", "Crop"), ("cut", "Cuts"), ("audio", "Audio"), ("quality", "Quality"),
            ("effects", "Effects"), ("color", "Color"), ("texts", "Texts"), ("export", "Export")]

    def _build_rail(self, tabs):
        w = QWidget()
        w.setObjectName("rail")
        w.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 10, 4, 10)
        lay.setSpacing(3)
        self._rail_btns = []
        for i, (icon, name) in enumerate(self.RAIL):
            b = QToolButton()
            b.setCheckable(True)
            b.setObjectName("railBtn")
            b.setIcon(_rail_icon(icon))
            b.setIconSize(QSize(19, 19))
            b.setText(tr(name))
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.clicked.connect(lambda _c, i=i: tabs.setCurrentIndex(i))
            lay.addWidget(b)
            self._rail_btns.append(b)
        lay.addStretch(1)
        tabs.currentChanged.connect(self._rail_sync)
        self._rail_sync(tabs.currentIndex())
        return w

    def _rail_sync(self, i):
        for j, b in enumerate(self._rail_btns):
            b.setChecked(j == i)

    def _page(self, form: QFormLayout):
        page = QWidget()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setVerticalSpacing(8)
        form.setContentsMargins(12, 12, 12, 12)
        page.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _build_bottom(self):
        box = QFrame()
        box.setObjectName("bottom")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 6, 10, 8)
        row = QHBoxLayout()
        self.stage_lab = QLabel(tr("Ready"))
        self.stage_lab.setObjectName("status")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.start_btn = QPushButton(tr("Create short"))
        self.start_btn.setObjectName("accent")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setMinimumWidth(150)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.stage_lab)
        row.addWidget(self.bar, 1)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lay.addWidget(self.log_view, 1)
        return box

    # ================================================================ settings
    def _val(self, name):
        wd = self.w[name]
        if isinstance(wd, (QDoubleSpinBox, QSpinBox)):
            return wd.value()
        if isinstance(wd, QComboBox):
            return wd.currentData()
        if isinstance(wd, SegControl):
            return wd.value()
        if isinstance(wd, QCheckBox):
            return wd.isChecked()
        return None

    def _set(self, name, value):
        wd = self.w.get(name)
        if wd is None:
            return
        if isinstance(wd, QDoubleSpinBox):
            wd.setValue(float(value))
        elif isinstance(wd, QSpinBox):
            wd.setValue(int(value))
        elif isinstance(wd, QComboBox):
            i = wd.findData(value)
            if i < 0:
                i = wd.findText(str(value))
            if i >= 0:
                wd.setCurrentIndex(i)
        elif isinstance(wd, SegControl):
            wd.setValue(value)
        elif isinstance(wd, QCheckBox):
            wd.setChecked(bool(value))

    def reframe_options(self) -> Options:
        v = self._val
        return Options(
            aspect=v("aspect"), zoom=v("zoom"), follow_vertical=v("follow_vertical"),
            face_height_pos=v("face_pos"), stride=int(v("stride")),
            track=TrackSettings(smoothing=v("smoothing"), dead_zone=v("dead_zone"), max_speed=v("max_speed"),
                                subject=v("subject"), noface=v("noface"),
                                look_room=v("look_room_amount") if v("look_room") else 0.0),
            shots=ShotSettings(wide_mode=v("wide_mode"), wide_threshold=v("wide_threshold"),
                               max_zoom=v("max_zoom")),
        )

    def job_settings(self) -> JobSettings:
        from .look import Grade
        v = self._val
        formats = [a for k, a in (("fmt_45", "4:5"), ("fmt_11", "1:1"), ("fmt_169", "16:9")) if v(k)]
        return JobSettings(
            reframe=self.reframe_options(),
            output_height=int(v("resolution")),
            enhance=EnhanceOptions(mode=v("enhance"), sharpen=v("sharpen"), face_boost=v("face_boost")),
            skin=v("skin"), skin_strength=v("skin_strength"),
            beauty=self.beauty_settings(),
            overlays=self.overlay_settings(),
            hook_by_file=dict(self.hook_text),
            cover=v("cover"), social=v("social"), sub_lang=v("sub_lang"),
            look=v("look"), look_strength=v("look_strength"), auto_wb=v("auto_wb"),
            grade=Grade(temp=v("color_temp"), tint=v("color_tint"), exposure=v("color_exp"),
                        contrast=v("color_con"), saturation=v("color_sat"),
                        shadows=v("color_shad"), highlights=v("color_high"), vignette=v("color_vig")),
            lowlight=v("lowlight"), lowlight_strength=v("lowlight_strength"),
            stabilize=v("stabilize"), stab_amount=v("stab_amount"), encoder=v("encoder"),
            bg_blur=v("bg_blur_amount") if v("bg_blur") else 0.0,
            bg_engine=v("bg_engine"),
            restore=v("restore"), restore_blend=v("restore_blend"), restore_fidelity=v("restore_fidelity"),
            formats=formats, layout=v("layout"), punch=v("punch"), punch_zoom=v("punch_zoom"),
            cut_pauses=v("cut_pauses"),
            speech=SpeechParams(vad=v("speech_engine") != "db",
                                min_pause=v("min_pause"), padding=v("padding"), margin_db=v("margin_db")),
            speaker_engine=v("speaker_engine"),
            cut_breaths=v("cut_breaths"), cut_fillers=v("cut_fillers"), cut_retakes=v("cut_retakes"),
            cinematic=v("cinematic"),
            cinematic_params=CinematicParams(denoise_db=v("denoise"), denoise_engine=v("denoise_engine"),
                                             dereverb_db=v("dereverb"),
                                             warmth_db=v("warmth"), presence_db=v("presence"),
                                             air_db=v("air")),
            music_path=self.music_path or None, music_db=v("music_db"), duck_ratio=v("duck"),
            subtitles=v("subtitles"), caption_style=v("caption_style"), language=v("language"),
            caption_words=int(v("caption_words")), caption_max_dur=v("caption_max_dur"),
            caption_pos=v("caption_pos"),
            person_by_file=dict(self.person_pick),
            motion=self.motion_settings(),
            manual_by_file={k: list(m) for k, m in self.manual.items() if m},
            preset=v("preset"), crf=int(v("crf")),
            slowmo=int(v("slowmo")), slowmo_engine=v("slowmo_engine"),
            output_dir=self.out_dir.text().strip() or None,
            suffix=self.suffix.text() or "_short",
        )

    def overlay_settings(self):
        from .overlays import OverlaySettings
        v = self._val
        return OverlaySettings(hook=v("hook"), hook_dur=v("hook_dur"), cta=self.cta_edit.text(),
                               cta_dur=v("cta_dur"), logo_path=self.logo_path or None,
                               logo_corner=v("logo_corner"), logo_size=v("logo_size"),
                               logo_opacity=v("logo_opacity"))

    def _hook_edited(self, text):
        if self.current:
            key = os.path.abspath(self.current)
            self._push_undo(tr("Opening title"), "hook", key, self.hook_text.get(key, ""))
            self.hook_text[key] = text
            if text and self._val("hook") != "custom":
                self._set("hook", "custom")
            self.replan_timer.start()

    def _pick_logo(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("Logo"), "", tr("Images (*.png *.jpg *.jpeg *.webp)"))
        if f:
            self._set_logo(f)

    def _set_logo(self, path):
        self.logo_path = path if path and os.path.exists(path) else ""
        self.logo_edit.setText(os.path.basename(self.logo_path))
        self._settings_changed()

    def beauty_settings(self):
        if not self._val("beauty"):
            return None
        from ..beauty import BeautySettings
        return BeautySettings(self._val("beauty_smooth"), self._val("beauty_blemish"),
                              self._val("beauty_wrinkles"))

    def _beautifier(self):
        b = self.beauty_settings()
        if b is None or not b.active:
            return None
        key = (b.smooth, b.blemish, b.wrinkles)
        if getattr(self, "_beauty_key", None) != key:
            from ..beauty import Beautifier, FaceParser, parse_available
            parser = FaceParser() if parse_available() else None
            self._beauty_obj, self._beauty_key = Beautifier(b, parser=parser), key
        return self._beauty_obj

    def motion_settings(self):
        from .motion import MotionSettings
        v = self._val
        return MotionSettings(
            handheld=v("mv_handheld"), handheld_amount=v("mv_handheld_amount"),
            push=v("mv_push"), push_amount=v("mv_push_amount"),
            crash=v("mv_crash"), crash_amount=v("mv_crash_amount"),
            shake=v("mv_shake"), shake_amount=v("mv_shake_amount"),
            transition=v("mv_transition"), transition_every=v("mv_every"), min_gap=v("mv_gap"))

    def _settings(self):
        return QSettings("face_reframe", "ShortsStudio")

    def _load_settings(self):
        st = self._settings()
        for name, wd in self.w.items():
            val = st.value("opt/" + name)
            if val is None:
                continue
            try:
                if isinstance(wd, QDoubleSpinBox):
                    wd.setValue(float(val))
                elif isinstance(wd, QSpinBox):
                    wd.setValue(int(float(val)))
                elif isinstance(wd, QComboBox):
                    i = wd.findData(val)
                    if i < 0:
                        try:
                            i = wd.findData(int(val))
                        except (TypeError, ValueError):
                            pass
                    if i < 0:
                        i = wd.findText(str(val))
                    if i >= 0:
                        wd.setCurrentIndex(i)
                elif isinstance(wd, SegControl):
                    wd.setValue(val)
                elif isinstance(wd, QCheckBox):
                    wd.setChecked(str(val).lower() in ("true", "1"))
            except (TypeError, ValueError):
                pass
        self.out_dir.setText(st.value("out_dir", "") or "")
        self.suffix.setText(st.value("suffix", "_short") or "_short")
        self._set_music(st.value("music", "") or "")
        self.cta_edit.setText(st.value("cta", "") or "")
        self._set_logo(st.value("logo", "") or "")
        self.user_presets = presets.load_user(st.value("presets", "") or "")
        self._fill_presets()
        geo = st.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def _save_settings(self):
        st = self._settings()
        for name, wd in self.w.items():
            if isinstance(wd, (QDoubleSpinBox, QSpinBox)):
                st.setValue("opt/" + name, wd.value())
            elif isinstance(wd, QComboBox):
                st.setValue("opt/" + name, wd.currentData())
            elif isinstance(wd, SegControl):
                st.setValue("opt/" + name, wd.value())
            elif isinstance(wd, QCheckBox):
                st.setValue("opt/" + name, wd.isChecked())
        st.setValue("out_dir", self.out_dir.text())
        st.setValue("suffix", self.suffix.text())
        st.setValue("music", self.music_path)
        st.setValue("cta", self.cta_edit.text())
        st.setValue("logo", self.logo_path)
        st.setValue("presets", presets.dump_user(self.user_presets))
        st.setValue("geometry", self.saveGeometry())

    def _sync_enabled(self):
        v = self._val
        for k in ("speech_engine", "min_pause", "padding"):
            self.w[k].setEnabled(v("cut_pauses"))
        # margin_db is the RMS speech threshold; breaths are always dB-based
        self.w["margin_db"].setEnabled(
            v("cut_breaths") or (v("cut_pauses") and v("speech_engine") == "db"))
        for k in ("denoise", "denoise_engine", "dereverb", "warmth", "presence", "air"):
            self.w[k].setEnabled(v("cinematic"))
        for k in ("music_db", "duck"):
            self.w[k].setEnabled(bool(self.music_path))
        for k in ("sharpen", "face_boost"):
            self.w[k].setEnabled(v("enhance") != "off")
        self.w["restore_blend"].setEnabled(v("restore") != "off")
        self.w["bg_blur_amount"].setEnabled(v("bg_blur"))
        self.w["bg_engine"].setEnabled(v("bg_blur"))
        self.w["look_room_amount"].setEnabled(v("look_room"))
        self.w["skin_strength"].setEnabled(v("skin"))
        for k in ("beauty_smooth", "beauty_blemish", "beauty_wrinkles"):
            self.w[k].setEnabled(v("beauty"))
        self.w["restore_fidelity"].setEnabled(v("restore") == "codeformer")
        for k in ("language", "caption_words", "caption_max_dur"):
            self.w[k].setEnabled(v("subtitles") != "off")
        for k in ("caption_style", "caption_pos", "caption_color", "caption_upper"):
            self.w[k].setEnabled(v("subtitles") == "burn")
        self.w["caption_hl"].setEnabled(v("subtitles") == "burn" and v("caption_style") == "word")
        self.w["punch_zoom"].setEnabled(v("punch"))
        for k in ("handheld", "push", "crash", "shake"):
            self.w["mv_%s_amount" % k].setEnabled(v("mv_" + k))
        self.w["mv_every"].setEnabled(v("mv_transition") != "off")
        self.w["look_strength"].setEnabled(v("look") != "none")
        self.w["stab_amount"].setEnabled(v("stabilize"))
        self.w["hook_dur"].setEnabled(v("hook") != "off")
        self.hook_edit.setEnabled(v("hook") != "off")
        self.w["sub_lang"].setEnabled(v("subtitles") != "off")
        has_logo = bool(self.logo_path)
        for k in ("logo_corner", "logo_size", "logo_opacity"):
            self.w[k].setEnabled(has_logo)
        self.w["face_pos"].setEnabled(v("follow_vertical") or v("zoom") > 1.0 or v("wide_mode") == "zoom")

    def _settings_changed(self, *_):
        self._sync_enabled()
        self._enh = None
        self._renderer = None
        self.replan_timer.start()
        if self.play_timer.isActive() and self._pa_key_changed():
            self.pa_timer.start()

    # ---------------------------------------------------------- presets
    def _fill_presets(self):
        self.preset_box.blockSignals(True)
        self.preset_box.clear()
        self.preset_box.addItem(tr("— choose —"), None)
        for name in presets.BUILTIN:
            self.preset_box.addItem(name, ("builtin", name))
        for name in sorted(self.user_presets):
            self.preset_box.addItem("★ " + name, ("user", name))
        self.preset_box.blockSignals(False)

    def _apply_preset(self, idx):
        data = self.preset_box.itemData(idx)
        if not data:
            return
        kind, name = data
        values = presets.BUILTIN[name] if kind == "builtin" else self.user_presets.get(name, {})
        for k, val in values.items():
            self._set(k, val)
        self._log(tr("Preset “%s” applied") % name)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, tr("Save preset"), tr("Preset name:"))
        name = (name or "").strip()
        if not ok or not name:
            return
        self.user_presets[name] = {k: self._val(k) for k in self.w}
        self._fill_presets()
        self.preset_box.setCurrentIndex(self.preset_box.findData(("user", name)))
        self._save_settings()

    def _delete_preset(self):
        data = self.preset_box.currentData()
        if not data or data[0] != "user":
            QMessageBox.information(self, APP_NAME, tr("Only presets you saved yourself (★) can be deleted."))
            return
        self.user_presets.pop(data[1], None)
        self._fill_presets()
        self._save_settings()

    # ---------------------------------------------------------- misc settings
    def _pick_music(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("Background music"), "", tr("Audio (%s)") % AUDIO_EXT)
        if f:
            self._set_music(f)

    def _set_music(self, path):
        self.music_path = path if path and os.path.exists(path) else ""
        self.music_edit.setText(os.path.basename(self.music_path))
        if "music_db" in self.w:
            self._settings_changed()

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("Output folder"), self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _refresh_cache_label(self):
        try:
            self.cache_lab.setText("%.0f MB" % diskcache.size_mb())
        except OSError:
            self.cache_lab.setText("")

    def _clear_cache(self):
        diskcache.clear()
        self._refresh_cache_label()
        self._log(tr("Cache cleared"))

    # ================================================================ queue
    def _items(self) -> List[dict]:
        return [self.queue.item(i).data(Qt.UserRole) for i in range(self.queue.count())]

    def _paths(self) -> List[str]:
        return [it["path"] for it in self._items() if not it.get("clip")]

    def _add_item(self, data: dict, row: Optional[int] = None):
        it = QListWidgetItem("○  " + data["label"])
        it.setIcon(_icon("film"))
        it.setData(Qt.UserRole, data)
        it.setToolTip(data["path"])
        if row is None:
            self.queue.addItem(it)
        else:
            self.queue.insertItem(row, it)
        self._schedule_autosave()
        return it

    def add_files(self, paths):
        have = set(self._paths())
        for p in paths:
            if os.path.isdir(p):
                self.add_files(sorted(os.path.join(p, n) for n in os.listdir(p)))
                continue
            if os.path.splitext(p)[1].lower() not in VIDEO_EXT or p in have:
                continue
            self._add_item({"path": p, "label": os.path.basename(p)})
            have.add(p)
        self._update_count()
        if self.queue.currentRow() < 0 and self.queue.count():
            self.queue.setCurrentRow(0)

    def _add_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("Add videos"), "", tr("Video (%s)") % " ".join("*" + e for e in sorted(VIDEO_EXT)))
        self.add_files(files)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("Add every video in a folder"))
        if d:
            self.add_files([d])

    def _update_count(self):
        n = self.queue.count()
        self.drop_hint.setVisible(n == 0)
        self.start_btn.setText(tr("Create short") if n <= 1 else tr("Create %d shorts") % n)

    def _remove_selected(self):
        for it in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(it))
        self._update_count()

    def _set_status(self, row: int, mark: str):
        it = self.queue.item(row)
        if it is not None:
            it.setText("%s  %s" % (mark, it.data(Qt.UserRole)["label"]))

    def _open_output(self, it):
        data = it.data(Qt.UserRole)
        outs = data.get("outs") or [output_path(data["path"], self.job_settings(), data.get("stem"))]
        if outs and os.path.exists(outs[0]):
            QDesktopServices.openUrl(QUrl.fromLocalFile(outs[0]))

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        self.add_files([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])

    # ================================================================ background work
    def _run(self, fn, on_result, label: str, on_fail=None):
        """Run ``fn(progress)`` in a thread; ``on_result(value)`` / ``on_fail(err)``
        are called on the GUI thread."""
        worker = Worker(fn)
        th = QThread()
        worker.moveToThread(th)
        th.started.connect(worker.run)
        rec = {"th": th, "w": worker, "label": label}

        def finish():
            th.quit()
            th.wait()
            if rec in self._workers:
                self._workers.remove(rec)
            if not self._workers:
                self.stage_lab.setText(tr("Ready"))

        def ok(value):
            finish()
            on_result(value)

        def bad(err):
            finish()
            self._bg_failed(err)
            if on_fail is not None:
                on_fail(err)

        rec["relays"] = (Relay(ok, self), Relay(bad, self))
        worker.progress.connect(self._bg_progress)
        worker.result.connect(rec["relays"][0].deliver)
        worker.failed.connect(rec["relays"][1].deliver)
        self._workers.append(rec)
        self.stage_lab.setText(label + "…")
        th.start()
        return worker

    def _bg_progress(self, f):
        self.bar.setValue(int(f * 1000))
        if self._workers:
            self.stage_lab.setText("%s… %d%%" % (self._workers[-1]["label"], int(f * 100)))

    def _bg_failed(self, err):
        if err != CANCELLED:
            self._log(tr("Error: %s") % err)

    def _busy(self, label: str) -> bool:
        return any(r["label"] == label for r in self._workers)

    # ================================================================ preview
    def _select(self, cur, _prev=None):
        self._pause()
        data = cur.data(Qt.UserRole) if cur else None
        path = data["path"] if data else None
        if path != self.current:
            self._stop_audio()
            if self.reader:
                self.reader.close()
                self.reader = None
            if self.proxy:
                self.proxy.close()
                self.proxy = None
            self.pcm = {}
            self.removed = []
            self.current = path
            if path:
                try:
                    self.reader = VideoReader(path)
                except IOError as exc:
                    self._log(str(exc))
                    self.current = None
        self.result = None
        if not self.current or not self.reader:
            self.src_view.set_frame(None)
            self.out_view.set_frame(None)
            return
        clip = data.get("clip")
        fps = self.reader.fps
        if clip:
            self.clip_range = (int(clip[0] * fps), min(self.reader.count - 1, int(clip[1] * fps)))
        else:
            self.clip_range = None
        lo, hi = self.clip_range or (0, self.reader.count - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(lo, hi)
        self.slider.setValue(lo)
        self.slider.blockSignals(False)
        self._load_audio()
        self._load_proxy()
        self._replan()

    # ---------------------------------------------------------- playback proxy
    def _load_proxy(self):
        """Open the cached proxy, or build one if the source is heavy."""
        from . import proxy
        path = self.current
        if not path or self.proxy is not None:
            return
        p = proxy.existing(path)
        if p:
            self._open_proxy(path, p)
            return
        if not proxy.needs_proxy(self.reader) or self._busy(tr("Preview proxy")):
            return
        out = proxy.proxy_path(path)
        dur = self.reader.count / self.reader.fps
        self._log(tr("Heavy video: making a light copy for smooth preview…"))

        def job(progress):
            return path, proxy.build(path, out, progress, dur, self.reader.count)

        def done(res):
            src, p2 = res
            if src == self.current:
                self._open_proxy(src, p2)

        self._run(job, done, tr("Preview proxy"))

    def _open_proxy(self, src, p):
        from . import proxy
        try:
            pr = VideoReader(p)
        except IOError:
            return
        # a proxy built here is verified against the decoder, not cv2's
        # metadata estimate (which is off on some VFR / edit-list files)
        exp = proxy.verified_count(p) or self.reader.count
        if abs(pr.count - exp) > 2:
            self._log(tr("Proxy discarded (frame count %d, source has %d)")
                      % (pr.count, exp))
            pr.close()
            try:
                os.unlink(p)   # poisoned cache entry: let the next open rebuild it
                os.unlink(p + ".frames")
            except OSError:
                pass
            if src not in self._proxy_retries:
                self._proxy_retries.add(src)
                QTimer.singleShot(0, self._load_proxy)
            return
        self.proxy = pr
        self.proxy.sequential = self.play_timer.isActive()
        self._log(tr("Preview: playing the light copy"))

    def _view_reader(self):
        """Proxy while playing or scrubbing (fluid); the original on a still."""
        if self.proxy is not None and (self.play_timer.isActive() or self.slider.isSliderDown()):
            return self.proxy
        return self.reader

    def _load_audio(self):
        path = self.current
        if not path or os.path.abspath(path) in self.pcm or self._busy("Audio"):
            return

        def job(_progress):
            from ..silence import extract_audio
            try:
                return path, extract_audio(path)
            except Exception:
                return path, None

        self._run(job, self._audio_loaded, "Audio")

    def _audio_loaded(self, res):
        path, pcm = res
        if pcm is not None and path == self.current:
            self.pcm = {os.path.abspath(path): pcm}
            self.replan_timer.start()

    def preview_options(self) -> Options:
        o = self.reframe_options()
        key = os.path.abspath(self.current or "")
        return dataclasses.replace(o, track=dataclasses.replace(
            o.track, person=self.person_pick.get(key) or None, manual=self.manual.get(key) or None,
            voice=self._voice.get(key) if (o.track.subject == "speaker"
                                           and self._val("speaker_engine") == "voice") else None))

    def _words(self):
        return self.cache.words(self.current, self._val("language")) if self.current else None

    def _replan(self):
        """Recompute everything the preview shows from the cached analyses."""
        self.result = None
        self._split = self._punch = self._burner = self._motion = None
        self._schedule_autosave()
        if not self.current or not self.reader:
            return
        key = os.path.abspath(self.current)
        self.hook_edit.blockSignals(True)
        self.hook_edit.setText(self.hook_text.get(key, ""))
        self.hook_edit.blockSignals(False)
        s = self.job_settings()
        pcm = self.pcm.get(key)
        words = self._words()
        fps = self.reader.fps
        # cut ranges (source timeline) for the timeline strip
        self.removed = []
        if pcm is not None and s.cuts:
            from .jobs import cut_spans
            dur = len(pcm) / 16000.0
            spans = cut_spans(pcm, dur, s, words, lambda *_: None)
            if spans:
                edges = [0.0] + [t for sp in spans for t in sp] + [dur]
                self.removed = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a > 0.02]
        a = self.cache.get(self.current, self.reframe_options())
        cuts, info = [], ""
        if a is None:
            self._clear_people()
            info = tr("Press “Analyze faces” to see the vertical crop.")
        else:
            from ..pipeline import plan_from_analysis
            o = self.preview_options()
            if (o.track.noface == "object" or o.track.subject == "object") \
                    and not getattr(a, "salient_pass", False):
                self._object_fill(a)
            if o.track.subject == "speaker" and self._val("speaker_engine") == "voice" \
                    and key not in self._voice:
                self._voice_fill()
            self._fill_people(a)
            if s.skin:
                self._measure_skin(a)
            self.result = plan_from_analysis(a, o)
            cuts = [x for x, _ in self.result.path.segments[1:]]
            shots = self.result.path.shots
            wide = sum(1 for sh in shots if sh.kind == "wide")
            found = int(self.result.path.has_face.sum())
            info = tr("%d shots (%d wide) · face on %d%% of frames") % (
                len(shots), wide, 100 * found // max(1, a.frame_count))
            ow, oh = self._preview_size()
            if s.layout == "split":
                from .compose import build_split
                self._split = build_split(a, o, ow, oh, self.person_pick.get(key))
                info += tr(" · split on %d%% of frames") % (
                    100 * int(self._split.mask.sum()) // max(1, a.frame_count)) if self._split else \
                    tr(" · split: need 2 people in the same shot")
            if s.punch and pcm is not None:
                from ..silence import speech_intervals
                from .compose import punch_curve
                self._punch = punch_curve(speech_intervals(pcm, s.speech), a.frame_count, fps, s.punch_zoom)
            if s.motion is not None and s.motion.active:
                from ..silence import speech_intervals
                from .jobs import _seed
                from .motion import build_motion
                shot_cuts = [x for x, _ in self.result.path.segments[1:]]
                rem = [(int(round(a0 * fps)), int(round(b0 * fps))) for a0, b0 in self.removed]
                edits = sorted([(c, c) for c in shot_cuts] + rem)
                phrases = speech_intervals(pcm, s.speech) if pcm is not None else []
                self._motion = build_motion(a.frame_count, fps, s.motion, edits, shot_cuts, phrases,
                                            words, pcm, seed=_seed(self.current), skip=rem)
            if s.stabilize:
                st = self._stab_for(a)
                if st is not None:
                    from .motion import MotionCurves
                    n = min(a.frame_count, len(st.dx))
                    c = self._motion if self._motion is not None else MotionCurves.empty(a.frame_count)
                    sd = np.zeros(a.frame_count)
                    sy = np.zeros(a.frame_count)
                    sr = np.zeros(a.frame_count)
                    k = s.stab_amount
                    sd[:n], sy[:n], sr[:n] = st.dx[:n] * k, st.dy[:n] * k, st.rot[:n] * k
                    pl = self.result.plan
                    self._motion = MotionCurves(c.dx + sd / np.maximum(1.0, pl.crop_w),
                                                c.dy + sy / np.maximum(1.0, pl.crop_h), c.zoom * 1.04,
                                                c.rot + sr, c.whip, c.zblur, c.glitch)
                    info += tr(" · stabilized")
        if s.cuts and self.removed:
            info += tr(" · cuts: −%.1fs") % sum(b - a2 for a2, b in self.removed)
        ow, oh = self._preview_size()
        parts = []
        if words and s.subtitles == "burn":
            from .jobs import CaptionBurner, group_word_events
            cap_words, second = words, None
            if s.sub_lang in ("en", "both"):
                from .translate import cached_english
                en = cached_english(self.current)
                if en is None:
                    info += tr(" · translation: done at the first export (or with “Transcribe”)")
                elif s.sub_lang == "en":
                    cap_words = en
                else:
                    second = en
            parts.append(CaptionBurner(group_word_events(cap_words, s.caption_words, s.caption_max_dur),
                                       fps, ow, oh, self._caption_renderer(),
                                       0.5 if self._split is not None else s.caption_pos,
                                       s.caption_style, second_words=second))
        ov = s.overlays
        if ov is not None:
            from .overlays import build as build_overlays, first_sentence
            hook = (self.hook_text.get(key, "") if ov.hook == "custom" else first_sentence(words or [])) \
                if ov.hook != "off" else ""
            parts += build_overlays(ov, hook, self.reader.count, fps, ow, oh, self._title_renderer(),
                                    lambda *_: None)
        if s.bg_blur > 0:
            from .rvm import BlurBackground, rvm_available
            if rvm_available():
                if self._matter is None:
                    from .rvm import Matter
                    self._matter = Matter()
                parts.insert(0, BlurBackground(self._matter, s.bg_blur))  # first: don't soften text
            else:
                info += tr(" · background blur: the model downloads on first export")
        if parts:
            from .overlays import Chain
            self._burner = Chain(parts)
        self._look = None
        grade_on = s.grade is not None and (s.grade.active or s.grade.vignette > 0)
        if s.look != "none" or s.auto_wb or grade_on:
            from .look import Look
            self._look = Look(s.look, s.look_strength,
                              self._gains_for() if s.auto_wb else None, s.grade)
        man = self.manual.get(key) or []
        if man:
            info += tr(" · %d manual corrections") % len(man)
        self.info_lab.setText(info)
        vsegs = self._voice.get(key) or []
        events = [vsegs[i][0] for i in range(1, len(vsegs))
                  if vsegs[i][2] != vsegs[i - 1][2] and vsegs[i][2] >= 0]
        self.timeline.set_data(self.reader.count, fps, self.removed, cuts, [m[0] for m in man],
                               self.clip_range, bool(words), events=events)
        self._show_frame(self.slider.value())

    def _preview_size(self):
        ow, oh = output_size(self._val("aspect"), 1920)
        s = min(1.0, 960.0 / max(ow, oh))
        return int(ow * s) // 2 * 2, int(oh * s) // 2 * 2

    def _caption_renderer(self):
        if self._renderer is None:
            try:
                from .captions_qt import make_renderer
                self._renderer = make_renderer(self._val("caption_color"), self._val("caption_upper"),
                                               self._val("caption_hl"))
            except Exception:
                self._renderer = None
        return self._renderer

    def _show_frame(self, i):
        if not self.reader:
            return
        self.timeline.set_pos(i)
        rd = self._view_reader()
        frame = rd.read(i)
        if frame is None and rd is not self.reader:
            rd, frame = self.reader, self.reader.read(i)
        if frame is None:
            return
        k = rd.width / float(self.reader.width) if self.reader.width else 1.0
        fps = self.reader.fps
        self.time_lab.setText("%s / %s" % (_fmt(i / fps), _fmt(self.reader.count / fps)))
        r = self.result
        t = i / fps
        cut_here = any(a <= t < b for a, b in self.removed)
        if r is None or i >= len(r.plan.cx):
            self.src_view.set_frame(to_qimage(frame), badge="%dx%d" % (self.reader.width, self.reader.height))
            self.out_view.set_frame(None)
            return
        from .compose import Framer, crop_box, scaled, scaled_split
        src_r = r
        r = scaled(r, k)
        split_v = scaled_split(self._split, k)
        z = float(self._punch[i]) if self._punch is not None and i < len(self._punch) else 1.0
        x, y, w, h = crop_box(r.plan, i, z)
        shot = next((sh for sh in r.path.shots if sh.start <= i < sh.end), None)
        badge = ""
        if shot is not None:
            badge = {"close": tr("Close-up"), "wide": tr("Wide · zoom %.2f×") % shot.zoom,
                     "noface": tr("No face")}.get(shot.kind, shot.kind)
        if cut_here:
            badge = tr("✂ cut · ") + badge
        split_here = self._split is not None and i < len(self._split.mask) and self._split.mask[i]
        self.src_view.set_frame(to_qimage(frame), crop=None if split_here else (x, y, w, h),
                                face=(float(src_r.path.x[i]) * k, float(src_r.path.y[i]) * k)
                                if src_r.path.has_face[i] else None,
                                badge=badge + (" · split" if split_here else ""))
        self._view_k = k
        ow, oh = self._preview_size()
        improve = self.enh_preview.isChecked()
        playing = self.play_timer.isActive()
        # the AI upscaler costs ~0.15 s a frame: off while playing, like face restoration
        enh = self._enhancer() if improve and self._val("enhance") != "off" and not (
            playing and self._val("enhance") == "ai") else None
        from .jobs import _seed
        restore = improve and self._val("restore") != "off" and not self.play_timer.isActive()
        # ~0.2 s per face with the neural mask: off while frames keep coming
        # (playing, slider/timeline drag); _beauty_settle puts it back on the
        # still frame once the motion ends
        now = time.monotonic()
        moving = playing or self.slider.isSliderDown() or (now - self._frame_at < 0.12)
        self._frame_at = now
        beauty = None if moving else self._beautifier()
        if beauty is None and self._beautifier() is not None:
            # every skipped frame arms a settle; only the last one's survives
            QTimer.singleShot(150, self._beauty_settle)
        lowlight = None if moving else self._lifter()
        if lowlight is None and self._lifter() is not None:
            QTimer.singleShot(150, self._beauty_settle)
        out = Framer(r, ow, oh, enh, self._punch, split_v, None, self._motion,
                     _seed(self.current), self._skin_now(), None if restore else beauty,
                     getattr(self, "_look", None), lowlight=lowlight)(i, frame)
        if restore:
            # restoration runs on the frame before beauty (same order as the export)
            self._schedule_restore(i, out)
            if beauty is not None:
                out = beauty(out)
        self._present(i, out, tr("Restoring faces…") if restore else (tr("Enhanced") if enh is not None else tr("Preview")))

    # ---------------------------------------------------------- skin tone
    def _measure_skin(self, analysis):
        key = os.path.abspath(self.current)
        if key in self._skin:
            c = self._skin[key]
            self.skin_lab.setText(c.describe() if c else tr("No measurable face in this video."))
            return
        if self._busy(tr("Skin measure")):
            return
        path = self.current
        self.skin_lab.setText(tr("Measuring skin…"))

        def job(_progress):
            from ..skin import measure_video
            return key, measure_video(path, analysis)

        def done(res):
            k, c = res
            self._skin[k] = c
            if k == os.path.abspath(self.current or ""):
                self._replan()

        self._run(job, done, tr("Skin measure"))

    def _lifter(self):
        """Neural low-light lift for the preview — only when the model is
        already downloaded (it fetches on first export otherwise)."""
        key = self._val("lowlight")
        if key == "off":
            return None
        from .. import lowlight
        if not lowlight.available(key):
            return None
        k = (key, round(self._val("lowlight_strength"), 2))
        if getattr(self, "_ll_key", None) != k:
            try:
                self._ll = lowlight.Lifter(key, k[1])
            except Exception:
                self._ll = None
            self._ll_key = k
        return self._ll

    def _voice_fill(self):
        """Tag speech spans by voice (ECAPA) for the current video — once,
        in the background; then the speaker engine and the timeline marks."""
        if self._voice_on or self._busy(tr("Voice identification")):
            return
        self._voice_on = True
        path, key = self.current, os.path.abspath(self.current)
        speech = SpeechParams(vad=self._val("speech_engine") != "db",
                              min_pause=self._val("min_pause"), padding=self._val("padding"),
                              margin_db=self._val("margin_db"))

        def job(progress):
            from ..silence import extract_audio, speech_intervals
            from ..speakerid import analyze
            pcm = self.pcm.get(key)
            if pcm is None:
                pcm = extract_audio(path)
            return analyze(pcm, speech_intervals(pcm, speech), lambda *_: None, progress)

        def done(vsegs):
            self._voice_on = False
            if vsegs:
                self._voice[key] = vsegs
                n = len({lab for _, _, lab in vsegs if lab >= 0})
                self._log(tr("Voice ID: %d speaker%s") % (n, "s" if n != 1 else ""))
            else:
                self._voice[key] = []        # don't retry every replan
            if path == self.current:
                self._replan()

        self._run(job, done, tr("Voice identification"),
                  on_fail=lambda _e: setattr(self, "_voice_on", False))

    def _pick_object(self):
        """Arm the source view: the next drag draws the box NanoTrack follows."""
        if not self.current or not self.reader:
            return
        if self.cache.get(self.current, self.reframe_options()) is None:
            self._log(tr("Analyze faces first, then pick the object."))
            return
        self.src_view.pick_mode = True
        self.src_view.setCursor(Qt.CrossCursor)
        self._log(tr("Drag a box around the object to follow in the left preview"))

    def _object_picked(self, x: float, y: float, w: float, h: float):
        self.src_view.pick_mode = False
        self.src_view.unsetCursor()
        a = self.cache.get(self.current, self.reframe_options()) if self.current else None
        if a is None:
            return
        k = getattr(self, "_view_k", 1.0) or 1.0
        box = (x / k, y / k, max(8.0, w / k), max(8.0, h / k))
        start = self.slider.value()
        if self._busy(tr("Object tracking")):
            return
        self._log(tr("Tracking the object over the video…"))
        path, analysis = self.current, a

        def job(progress):
            from ..objtrack import track
            n = track(path, start, box, analysis, self._log, progress)
            analysis.salient_pass = True
            return n

        def done(n):
            if n and self.current == path:
                self._set("subject", "object")
                self.cache.put(path, self.reframe_options(), analysis)
                self._log(tr("Object locked: %d samples tracked") % n)
                self._replan()
            elif not n:
                self._log(tr("Could not track the object (model missing or lost immediately)"))

        self._run(job, done, tr("Object tracking"),
                  on_fail=lambda _e: self._log(tr("Object tracking failed")))

    def _object_fill(self, analysis):
        """Backfill salient objects on a cached analysis (only faceless
        frames are re-read), then replan — turning "Follow object" on after
        the faces were analysed doesn't need a full re-analysis."""
        if getattr(self, "_object_fill_on", False) or self._busy(tr("Object detection")):
            return
        if not any(not smp.faces for smp in analysis.samples):
            analysis.salient_pass = True
            return
        self._object_fill_on = True
        path, key = self.current, os.path.abspath(self.current)

        def job(progress):
            from ..saliency import LazySaliency, backfill
            sal = LazySaliency(lambda *_: None, progress)
            backfill(path, analysis, sal, progress)
            return sal.ok

        def done(ok):
            self._object_fill_on = False
            if ok and self.current:
                self.cache.put(self.current, self.reframe_options(), analysis)
                self._replan()
            elif not ok:
                self._log(tr("Object detection unavailable (offline): staying centred"))

        self._run(job, done, tr("Object detection"),
                on_fail=lambda _e: setattr(self, "_object_fill_on", False))

    def _skin_now(self):
        if not self._val("skin") or not self.current:
            return None
        c = self._skin.get(os.path.abspath(self.current))
        if c is None:
            return None
        key = (id(c), round(self._val("skin_strength"), 3))
        if self._skin_view is None or self._skin_view[0] != key:
            self._skin_view = (key, c.with_strength(key[1]))
        return self._skin_view[1]

    def _save_skin_cube(self):
        c = self._skin_now()
        if c is None:
            QMessageBox.information(self, APP_NAME, tr("Enable “Normalize skin color” and press “Analyze faces”: the LUT is based on the skin measured in the video."))
            return
        base = os.path.splitext(os.path.basename(self.current))[0]
        path, _ = QFileDialog.getSaveFileName(self, tr("Save LUT"), base + "_skin.cube", "LUT (*.cube)")
        if path:
            from ..skin import write_cube
            write_cube(c, path)
            self._log(tr("LUT saved: %s (in Resolve: right-click the clip > LUT)") % path)

    def _present(self, i, out, tag):
        if self._burner is not None:
            out = self._burner(i, out)
        ow2, oh2 = output_size(self._val("aspect"), int(self._val("resolution")))
        z = self.zones_box.currentData() if hasattr(self, "zones_box") else None
        self.out_view.zones = SAFE_ZONES.get(z) if z and self._val("aspect") == "9:16" else None
        self.out_view.set_frame(to_qimage(out), badge="%s · %dx%d" % (tag, ow2, oh2))

    def _beauty_settle(self):
        """The motion that skipped beauty just ended: repaint the still frame."""
        if (self.play_timer.isActive() or self.slider.isSliderDown()
                or time.monotonic() - self._frame_at < 0.14):
            return    # still moving: the newest frame queued its own settle
        self._refresh_frame()

    # ---------------------------------------------------------- preview face restoration
    def _restore_key(self):
        return (self._val("restore"), round(self._val("restore_fidelity"), 3), round(self._val("restore_blend"), 3))

    def _schedule_restore(self, i, out):
        """Restore the shown frame in the background; only the latest request
        survives while one is running (scrubbing doesn't queue up work)."""
        self._restore_pending = (i, out.copy(), self._restore_key())
        if not self._busy(tr("Face restoration")):
            self._start_restore()

    def _start_restore(self):
        i, out, key = self._restore_pending
        self._restore_pending = None
        s = self.job_settings()
        cached = self._restorer if self._restorer_key == key else None

        def job(progress):
            from .jobs import make_restorer
            r = cached or make_restorer(s, lambda *_: None, progress)
            r._prev = []  # preview frames aren't consecutive: no landmark smoothing
            img = r(out)
            if s.beauty is not None and s.beauty.active:
                from ..beauty import Beautifier, FaceParser, parse_available
                b = Beautifier(s.beauty, parser=FaceParser() if parse_available() else None)
                img = b(img)   # own instance: not shared across threads
            return i, key, r, img

        def done(res):
            fi, k, r, img = res
            self._restorer, self._restorer_key = r, k
            if self._restore_pending is not None:
                self._start_restore()
            elif fi == self.slider.value() and not self.play_timer.isActive():
                self._present(fi, img, tr("Face restored"))

        self._run(job, done, tr("Face restoration"))

    def _refresh_frame(self, *_):
        self._show_frame(self.slider.value())

    def _enhancer(self):
        key = (self._val("enhance"), self._val("sharpen"), self._val("face_boost"))
        if self._enh is None or self._enh_key != key:
            from ..enhance import Enhancer
            self._enh = Enhancer(EnhanceOptions(mode=key[0], sharpen=key[1], face_boost=key[2]),
                                 log=lambda *_: None)
            self._enh_key = key
        return self._enh

    def _seek(self, f):
        lo, hi = self.slider.minimum(), self.slider.maximum()
        f = max(lo, min(hi, f))
        self.slider.setValue(f)
        self._user_seek(f)

    # ---------------------------------------------------------- playback (+ audio)
    def _ensure_player(self) -> bool:
        """QMediaPlayer: we hear the audio and use its position as the clock
        (frames are drawn to match it). It plays the source file, or a WAV
        window of the treated soundtrack (cinematic audio, music). Without
        QtMultimedia or an audio device, playback falls back to silent."""
        if self.player is None:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
                self.audio_out = QAudioOutput(self)
                self.audio_out.setMuted(self._muted)
                self.player = QMediaPlayer(self)
                self.player.setAudioOutput(self.audio_out)
                self.player.errorOccurred.connect(self._player_error)
                self.player.mediaStatusChanged.connect(self._media_status)
            except Exception as exc:
                self._log(tr("Preview audio unavailable (%s)") % exc)
                self.player = False
        return bool(self.player) and bool(self.current)

    def _player_error(self, *_):
        if self.player:
            self._log(tr("Preview audio: %s") % self.player.errorString())

    def _audio_active(self) -> bool:
        if not self.player:
            return False
        from PySide6.QtMultimedia import QMediaPlayer
        return self.player.playbackState() == QMediaPlayer.PlayingState

    def _player_time(self) -> float:
        return self._pa_offset + self.player.position() / 1000.0

    def _set_source(self, path: str, offset: float, t: float):
        """Play ``path`` (whose 0 is source time ``offset``) from source time ``t``.

        Arms the seek guard: until the player really reports the new position
        the clock is extrapolated from ``t`` (the player's position lags a
        seek for a few ticks; reading the stale value used to bounce playback
        back into the part just skipped — the "stuck" loop)."""
        import time
        self._pa_offset = offset
        pos = int(max(0.0, t - offset) * 1000)
        self._seek_guard = (t, time.monotonic())
        if self._player_src != path:
            self._player_src = path
            self._pending_seek = pos
            self.player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
            self.player.play()
        elif self._pending_seek is not None:
            self._pending_seek = pos        # still loading: just retarget the seek
        else:
            self.player.setPosition(pos)
            self.player.play()

    def _clock(self):
        """Playback time in source seconds, or None without an audio clock."""
        import time
        g = self._seek_guard
        if g is not None:
            tgt, t0 = g
            est = tgt + (time.monotonic() - t0)
            if self._pending_seek is None and self._audio_active():
                pt = self._player_time()
                # the seek has landed once the player is past the target and
                # roughly where we expect it; give up waiting after 2 s
                if (pt >= tgt - 0.03 and abs(pt - est) < 0.35) or time.monotonic() - t0 > 2.0:
                    self._seek_guard = None
                    return max(pt, tgt)
            return est
        if self._audio_active():
            return self._player_time()
        return None

    def _media_status(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        # seek once the new file is buffered: a seek issued at LoadedMedia is
        # lost when playback then starts (Qt's FFmpeg backend)
        if self._pending_seek is not None and status == QMediaPlayer.BufferedMedia:
            pos, self._pending_seek = self._pending_seek, None
            self.player.setPosition(pos)
            if self.play_timer.isActive():
                self.player.play()

    # -- treated soundtrack (cinematic audio + music) for the preview
    def _pa_wanted(self) -> bool:
        return (self._val("cinematic") or bool(self.music_path)) and not self.ab_btn.isChecked()

    def _pa_key(self):
        v = self._val
        cine = tuple(v(k) for k in ("denoise", "dereverb", "warmth", "presence", "air")) if v("cinematic") else None
        return (self.current, cine, self.music_path, round(v("music_db"), 1), v("duck"))

    def _pa_key_changed(self) -> bool:
        return not any(w["key"] == self._pa_key() for w in self._pa_windows) or not self._pa_wanted()

    def _pa_find(self, t: float):
        """Treated window covering ``t``: the one playing if it does (switching
        files costs a reload), else the newest."""
        key = self._pa_key()
        ok = [w for w in self._pa_windows if w["key"] == key and w["start"] <= t < w["end"] - 0.5]
        for w in ok:
            if w["path"] == self._player_src:
                return w
        return ok[-1] if ok else None

    def _play_from(self, t: float):
        """(Re)start the audio at source time ``t`` with the right soundtrack."""
        if not self._ensure_player():
            return
        if self._pa_wanted():
            w = self._pa_find(t)
            if w is not None:
                self._set_source(w["path"], w["start"], t)
                return
            self._pa_build(t)  # meanwhile: original audio, swapped when ready
        self._set_source(self.current, 0.0, t)

    def _pa_refresh(self):
        if self.play_timer.isActive() and self.player:
            t = self._player_time() if self._audio_active() else self.slider.value() / self.reader.fps
            self._play_from(t)

    def _pa_build(self, t: float):
        if self._busy(tr("Preview audio")):
            self._pa_pending = t
            return
        if self._pa_dir is None:
            import tempfile
            self._pa_dir = tempfile.mkdtemp(prefix="shorts_preview_")
        key, s, src = self._pa_key(), self.job_settings(), self.current
        start = max(0.0, t - 0.5)
        dur = 40.0

        def job(_progress):
            from .jobs import preview_audio
            # a fresh file name each time: the player may still hold the old one
            import time
            out = os.path.join(self._pa_dir, "pa_%d.wav" % int(time.time() * 1000))
            preview_audio(src, start, dur, s, out)
            return key, out, start, start + dur

        def done(res):
            k, out, a, b = res
            self._pa_windows.append({"key": k, "path": out, "start": a, "end": b})
            # keep the playing window and the newest two; drop the rest
            keep = [w for w in self._pa_windows if w["path"] == self._player_src] + self._pa_windows[-2:]
            for w in self._pa_windows:
                if w not in keep:
                    try:
                        os.unlink(w["path"])
                    except OSError:
                        pass
            self._pa_windows = [w for w in self._pa_windows if w in keep]
            if self._pa_pending is not None:
                t2, self._pa_pending = self._pa_pending, None
                self._pa_build(t2)
                return
            if self.play_timer.isActive() and self._pa_wanted() and k == self._pa_key():
                now = self._player_time() if self._audio_active() else self.slider.value() / self.reader.fps
                if a <= now < b - 0.5:
                    self._set_source(out, a, now)
                    self.stage_lab.setText(tr("Audio: as exported"))

        self._run(job, done, tr("Preview audio"))

    def _ab_toggled(self, on):
        self.ab_btn.setToolTip(tr("You're hearing the original") if on else tr("You're hearing the treated audio"))
        self._pa_refresh()

    def _stop_audio(self):
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        self._player_src = None
        self._pa_windows = []

    def _pause(self):
        self._seek_guard = None
        was_playing = self.play_timer.isActive()
        if self.reader:
            self.reader.sequential = False
        if self.proxy is not None:
            self.proxy.sequential = False
        self.play_timer.stop()
        if was_playing and self.proxy is not None:
            QTimer.singleShot(0, self._refresh_frame)  # the still from the original
        self.play_btn.setIcon(_icon("play"))
        if self.player:
            self.player.pause()

    def _toggle_play(self):
        if not self.reader:
            return
        if self.play_timer.isActive():
            self._pause()
            return
        fps = self.reader.fps
        self.reader.sequential = True
        if self.proxy is not None:
            self.proxy.sequential = True
        self.play_timer.start(max(15, int(1000 / fps)))
        self.play_btn.setIcon(_icon("pause"))
        self._play_from(self.slider.value() / fps)

    def _user_seek(self, f):
        if self.play_timer.isActive() and self.player and self.reader:
            self._play_from(f / self.reader.fps)

    def _toggle_mute(self):
        self._muted = not self._muted
        if self.audio_out is not None:
            self.audio_out.setMuted(self._muted)
        self.mute_btn.setIcon(_icon("vol_off" if self._muted else "vol"))

    def _tick(self):
        if not self.reader:
            return
        fps = self.reader.fps
        lo, hi = self.slider.minimum(), self.slider.maximum()
        t = self._clock() if self.player else None
        audio = t is not None
        if audio:
            nxt = int(t * fps)
            if self._seek_guard is None:
                cur = next((w for w in self._pa_windows if w["path"] == self._player_src), None)
                if cur is not None and t > cur["end"] - 0.6:
                    self._play_from(t)               # this window is over: next one (or original)
                    return
                if (cur is not None and t > cur["end"] - 12 and not self._busy(tr("Preview audio"))
                        and self._pa_find(cur["end"] - 0.6) is None):
                    self._pa_build(cur["end"] - 1.0)  # prepare the next window ahead of time
        else:
            nxt = self.slider.value() + 1
        jump_t = None
        if self.skip_cuts.isChecked() and self.removed:
            tt = nxt / fps
            for a, b in self.removed:
                if a <= tt < b:
                    jump_t = b + 0.06               # land just past the cut
                    break
        if jump_t is not None and int(jump_t * fps) > hi:
            jump_t = lo / fps
        elif jump_t is None and nxt > hi:
            jump_t = lo / fps                       # loop the clip / range
        if jump_t is not None:
            nxt = int(round(jump_t * fps))
            if audio:
                self._play_from(jump_t)
        elif audio and nxt < self.slider.value():
            # the audio clock settling a frame or two back after a seek: hold
            # the picture (stepping back would force a slow decoder seek)
            nxt = self.slider.value()
        nxt = max(lo, min(hi, nxt))
        if nxt != self.slider.value():
            self.slider.setValue(nxt)

    # ---------------------------------------------------------- analysis
    def _analyze(self):
        if not self.current or self._busy(tr("Face analysis")):
            return
        path, opts = self.current, self.reframe_options()
        self.analyze_btn.setEnabled(False)

        def job(progress):
            from ..detect import analyze_video
            sal = None
            if opts.track.noface == "object" or opts.track.subject == "object":
                from ..saliency import LazySaliency
                sal = LazySaliency(lambda *_: None, progress)
            return path, analyze_video(path, 0, None, stride=opts.stride, min_score=opts.min_score,
                                       progress=progress, saliency=sal)

        def done(res):
            p, a = res
            self.cache.put(p, opts, a)
            self.analyze_btn.setEnabled(True)
            self._log(tr("Analysing %s: %d frames") % (os.path.basename(p), a.frame_count))
            self._refresh_cache_label()
            if p == self.current:
                self._people_for = None
                self._replan()

        self._run(job, done, tr("Face analysis"), on_fail=lambda _e: self.analyze_btn.setEnabled(True))

    def _transcribe(self, then=None):
        if not self.current or self._busy(tr("Transcription")):
            return
        path, lang = self.current, self._val("language")
        if self.cache.words(path, lang) is not None:
            self._log(tr("Transcript already cached"))
            self._replan()
            if then:
                then()
            return
        self.tr_btn.setEnabled(False)

        def job(progress):
            from .jobs import transcribe_file
            return transcribe_file(path, lang, self.cache, progress, lambda *_: None)

        def done(words):
            self.tr_btn.setEnabled(True)
            self._log(tr("Transcript: %d words") % len(words))
            self._refresh_cache_label()
            self._replan()
            if then:
                then()

        self._run(job, done, tr("Transcription"), on_fail=lambda _e: self.tr_btn.setEnabled(True))

    # ---------------------------------------------------------- subtitle editor
    def _open_subeditor(self):
        if not self.current:
            return
        if self._words() is None:
            self._log(tr("First I'll transcribe the video, then I'll open the caption editor…"))
            self._transcribe(then=self._open_subeditor)
            return
        ed = getattr(self, "_subeditor", None)
        if ed is not None and ed.isVisible() and ed.path == self.current:
            ed.raise_()
            ed.activateWindow()
            return
        if ed is not None:
            ed.close()
        from .subedit_qt import SubtitleEditor
        path, lang = self.current, self._val("language")
        fps = self.reader.fps

        def get_words():
            return self.cache.words(path, lang)

        def set_words(words):
            from . import diskcache
            from ..transcribe import MODEL_NAME
            extra = "%s|%s" % (lang or "", MODEL_NAME)
            if diskcache.load(path, "words_orig", extra) is None:
                diskcache.save(path, "words_orig", self.cache.words(path, lang), extra)  # keep Whisper's
            self._push_undo(tr("Edit captions"), "words", os.path.abspath(path), self.cache.words(path, lang))
            self.cache.put_words(path, lang, words)
            if path == self.current:
                self.replan_timer.start()

        def restore():
            from . import diskcache
            from ..transcribe import MODEL_NAME
            orig = diskcache.load(path, "words_orig", "%s|%s" % (lang or "", MODEL_NAME))
            if orig is not None:
                self.cache.put_words(path, lang, orig)
                if path == self.current:
                    self.replan_timer.start()
            return orig

        def settings():
            return int(self._val("caption_words")), float(self._val("caption_max_dur"))

        def seek(t):
            if path == self.current:
                self._seek(int(round(t * fps)))

        def now():
            return self.slider.value() / fps if path == self.current and self.reader else None

        ed = SubtitleEditor(self, os.path.basename(path), get_words, set_words, settings, seek, now, restore)
        ed.path = path
        self._subeditor = ed
        ed.show()

    # ---------------------------------------------------------- highlights
    def _find_highlights(self):
        it = self.queue.currentItem()
        if it is None:
            QMessageBox.information(self, APP_NAME, tr("Select a video in the list first."))
            return
        src = it.data(Qt.UserRole)
        if src.get("clip"):
            QMessageBox.information(self, APP_NAME, tr("Select the original video, not a clip."))
            return
        self._transcribe(then=lambda: self._highlights_ready(src))

    def _highlights_ready(self, src):
        words = self.cache.words(src["path"], self._val("language"))
        if not words:
            self._log(tr("No transcript: can't look for the best moments"))
            return
        if self._val("audio_events"):
            self._log(tr("Listening for laughter and applause (YAMNet)…"))

            def job(progress):
                from ..silence import extract_audio
                from ..audioevents import analyze
                pcm = extract_audio(src["path"])
                return analyze(pcm, lambda *_: None, progress)

            self._run(job, lambda ev: self._finish_highlights(src, words, ev),
                      tr("Audio events"), on_fail=lambda _e: self._finish_highlights(src, words, None))
            return
        self._finish_highlights(src, words, None)

    def _finish_highlights(self, src, words, events):
        from .highlights import find_highlights
        dur = self.reader.count / self.reader.fps if self.reader and self.current == src["path"] else words[-1].end
        hls = find_highlights(words, dur, count=self.hl_count.value(), events=events)
        if not hls:
            self._log(tr("Video too short or no speech: no moments suggested"))
            return
        row = next(i for i in range(self.queue.count()) if self.queue.item(i).data(Qt.UserRole) is src
                   or self.queue.item(i).data(Qt.UserRole) == src)
        base = os.path.splitext(os.path.basename(src["path"]))[0]
        n_existing = sum(1 for d in self._items() if d.get("clip") and d["path"] == src["path"])
        for k, h in enumerate(hls, 1 + n_existing):
            title = h.title if len(h.title) <= 38 else h.title[:36] + "…"
            self._add_item({"path": src["path"], "clip": (h.start, h.end), "stem": "%s_clip%d" % (base, k),
                            "label": "   ↳ %s–%s  %s" % (_fmt(h.start), _fmt(h.end), title)},
                           row + k - n_existing)
        self._log(tr("Found %d moments in %s: they're in the list under the video") % (len(hls), src["label"]))
        self._update_count()

    # ---------------------------------------------------------- people
    def _clear_people(self):
        for b in self.people_group.buttons():
            self.people_group.removeButton(b)
            self.people_layout.removeWidget(b)
            b.hide()
            b.deleteLater()
        self.people_bar.setVisible(False)
        self._people_for = None

    def _fill_people(self, analysis):
        """Thumbnail strip of everyone found in the video (rebuilt per video)."""
        key = (self.current, id(analysis))
        if self._people_for == key:
            return
        self._clear_people()
        self._people_for = key
        from ..track import find_people, match_person
        people = [p for p in find_people(analysis) if p.samples >= 5][:8]
        self._people = people
        if len(people) < 2 or not self.reader:
            return
        pick = self.person_pick.get(os.path.abspath(self.current)) or []
        chosen = {p.id for p in (match_person(people, sig) for sig in pick) if p is not None}
        auto = QToolButton()
        auto.setText(tr("Auto"))
        auto.setCheckable(True)
        auto.setFixedSize(64, 64)
        auto.setObjectName("person")
        auto.setToolTip(tr("Use the “Follow” setting (main person or whoever speaks)"))
        self.people_group.addButton(auto, -2)
        self.people_layout.insertWidget(self.people_layout.count() - 1, auto)
        auto.setChecked(not chosen)
        fps = analysis.fps or 25.0
        for p in people:
            frame = self.reader.read(p.best[0])
            f = p.best[1]
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setFixedSize(64, 64)
            btn.setObjectName("person")
            btn.setIconSize(QSize(56, 56))
            if frame is not None:
                import cv2
                r = int(max(f.w, f.h) * 1.0)
                x0, y0 = int(max(0, f.cx - r)), int(max(0, f.cy - 0.8 * r))
                crop = frame[y0:int(f.cy + 1.2 * r), x0:int(f.cx + r)]
                if crop.size:
                    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                    btn.setIcon(QIcon(QPixmap.fromImage(to_qimage(crop))))
            secs = int(p.samples * analysis.frame_count / max(1, len(analysis.samples)) / fps)
            btn.setToolTip(tr("Person %d · in %d shots · ~%ds on screen") % (p.id + 1, len(p.tracks), secs))
            self.people_group.addButton(btn, p.id)
            self.people_layout.insertWidget(self.people_layout.count() - 1, btn)
            btn.setChecked(p.id in chosen)
        self.people_bar.setVisible(True)
        self.reader.read(self.slider.value())

    def _pick_person(self, pid):
        """Multi-select: tick every thumbnail of the person to follow (the
        same person can show up twice, e.g. close-up and wide shot)."""
        path = os.path.abspath(self.current or "")
        buttons = {self.people_group.id(b): b for b in self.people_group.buttons()}
        if pid < 0:
            for k, b in buttons.items():
                b.setChecked(k < 0)
        else:
            any_on = any(b.isChecked() for k, b in buttons.items() if k >= 0)
            if -2 in buttons:
                buttons[-2].setChecked(not any_on)
        picked = [p for p in self._people if p.id in buttons and buttons[p.id].isChecked()]
        self._push_undo(tr("Person choice"), "person", path, self.person_pick.get(path))
        if picked:
            self.person_pick[path] = [p.sig for p in picked]
            self._log(tr("%s: following %s") % (os.path.basename(path),
                                         ", ".join(tr("person %d") % (p.id + 1) for p in picked)))
        else:
            self.person_pick.pop(path, None)
        self._replan()

    def _reset_grade(self):
        for k in ("color_temp", "color_tint", "color_exp", "color_shad", "color_high", "color_vig"):
            self._set(k, 0.0)
        self._set("color_con", 1.0)
        self._set("color_sat", 1.0)

    # ---------------------------------------------------------- look / stab / titles
    def _title_renderer(self):
        if getattr(self, "_trender", None) is None:
            try:
                from .captions_qt import make_title_renderer
                self._trender = make_title_renderer()
            except Exception:
                from .overlays import cv_title
                self._trender = cv_title
        return self._trender

    def _gains_for(self):
        """Auto white balance gains of the current file (measured in the background)."""
        key = os.path.abspath(self.current or "")
        cache = self.__dict__.setdefault("_gains", {})
        if key in cache:
            return cache[key]
        if not self._busy(tr("White balance")):
            path = self.current

            def job(_progress):
                from .look import auto_gains
                try:
                    return path, auto_gains(path)
                except Exception:
                    return path, None

            self._run(job, self._gains_done, tr("White balance"))
        return None

    def _gains_done(self, res):
        path, g = res
        self.__dict__.setdefault("_gains", {})[os.path.abspath(path)] = g
        if path == self.current:
            self.replan_timer.start()

    def _stab_for(self, a):
        key = os.path.abspath(self.current or "")
        cache = self.__dict__.setdefault("_stab", {})
        if key in cache:
            return cache[key]
        from . import diskcache
        st = diskcache.load(self.current, "stab", "v1")
        if st is not None:
            cache[key] = st
            return st
        if not self._busy(tr("Stabilization")):
            path = self.current
            cuts = [x.frame for x in a.samples if x.cut]
            self._log(tr("Stabilization: motion analysis (once per file)…"))

            def job(progress):
                from .stabilize import estimate
                st = estimate(path, cuts, progress=progress)
                if st is not None:
                    diskcache.save(path, "stab", st, "v1")
                return path, st

            self._run(job, self._stab_done, tr("Stabilization"))
        return None

    def _stab_done(self, res):
        path, st = res
        if st is not None:
            self.__dict__.setdefault("_stab", {})[os.path.abspath(path)] = st
            if path == self.current:
                self.replan_timer.start()

    # ---------------------------------------------------------- undo / redo
    def _push_undo(self, label, kind, key, old):
        now = time.monotonic()
        top = self._undo[-1] if self._undo else None
        # typing in the hook field: one step per burst
        if kind == "hook" and top and top[1] == "hook" and top[2] == key and now - top[4] < 1.5:
            self._undo[-1] = top[:4] + (now,)
            return
        self._undo.append((label, kind, key, old, now))
        del self._undo[:-100]
        self._redo.clear()
        self._sync_undo()
        self._schedule_autosave()

    def _state_of(self, kind, key):
        if kind == "manual":
            return list(self.manual.get(key, []))
        if kind == "person":
            return self.person_pick.get(key)
        if kind == "hook":
            return self.hook_text.get(key, "")
        if kind == "words":
            lang = self._val("language")
            return self.cache.words(key, lang)
        return None

    def _put_state(self, kind, key, val):
        if kind == "manual":
            if val:
                self.manual[key] = list(val)
            else:
                self.manual.pop(key, None)
        elif kind == "person":
            if val:
                self.person_pick[key] = val
            else:
                self.person_pick.pop(key, None)
        elif kind == "hook":
            self.hook_text[key] = val or ""
        elif kind == "words" and val is not None:
            self.cache.put_words(key, self._val("language"), val)

    def _undo_redo(self, src, dst, verb):
        if not src:
            return
        label, kind, key, old, _t = src.pop()
        dst.append((label, kind, key, self._state_of(kind, key), time.monotonic()))
        self._put_state(kind, key, old)
        if kind == "person" and self.current and os.path.abspath(self.current) == key:
            self._people_for = None   # rebuild the strip with the restored ticks
        self._log("%s: %s" % (verb, label))
        self._sync_undo()
        self._schedule_autosave()
        if self.current and os.path.abspath(self.current) == key:
            self._replan()

    def _undo_action(self):
        self._undo_redo(self._undo, self._redo, tr("Undone"))

    def _redo_action(self):
        self._undo_redo(self._redo, self._undo, tr("Redone"))

    def _sync_undo(self):
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(bool(self._redo))
        self.undo_btn.setToolTip(tr("Undo: ") + self._undo[-1][0] if self._undo else tr("Nothing to undo"))
        self.redo_btn.setToolTip(tr("Redo: ") + self._redo[-1][0] if self._redo else tr("Nothing to redo"))

    def _shortcut_undo(self):
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            fw.undo()
        else:
            self._undo_action()

    def _shortcut_redo(self):
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            fw.redo()
        else:
            self._redo_action()

    # ---------------------------------------------------------- projects
    def _project_data(self) -> dict:
        opts = {}
        for name, wd in self.w.items():
            if isinstance(wd, (QDoubleSpinBox, QSpinBox)):
                opts[name] = wd.value()
            elif isinstance(wd, QComboBox):
                d = wd.currentData()
                opts[name] = d if isinstance(d, (str, int, float, bool)) or d is None else wd.currentText()
            elif isinstance(wd, QCheckBox):
                opts[name] = wd.isChecked()
        items = []
        for it in self._items():
            d = {k: it[k] for k in ("path", "label", "clip", "stem") if k in it}
            if "clip" in d:
                d["clip"] = list(d["clip"])
            items.append(d)

        def sigs(v):
            return [np.asarray(x, dtype=float).tolist() for x in v]

        return {
            "app": "Shorts Studio", "version": 1, "items": items, "options": opts,
            "person_pick": {k: sigs(v) for k, v in self.person_pick.items() if v},
            "manual": {k: [[m[0], m[1], m[2]] for m in v] for k, v in self.manual.items() if v},
            "hook_text": {k: t for k, t in self.hook_text.items() if t},
            "music": self.music_path or "", "logo": self.logo_path or "", "cta": self.cta_edit.text(),
            "out_dir": self.out_dir.text(), "suffix": self.suffix.text(),
            "current": self.current or "",
        }

    def _apply_project(self, d: dict):
        if not isinstance(d, dict) or "items" not in d:
            raise ValueError(tr("invalid project file"))
        self._pause()
        for name, val in (d.get("options") or {}).items():
            try:
                self._set(name, val)
            except (TypeError, ValueError):
                pass
        self.queue.clear()
        self.current = None
        for it in d["items"]:
            if not os.path.exists(it.get("path", "")):
                self._log(tr("Missing: %s") % it.get("path"))
                continue
            it = dict(it)
            if it.get("clip"):
                it["clip"] = tuple(it["clip"])
            it.setdefault("label", os.path.basename(it["path"]))
            self._add_item(it)
        self.person_pick = {k: [np.asarray(x, dtype=np.float32) for x in v]
                            for k, v in (d.get("person_pick") or {}).items()}
        self.manual = {k: [(int(m[0]), m[1], m[2]) for m in v] for k, v in (d.get("manual") or {}).items()}
        self.hook_text = dict(d.get("hook_text") or {})
        self._set_music(d.get("music") or "")
        self._set_logo(d.get("logo") or "")
        self.cta_edit.setText(d.get("cta") or "")
        self.out_dir.setText(d.get("out_dir") or "")
        self.suffix.setText(d.get("suffix") or "_short")
        self._undo.clear()
        self._redo.clear()
        self._sync_undo()
        self._sync_enabled()
        cur = d.get("current")
        row = 0
        for i in range(self.queue.count()):
            if self.queue.item(i).data(Qt.UserRole)["path"] == cur:
                row = i
                break
        if self.queue.count():
            self.queue.setCurrentRow(row)

    def save_project(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._project_data(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def load_project(self, path: str):
        with open(path, encoding="utf-8") as f:
            self._apply_project(json.load(f))

    def _save_project(self):
        start = getattr(self, "_project_path", "") or os.path.expanduser("~/project.shorts")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save project"), start, tr("Shorts Project (*.shorts)"))
        if not path:
            return
        if not path.endswith(".shorts"):
            path += ".shorts"
        try:
            self.save_project(path)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, tr("Saving failed: %s") % exc)
            return
        self._project_path = path
        self._log(tr("Project saved: %s") % path)

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open project"), os.path.dirname(
            getattr(self, "_project_path", "") or os.path.expanduser("~")), tr("Shorts Project (*.shorts)"))
        if not path:
            return
        try:
            self.load_project(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, tr("Can't open the project: %s") % exc)
            return
        self._project_path = path
        self._log(tr("Project opened: %s") % path)

    @staticmethod
    def autosave_path() -> str:
        from . import diskcache
        return os.path.join(diskcache.cache_dir(), "autosave.shorts")

    def _schedule_autosave(self):
        t = getattr(self, "_autosave_timer", None)
        if t is None:
            t = self._autosave_timer = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(3000)
            t.timeout.connect(self._autosave)
        t.start()

    def _autosave(self):
        if not self.queue.count():
            return
        try:
            os.makedirs(os.path.dirname(self.autosave_path()), exist_ok=True)
            self.save_project(self.autosave_path())
        except OSError:
            pass

    def _offer_restore(self):
        p = self.autosave_path()
        if self.queue.count() or not os.path.exists(p) or os.environ.get("SHORTS_NO_RESTORE"):
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        n = len([i for i in d.get("items", []) if os.path.exists(i.get("path", ""))])
        if not n:
            return
        if QMessageBox.question(self, APP_NAME, tr("Resume the previous session (%d videos)?") % n) \
                == QMessageBox.Yes:
            try:
                self._apply_project(d)
            except ValueError:
                pass

    # ---------------------------------------------------------- social texts
    def _social_words(self, data):
        words = self.cache.words(data["path"], self._val("language"))
        if words and data.get("clip"):
            a, b = data["clip"]
            words = [w for w in words if a <= w.start < b]
        return words

    def _show_social(self):
        from .social import make_texts
        from .overlays import first_sentence
        it = self.queue.currentItem()
        data = it.data(Qt.UserRole) if it else None
        if not data:
            QMessageBox.information(self, APP_NAME, tr("Select a video."))
            return
        words = self._social_words(data)
        if not words:
            QMessageBox.information(self, APP_NAME, tr("You need a transcript first: press “Transcribe”."))
            return
        key = os.path.abspath(data["path"])
        hook = self.hook_text.get(key) or first_sentence(words)
        t = make_texts(words, hook, self._val("language"))
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Social texts"))
        lay = QVBoxLayout(dlg)
        for name, field, multi in ((tr("Title"), "title", False), ("Descrizione", "description", True),
                                   ("Hashtag", "hashtags", False)):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            row.addStretch(1)
            b = QPushButton(tr("Copy"))
            row.addWidget(b)
            lay.addLayout(row)
            ed = QPlainTextEdit(t[field]) if multi else QLineEdit(t[field])
            if multi:
                ed.setMinimumHeight(110)
            lay.addWidget(ed)
            b.clicked.connect(lambda _=False, e=ed: QApplication.clipboard().setText(
                e.toPlainText() if isinstance(e, QPlainTextEdit) else e.text()))
        close = QPushButton(tr("Close"))
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.resize(520, 380)
        self._social_dlg = dlg
        dlg.show()

    # ---------------------------------------------------------- thumbnails
    def _open_thumbs(self):
        it = self.queue.currentItem()
        data = it.data(Qt.UserRole) if it else None
        if not data:
            QMessageBox.information(self, APP_NAME, tr("Select a video."))
            return
        path = data["path"]
        a = self.cache.get(path, self.reframe_options()) if path == self.current else None
        if a is None:
            QMessageBox.information(self, APP_NAME, tr("Press “Analyze faces” first: thumbnails start from the frames with the best face."))
            return
        if data.get("clip"):
            from .jobs import slice_analysis
            f0, f1 = (int(round(t * self.reader.fps)) for t in data["clip"])
            try:
                a = slice_analysis(a, f0, f1)
            except Exception:
                pass
        if self._busy(tr("Thumbnails")):
            return
        words = self._social_words(data) or []
        key = os.path.abspath(path)
        hook = self.hook_text.get(key, "")
        self._log(tr("Thumbnails: picking frames and cutting out the subject…"))

        def job(progress):
            from . import thumbnail as T
            return data, T.prepare(path, a, count=4, progress=progress, log=lambda *_: None), words, hook

        self._run(job, self._thumbs_ready, tr("Thumbnails"))

    def _thumbs_ready(self, res):
        data, variants, words, hook = res
        if not variants:
            QMessageBox.information(self, APP_NAME, tr("No usable face for a thumbnail."))
            return
        from . import thumbnail as T
        from .captions_qt import make_thumb_text_renderer
        from .thumb_qt import ThumbDialog
        text, hl = T.thumb_text(words, hook)
        out_dir = self.out_dir.text().strip() or os.path.dirname(data["path"])
        stem = data.get("stem") or os.path.splitext(os.path.basename(data["path"]))[0]
        dlg = ThumbDialog(self, variants, text, hl, make_thumb_text_renderer(), out_dir, stem)
        self._thumb_dlg = dlg
        dlg.show()

    # ---------------------------------------------------------- hand corrections
    def _crop_dragged(self, dx, dy):
        k = getattr(self, "_view_k", 1.0) or 1.0
        dx, dy = dx / k, dy / k   # the view may be the proxy
        r = self.result
        if r is None or not self.current:
            return
        i = self.slider.value()
        if i >= len(r.plan.cx):
            return
        cx = float(r.plan.cx[i] + dx)
        cy = float(r.plan.cy[i] + dy)
        o = self.reframe_options()
        vertical = o.follow_vertical or (r.path.zoom[i] * o.zoom) > 1.0
        y = cy - (0.5 - o.face_height_pos) * float(r.plan.crop_h[i]) if vertical else None
        key = os.path.abspath(self.current)
        self._push_undo(tr("Crop correction"), "manual", key, list(self.manual.get(key, [])))
        man = [m for m in self.manual.get(key, []) if m[0] != i]
        man.append((i, cx, y))
        self.manual[key] = sorted(man)
        self._log(tr("Correction at %s") % _fmt(i / self.reader.fps))
        self._replan()

    def _manual_auto_here(self):
        if not self.current or self.result is None:
            return
        i = self.slider.value()
        key = os.path.abspath(self.current)
        self._push_undo(tr("Back to auto"), "manual", key, list(self.manual.get(key, [])))
        man = [m for m in self.manual.get(key, []) if m[0] != i] + [(i, None, None)]
        self.manual[key] = sorted(man, key=lambda m: m[0])
        self._replan()

    def _manual_clear(self):
        if self.current:
            key = os.path.abspath(self.current)
            if self.manual.get(key):
                self._push_undo(tr("Clear all corrections"), "manual", key, list(self.manual[key]))
            self.manual.pop(key, None)
            self._replan()

    # ================================================================ batch
    def _start(self):
        items = self._items()
        if not items:
            QMessageBox.information(self, APP_NAME, tr("Add at least one video."))
            return
        self._save_settings()
        if not self._confirm_restore_time(items):
            return
        renderer = None
        try:
            from .captions_qt import make_renderer
            renderer = make_renderer(self._val("caption_color"), self._val("caption_upper"),
                                     self._val("caption_hl"))
        except Exception as exc:
            self._log(tr("Qt captions unavailable (%s), using OpenCV") % exc)
        for i in range(len(items)):
            self._set_status(i, "○")
        worker = BatchWorker(items, self.job_settings(), self.cache, renderer)
        try:
            from .captions_qt import make_thumb_text_renderer, make_title_renderer
            worker.title_renderer = make_title_renderer()
            worker.thumb_renderer = make_thumb_text_renderer()
        except Exception:
            pass
        th = QThread()
        worker.moveToThread(th)
        th.started.connect(worker.run)
        worker.file_started.connect(self._on_file_started)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._log)
        worker.file_done.connect(self._on_file_done)
        worker.finished.connect(th.quit)
        th.finished.connect(self._batch_finished)
        self.batch = worker
        self._batch_thread = th
        self._total = len(items)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        th.start()

    def _confirm_restore_time(self, items) -> bool:
        """Slow face models: say how long it will take before starting."""
        key = self._val("restore")
        if key in ("off", "gpen256"):
            return True
        from .. import restore
        from .jobs import _dur, media_info
        s = self.job_settings()
        frames = 0.0
        for it in items:
            dur, fps = media_info(it["path"])
            if it.get("clip"):
                dur = it["clip"][1] - it["clip"][0]
            frames += dur * fps * (1 + len(s.formats))
        eta = restore.estimate_seconds(key, int(frames))
        if eta < 600:
            return True
        name = next(label for label, k in RESTORE if k == key).split(" —")[0]
        msg = (tr("With %s face restoration will take about %s (M1 estimate, one face per frame; cuts reduce it).\n\nContinue?") % (name, _dur(eta)))
        return QMessageBox.question(self, APP_NAME, msg) == QMessageBox.Yes

    def _on_file_started(self, i):
        self._set_status(i, "◐")

    def _on_progress(self, i, stage, f):
        self.stage_lab.setText(tr("Video %d/%d · %s %d%%") % (i + 1, self._total, stage, int(f * 100)))
        self.bar.setValue(int(f * 1000))

    def _on_file_done(self, i, ok, res):
        self._set_status(i, "●" if ok else "✕")
        if ok:
            it = self.queue.item(i)
            if it is not None:
                data = dict(it.data(Qt.UserRole))
                data["outs"] = list(res)
                it.setData(Qt.UserRole, data)
            for out in res:
                self._log(tr("✓ Created: %s") % out)
        else:
            self._log(tr("✕ Error: %s") % res)

    def _batch_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stage_lab.setText(tr("Done"))
        self.batch = None
        self._refresh_cache_label()

    def _stop(self):
        if self.batch:
            self.batch.cancel.set()
            self.stage_lab.setText(tr("Stopping…"))

    def _log(self, text):
        self.log_view.appendPlainText(text)

    def shutdown_workers(self, timeout_ms: int = 8000):
        """Cancel background jobs and wait for their threads to end."""
        import time
        for rec in self._workers:
            rec["w"].cancel.set()
            rec["th"].quit()  # leaves the thread's event loop once the job returns
        end = time.monotonic() + timeout_ms / 1000.0
        for rec in list(self._workers):
            rec["th"].wait(max(0, int((end - time.monotonic()) * 1000)))

    def threads_running(self) -> bool:
        running = any(rec["th"].isRunning() for rec in self._workers)
        bt = getattr(self, "_batch_thread", None)
        return running or bool(bt is not None and bt.isRunning())

    def closeEvent(self, ev):
        if self.batch:
            if QMessageBox.question(self, APP_NAME, tr("Processing. Stop and quit?")) \
                    != QMessageBox.Yes:
                ev.ignore()
                return
            self.batch.cancel.set()
            self._batch_thread.wait(8000)
        self._autosave()
        self.shutdown_workers()
        self._stop_audio()
        if self.proxy:
            self.proxy.close()
        if self._pa_dir:
            import shutil
            shutil.rmtree(self._pa_dir, ignore_errors=True)
        self._save_settings()
        super().closeEvent(ev)


STYLE = """
/* ---------- base ---------- */
QWidget { font-size: 13px; color: #e8e8e8; }
QFrame#header { background: #0c0c0c; border-bottom: 1px solid #1f1f1f; }
QLabel#brand { font-size: 13px; font-weight: 800; color: #e8e8e8; }
QLabel#brandAlt { font-size: 13px; font-weight: 800; color: #e5484d; }
QWidget#panel { background: #161616; }
QWidget#previewPane { background: #0f0f0f; }
QFrame#ctlcard { background: #141414; border: 1px solid #222222; border-radius: 10px; }
QLabel#h1 { font-size: 12px; font-weight: 700; color: #919191; }
QLabel#hint { color: #7f7f7f; font-size: 12px; }
QLabel#status { color: #adadad; font-weight: 600; }
QLabel#section {
    color: #919191; font-size: 11px; font-weight: 700;
    padding: 10px 0 5px; border-bottom: 1px solid #242424; margin-bottom: 3px;
}
QDialog, QMenu { background: #161616; }
QToolTip { background: #232323; color: #e8e8e8; border: 1px solid #363636; padding: 6px 8px; }

/* ---------- buttons ---------- */
QPushButton {
    padding: 6px 13px; border-radius: 7px;
    background: #232323; border: 1px solid #2f2f2f;
}
QPushButton:hover { background: #2a2a2a; border-color: #3a3a3a; }
QPushButton:pressed { background: #1d1d1d; }
QPushButton:disabled { color: #575757; background: #1c1c1c; border-color: #262626; }
QPushButton#accent { background: #e5484d; border: none; color: white; font-weight: 600; }
QPushButton#accent:hover { background: #f0565b; }
QPushButton#accent:pressed { background: #cf3d42; }
QPushButton#accent:disabled { background: #472628; color: #9a8485; }
QToolButton {
    padding: 5px 8px; border-radius: 6px;
    background: transparent; border: 1px solid transparent; color: #c7c7c7;
}
QToolButton:hover { background: #232323; }
QToolButton:pressed, QToolButton:checked { background: #2b2b2b; border-color: #343434; }
QToolButton#person { background: #191919; border: 2px solid #2d2d2d; border-radius: 8px; color: #cecece; }
QToolButton#person:hover { border-color: #4b4b4b; }
QToolButton#person:checked { border: 2px solid #e5484d; background: #32202a; }
QWidget#rail { background: #101010; }
QToolButton#railBtn {
    border-radius: 8px; padding: 9px 2px 7px; margin: 0 2px;
    font-size: 10px; color: #7f7f7f;
}
QToolButton#railBtn:hover { background: #1c1c1c; color: #bdbdbd; }
QToolButton#railBtn:checked { background: #332025; color: #f0a7ab; }

/* ---------- inputs ---------- */
QLineEdit, QComboBox, QAbstractSpinBox {
    background: #191919; border: 1px solid #2d2d2d; border-radius: 6px;
    padding: 4px 8px; min-height: 20px;
    selection-background-color: #40343a;
}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border-color: #e5484d; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { color: #575757; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #1e1e1e; border: 1px solid #343434; border-radius: 6px;
    padding: 4px; outline: none;
}
QComboBox QAbstractItemView::item { padding: 5px 9px; border-radius: 4px; min-height: 20px; }
QComboBox QAbstractItemView::item:selected { background: #40343a; color: #fff; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    width: 18px; border-left: 1px solid #242424; background: transparent;
}
QAbstractSpinBox::up-button { image: url(__PLUS__); }
QAbstractSpinBox::down-button { image: url(__MINUS__); }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: #252525; }
QComboBox::down-arrow { image: url(__ARROW__); width: 9px; height: 9px; }

/* ---------- segmented ---------- */
QWidget#segbox { background: #171717; border: 1px solid #272727; border-radius: 8px; }
QPushButton#segFirst, QPushButton#segMid, QPushButton#segLast {
    border: none; border-radius: 0; background: transparent;
    padding: 5px 6px; color: #9c9c9c; font-size: 12px;
}
QPushButton#segFirst { border-top-left-radius: 6px; border-bottom-left-radius: 6px; }
QPushButton#segLast { border-top-right-radius: 6px; border-bottom-right-radius: 6px; }
QPushButton#segFirst:hover, QPushButton#segMid:hover, QPushButton#segLast:hover {
    color: #d7d7d7; background: #202020;
}
QPushButton#segFirst:checked, QPushButton#segMid:checked, QPushButton#segLast:checked {
    background: #3c3c3c; color: #ffffff;
}
QPushButton#segFirst:disabled, QPushButton#segMid:disabled, QPushButton#segLast:disabled {
    color: #575757; background: transparent;
}

/* ---------- lists ---------- */
QListWidget {
    background: #121212; border: 1px solid #252525; border-radius: 8px; padding: 4px;
}
QListWidget::item { padding: 7px 8px; border-radius: 5px; }
QListWidget::item:hover { background: #1d1d1d; }
QListWidget::item:selected { background: #3a262b; color: #f3d4d6; }

/* ---------- tabs ---------- */
QTabWidget::pane { border: 1px solid #252525; border-radius: 8px; top: -1px; background: #131313; }
QTabBar::tab { padding: 7px 11px; background: transparent; color: #7f7f7f; }
QTabBar::tab:hover { color: #bdbdbd; }
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #e5484d; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }

/* ---------- sliders & progress ---------- */
QSlider::groove:horizontal { height: 4px; background: #2b2b2b; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -6px 0; border-radius: 7px;
    background: #e6e6e6; border: 1px solid #9c9c9c;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider::sub-page:horizontal { background: #e5484d; border-radius: 2px; }
QProgressBar { background: #1b1b1b; border: none; border-radius: 5px; height: 10px; }
QProgressBar::chunk { background: #e5484d; border-radius: 5px; }

/* ---------- checkboxes ---------- */
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 1px solid #3c3c3c; border-radius: 4px; background: #191919;
}
QCheckBox::indicator:hover { border-color: #4b4b4b; }
QCheckBox::indicator:checked {
    background: #e5484d; border-color: #e5484d; image: url(__CHECK__);
}
QCheckBox:disabled, QLabel:disabled { color: #575757; }

/* ---------- scrollbars ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px 2px 2px 0; }
QScrollBar::handle:vertical { background: #343434; min-height: 30px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #464646; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0 2px 2px 2px; }
QScrollBar::handle:horizontal { background: #343434; min-width: 30px; border-radius: 4px; }
QScrollBar::handle:horizontal:hover { background: #464646; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---------- misc ---------- */
QSplitter::handle { background: #1c1c1c; }
QSplitter::handle:hover { background: #e5484d; }
QPlainTextEdit {
    font-family: Menlo, Consolas, monospace; font-size: 11px; color: #9a9a9a;
    background: #0e0e0e; border: 1px solid #212121; border-radius: 8px; padding: 6px;
}
QFrame#bottom { background: #131313; border-top: 1px solid #232323; }
QHeaderView::section {
    background: #181818; color: #919191; border: none;
    border-bottom: 1px solid #252525; padding: 5px 8px;
}
QTableWidget { background: #121212; border: 1px solid #252525; border-radius: 8px; gridline-color: #212121; }
QTableWidget::item:selected { background: #3a262b; }
QMenu::item { padding: 5px 24px 5px 12px; border-radius: 4px; }
QMenu::item:selected { background: #40343a; }
QMenu::separator { height: 1px; background: #252525; margin: 4px 8px; }
"""


def _pixmap(kind: str, color: str) -> QPixmap:
    """Small geometric icons painted at runtime (no asset files, same look
    on every platform), on a 32-px logical canvas."""
    img = QImage(64, 64, QImage.Format_ARGB32)
    img.setDevicePixelRatio(2.0)
    img.fill(Qt.transparent)
    pt = QPainter(img)
    pt.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    line = QPen(c, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    if kind == "play":
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawPolygon(QPolygonF([QPointF(10, 5), QPointF(26, 16), QPointF(10, 27)]))
    elif kind == "pause":
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawRoundedRect(QRectF(8.5, 7, 5.5, 18), 1.8, 1.8)
        pt.drawRoundedRect(QRectF(18, 7, 5.5, 18), 1.8, 1.8)
    elif kind in ("vol", "vol_off"):
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawPolygon(QPolygonF([QPointF(6, 12), QPointF(11, 12), QPointF(17, 6),
                                  QPointF(17, 26), QPointF(11, 20), QPointF(6, 20)]))
        pt.setPen(QPen(c, 2.2, Qt.SolidLine, Qt.RoundCap))
        pt.setBrush(Qt.NoBrush)
        if kind == "vol":
            pt.drawArc(QRectF(16, 11, 9, 10), -65 * 16, 130 * 16)
            pt.drawArc(QRectF(16, 6, 14, 20), -55 * 16, 110 * 16)
        else:
            pt.drawLine(QPointF(21, 12), QPointF(28, 20))
            pt.drawLine(QPointF(28, 12), QPointF(21, 20))
    elif kind == "crop":
        pt.setPen(line)
        for pts in ([(9, 13), (9, 9), (13, 9)], [(19, 9), (23, 9), (23, 13)],
                    [(23, 19), (23, 23), (19, 23)], [(13, 23), (9, 23), (9, 19)]):
            pt.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
    elif kind == "cut":
        pt.setPen(QPen(c, 2.0, Qt.SolidLine, Qt.RoundCap))
        pt.setBrush(Qt.NoBrush)
        pt.drawEllipse(QPointF(10.5, 22), 3, 3)
        pt.drawEllipse(QPointF(21.5, 22), 3, 3)
        pt.setPen(line)
        pt.drawLine(QPointF(12.5, 19.5), QPointF(21, 8))
        pt.drawLine(QPointF(19.5, 19.5), QPointF(11, 8))
    elif kind == "audio":
        pt.setPen(QPen(c, 2.8, Qt.SolidLine, Qt.RoundCap))
        for x, hh in ((7, 7), (12, 14), (17, 22), (22, 14), (27, 9)):
            pt.drawLine(QPointF(x, 16 - hh / 2), QPointF(x, 16 + hh / 2))
    elif kind == "quality":
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawPolygon(QPolygonF([QPointF(16, 6), QPointF(18.4, 13.6), QPointF(26, 16),
                                  QPointF(18.4, 18.4), QPointF(16, 26), QPointF(13.6, 18.4),
                                  QPointF(6, 16), QPointF(13.6, 13.6)]))
        pt.drawEllipse(QPointF(24, 8), 1.6, 1.6)
    elif kind == "effects":
        pt.setPen(QPen(c, 3.0, Qt.SolidLine, Qt.RoundCap))
        pt.drawLine(QPointF(9, 23), QPointF(22, 10))
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawEllipse(QPointF(20, 20), 1.6, 1.6)
        pt.drawEllipse(QPointF(10, 9), 1.6, 1.6)
        pt.drawEllipse(QPointF(24, 16), 1.3, 1.3)
    elif kind == "color":
        pt.setPen(line)
        pt.setBrush(Qt.NoBrush)
        pt.drawEllipse(QRectF(8, 8, 16, 16))
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        pt.drawChord(QRectF(9, 9, 14, 14), 180 * 16, 180 * 16)
    elif kind == "texts":
        pt.setPen(QPen(c, 2.6, Qt.SolidLine, Qt.RoundCap))
        for y, x2 in ((10, 24), (16, 20), (22, 15)):
            pt.drawLine(QPointF(8, y), QPointF(x2, y))
    elif kind == "export":
        pt.setPen(line)
        pt.setBrush(Qt.NoBrush)
        pt.drawRoundedRect(QRectF(7, 15, 18, 10), 2.5, 2.5)
        pt.drawLine(QPointF(16, 6), QPointF(16, 17))
        pt.drawPolyline(QPolygonF([QPointF(11.5, 10.5), QPointF(16, 6), QPointF(20.5, 10.5)]))
    elif kind == "film":
        pt.setPen(QPen(c, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        pt.setBrush(Qt.NoBrush)
        pt.drawRoundedRect(QRectF(6, 9, 20, 14), 2.5, 2.5)
        pt.setPen(Qt.NoPen)
        pt.setBrush(c)
        for y in (11, 18):
            pt.drawRect(QRectF(8, y, 3, 3))
            pt.drawRect(QRectF(21, y, 3, 3))
    pt.end()
    return QPixmap.fromImage(img)


def _icon(kind: str) -> QIcon:
    return QIcon(_pixmap(kind, "#c7c7c7"))


def _rail_icon(kind: str) -> QIcon:
    """Two-state icon: muted normally, accent when the rail tab is active."""
    ic = QIcon()
    ic.addPixmap(_pixmap(kind, "#919191"), QIcon.Normal, QIcon.Off)
    ic.addPixmap(_pixmap(kind, "#e58a8e"), QIcon.Normal, QIcon.On)
    return ic


def _ui_pngs() -> dict:
    """Small UI glyphs drawn once into the app cache: QSS needs image files
    for the checkbox ✓, and the native spin +/− / combo-arrow glyphs come
    out dark on a dark theme (unreadable), so we supply our own."""
    d = diskcache.cache_dir()
    gray, white = QColor("#c7c7c7"), QColor("#ffffff")

    def draw(name, fn, pen):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            img = QImage(32, 32, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            pt = QPainter(img)
            pt.setRenderHint(QPainter.Antialiasing)
            pt.setPen(pen)
            fn(pt)
            pt.end()
            img.save(p)
        return p

    return {
        "check": draw("check.png",
                      lambda pt: pt.drawPolyline(QPolygonF(
                          [QPointF(9, 17), QPointF(14, 23), QPointF(24, 9)])),
                      QPen(white, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)),
        "spin_plus": draw("spin_plus.png",
                          lambda pt: (pt.drawLine(QPointF(16, 9), QPointF(16, 23)),
                                      pt.drawLine(QPointF(9, 16), QPointF(23, 16))),
                          QPen(gray, 4, Qt.SolidLine, Qt.RoundCap)),
        "spin_minus": draw("spin_minus.png",
                           lambda pt: pt.drawLine(QPointF(9, 16), QPointF(23, 16)),
                           QPen(gray, 4, Qt.SolidLine, Qt.RoundCap)),
        "combo_down": draw("combo_down.png",
                           lambda pt: pt.drawPolyline(QPolygonF(
                               [QPointF(9, 13), QPointF(16, 20), QPointF(23, 13)])),
                           QPen(gray, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)),
    }


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [
        (QPalette.Window, "#111111"), (QPalette.WindowText, "#e8e8e8"), (QPalette.Base, "#191919"),
        (QPalette.AlternateBase, "#1f1f1f"), (QPalette.Text, "#e8e8e8"), (QPalette.Button, "#232323"),
        (QPalette.ButtonText, "#e8e8e8"), (QPalette.Highlight, "#e5484d"),
        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#232323"),
        (QPalette.ToolTipText, "#e8e8e8"), (QPalette.PlaceholderText, "#5e5e5e"),
    ]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    pngs = _ui_pngs()
    style = (STYLE.replace("__CHECK__", pngs["check"])
                  .replace("__PLUS__", pngs["spin_plus"])
                  .replace("__MINUS__", pngs["spin_minus"])
                  .replace("__ARROW__", pngs["combo_down"]))
    app.setStyleSheet(style)


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    i18n.set_lang(QSettings("face_reframe", "ShortsStudio").value("ui/lang", "en") or "en")
    apply_theme(app)
    win = MainWindow()
    win.add_files([a for a in (sys.argv[1:] if argv is None else argv[1:]) if os.path.exists(a)])
    win.show()
    code = app.exec()
    if win.threads_running():
        # a native call (decoder, ONNX) that ignored the cancel is still busy:
        # skip Python's teardown, which would destroy the running QThread and
        # make Qt abort ("QThread: Destroyed while thread is still running").
        # Settings and caches are already saved by closeEvent.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    # tear the window down before QApplication goes away (avoids exit crashes)
    if win.reader:
        win.reader.close()
    win.deleteLater()
    del win
    return code
