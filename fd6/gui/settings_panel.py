from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget
)

from fd6.shapegen.profile import Profile, load_profile_from_file, list_bundled_profiles


SHAPE_TYPE_CHOICES = [
    ("rotated_ellipse", "Rotated Ellipse (default)"),
    ("rectangle", "Rectangle"),
    ("rotated_rectangle", "Rotated Rectangle"),
    ("ellipse", "Ellipse"),
    ("circle", "Circle"),
]


class SettingsPanel(QWidget):
    profile_changed = Signal(object)
    start_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    inject_clicked = Signal()
    wiggle_clicked = Signal()  # Signal to trigger manual post-optimization

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        prof_row = QHBoxLayout()
        prof_label = QLabel("Profile:")
        prof_row.addWidget(prof_label)
        self.profile_combo = QComboBox(self)
        self.profile_combo.setToolTip(prof_label.toolTip())
        self._populate_profiles()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        prof_row.addWidget(self.profile_combo, stretch=1)
        layout.addLayout(prof_row)

        adv = QGroupBox("Advanced", self)
        form = QFormLayout(adv)
        
        self.stop_at = QSpinBox(); self.stop_at.setRange(10, 100000); self.stop_at.setValue(3000)
        self.random_samples = QSpinBox(); self.random_samples.setRange(10, 1000000); self.random_samples.setValue(1000)
        self.mutated_samples = QSpinBox(); self.mutated_samples.setRange(1, 100000); self.mutated_samples.setValue(200)
        self.max_resolution = QSpinBox(); self.max_resolution.setRange(100, 4096); self.max_resolution.setValue(1200)
        self.max_threads = QSpinBox(); self.max_threads.setRange(0, 64); self.max_threads.setValue(0) 
        self.preview_every = QSpinBox(); self.preview_every.setRange(1, 100); self.preview_every.setValue(1)
        
        for label_text, field in (
            ("Stop at shapes", self.stop_at),
            ("Random samples", self.random_samples),
            ("Mutated samples", self.mutated_samples),
            ("Max resolution (px)", self.max_resolution),
            ("Preview every N", self.preview_every),
        ):
            row_label = QLabel(label_text, adv)
            row_label.setToolTip(field.toolTip())
            form.addRow(row_label, field)
            field.valueChanged.connect(self._on_adv_changed)
        layout.addWidget(adv)

        sticker_group = QGroupBox("Image options", self)
        sg_layout = QVBoxLayout(sticker_group)
        self.sticker_mode_cb = QCheckBox("Add white background to transparent images", sticker_group)
        self.sticker_mode_cb.setChecked(True)  
        sg_layout.addWidget(self.sticker_mode_cb)
        layout.addWidget(sticker_group)

        types_group = QGroupBox("Shape types", self)
        tg_layout = QVBoxLayout(types_group)
        self._shape_checks: dict[str, QCheckBox] = {}
        
        # Strictly limited to safe, mapped game primitives
        supported_codes = {"rotated_ellipse", "rectangle", "rotated_rectangle", "ellipse", "circle"}
        supported_tooltips = {
            "rotated_ellipse": "An oval that can be rotated to any angle. Fits organic/curvy content.",
            "rectangle": "An axis-aligned rectangle or square.",
            "rotated_rectangle": "A rectangle or square that can be rotated to any angle.",
            "ellipse": "A non-rotatable stretched oval.",
            "circle": "A uniform circle.",
        }
        
        for code, label in SHAPE_TYPE_CHOICES:
            cb = QCheckBox(label, types_group)
            cb.setChecked(code == "rotated_ellipse")
            if code in supported_codes:
                cb.setEnabled(True)
                cb.setToolTip(supported_tooltips.get(code, ""))
            else:
                cb.setEnabled(False)
                cb.setToolTip("Shape type currently disabled.")
            cb.stateChanged.connect(self._on_adv_changed)
            tg_layout.addWidget(cb)
            self._shape_checks[code] = cb
        layout.addWidget(types_group)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.setMinimumHeight(36)
        self.pause_btn = QPushButton("Pause"); self.pause_btn.setCheckable(True); self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.start_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        from fd6.inject.game_profiles import list_profiles
        target_row = QHBoxLayout()
        target_label = QLabel("Target:")
        target_row.addWidget(target_label)
        self.target_combo = QComboBox(self)
        self._target_profiles = list_profiles()
        for prof in self._target_profiles:
            self.target_combo.addItem(prof.label, prof.key)
        self.target_combo.setCurrentIndex(0)  
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, stretch=1)
        layout.addLayout(target_row)

        self.inject_btn = QPushButton("Inject into Forza Horizon 6")
        self.inject_btn.setEnabled(False)
        self.inject_btn.clicked.connect(self.inject_clicked.emit)
        layout.addWidget(self.inject_btn)

        # --- MANUAL POST-OPTIMIZATION OPTION ---
        self.wiggle_btn = QPushButton("Post-Optimize Loaded JSON", self)
        self.wiggle_btn.setEnabled(False)  
        self.wiggle_btn.clicked.connect(self.wiggle_clicked.emit)
        layout.addWidget(self.wiggle_btn)
        # ----------------------------------------

        layout.addStretch()
        self._on_profile_changed(self.profile_combo.currentIndex())

    def selected_target_profile_key(self) -> str:
        data = self.target_combo.currentData()
        return str(data) if data else "fh6"

    def _on_target_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._target_profiles):
            return
        prof = self._target_profiles[idx]
        clean_label = prof.label.replace(" (BETA)", "")
        self.inject_btn.setText(f"Inject into {clean_label}")

    def _populate_profiles(self) -> None:
        self.profile_combo.clear()
        for path in list_bundled_profiles():
            self.profile_combo.addItem(path.stem, str(path))
        if self.profile_combo.count() == 0:
            self.profile_combo.addItem("default", "")

    def _on_profile_changed(self, idx: int) -> None:
        path = self.profile_combo.itemData(idx)
        if not path:
            return
        try:
            prof = load_profile_from_file(path)
        except Exception:
            return
        for w in (self.stop_at, self.random_samples, self.mutated_samples, self.max_resolution, self.preview_every):
            w.blockSignals(True)
        self.stop_at.setValue(prof.stop_at)
        self.random_samples.setValue(prof.random_samples)
        self.mutated_samples.setValue(prof.mutated_samples)
        self.max_resolution.setValue(prof.max_resolution)
        self.preview_every.setValue(prof.preview_every)
        for w in (self.stop_at, self.random_samples, self.mutated_samples, self.max_resolution, self.preview_every):
            w.blockSignals(False)
            
        for code, cb in self._shape_checks.items():
            cb.blockSignals(True)
            if cb.isEnabled():
                cb.setChecked(code in prof.shape_types)
            else:
                cb.setChecked(False)
            cb.blockSignals(False)
        self.profile_changed.emit(self.build_profile())

    def _on_adv_changed(self, *_args) -> None:
        self.profile_changed.emit(self.build_profile())

    def build_profile(self) -> Profile:
        idx = self.profile_combo.currentIndex()
        path = self.profile_combo.itemData(idx) or ""
        base = Profile(name=self.profile_combo.itemText(idx) or "custom")
        if path:
            try:
                base = load_profile_from_file(path)
            except Exception:
                pass
        base.stop_at = self.stop_at.value()
        base.random_samples = self.random_samples.value()
        base.mutated_samples = self.mutated_samples.value()
        base.max_resolution = self.max_resolution.value()
        base.preview_every = self.preview_every.value()
        base.shape_types = [code for code, cb in self._shape_checks.items() if cb.isChecked()] or ["rotated_ellipse"]
        return base

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.wiggle_btn.setEnabled(not running)