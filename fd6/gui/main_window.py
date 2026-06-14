# --- START OF FILE main_window.py ---
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QMainWindow, QMessageBox, 
    QScrollArea, QSplitter, QStatusBar, QVBoxLayout, QWidget, QStackedWidget
)

from fd6.gui.brand_banner import BrandBanner, badge_path
from fd6.gui.preview_panel import PreviewPanel
from fd6.gui.themes import THEMES, apply_theme, saved_theme_name, badge_filename_for_theme
from fd6.gui.queue_panel import QueuePanel
from fd6.gui.settings_panel import SettingsPanel
from fd6.gui.upload_panel import UploadPanel
from fd6.shapegen.profile import Profile
from fd6.shapegen.shapes import Shape
from fd6.shapegen.worker import GenerationWorker
from fd6.inject.fh6_injector import patterns_are_populated, FH6_TARGET_BUILD

try:
    from fd6.gui.suite import SuiteMode, SUITE_DISPLAY, save_suite_mode, load_suite_mode, GameSuiteDialog
except ImportError:
    class SuiteMode:
        FORZA = "forza"
        AC = "ac"
        NFS = "nfs"
        CREW = "crew"
    SUITE_DISPLAY = {
        SuiteMode.FORZA: {"label": "Forza Suite", "enabled": True},
        SuiteMode.AC: {"label": "Assetto Corsa", "enabled": False},
        SuiteMode.NFS: {"label": "Need for Speed", "enabled": False},
        SuiteMode.CREW: {"label": "The Crew", "enabled": False}
    }
    def save_suite_mode(mode): pass
    def load_suite_mode(): return SuiteMode.FORZA


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Forza Designer 6+ — Inject custom decals into Forza (FH6 build {FH6_TARGET_BUILD})")
        self.resize(1280, 760)
        self.setStatusBar(QStatusBar(self))
        self._apply_dark_palette()

        try:
            self._suite_mode = load_suite_mode()
            self._suite_first_launch = False
        except Exception:
            self._suite_mode = SuiteMode.FORZA
            self._suite_first_launch = True

        # Panels
        self.upload = UploadPanel(self)
        self.preview = PreviewPanel(self)              
        self.queue = QueuePanel(self)
        self.settings_panel = SettingsPanel(self)      

        # UI Stacking configuration
        self.preview_stack = QStackedWidget(self)
        self.preview_stack.addWidget(self.preview)
        try:
            from fd6.gui.ac_preview import ACPreviewPanel
            self.ac_preview = ACPreviewPanel(self)
            self.preview_stack.addWidget(self.ac_preview)
        except ImportError:
            self.ac_preview = QWidget(self)
            self.preview_stack.addWidget(self.ac_preview)

        # Wrap tall settings pages in vertical scroll areas so they never grow window bounds
        self.settings_stack = QStackedWidget(self)
        self.settings_stack.addWidget(self._scrollable(self.settings_panel))
        try:
            from fd6.gui.ac_settings import ACSettingsPanel
            self.ac_settings = ACSettingsPanel(self)
            self.settings_stack.addWidget(self._scrollable(self.ac_settings))
        except ImportError:
            class DummyACSettings:
                export_clicked = type('Dummy', (), {'connect': lambda self, f: None})()
            self.ac_settings = DummyACSettings()
            self.settings_stack.addWidget(QWidget(self))

        # Layout: [upload | center (preview over queue) | settings]
        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Vertical, center)
        vsplit.addWidget(self.preview_stack)
        vsplit.addWidget(self.queue)
        vsplit.setSizes([520, 220])
        center_layout.addWidget(vsplit)

        hsplit = QSplitter(Qt.Horizontal, self)
        hsplit.addWidget(self._scrollable(self.upload))
        hsplit.addWidget(center)
        hsplit.addWidget(self.settings_stack)

        hsplit.setSizes([240, 760, 280])
        self.setCentralWidget(hsplit)

        # Signal connections
        self.upload.files_selected.connect(self._on_files_selected)
        self.upload.json_loaded.connect(self._on_json_loaded_for_preview)
        self.upload.download_json_requested.connect(self._on_download_json)
        self.queue.download_requested.connect(self._on_queue_download_json)
        self.settings_panel.start_clicked.connect(self._start_next)
        self.settings_panel.pause_clicked.connect(self._toggle_pause)
        self.settings_panel.stop_clicked.connect(self._stop_current)
        self.settings_panel.inject_clicked.connect(self._on_inject_clicked)
        self.settings_panel.wiggle_clicked.connect(self._on_post_optimize_clicked)  
        
        # AC path
        self.ac_settings.export_clicked.connect(self._on_ac_export_clicked)

        # Worker state
        self._worker = None
        self._thread: QThread | None = None
        self._current_path: Path | None = None
        self._current_uid: int | None = None  
        self._current_profile: Profile | None = None
        self._last_finished_json: Path | None = None  
        self._loaded_json_path: Path | None = None    
        self._inject_worker = None  
        self._inject_thread: QThread | None = None

        self._build_menus()
        self._refresh_inject_button()

        self.brand_banner = BrandBanner(self)
        self.brand_banner.show()
        
        _saved_theme = saved_theme_name()
        _bp = badge_path(badge_filename_for_theme(_saved_theme))
        if _bp:
            self.setWindowIcon(QIcon(str(_bp)))
            self.brand_banner.set_badge(_bp)

        from fd6.gui.particles import ParticleOverlay
        self.particles = ParticleOverlay(self)
        _pal = THEMES.get(_saved_theme, THEMES["Default"])
        self.particles.set_theme_colors(
            _pal["particle_1"], _pal["particle_2"], _pal["particle_3"],
        )
        self.particles.reposition()
        self.particles.set_exclude_provider(self._compute_particle_exclude_rect)
        QTimer.singleShot(0, self._sync_particle_exclude_rect)
        
        if hasattr(self, "_particles_enabled_act"):
            self._particles_enabled_act.setChecked(self.particles.enabled())
            self._sync_particle_count_check(self.particles.count())

        self._apply_suite_mode(self._suite_mode)
        self._suite_popup_shown_this_session = False

        from fd6.gui.music import MusicPlayer
        self.music = MusicPlayer(self)
        self.music.state_changed.connect(self._on_music_state)
        self.music.muted_changed.connect(self._on_music_muted)
        self.music.volume_changed.connect(self._on_music_volume)
        self.music.track_changed.connect(
            lambda name: self.statusBar().showMessage(f"♪ {name}", 4000)
        )
        if not self.music.has_tracks():
            for act in (self._music_play_act, self._music_mute_act):
                act.setEnabled(False)
            for act in self._music_vol_group.actions():
                act.setEnabled(False)

    def start_music(self) -> None:
        if not getattr(self, "music", None) or not self.music.has_tracks():
            return
        if getattr(self, "_music_started", False):
            return
        self._music_started = True
        self.music.start()
        self._music_play_act.setChecked(self.music.is_playing())
        self._music_mute_act.setChecked(self.music.muted())
        self._sync_volume_check(self.music.volume())

    def _apply_dark_palette(self) -> None:
        from PySide6.QtWidgets import QApplication
        apply_theme(QApplication.instance(), saved_theme_name())

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        sa = QScrollArea(self)
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        return sa

    def _build_menus(self) -> None:
        mbar = self.menuBar()
        file_menu = mbar.addMenu("&File")

        open_act = QAction("&Upload Image…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.upload._on_upload_clicked)
        file_menu.addAction(open_act)

        resume_act = QAction("&Resume Generation…", self)
        resume_act.setShortcut(QKeySequence("Ctrl+R"))
        resume_act.triggered.connect(self._on_resume_triggered)
        file_menu.addAction(resume_act)

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = mbar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        current_theme = saved_theme_name()
        for theme_name in THEMES.keys():
            act = QAction(theme_name, self, checkable=True)
            act.setChecked(theme_name == current_theme)
            act.triggered.connect(lambda _checked, n=theme_name: self._set_theme(n))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)

        view_menu.addSeparator()
        music_menu = view_menu.addMenu("&Music")
        self._music_play_act = QAction("&Play / Pause", self, checkable=True)
        self._music_play_act.setShortcut("Ctrl+M")
        self._music_play_act.triggered.connect(self._music_toggle_play)
        music_menu.addAction(self._music_play_act)
        self._music_mute_act = QAction("M&ute", self, checkable=True)
        self._music_mute_act.triggered.connect(self._music_toggle_mute)
        music_menu.addAction(self._music_mute_act)
        next_act = QAction("&Next track", self)
        next_act.setShortcut("Ctrl+Shift+M")
        next_act.triggered.connect(self._music_next)
        music_menu.addAction(next_act)
        music_menu.addSeparator()
        vol_menu = music_menu.addMenu("&Volume")
        self._music_vol_group = QActionGroup(self)
        self._music_vol_group.setExclusive(True)
        for pct in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            a = QAction(f"{pct}%", self, checkable=True)
            a.triggered.connect(lambda _c, p=pct: self._music_set_volume(p / 100.0))
            self._music_vol_group.addAction(a)
            vol_menu.addAction(a)

        view_menu.addSeparator()
        fonts_menu = view_menu.addMenu("F&onts")
        from fd6.gui.fonts import available_family_names as _font_names, saved_font_name as _saved_font
        self._font_group = QActionGroup(self)
        self._font_group.setExclusive(True)
        current_font = _saved_font()
        for fname in _font_names():
            a = QAction(fname, self, checkable=True)
            a.setChecked(fname == current_font)
            a.triggered.connect(lambda _c, name=fname: self._on_font_pick(name))
            self._font_group.addAction(a)
            fonts_menu.addAction(a)

        view_menu.addSeparator()
        loc_menu = view_menu.addMenu("&Localization")
        self._loc_group = QActionGroup(self)
        self._loc_group.setExclusive(True)
        loc_options = [
            ("English", True),
            ("Spanish (Español)", False),
            ("French (Français)", False),
            ("German (Deutsch)", False),
            ("Italian (Italiano)", False),
            ("Portuguese (Português)", False),
            ("Dutch (Nederlands)", False),
            ("Polish (Polski)", False),
            ("Russian (Русский)", False),
            ("Japanese (日本語)", False),
            ("Korean (한국어)", False),
            ("Simplified Chinese (简体中文)", False),
            ("Traditional Chinese (繁體中文)", False),
            ("Arabic (العربية)", False),
        ]
        for label, enabled in loc_options:
            act = QAction(label if enabled else f"{label} (coming soon)", self, checkable=True)
            act.setEnabled(enabled)
            act.setChecked(enabled)
            self._loc_group.addAction(act)
            loc_menu.addAction(act)

        view_menu.addSeparator()
        custom_menu = view_menu.addMenu("&Customizations")

        suite_menu = custom_menu.addMenu("Change Game &Suite")
        self._suite_action_group = QActionGroup(self)
        self._suite_action_group.setExclusive(True)
        self._suite_actions: dict[SuiteMode, QAction] = {}
        for mode in (SuiteMode.FORZA, SuiteMode.AC, SuiteMode.NFS, SuiteMode.CREW):
            meta = SUITE_DISPLAY[mode]
            label = meta["label"] + ("" if meta["enabled"] else " (Coming Soon)")
            act = QAction(label, self, checkable=True)
            act.setEnabled(bool(meta["enabled"]))
            act.setChecked(mode == self._suite_mode)
            act.triggered.connect(lambda checked=False, m=mode: self._on_suite_menu_selected(m))
            self._suite_action_group.addAction(act)
            suite_menu.addAction(act)
            self._suite_actions[mode] = act
        custom_menu.addSeparator()

        self._swap_recents_act = QAction("&Swap recents with image searcher", self, checkable=True)
        self._swap_recents_act.setStatusTip("Replace the Recent files list with a Google-style image search panel.")
        from PySide6.QtCore import QSettings as _QS
        _cs = _QS("FD6", "Forza Designer 6")
        _cs.beginGroup("customizations")
        _init_swap = _cs.value("swap_recents_with_image_searcher", False, type=bool)
        _cs.endGroup()
        self._swap_recents_act.setChecked(_init_swap)
        self._swap_recents_act.triggered.connect(self._on_swap_recents_toggled)
        custom_menu.addAction(self._swap_recents_act)
        
        if hasattr(self, "upload"):
            self.upload.set_use_image_searcher(_init_swap)

        view_menu.addSeparator()
        particles_menu = view_menu.addMenu("&Particles")
        self._particles_enabled_act = QAction("&Show particles", self, checkable=True)
        self._particles_enabled_act.triggered.connect(self._on_particles_toggle)
        particles_menu.addAction(self._particles_enabled_act)
        particles_menu.addSeparator()
        density_menu = particles_menu.addMenu("&Density")
        self._particle_count_group = QActionGroup(self)
        self._particle_count_group.setExclusive(True)
        from fd6.gui.particles import COUNT_OPTIONS
        for n in COUNT_OPTIONS:
            label = "Off (0)" if n == 0 else f"{n} particles"
            a = QAction(label, self, checkable=True)
            a.triggered.connect(lambda _c, count=n: self._on_particles_count(count))
            self._particle_count_group.addAction(a)
            density_menu.addAction(a)

        fh6_menu = mbar.addMenu("F&H6")
        status_act = QAction("FH6 &Status…", self)
        status_act.triggered.connect(self._show_fh6_status)
        fh6_menu.addAction(status_act)
        discovery_act = QAction("&Discovery Workflow…", self)
        discovery_act.triggered.connect(self._show_discovery_help)
        fh6_menu.addAction(discovery_act)
        fh6_menu.addSeparator()
        reload_act = QAction("&Reload Patterns", self)
        reload_act.triggered.connect(self._refresh_inject_button)
        fh6_menu.addAction(reload_act)

        help_menu = mbar.addMenu("&Help")
        about_act = QAction("&About FD6…", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _set_theme(self, theme_name: str) -> None:
        apply_theme(QApplication.instance(), theme_name)
        bp = badge_path(badge_filename_for_theme(theme_name))
        if bp:
            QApplication.instance().setWindowIcon(QIcon(str(bp)))
            self.setWindowIcon(QIcon(str(bp)))
            if hasattr(self, "brand_banner"):
                self.brand_banner.set_badge(bp)
        if hasattr(self, "particles"):
            pal = THEMES.get(theme_name, THEMES["Default"])
            self.particles.set_theme_colors(
                pal["particle_1"], pal["particle_2"], pal["particle_3"],
            )
        self.statusBar().showMessage(f"Theme: {theme_name}", 3000)

    def _on_swap_recents_toggled(self, checked: bool) -> None:
        from PySide6.QtCore import QSettings
        if hasattr(self, "upload"):
            self.upload.set_use_image_searcher(checked)
        s = QSettings("FD6", "Forza Designer 6")
        s.beginGroup("customizations")
        s.setValue("swap_recents_with_image_searcher", checked)
        s.endGroup()

    def _apply_suite_mode(self, mode: SuiteMode) -> None:
        self._suite_mode = mode
        is_ac = (mode == SuiteMode.AC)
        self.preview_stack.setCurrentIndex(1 if is_ac else 0)
        self.settings_stack.setCurrentIndex(1 if is_ac else 0)
        if hasattr(self.upload, "upload_json_btn"):
            self.upload.upload_json_btn.setVisible(not is_ac)
        if hasattr(self.upload, "download_json_btn"):
            self.upload.download_json_btn.setVisible(not is_ac)
        if hasattr(self, "_suite_actions"):
            for m, act in self._suite_actions.items():
                act.setChecked(m == mode)
        meta = SUITE_DISPLAY[mode]
        self.statusBar().showMessage(f"Game suite: {meta['label']}", 4000)

    def _on_suite_menu_selected(self, mode: SuiteMode) -> None:
        meta = SUITE_DISPLAY[mode]
        if not meta["enabled"]:
            return
        if mode == self._suite_mode:
            return
        self._apply_suite_mode(mode)
        save_suite_mode(mode)

    def _prompt_suite_on_first_launch(self) -> None:
        if not self._suite_first_launch:
            return
        dlg = GameSuiteDialog(self, current=None)
        result = dlg.exec()
        if result and dlg.selected is not None:
            self._apply_suite_mode(dlg.selected)
            save_suite_mode(dlg.selected)
        else:
            self._apply_suite_mode(SuiteMode.FORZA)
            save_suite_mode(SuiteMode.FORZA)
        self._suite_first_launch = False

    def _on_ac_export_clicked(self, cfg: dict) -> None:
        from fd6.ac.livery_writer import write_acc_livery
        from fd6.ac.slot_planner import plan_slots
        from fd6.ac.texture_pipeline import build_decal_texture

        if not getattr(self, "_current_path", None):
            QMessageBox.information(self, "No image", "Upload an image first via 'Upload Image…' before exporting.")
            return
        if not cfg.get("car_model"):
            QMessageBox.information(self, "Pick a car", "Select an ACC car model from the dropdown before exporting.")
            return
        try:
            rgba, applied_aspect = build_decal_texture(
                self._current_path,
                target_long_edge=int(cfg["resolution"]),
                aspect_choice=str(cfg["aspect"]),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Texture build failed", f"{type(exc).__name__}: {exc}")
            return

        slot_filenames = plan_slots(
            auto=bool(cfg["auto_slot"]),
            manual_main=cfg.get("manual_main_slots"),
            manual_sponsors=cfg.get("manual_sponsor_slots"),
        )
        self.ac_preview.set_slots([(s, rgba) for s in slot_filenames])

        result = write_acc_livery(
            profile=cfg["profile"],
            car_model=cfg["car_model"],
            team_name=cfg["team_name"] or f"FD6_{Path(self._current_path).stem}",
            rgba=rgba,
            slot_filenames=slot_filenames,
            display_name=cfg["display_name"],
            race_number=int(cfg["race_number"]),
            paint=cfg.get("paint"),
        )
        if result.success:
            self.ac_preview.progress.setValue(100)
            self.ac_preview.status_label.setText(result.message)
            self.statusBar().showMessage(result.message, 8000)
            QMessageBox.information(self, "Livery exported", f"{result.message}\n\nFolder:\n{result.team_folder}")
        else:
            self.ac_preview.progress.setValue(0)
            QMessageBox.critical(self, "Export failed", result.message)

    def _on_font_pick(self, display_name: str) -> None:
        apply_font(QApplication.instance(), display_name)
        self.statusBar().showMessage(f"Font: {display_name}", 3000)

    def _on_particles_toggle(self, checked: bool) -> None:
        if hasattr(self, "particles"):
            self.particles.set_enabled(checked)

    def _on_particles_count(self, count: int) -> None:
        if hasattr(self, "particles"):
            self.particles.set_count(count)
            self._particles_enabled_act.setChecked(self.particles.enabled() and count > 0)

    def _sync_particle_count_check(self, n: int) -> None:
        for act in self._particle_count_group.actions():
            t = act.text()
            digits = "".join(c for c in t.split()[0] if c.isdigit())
            if digits.isdigit() and int(digits) == n:
                act.setChecked(True)
                return

    def _music_toggle_play(self) -> None:
        playing = self.music.toggle_play()
        self._music_play_act.setChecked(playing)

    def _music_toggle_mute(self) -> None:
        muted = self.music.toggle_mute()
        self._music_mute_act.setChecked(muted)

    def _music_next(self) -> None:
        self.music.next_track()

    def _music_set_volume(self, vol: float) -> None:
        self.music.set_volume(vol)

    def _on_music_state(self, playing: bool) -> None:
        self._music_play_act.setChecked(playing)

    def _on_music_muted(self, muted: bool) -> None:
        self._music_mute_act.setChecked(muted)

    def _on_music_volume(self, vol: float) -> None:
        self._sync_volume_check(vol)

    def _sync_volume_check(self, vol: float) -> None:
        nearest = round(vol * 10) * 10
        for act in self._music_vol_group.actions():
            label = act.text().rstrip("%")
            if label.isdigit() and int(label) == nearest:
                act.setChecked(True)
                break

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Forza Designer 6+",
            f"<b>Forza Designer 6+</b><br>v0.5.0<br>"
            f"<i>For Forza Horizon 3 / 4 / 5 / 6 (FH6 build {FH6_TARGET_BUILD}) "
            f"and Assetto Corsa Competizione</i><br><br>"
            "Multi-game livery suite. Forza titles: live memory injection of "
            "vinyl-group shapes (position, scale, rotation, color). Assetto "
            "Corsa Competizione: file-based PNG livery export to the user's "
            "Documents folder.<br><br>"
            "Inspired by forza-painter (the_adawg), built on the techniques of "
            "geometrize-lib (Sam Twidale) and Primitive (Michael Fogleman). "
            "LiveryGroup discovery approach adapted from bvzrays/forza-painter-fh6.<br><br>"
        )

    def _refresh_inject_button(self) -> None:
        ready = patterns_are_populated()
        self.settings_panel.inject_btn.setEnabled(ready)
        tip = (
            "Requires FH6 memory patterns. See README."
            if not ready else
            "Inject the most recent shapes into a running FH6 vinyl group."
        )
        self.settings_panel.inject_btn.setToolTip(tip)

    def _on_files_selected(self, paths: list[Path]) -> None:
        if self._suite_mode == SuiteMode.AC:
            if paths:
                self._current_path = Path(paths[-1])
                self.ac_preview.set_source(self._current_path)
                self._refresh_ac_preview()
                self.statusBar().showMessage(f"Loaded {self._current_path.name}. Adjust settings then click Export.", 6000)
            return
        for p in paths:
            self.queue.add(p)
        if self._worker is None:
            self._start_next()

    def _refresh_ac_preview(self) -> None:
        if not getattr(self, "_current_path", None):
            return
        try:
            from fd6.ac.slot_planner import plan_slots
            from fd6.ac.texture_pipeline import build_decal_texture
            cfg = self.ac_settings._gather_export_config()
            rgba, applied_aspect = build_decal_texture(
                self._current_path,
                target_long_edge=int(cfg["resolution"]),
                aspect_choice=str(cfg["aspect"]),
            )
            slot_filenames = plan_slots(
                auto=bool(cfg["auto_slot"]),
                manual_main=cfg.get("manual_main_slots"),
                manual_sponsors=cfg.get("manual_sponsor_slots"),
            )
            self.ac_preview.set_slots([(s, rgba) for s in slot_filenames])
            h, w, _ = rgba.shape
            self.ac_preview.status_label.setText(f"Preview ready — {w}×{h}  •  aspect {applied_aspect}  •  {len(slot_filenames)} slot(s) ready to write.")
            self.ac_preview.progress.setValue(100)
        except Exception as exc:
            self.ac_preview.status_label.setText(f"Preview build failed: {type(exc).__name__}: {exc}")
            self.ac_preview.progress.setValue(0)

    def _start_next(self) -> None:
        if self._worker is not None:
            return  
        nxt = self.queue.pop_next_queued()
        if nxt is None:
            self.statusBar().showMessage("Nothing queued.")
            return
        next_uid, next_path = nxt
        profile = self.settings_panel.build_profile()
        self._current_path = next_path
        self._current_uid = next_uid
        self._current_profile = profile
        self.preview.set_source(next_path)
        self.queue.set_status(next_uid, "running")

        add_white_bg = self.settings_panel.sticker_mode_cb.isChecked()
        self._worker = GenerationWorker(next_path, profile, sticker_mode=not add_white_bg)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.preview.connect(self.preview.on_preview)
        self._worker.checkpoint_written.connect(lambda p: self.statusBar().showMessage(f"Checkpoint: {p}", 4000))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()
        
        self.settings_panel.set_running(True)
        self.statusBar().showMessage(f"Generating: {next_path.name}")

    def _on_finished(self, out_path: str) -> None:
        if self._current_uid is not None:
            self.queue.set_json_path(self._current_uid, Path(out_path))
            self.queue.set_status(self._current_uid, "done")
        self._last_finished_json = Path(out_path)
        self.statusBar().showMessage(f"Saved: {out_path}", 8000)
        self.upload.mark_json_ready(self._last_finished_json)
        self.settings_panel.wiggle_btn.setEnabled(True)  
        self._loaded_json_path = self._last_finished_json
        self._teardown_thread()
        self._start_next()

    def _on_error(self, msg: str) -> None:
        if self._current_uid is not None:
            self.queue.set_status(self._current_uid, "error")
        QMessageBox.critical(self, "Generation error", msg)
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None
        self._current_path = None
        self._current_uid = None
        self.settings_panel.set_running(False)
        self.settings_panel.pause_btn.setChecked(False)

    def _toggle_pause(self) -> None:
        if not self._worker:
            return
        paused = self.settings_panel.pause_btn.isChecked()
        self._worker.set_pause(paused)
        self.statusBar().showMessage("Paused." if paused else "Resumed.", 3000)

    def _stop_current(self) -> None:
        if self._worker:
            self._worker.stop()

    def _on_inject_clicked(self) -> None:
        if self._loaded_json_path and self._loaded_json_path.exists():
            self._on_inject_json_path(self._loaded_json_path)
            return
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Pick shapes JSON to inject", "", "FD6 shapes (*.json);;All files (*)"
        )
        if json_path:
            self._on_inject_json_path(Path(json_path))

    def _on_json_loaded_for_preview(self, json_path: Path) -> None:
        from fd6.io.exporter import load_json
        from fd6.shapegen.render import render_shapes
        try:
            doc = load_json(str(json_path))
            shapes = doc.materialize_shapes()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"{type(exc).__name__}: {exc}")
            return
        w, h = doc.image_size if doc.image_size and doc.image_size[0] > 0 else (1200, 800)
        self.statusBar().showMessage(f"Rendering preview of {len(shapes)} shapes from {json_path.name}...")
        
        white_bg_checked = self.settings_panel.sticker_mode_cb.isChecked()
        render_transparent = bool(getattr(doc, "sticker_mode", False)) or not white_bg_checked
        canvas = render_shapes(shapes, w, h, background=(255, 255, 255),
                               transparent_bg=render_transparent)
        self.preview.source_view.clear_image()
        self.preview.preview_view.set_numpy(canvas)
        self.preview.status_label.setText(
            f"Loaded {len(shapes)} shapes from '{json_path.name}'  •  {w}x{h}  •  ready to inject"
        )
        self.preview.progress.setValue(100)
        self._loaded_json_path = json_path
        self.settings_panel.wiggle_btn.setEnabled(True)  
        self.statusBar().showMessage(f"Preview ready. Click 'Inject into FH6' or 'Post-Optimize Loaded JSON'.", 8000)

    def _on_inject_json_path(self, json_path: Path) -> None:
        from fd6.inject import patterns_are_populated
        from fd6.gui.inject_worker import InjectionWorker
        from fd6.gui.inject_dialog import InjectionDialog

        if not patterns_are_populated():
            QMessageBox.warning(self, "FH6 Injection", "Patterns file is incomplete. Use FH6 → Discovery Workflow… to populate it.")
            return

        if getattr(self, "_inject_thread", None) is not None:
            QMessageBox.information(self, "Inject in progress", "An injection is already running. Wait for it to finish.")
            return

        target_key = self.settings_panel.selected_target_profile_key()
        self._inject_worker = InjectionWorker(json_path, profile_key=target_key)
        self._inject_thread = QThread(self)
        self._inject_worker.moveToThread(self._inject_thread)

        from fd6.inject.game_profiles import get_profile, default_profile
        try:
            game_label = get_profile(target_key).label
        except ValueError:
            game_label = default_profile().label
        self._inject_dialog = InjectionDialog(self, json_name=json_path.name, game_label=game_label)

        self._inject_worker.scan_progress.connect(self._inject_dialog.on_scan_progress)
        self._inject_worker.write_progress.connect(self._inject_dialog.on_write_progress)
        self._inject_worker.status.connect(self._inject_dialog.on_status)
        self._inject_worker.done.connect(self._inject_dialog.on_done)

        self._inject_worker.scan_progress.connect(self._on_inject_scan_progress)
        self._inject_worker.write_progress.connect(self._on_inject_write_progress)
        self._inject_worker.status.connect(self._on_inject_status)
        self._inject_worker.done.connect(self._on_inject_done)

        self._inject_thread.started.connect(self._inject_worker.run)
        self._set_inject_status("Starting injection…", "info")
        self._inject_thread.start()
        self._inject_dialog.exec()

    def _set_inject_status(self, message: str, severity: str = "info") -> None:
        color = {
            "info":    "#cccccc",
            "success": "#2ecc71",
            "warning": "#f1c40f",
            "error":   "#ff4d4d",
        }.get(severity, "#cccccc")
        bg = {
            "info":    "#1f1f1f",
            "success": "#0c2417",
            "warning": "#2a2410",
            "error":   "#2a1414",
        }.get(severity, "#1f1f1f")
        sb = self.statusBar()
        sb.setStyleSheet(f"QStatusBar {{ background: {bg}; color: {color}; font-weight: bold; }}")
        sb.showMessage(message, 0 if severity in ("info", "error", "warning") else 10000)

    def _on_inject_status(self, message: str, severity: str) -> None:
        self._set_inject_status(message, severity)

    def _on_inject_scan_progress(self, scanned: int, total: int, hits: int) -> None:
        pct = int(round(100 * scanned / max(1, total)))
        try:
            from fd6.inject.game_profiles import get_profile
            target_key = self.settings_panel.selected_target_profile_key()
            short = get_profile(target_key).label.replace(" (BETA)", "")
            if short.startswith("Forza Horizon ") and short[len("Forza Horizon "):].strip().isdigit():
                short = "FH" + short[len("Forza Horizon "):].strip()
        except Exception as exc:
            short = "FH6"
        self._set_inject_status(f"Scanning {short} memory… {scanned}/{total} regions ({pct}%) — {hits} shape structs found so far", "info")

    def _on_inject_write_progress(self, written: int, total: int) -> None:
        pct = int(round(100 * written / max(1, total)))
        self._set_inject_status(f"Writing shapes… {written}/{total} ({pct}%)", "info")

    def _on_inject_done(self) -> None:
        if self._inject_thread:
            self._inject_thread.quit()
            self._inject_thread.wait(3000)
        self._inject_worker = None
        self._inject_thread = None
        self._inject_dialog = None

    def _on_download_json(self) -> None:
        import shutil
        if not self._last_finished_json or not self._last_finished_json.exists():
            QMessageBox.information(self, "No JSON yet", "No completed generation yet. Generate from an image first (or use Upload JSON to load an existing one).")
            return
        self._save_json_copy(self._last_finished_json)

    def _on_queue_download_json(self, uid: int) -> None:
        json_path = self.queue.json_path_for(uid)
        if not json_path or not Path(json_path).exists():
            QMessageBox.information(
                self, "No JSON for this item",
                "This queue item hasn't finished generating, or its JSON is missing."
            )
            return
        self._save_json_copy(Path(json_path))

    def _save_json_copy(self, src: Path) -> None:
        import shutil
        dest, _ = QFileDialog.getSaveFileName(self, "Save shapes JSON as…", src.name, "FD6 shapes (*.json);;All files (*)")
        if not dest:
            return
        try:
            shutil.copy2(str(src), dest)
            self.statusBar().showMessage(f"Exported to {dest}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _show_fh6_status(self) -> None:
        from fd6.inject import discovery as disc
        from fd6.inject.patterns_io import load_patterns, has_usable_patterns

        pid = disc.find_game_pid()
        if pid is None:
            game_line = "forzahorizon6.exe: <b>NOT RUNNING</b>"
        else:
            info = disc.process_summary(pid)
            game_line = (
                f"forzahorizon6.exe PID <b>{info.pid}</b><br>"
                f"&nbsp;&nbsp;committed regions: {info.region_count}<br>"
                f"&nbsp;&nbsp;private+writable bytes: {info.private_writable_bytes:,}<br>"
                f"&nbsp;&nbsp;image bytes: {info.image_bytes:,}"
            )
        pf = load_patterns()
        QMessageBox.information(
            self,
            "FH6 Status",
            f"{game_line}<br><br>"
            f"<b>Patterns file</b><br>"
            f"&nbsp;&nbsp;patterns: {len(pf.patterns)}<br>"
            f"&nbsp;&nbsp;shape_struct.stride: {pf.shape_struct.stride_bytes}<br>"
            f"&nbsp;&nbsp;shape_struct.fields: {len(pf.shape_struct.fields)}<br>"
            f"&nbsp;&nbsp;injector ready: <b>{has_usable_patterns(pf)}</b>"
        )

    def _show_discovery_help(self) -> None:
        QMessageBox.information(
            self,
            "FH6 Discovery Workflow",
            "<p>Discovery is done from the command line, run from the FD6 project root:</p>"
            "<pre>"
            "python -m fd6.inject status\n"
            "python -m fd6.inject scan-float &lt;known sphere coord&gt;\n"
            "python -m fd6.inject narrow &lt;moved coord&gt;   (repeat until ~1 hit)\n"
            "python -m fd6.inject dump &lt;addr&gt; 256\n"
            "python -m fd6.inject find-refs &lt;struct_addr&gt;\n"
            "python -m fd6.inject save-pattern shape_array_ref '&lt;AOB&gt;' --offset 3\n"
            "python -m fd6.inject test-injector\n"
            "</pre>"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "brand_banner") and self.brand_banner is not None:
            self.brand_banner.reposition()
        if hasattr(self, "particles") and self.particles is not None:
            self.particles.reposition()
            self._sync_particle_exclude_rect()
            if hasattr(self, "brand_banner"):
                self.brand_banner.raise_()

    def _compute_particle_exclude_rect(self):
        if not hasattr(self, "preview") or self.preview is None:
            return None
        if self.preview.width() <= 0 or self.preview.height() <= 0:
            return None
        from PySide6.QtCore import QRect
        top_left = self.preview.mapTo(self, self.preview.rect().topLeft())
        return QRect(top_left, self.preview.size())

    def _sync_particle_exclude_rect(self) -> None:
        if not hasattr(self, "particles") or self.particles is None:
            return
        excl = self._compute_particle_exclude_rect()
        if excl is not None:
            self.particles.set_exclude_rect(excl)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
            if self._thread:
                self._thread.quit()
                self._thread.wait(3000)
        try:
            if (hasattr(self, "upload")
                    and getattr(self.upload, "image_search", None) is not None):
                self.upload.image_search.shutdown()
        except Exception:
            pass
        super().closeEvent(event)

    def _on_resume_triggered(self) -> None:
        json_path_str, _ = QFileDialog.getOpenFileName(
            self, "Select partial FD6 JSON to resume", "", "FD6 shapes (*.json);;All files (*)"
        )
        if not json_path_str:
            return
        json_path = Path(json_path_str)

        from fd6.io.resume import load_resume
        try:
            doc, shapes = load_resume(json_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not parse JSON document:\n{exc}")
            return

        if not shapes:
            QMessageBox.warning(self, "No shapes found", "This JSON file contains zero shapes. Cannot resume.")
            return

        image_path_str, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select original source image (Hint: {doc.source_image})", 
            str(json_path.parent), 
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All files (*)"
        )
        if not image_path_str:
            return
        image_path = Path(image_path_str)

        self._start_resume(json_path, image_path, doc, shapes)

    def _start_resume(self, json_path: Path, image_path: Path, doc: FD6Document, shapes: list[Shape]) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Generation in progress", "Please stop or wait for the current generation to finish first.")
            return

        from fd6.shapegen.profile import load_profile_from_file, list_bundled_profiles
        
        ui_profile = self.settings_panel.build_profile()
        ui_stop_at = ui_profile.stop_at

        profile = ui_profile
        if doc.profile:
            for p_path in list_bundled_profiles():
                if p_path.stem == doc.profile:
                    try:
                        profile = load_profile_from_file(p_path)
                    except Exception:
                        pass
                    break

        profile.stop_at = ui_stop_at
        if profile.stop_at <= len(shapes):
            profile.stop_at = len(shapes) + 1000  

        self._current_path = image_path
        self._current_profile = profile
        self.preview.set_source(image_path)

        output_dir = json_path.parent
        add_white_bg = self.settings_panel.sticker_mode_cb.isChecked()

        self._worker = GenerationWorker(
            image_path, profile, output_dir=output_dir, 
            sticker_mode=not add_white_bg, seed_shapes=shapes,
            target_size=doc.image_size  
        )

        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.preview.connect(self.preview.on_preview)
        self._worker.checkpoint_written.connect(lambda p: self.statusBar().showMessage(f"Checkpoint: {p}", 4000))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        
        self._thread.start()

        self.settings_panel.set_running(True)
        self.statusBar().showMessage(f"Resuming generation ({len(shapes)} shapes): {image_path.name}")

    def _on_post_optimize_clicked(self) -> None:
        if not self._loaded_json_path or not self._loaded_json_path.exists():
            return
            
        image_path_str, _ = QFileDialog.getOpenFileName(
            self, f"Select original source image for {self._loaded_json_path.name}", 
            str(self._loaded_json_path.parent), 
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All files (*)"
        )
        if not image_path_str:
            return
        image_path = Path(image_path_str)
        
        self._start_post_optimization(self._loaded_json_path, image_path)

    def _start_post_optimization(self, json_path: Path, image_path: Path) -> None:
        if self._worker is not None:
            QMessageBox.warning(self, "Generation in progress", "Please stop or wait for the current generation to finish first.")
            return

        from fd6.shapegen.engine import PostOptimizeWorker
        profile = self.settings_panel.build_profile()
        
        self._current_path = image_path
        self.preview.set_source(image_path)
        
        add_white_bg = self.settings_panel.sticker_mode_cb.isChecked()
        self._worker = PostOptimizeWorker(json_path, image_path, profile, sticker_mode=not add_white_bg)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.preview.connect(self.preview.on_preview)
        self._worker.status.connect(self._on_inject_status)  
        self._worker.finished.connect(self._on_post_optimize_finished)
        self._worker.error.connect(self._on_error)
        
        self._thread.start()
        self.settings_panel.set_running(True)

    def _on_post_optimize_finished(self, out_path_str: str) -> None:
        self._last_finished_json = Path(out_path_str)
        self.upload.mark_json_ready(self._last_finished_json)
        self._on_json_loaded_for_preview(self._last_finished_json)  
        self._teardown_thread()
        QMessageBox.information(
            self, "Optimization complete", 
            f"Successfully optimized, wiggled, and regenerated shapes.\n\nSaved as:\n{self._last_finished_json.name}"
        )