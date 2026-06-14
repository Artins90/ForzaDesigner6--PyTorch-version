# --- START OF FILE queue_panel.py ---
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from itertools import count

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
)

# Monotonic id source so every queued entry is uniquely identified even when the
# SAME image path is added multiple times.
_uid_counter = count(1)

@dataclass
class QueueItem:
    path: Path
    status: str = "queued"  # queued | running | done | error
    uid: int = field(default_factory=lambda: next(_uid_counter))
    json_path: Path | None = None  # set when generation finishes — enables per-row download


STATUS_ICON = {"queued": "⏳", "running": "▶", "done": "✓", "error": "✗"}


class QueueRow(QWidget):
    """One row of the queue: status-icon + filename + X (remove) button."""

    remove_requested = Signal(int)   # uid
    download_requested = Signal(int)  # uid — per-row "Download JSON" for a finished item

    def __init__(self, path: Path, uid: int, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._uid = uid
        self._status = "queued"
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 4, 2)
        h.setSpacing(8)
        self.label = QLabel(f"{STATUS_ICON['queued']} {path.name}", self)
        h.addWidget(self.label, stretch=1)
        
        # Per-row "Download JSON" — only shown once this item finishes generating
        self.dl_btn = QPushButton("⬇ JSON", self)
        self.dl_btn.setFixedHeight(22)
        self.dl_btn.setToolTip("Download this image's generated shapes JSON")
        self.dl_btn.setCursor(Qt.PointingHandCursor)
        self.dl_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #2ecc71;"
            " color: #2ecc71; border-radius: 11px; padding: 0 8px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(46, 204, 113, 0.18); color: #fff; }"
        )
        self.dl_btn.clicked.connect(lambda: self.download_requested.emit(self._uid))
        self.dl_btn.setVisible(False)  # appears only when status == "done"
        h.addWidget(self.dl_btn)
        
        self.x_btn = QPushButton("✕", self)
        self.x_btn.setFixedSize(QSize(22, 22))
        self.x_btn.setToolTip("Remove from queue")
        self.x_btn.setCursor(Qt.PointingHandCursor)
        self.x_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #555;"
            " color: #ccc; border-radius: 11px; padding: 0; font-size: 12px; }"
            "QPushButton:hover { background: rgba(255, 80, 80, 0.18);"
            " border-color: #ff5555; color: #fff; }"
            "QPushButton:disabled { color: #666; border-color: #333; }"
        )
        self.x_btn.clicked.connect(lambda: self.remove_requested.emit(self._uid))
        h.addWidget(self.x_btn)

    def set_status(self, status: str) -> None:
        self._status = status
        self.label.setText(f"{STATUS_ICON.get(status, '?')} {self._path.name}")
        self.x_btn.setEnabled(status != "running")
        self.dl_btn.setVisible(status == "done")


class QueuePanel(QWidget):
    cleared = Signal()
    item_removed = Signal(int)       # uid — emitted when user clicks the X
    download_requested = Signal(int)  # uid — emitted when user clicks a row's Download JSON

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel('Queue "fixed in 0.5.0"'))
        header.addStretch()
        self.clear_btn = QPushButton("Clear done")
        self.clear_btn.clicked.connect(self._clear_done)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        self.list = QListWidget(self)
        self.list.setObjectName("ThemeGlow")
        layout.addWidget(self.list, stretch=1)

        self._items: list[QueueItem] = []

    def _row_widget_at(self, idx: int) -> QueueRow | None:
        item = self.list.item(idx)
        if item is None:
            return None
        w = self.list.itemWidget(item)
        return w if isinstance(w, QueueRow) else None

    def _index_of(self, uid: int) -> int:
        for i, it in enumerate(self._items):
            if it.uid == uid:
                return i
        return -1

    def add(self, path: Path) -> int:
        """Add a path to the queue. Returns the unique id of the new entry."""
        item = QueueItem(path=path)
        self._items.append(item)
        li = QListWidgetItem()
        li.setData(Qt.UserRole, item.uid)
        row = QueueRow(path, item.uid, self.list)
        row.remove_requested.connect(self._on_row_remove)
        row.download_requested.connect(self.download_requested.emit)
        li.setSizeHint(row.sizeHint())
        self.list.addItem(li)
        self.list.setItemWidget(li, row)
        return item.uid

    def set_json_path(self, uid: int, json_path: Path) -> None:
        """Record the generated JSON for a finished item (enables its row download)."""
        idx = self._index_of(uid)
        if idx >= 0:
            self._items[idx].json_path = Path(json_path)

    def json_path_for(self, uid: int) -> Path | None:
        idx = self._index_of(uid)
        return self._items[idx].json_path if idx >= 0 else None

    def path_for(self, uid: int) -> Path | None:
        idx = self._index_of(uid)
        return self._items[idx].path if idx >= 0 else None

    def set_status(self, uid: int, status: str) -> None:
        idx = self._index_of(uid)
        if idx < 0:
            return
        self._items[idx].status = status
        row = self._row_widget_at(idx)
        if row is not None:
            row.set_status(status)

    def pop_next_queued(self) -> tuple[int, Path] | None:
        """Return (uid, path) of the first still-queued entry, or None."""
        for it in self._items:
            if it.status == "queued":
                return it.uid, it.path
        return None

    def remove(self, uid: int) -> bool:
        """Remove an item by uid. Refuses to remove a `running` item."""
        idx = self._index_of(uid)
        if idx < 0:
            return False
        if self._items[idx].status == "running":
            return False
        self.list.takeItem(idx)
        self._items.pop(idx)
        return True

    def _on_row_remove(self, uid: int) -> None:
        if self.remove(uid):
            self.item_removed.emit(uid)

    def _clear_done(self) -> None:
        i = 0
        while i < len(self._items):
            if self._items[i].status in ("done", "error"):
                self.list.takeItem(i)
                self._items.pop(i)
            else:
                i += 1
            self.cleared.emit()