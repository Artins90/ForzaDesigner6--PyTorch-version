from __future__ import annotations

from pathlib import Path

from fd6.io.exporter import load_json
from fd6.io.json_schema import FD6Document
from fd6.shapegen.shapes import Shape


def load_resume(path: str | Path) -> tuple[FD6Document, list[Shape]]:
    """Load a prior FD6 JSON to continue generation from."""
    doc = load_json(path)
    return doc, doc.materialize_shapes()
