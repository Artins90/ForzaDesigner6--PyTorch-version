from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import random
import math
import numpy as np

ShapeType = str
SHAPE_REGISTRY: dict[ShapeType, type["Shape"]] = {}

def _register(cls: type["Shape"]) -> type["Shape"]:
    SHAPE_REGISTRY[cls.type_name] = cls
    return cls

@dataclass
class Shape(ABC):
    type_name: ClassVar[ShapeType] = "shape"
    # Base fallback is 100% opaque just in case
    color: tuple[int, int, int, int] = (0, 0, 0, 255)

    @abstractmethod
    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        pass

    @abstractmethod
    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        pass

    @abstractmethod
    def mutate(self, rng: random.Random, w: int, h: int) -> "Shape":
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> "Shape":
        pass

    @classmethod
    @abstractmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "Shape":
        pass

    def with_color(self, color: tuple[int, int, int, int]) -> "Shape":
        from copy import copy as shallow_copy
        new = shallow_copy(self)
        new.color = color
        return new

def random_shape(rng: random.Random, w: int, h: int, allowed_types: list[ShapeType]) -> Shape:
    type_name = rng.choice(allowed_types)
    cls = SHAPE_REGISTRY[type_name]
    return cls.random(rng, w, h)

def shape_from_json(data: dict) -> Shape:
    type_name = data.get("type")
    # Native fallback for old JSONs that used "rotated_rectangle"
    if type_name == "rotated_rectangle" and "rotated_rectangle" not in SHAPE_REGISTRY:
        type_name = "rectangle"
    if type_name not in SHAPE_REGISTRY:
        raise ValueError(f"Unknown shape type: {type_name!r}")
    return SHAPE_REGISTRY[type_name].from_json(data)

def _clip_bbox(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, int, int, int]:
    cx0 = max(0, int(np.floor(x0)))
    cy0 = max(0, int(np.floor(y0)))
    cx1 = min(w, int(np.ceil(x1)))
    cy1 = min(h, int(np.ceil(y1)))
    if cx1 <= cx0 or cy1 <= cy0:
        return (0, 0, 0, 0)
    return cx0, cy0, cx1, cy1

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))