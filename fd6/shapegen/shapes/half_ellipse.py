from __future__ import annotations
from dataclasses import dataclass
import math
import random
import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register

@dataclass
class HalfEllipse(Shape):
    type_name = "half_ellipse"
    x: float = 0.0
    y: float = 0.0
    rx: float = 1.0
    ry: float = 1.0
    angle: float = 0.0  

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        r = max(self.rx, self.ry)
        return _clip_bbox(self.x - r, self.y - r, self.x + r + 1, self.y + r + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        ys = np.arange(y0, y1, dtype=np.float32) - self.y
        xs = np.arange(x0, x1, dtype=np.float32) - self.x
        xg, yg = np.meshgrid(xs, ys)
        
        xr = cos_a * xg + sin_a * yg
        yr = -sin_a * xg + cos_a * yg
        
        dx = xr / max(self.rx, 1e-6)
        dy = yr / max(self.ry, 1e-6)
        
        # Ellipse bound cut in half along the local Y-axis
        mask = ((dx ** 2 + dy ** 2) <= 1.0) & (yr >= 0.0)
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "HalfEllipse":
        from copy import copy as shallow_copy
        new = shallow_copy(self)
        which = rng.randint(0, 2)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        elif which == 1:
            new.rx = _clamp(new.rx + rng.gauss(0, 16), 1, w)
            new.ry = _clamp(new.ry + rng.gauss(0, 16), 1, h)
        else:
            new.angle = (new.angle + rng.gauss(0, 25)) % 360.0
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name, 
            "x": round(self.x, 3), "y": round(self.y, 3),
            "rx": round(self.rx, 3), "ry": round(self.ry, 3),
            "angle": round(self.angle, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "HalfEllipse":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            rx=float(data.get("rx", 1.0)), ry=float(data.get("ry", 1.0)),
            angle=float(data.get("angle", 0.0)),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "HalfEllipse":
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            rx=rng.uniform(1, max(2, w / 8)), ry=rng.uniform(1, max(2, h / 8)),
            angle=rng.uniform(0, 360),
        )

_register(HalfEllipse)