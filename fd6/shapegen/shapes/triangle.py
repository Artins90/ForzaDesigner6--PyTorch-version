from __future__ import annotations
from dataclasses import dataclass
import math
import random
import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register

@dataclass
class Triangle(Shape):
    type_name = "triangle"
    x: float = 0.0
    y: float = 0.0
    rx: float = 1.0
    ry: float = 1.0
    angle: float = 0.0

    def _get_vertices(self) -> list[tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        def rotate(px, py):
            return (self.x + px * cos_a - py * sin_a, self.y + px * sin_a + py * cos_a)

        v1 = rotate(0, -self.ry)         
        v2 = rotate(-self.rx, self.ry)   
        v3 = rotate(self.rx, self.ry)    
        return [v1, v2, v3]

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        verts = self._get_vertices()
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return _clip_bbox(min(xs), min(ys), max(xs) + 1, max(ys) + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
            
        v1, v2, v3 = self._get_vertices()
        ys = np.arange(y0, y1, dtype=np.float32)
        xs = np.arange(x0, x1, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        
        def edge(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            
        d1 = edge(v1[0], v1[1], v2[0], v2[1], xg, yg)
        d2 = edge(v2[0], v2[1], v3[0], v3[1], xg, yg)
        d3 = edge(v3[0], v3[1], v1[0], v1[1], xg, yg)
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        mask = ~(has_neg & has_pos)
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "Triangle":
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
            new.angle = (new.angle + rng.gauss(0, 25)) % 180.0
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
    def from_json(cls, data: dict) -> "Triangle":
        return cls(
            color=tuple(data["color"]),
            x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)),
            rx=float(data.get("rx", 1.0)), ry=float(data.get("ry", 1.0)),
            angle=float(data.get("angle", 0.0)),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "Triangle":
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            rx=rng.uniform(1, max(2, w / 8)), ry=rng.uniform(1, max(2, h / 8)),
            angle=rng.uniform(0, 180),
        )

_register(Triangle)