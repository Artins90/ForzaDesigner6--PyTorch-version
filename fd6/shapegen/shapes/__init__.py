# --- START OF FILE fd6/shapegen/shapes/__init__.py ---
from __future__ import annotations

# Import base primitives to expose them to the rest of the application
from .base import Shape, random_shape, shape_from_json, SHAPE_REGISTRY

# Explicitly import all shape submodules to trigger their 
# @registration decorators and populate SHAPE_REGISTRY at startup:
from . import circle
from . import ellipse
from . import half_ellipse
from . import rectangle
from . import right_triangle
from . import rounded_rectangle
from . import triangle
# --- END OF FILE ---