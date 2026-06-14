from __future__ import annotations

import math
import numpy as np
from fd6.shapegen.shapes.base import Shape

def get_true_area(shape: Shape) -> float:
    tname = shape.type_name
    if tname == "circle": return math.pi * (max(getattr(shape, 'r', 1e-6), 1e-6)**2)
    elif tname in ("ellipse", "rotated_ellipse"): return math.pi * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname in ("rectangle", "rotated_rectangle"): return 4.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname == "triangle": return 2.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    return 0.0

def compute_diff_sq_sum(current: np.ndarray, target: np.ndarray, alpha_mask: np.ndarray | None = None):
    diff = current.astype(np.float32) - target.astype(np.float32)
    sq = diff * diff
    if alpha_mask is None:
        return float(sq.sum()), float(current.size)
    weight = (alpha_mask > 0)[:, :, None].astype(np.float32)
    return float((sq * weight).sum()), float(weight.sum() * 3)

def rms_error(a: np.ndarray, b: np.ndarray, alpha_mask: np.ndarray | None = None) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    sq = diff * diff
    if alpha_mask is None:
        return float(np.sqrt(sq.mean()))
    weight = (alpha_mask > 0)[:, :, None].astype(np.float32)
    total = float((sq * weight).sum())
    n = float(weight.sum() * 3)
    if n < 1: return 0.0
    return float(np.sqrt(total / n))

def compute_optimal_color(region_tgt, region_cur, mask_local, alpha):
    if mask_local.size == 0: return (0, 0, 0, alpha)
    m = mask_local.astype(np.float32) / 255.0
    weight = m.sum()
    if weight < 0.5: return (0, 0, 0, alpha)
    a = alpha / 255.0
    if a < 1e-6: return (0, 0, 0, alpha)
    src = (region_tgt - (1.0 - a) * region_cur) / a
    src_masked = src * m[:, :, None]
    avg = src_masked.reshape(-1, 3).sum(axis=0) / weight
    avg = np.clip(avg, 0, 255).astype(np.int32)
    return (int(avg[0]), int(avg[1]), int(avg[2]), alpha)

def composite(current, shape, target, alpha_mask=None):
    h, w = current.shape[:2]
    mask_local, bbox = shape.rasterize_mask(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0: return current, rms_error(current, target, alpha_mask)
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        effective_mask = np.minimum(mask_local, region_alpha)
    else: effective_mask = mask_local
        
    region_cur = current[y0:y1, x0:x1].astype(np.float32)
    region_tgt = target[y0:y1, x0:x1].astype(np.float32)
    color = compute_optimal_color(region_tgt, region_cur, effective_mask, shape.color[3])
    
    new = current.copy()
    a = color[3] / 255.0
    src = np.array(color[:3], dtype=np.float32)
    m = (effective_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = m * (a * src + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
    new[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    shape.color = color
    return new, rms_error(new, target, alpha_mask)

STICKER_OVERLAP_MIN = 0.995

def score_shape(shape, current, target, alpha_mask=None, current_diff_sq=None, n_px=None):
    if current_diff_sq is None or n_px is None:
        c_sq, c_n = compute_diff_sq_sum(current, target, alpha_mask)
        current_diff_sq = current_diff_sq if current_diff_sq is not None else c_sq
        n_px = n_px if n_px is not None else c_n

    h, w = current.shape[:2]
    mask_local, bbox = shape.rasterize_mask(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0: return float("inf"), shape.color
        
    region_cur = current[y0:y1, x0:x1].astype(np.float32)
    region_tgt = target[y0:y1, x0:x1].astype(np.float32)
    
    true_area = get_true_area(shape)
    shape_body = mask_local >= 128
    canvas_area = float(np.count_nonzero(shape_body))
    
    effective_mask = mask_local
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        opaque_body = region_alpha >= 128
        if not np.any(opaque_body): return float("inf"), shape.color
        inside = float(np.count_nonzero(shape_body & opaque_body)) 
        
        # Sticker overlap constraint based strictly on rasterized canvas_area
        if inside / max(canvas_area, 1.0) < STICKER_OVERLAP_MIN: return float("inf"), shape.color
        effective_mask = np.minimum(mask_local, region_alpha)
        
    color = compute_optimal_color(region_tgt, region_cur, effective_mask, shape.color[3])
    
    a = color[3] / 255.0
    src = np.array(color[:3], dtype=np.float32)
    m = (mask_local.astype(np.float32) / 255.0)[:, :, None]
    blended = region_cur + m * (a * src + (1.0 - a) * region_cur - region_cur)
    
    diff_in = blended - region_tgt
    diff_old = region_cur - region_tgt
    delta_sq = (diff_in ** 2) - (diff_old ** 2)
    
    # Bleed penalty is only applied if the shape bounding box is clipped by canvas boundaries
    rx_val = getattr(shape, 'rx', getattr(shape, 'r', 1.0))
    ry_val = getattr(shape, 'ry', getattr(shape, 'r', 1.0))
    is_clipped = (shape.x - rx_val < 0 or shape.x + rx_val > w or 
                  shape.y - ry_val < 0 or shape.y + ry_val > h)
                  
    if not is_clipped:
        bleed_penalty = 0.0
    else:
        bleed_pixels = max(0.0, true_area - canvas_area - (true_area * 0.05))
        bleed_penalty = bleed_pixels * (255.0 ** 2)
    
    if alpha_mask is None:
        region_delta_sq = float(delta_sq.sum())
        total_sq = current_diff_sq + region_delta_sq + bleed_penalty
        return float(np.sqrt(max(0.0, total_sq) / n_px)), color
        
    weight_region = (alpha_mask[y0:y1, x0:x1] > 0)[:, :, None].astype(np.float32)
    region_delta_sq = float((delta_sq * weight_region).sum())
    total_sq = current_diff_sq + region_delta_sq + bleed_penalty
    if n_px < 1: return 0.0, color
    return float(np.sqrt(max(0.0, total_sq) / n_px)), color

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, RuntimeError): return False

def compute_diff_sq_dispatch(current, target, alpha_mask, use_gpu):
    if use_gpu and _torch_cuda_available():
        from fd6.shapegen.gpu_scoring import compute_diff_sq_sum_gpu
        return compute_diff_sq_sum_gpu(current, target, alpha_mask)
    return compute_diff_sq_sum(current, target, alpha_mask)

def score_shape_dispatch(shape, current, target, alpha_mask, use_gpu, current_diff_sq=None, n_px=None):
    if use_gpu and _torch_cuda_available():
        import torch
        from fd6.shapegen.gpu_scoring import score_shape_gpu
        return score_shape_gpu(shape, current, target, alpha_mask, current_diff_sq, n_px)
    else:
        return score_shape(shape, current, target, alpha_mask, current_diff_sq, n_px)

def composite_dispatch(current, shape, target, alpha_mask, use_gpu):
    if use_gpu and _torch_cuda_available():
        import torch
        from fd6.shapegen.gpu_scoring import composite_gpu
        return composite_gpu(current, shape, target, alpha_mask)
    else:
        return composite(current, shape, target, alpha_mask)

def rms_error_dispatch(a, b, alpha_mask, use_gpu):
    if use_gpu and _torch_cuda_available():
        import torch
        if isinstance(a, torch.Tensor):
            from fd6.shapegen.gpu_scoring import _rms_error_gpu
            return _rms_error_gpu(a, b, alpha_mask)
    return rms_error(a, b, alpha_mask)