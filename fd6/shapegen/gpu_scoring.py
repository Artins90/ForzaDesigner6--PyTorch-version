from __future__ import annotations
import math
from typing import Tuple
import torch
from fd6.shapegen.shapes import Shape

_cached_grids = {}

# Dynamic global caches to prevent CPU-GPU syncs on allocation
_INF_TENSOR: torch.Tensor | None = None
_FALSE_TENSOR: torch.Tensor | None = None

def get_true_area_gpu(shape: Shape) -> float:
    tname = shape.type_name
    if tname == "circle": return math.pi * (max(getattr(shape, 'r', 1e-6), 1e-6)**2)
    elif tname in ("ellipse", "rotated_ellipse"): return math.pi * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname in ("rectangle", "rotated_rectangle"): return 4.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname == "triangle": return 2.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    return 0.0

def _get_meshgrid_gpu(w: int, h: int, device: torch.device):
    key = (w, h, device)
    if key not in _cached_grids:
        ys = torch.arange(h, dtype=torch.float32, device=device)
        xs = torch.arange(w, dtype=torch.float32, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing='xy')
        _cached_grids[key] = (xg, yg)
    return _cached_grids[key]

def _rasterize_mask_gpu(shape: Shape, w: int, h: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    bbox = shape.bbox(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0: 
        return torch.zeros((0, 0), dtype=torch.bool, device='cuda'), bbox
    tname = shape.type_name
    xg_full, yg_full = _get_meshgrid_gpu(w, h, torch.device('cuda'))
    xg = xg_full[y0:y1, x0:x1]
    yg = yg_full[y0:y1, x0:x1]

    xg = xg - getattr(shape, 'x', 0.0)
    yg = yg - getattr(shape, 'y', 0.0)

    # Sharp boundaries preserve sharp details in high-frequency regions
    if tname == "rotated_ellipse":
        rad = math.radians(getattr(shape, 'angle', 0.0))
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        xr = cos_a * xg + sin_a * yg
        yr = -sin_a * xg + cos_a * yg
        dx = xr / max(getattr(shape, 'rx', 1.0), 1e-6)
        dy = yr / max(getattr(shape, 'ry', 1.0), 1e-6)
        mask = (dx ** 2 + dy ** 2) <= 1.0
    elif tname == "ellipse":
        dx = xg / max(getattr(shape, 'rx', 1.0), 1e-6)
        dy = yg / max(getattr(shape, 'ry', 1.0), 1e-6)
        mask = (dx ** 2 + dy ** 2) <= 1.0
    elif tname == "circle":
        r2 = max(getattr(shape, 'r', 1.0), 1e-6) ** 2
        mask = (xg ** 2 + yg ** 2) <= r2
    else:
        raise ValueError(f"Unsupported shape type for GPU rasterization: {tname}")
        
    return mask, bbox

def _compute_optimal_color_gpu_tensor(
    region_tgt: torch.Tensor,
    region_cur: torch.Tensor,
    mask_local: torch.Tensor,
    alpha: int,
) -> torch.Tensor:
    if mask_local.numel() == 0: 
        return torch.zeros(3, dtype=torch.float32, device=region_tgt.device)
    
    m = mask_local.float()
    weight = m.sum()
    
    a = alpha / 255.0
    if weight < 0.5 or a < 1e-6:
        return torch.zeros(3, dtype=torch.float32, device=region_tgt.device)
        
    src = (region_tgt - (1.0 - a) * region_cur) / a
    src_masked = src * m.unsqueeze(-1)
    avg = src_masked.reshape(-1, 3).sum(dim=0) / weight
    return torch.clamp(avg, 0, 255)

def score_shape_gpu_tensor(
    shape: Shape,
    current: torch.Tensor,
    target: torch.Tensor,
    alpha_mask: torch.Tensor | None = None,
    current_diff_sq: float = None,
    n_px: float = None,
    sticker_overlap_min: float = 0.995,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global _INF_TENSOR, _FALSE_TENSOR
    if _INF_TENSOR is None:
        _INF_TENSOR = torch.tensor(float('inf'), device=current.device)
    if _FALSE_TENSOR is None:
        _FALSE_TENSOR = torch.tensor(False, device=current.device)

    h, w = current.shape[:2]
    mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
    x0, y0, x1, y1 = bbox
    
    if x1 <= x0 or y1 <= y0: 
        shape_color = getattr(shape, 'color', (0, 0, 0, 128))
        color_fallback = torch.tensor([*shape_color[:3], shape_color[3]], dtype=torch.float32, device=current.device)
        return _INF_TENSOR, color_fallback
        
    region_cur = current[y0:y1, x0:x1].float()
    region_tgt = target[y0:y1, x0:x1].float()
    
    true_area = get_true_area_gpu(shape)
    shape_body = mask_local
    effective_mask = mask_local
    canvas_area_t = shape_body.sum().float()
    
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        opaque_body = region_alpha >= 128
        inside = (shape_body & opaque_body).sum().float()
        # Normalizes strictly against actual discrete rasterized area, bypassing grid pixel discretization errors
        invalid_mask = (~opaque_body.any()) | ((inside / torch.clamp(canvas_area_t, min=1.0)) < sticker_overlap_min)
        effective_mask = mask_local & opaque_body
    else:
        invalid_mask = _FALSE_TENSOR

    color_rgb = _compute_optimal_color_gpu_tensor(region_tgt, region_cur, effective_mask, shape.color[3])
    
    a = shape.color[3] / 255.0
    m = mask_local.float().unsqueeze(-1)
    blended = region_cur + m * (a * (color_rgb - region_cur))
    
    diff_in = blended - region_tgt
    diff_old = region_cur - region_tgt
    delta_sq = (diff_in ** 2) - (diff_old ** 2)
    
    bleed_pixels_t = torch.clamp(true_area - canvas_area_t - (true_area * 0.05), min=0.0)
    bleed_penalty = bleed_pixels_t * (255.0 ** 2)

    if alpha_mask is None:
        region_delta_sq = delta_sq.sum()
        total_sq = current_diff_sq + region_delta_sq + bleed_penalty
        rms = torch.sqrt(torch.clamp(total_sq, min=0.0) / n_px)
    else:
        weight_region = (alpha_mask[y0:y1, x0:x1] > 0).float().unsqueeze(-1)
        region_delta_sq = (delta_sq * weight_region).sum()
        total_sq = current_diff_sq + region_delta_sq + bleed_penalty
        
        if n_px < 1.0:
            rms = torch.zeros(1, dtype=torch.float32, device=current.device)
        else:
            rms = torch.sqrt(torch.clamp(total_sq, min=0.0) / n_px)
            
    color_t = torch.cat([color_rgb, torch.tensor([shape.color[3]], dtype=torch.float32, device=current.device)])
    rms = torch.where(invalid_mask, _INF_TENSOR, rms)
    return rms, color_t

def compute_diff_sq_sum_gpu(
    current: torch.Tensor, target: torch.Tensor, alpha_mask: torch.Tensor | None = None
) -> Tuple[float, float]:
    diff = current.float() - target.float()
    sq = diff * diff
    if alpha_mask is None:
        return sq.sum().item(), float(current.numel())
    weight = (alpha_mask > 0).float().unsqueeze(-1)
    return (sq * weight).sum().item(), float(weight.sum().item() * 3)

def score_shape_gpu_batch(
    shapes: list[Shape],
    current: torch.Tensor,
    target: torch.Tensor,
    alpha_mask: torch.Tensor | None = None,
    current_diff_sq: float = None,
    n_px: float = None,
) -> Tuple[list[float], list[Tuple[int, int, int, int]]]:
    global _INF_TENSOR, _FALSE_TENSOR
    if _INF_TENSOR is None:
        _INF_TENSOR = torch.tensor(float('inf'), device=current.device)
    if _FALSE_TENSOR is None:
        _FALSE_TENSOR = torch.tensor(False, device=current.device)
        
    scores_tensors = []
    colors_tensors = []
    if current_diff_sq is None or n_px is None:
        c_sq, c_n = compute_diff_sq_sum_gpu(current, target, alpha_mask)
        current_diff_sq = current_diff_sq if current_diff_sq is not None else c_sq
        n_px = n_px if n_px is not None else c_n
        
    for shape in shapes:
        rms_t, color_t = score_shape_gpu_tensor(shape, current, target, alpha_mask, current_diff_sq, n_px)
        scores_tensors.append(rms_t)
        colors_tensors.append(color_t)
        
    scores_stacked = torch.stack(scores_tensors)
    colors_stacked = torch.stack(colors_tensors)
    scores_list = scores_stacked.cpu().tolist()
    colors_list = colors_stacked.cpu().tolist()
    colors_out = [(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in colors_list]
    return scores_list, colors_out

def score_shape_gpu(
    shape: Shape,
    current: torch.Tensor,
    target: torch.Tensor,
    alpha_mask: torch.Tensor | None = None,
    current_diff_sq: float = None,
    n_px: float = None,
) -> Tuple[float, Tuple[int, int, int, int]]:
    global _INF_TENSOR, _FALSE_TENSOR
    if _INF_TENSOR is None:
        _INF_TENSOR = torch.tensor(float('inf'), device=current.device)
    if _FALSE_TENSOR is None:
        _FALSE_TENSOR = torch.tensor(False, device=current.device)

    if current_diff_sq is None or n_px is None:
        c_sq, c_n = compute_diff_sq_sum_gpu(current, target, alpha_mask)
        current_diff_sq = current_diff_sq if current_diff_sq is not None else c_sq
        n_px = n_px if n_px is not None else c_n
    rms_t, color_t = score_shape_gpu_tensor(shape, current, target, alpha_mask, current_diff_sq, n_px)
    return rms_t.item(), (int(color_t[0].item()), int(color_t[1].item()), int(color_t[2].item()), int(color_t[3].item()))

def composite_gpu(
    current: torch.Tensor,
    shape: Shape,
    target: torch.Tensor,
    alpha_mask: torch.Tensor | None = None,
    recalculate_color: bool = True,  # Cleanly prevents color-reoptimization drift during seeding
) -> Tuple[torch.Tensor, float]:
    h, w = current.shape[:2]
    mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.numel() == 0: 
        return current, _rms_error_gpu(current, target, alpha_mask)
        
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1].float() / 255.0
        effective_mask = mask_local * region_alpha
    else:
        effective_mask = mask_local
        
    region_cur = current[y0:y1, x0:x1].float()
    region_tgt = target[y0:y1, x0:x1].float()
    
    if recalculate_color:
        color_rgb = _compute_optimal_color_gpu_tensor(region_tgt, region_cur, effective_mask, shape.color[3])
        color = (int(color_rgb[0].item()), int(color_rgb[1].item()), int(color_rgb[2].item()), shape.color[3])
        shape.color = color
    else:
        color = shape.color
        color_rgb = torch.tensor(color[:3], dtype=torch.float32, device=current.device)
    
    a = color[3] / 255.0
    m = effective_mask.float().unsqueeze(-1)  # FORCED FLOAT CASTING PREVENTS SEED-STAGE CANVAS CORRUPTION
    blended = region_cur + m * (a * (color_rgb - region_cur))
    
    new = current.clone()
    new[y0:y1, x0:x1] = torch.clamp(blended, 0, 255).to(torch.uint8)
    return new, _rms_error_gpu(new, target, alpha_mask)

def _rms_error_gpu(
    a: torch.Tensor, b: torch.Tensor, alpha_mask: torch.Tensor | None = None
) -> float:
    diff = a.to(torch.int32) - b.to(torch.int32)
    sq = diff * diff
    if alpha_mask is None: return float(torch.sqrt(sq.float().mean()).item())
    weight = (alpha_mask > 0).float().unsqueeze(-1)
    total = float((sq.float() * weight).sum().item())
    n = float(weight.sum().item() * 3)
    if n < 1: return 0.0
    return float(torch.sqrt(torch.tensor(total / n)).item())