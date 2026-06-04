from __future__ import annotations
import torch
import math
from typing import Tuple
from fd6.shapegen.shapes import Shape

def is_shape_out_of_bounds(shape: Shape, w: int, h: int) -> bool:
    """Enforce strict sub-pixel boundary safety (0.1px tolerance)."""
    if hasattr(shape, 'rx') and hasattr(shape, 'ry'):
        a, b = shape.rx, shape.ry
        if hasattr(shape, 'angle'):
            rad = math.radians(shape.angle)
            cos_t, sin_t = math.cos(rad), math.sin(rad)
            x_ext = math.sqrt(a**2 * cos_t**2 + b**2 * sin_t**2)
            y_ext = math.sqrt(a**2 * sin_t**2 + b**2 * cos_t**2)
        else: x_ext, y_ext = a, b
        if (shape.x - x_ext < -0.1 or shape.x + x_ext > w + 0.1 or 
            shape.y - y_ext < -0.1 or shape.y + y_ext > h + 0.1): return True
    elif hasattr(shape, 'r'):
        r = shape.r
        if (shape.x - r < -0.1 or shape.x + r > w + 0.1 or 
            shape.y - r < -0.1 or shape.y + r > h + 0.1): return True
    return False

def compensated_ellipse_size(w: float, h: float) -> tuple[float, float]:
    major = max(w, h)
    minor = max(1.0, min(w, h))
    aspect = major / minor

    uniform_scale = 1.0
    if major >= 220:
        uniform_scale *= 0.985
    if major >= 300:
        uniform_scale *= 0.975

    major_axis_scale = 1.0
    if aspect >= 2.0:
        major_axis_scale *= 0.985
    if aspect >= 3.5:
        major_axis_scale *= 0.970
    if aspect >= 6.0:
        major_axis_scale *= 0.955

    if w >= h:
        sx = uniform_scale * major_axis_scale
        sy = uniform_scale
    else:
        sx = uniform_scale
        sy = uniform_scale * major_axis_scale

    return max(1.0, w * sx), max(1.0, h * sy)

def _rasterize_mask_gpu(shape: Shape, w: int, h: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    bbox = shape.bbox(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0: return torch.zeros((0, 0), device='cuda'), bbox
    ys = torch.arange(y0, y1, device='cuda') - shape.y
    xs = torch.arange(x0, x1, device='cuda') - shape.x
    yg, xg = ys.unsqueeze(1), xs.unsqueeze(0)
    
    tname = shape.type_name
    
    # 1. Triangle & RightTriangle Math
    if "triangle" in tname:
        verts = shape._get_vertices()
        v1_x, v1_y = verts[0][0] - shape.x, verts[0][1] - shape.y
        v2_x, v2_y = verts[1][0] - shape.x, verts[1][1] - shape.y
        v3_x, v3_y = verts[2][0] - shape.x, verts[2][1] - shape.y
        
        d1 = (v2_x - v1_x) * (yg - v1_y) - (v2_y - v1_y) * (xg - v1_x)
        d2 = (v3_x - v2_x) * (yg - v2_y) - (v3_y - v2_y) * (xg - v2_x)
        d3 = (v1_x - v3_x) * (yg - v3_y) - (v1_y - v3_y) * (xg - v3_x)
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        mask = ~(has_neg & has_pos)
    else:
        if hasattr(shape, 'angle'):
            rad = math.radians(shape.angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            xr, yr = cos_a * xg + sin_a * yg, -sin_a * xg + cos_a * yg
        else:
            xr, yr = xg, yg
            
        # 2. Rectangle & RoundedRectangle Math
        if "rectangle" in tname:
            rx = getattr(shape, 'rx', 1.0)
            ry = getattr(shape, 'ry', 1.0)
            if tname == "rounded_rectangle":
                cr = min(rx, ry) * 0.25
                dx = torch.clamp(torch.abs(xr) - (rx - cr), min=0.0)
                dy = torch.clamp(torch.abs(yr) - (ry - cr), min=0.0)
                mask = (torch.abs(xr) <= rx) & (torch.abs(yr) <= ry) & (~((dx > 0) & (dy > 0) & ((dx ** 2 + dy ** 2) > cr ** 2)))
            else:
                mask = (torch.abs(xr) <= rx) & (torch.abs(yr) <= ry)
                
        # 3. HalfEllipse / Semicircle Math
        elif tname == "half_ellipse":
            rx = getattr(shape, 'rx', 1.0)
            ry = getattr(shape, 'ry', 1.0)
            dx, dy = xr / max(rx, 1e-6), yr / max(ry, 1e-6)
            mask = ((dx ** 2 + dy ** 2) <= 1.0) & (yr >= 0.0)
            
        # 4. Standard Ellipse and Circle Math
        else:
            rx = getattr(shape, 'rx', getattr(shape, 'r', 1.0))
            ry = getattr(shape, 'ry', getattr(shape, 'r', 1.0))
            
            # Apply ellipse size correction for consistent GPU/game evaluation
            rx, ry = compensated_ellipse_size(rx, ry)
            
            dx, dy = xr / max(rx, 1e-6), yr / max(ry, 1e-6)
            mask = (dx ** 2 + dy ** 2) <= 1.0
            
    return (mask.to(torch.uint8) * 255), bbox

def compute_diff_sq_sum_gpu(canvas: torch.Tensor, target: torch.Tensor, alpha_mask: torch.Tensor | None) -> Tuple[float, float]:
    c_pm, c_a = canvas[..., :3].float(), canvas[..., 3].float()
    t_pm = target.float()
    
    if alpha_mask is not None:
        t_a = alpha_mask.float()
        t_a = torch.where(t_a < 15.0, 0.0, t_a)
    else:
        t_a = torch.full_like(c_a, 255.0)
        
    diff_rgb_sq, diff_a_sq = (c_pm - t_pm)**2, (c_a - t_a)**2
    return (torch.sum(diff_rgb_sq) + torch.sum(diff_a_sq)).item(), float(canvas.shape[0] * canvas.shape[1])

def _rms_error_gpu(canvas: torch.Tensor, target: torch.Tensor, alpha_mask: torch.Tensor | None) -> float:
    diff_sq, n_px = compute_diff_sq_sum_gpu(canvas, target, alpha_mask)
    return math.sqrt(max(0.0, diff_sq) / max(n_px * 4.0, 1e-6))

def estimate_optimal_color_gpu(mask_local: torch.Tensor, bbox: Tuple[int, int, int, int], canvas: torch.Tensor, target: torch.Tensor, shape_alpha: float) -> Tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    canvas_region, target_region = canvas[y0:y1, x0:x1, :3].float(), target[y0:y1, x0:x1, :3].float()
    mask_f = (mask_local > 0).float().unsqueeze(-1)
    sum_t_pm, sum_c_pm, n_pixels = torch.sum(target_region * mask_f, dim=(0, 1)), torch.sum(canvas_region * mask_f, dim=(0, 1)), torch.sum(mask_f, dim=(0, 1))
    if n_pixels.item() == 0: return (128, 128, 128)
    a_s = shape_alpha / 255.0
    optimal_rgb = torch.clamp((sum_t_pm - (1.0 - a_s) * sum_c_pm) / max(a_s * n_pixels, 1e-6), 0.0, 255.0)
    return (int(optimal_rgb[0].item()), int(optimal_rgb[1].item()), int(optimal_rgb[2].item()))

def composite_gpu(canvas: torch.Tensor, shape: Shape, target: torch.Tensor, alpha_mask: torch.Tensor | None, recalculate_color: bool = True) -> Tuple[torch.Tensor, float]:
    w, h = canvas.shape[1], canvas.shape[0]
    mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
    x0, y0, x1, y1 = bbox
    if mask_local.numel() == 0: return canvas.clone(), _rms_error_gpu(canvas, target, alpha_mask)
    shape_alpha = float(shape.color[3]) if len(shape.color) >= 4 else 255.0
    if recalculate_color:
        rgb = estimate_optimal_color_gpu(mask_local, bbox, canvas, target, shape_alpha)
        shape.color = (*rgb, int(shape_alpha))
    else: rgb = shape.color[:3]
    shape_rgb = torch.tensor(rgb, dtype=torch.float32, device='cuda')
    canvas_new = canvas.clone()
    canvas_region = canvas_new[y0:y1, x0:x1].float()
    mask_f = (mask_local > 0).float().unsqueeze(-1)
    a_src = (shape_alpha / 255.0) * mask_f
    canvas_region[..., :3] = shape_rgb * a_src + canvas_region[..., :3] * (1.0 - a_src)
    canvas_region[..., 3:4] = 255.0 * a_src + canvas_region[..., 3:4] * (1.0 - a_src)
    canvas_new[y0:y1, x0:x1] = torch.clamp(canvas_region, 0.0, 255.0).to(torch.uint8)
    return canvas_new, _rms_error_gpu(canvas_new, target, alpha_mask)

def score_shape_gpu(shape: Shape, canvas: torch.Tensor, target: torch.Tensor, alpha_mask: torch.Tensor | None, current_diff_sq: float, n_px: float) -> Tuple[float, Tuple[int, int, int, int]]:
    w, h = canvas.shape[1], canvas.shape[0]
    if is_shape_out_of_bounds(shape, w, h): return float('inf'), (128, 128, 128, 255)
    mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
    x0, y0, x1, y1 = bbox
    if mask_local.numel() == 0: return float('inf'), (128, 128, 128, 255)
    
    if alpha_mask is not None:
        target_alpha_region = alpha_mask[y0:y1, x0:x1].float()
        if ((mask_local > 0) & (target_alpha_region < 15.0)).any(): 
            return float('inf'), (128, 128, 128, 255)
        target_alpha_region = target_alpha_region.unsqueeze(-1)
    else: 
        target_alpha_region = torch.full((y1-y0, x1-x0, 1), 255.0, device='cuda')
        
    shape_alpha = float(shape.color[3]) if len(shape.color) >= 4 else 255.0
    rgb = estimate_optimal_color_gpu(mask_local, bbox, canvas, target, shape_alpha)
    shape_rgb = torch.tensor(rgb, dtype=torch.float32, device='cuda')
    canvas_region, target_region = canvas[y0:y1, x0:x1].float(), target[y0:y1, x0:x1, :3].float()
    old_local_diff_sq = torch.sum((canvas_region[..., :3]-target_region)**2) + torch.sum((canvas_region[..., 3:4]-target_alpha_region)**2)
    mask_f = (mask_local > 0).float().unsqueeze(-1)
    a_src = (shape_alpha / 255.0) * mask_f
    new_local_diff_sq = torch.sum((shape_rgb * a_src + canvas_region[..., :3] * (1.0 - a_src) - target_region)**2) + torch.sum((255.0 * a_src + canvas_region[..., 3:4] * (1.0 - a_src) - target_alpha_region)**2)
    return math.sqrt(max(0.0, current_diff_sq - old_local_diff_sq.item() + new_local_diff_sq.item()) / max(n_px * 4.0, 1e-6)), (*rgb, int(shape_alpha))

def score_shape_gpu_batch(candidates, canvas, target, alpha_mask, current_diff_sq, n_px):
    scores, colors = [], []
    for s in candidates:
        rms, color = score_shape_gpu(s, canvas, target, alpha_mask, current_diff_sq, n_px)
        scores.append(rms); colors.append(color)
    return scores, colors