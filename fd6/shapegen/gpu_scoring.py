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

def get_true_area_gpu(shape: Shape) -> float:
    tname = shape.type_name
    if tname == "circle": return math.pi * (max(getattr(shape, 'r', 1e-6), 1e-6)**2)
    elif tname in ("ellipse", "rotated_ellipse"): return math.pi * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname in ("rectangle", "rotated_rectangle"): return 4.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    elif tname == "triangle": return 2.0 * max(getattr(shape, 'rx', 1e-6), 1e-6) * max(getattr(shape, 'ry', 1e-6), 1e-6)
    return 0.0

def _rasterize_mask_gpu(shape: Shape, w: int, h: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    bbox = shape.bbox(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0: return torch.zeros((0, 0), device='cuda'), bbox
    ys = torch.arange(y0, y1, device='cuda') - shape.y
    xs = torch.arange(x0, x1, device='cuda') - shape.x
    yg, xg = ys.unsqueeze(1), xs.unsqueeze(0)
    
    tname = shape.type_name
    
    # 1. Triangle Math with smooth analytical SDF anti-aliasing
    if "triangle" in tname:
        verts = shape._get_vertices()
        v1_x, v1_y = verts[0][0] - shape.x, verts[0][1] - shape.y
        v2_x, v2_y = verts[1][0] - shape.x, verts[1][1] - shape.y
        v3_x, v3_y = verts[2][0] - shape.x, verts[2][1] - shape.y
        
        len1 = math.sqrt((v2_x - v1_x)**2 + (v2_y - v1_y)**2 + 1e-8)
        len2 = math.sqrt((v3_x - v2_x)**2 + (v3_y - v2_y)**2 + 1e-8)
        len3 = math.sqrt((v1_x - v3_x)**2 + (v1_y - v3_y)**2 + 1e-8)
        
        sd1 = ((v2_x - v1_x) * (yg - v1_y) - (v2_y - v1_y) * (xg - v1_x)) / len1
        sd2 = ((v3_x - v2_x) * (yg - v2_y) - (v3_y - v2_y) * (xg - v2_x)) / len2
        sd3 = ((v1_x - v3_x) * (yg - v3_y) - (v1_y - v3_y) * (xg - v3_x)) / len3
        
        cross = (v2_x - v1_x) * (v3_y - v1_y) - (v2_y - v1_y) * (v3_x - v1_x)
        is_ccw = cross > 0
        if is_ccw:
            sdf = torch.max(torch.max(-sd1, -sd2), -sd3)
        else:
            sdf = torch.max(torch.max(sd1, sd2), sd3)
        
        coverage = torch.clamp(0.5 - sdf, 0.0, 1.0)
        mask = coverage * 255.0
    else:
        if hasattr(shape, 'angle'):
            rad = math.radians(shape.angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            xr, yr = cos_a * xg + sin_a * yg, -sin_a * xg + cos_a * yg
        else:
            xr, yr = xg, yg
            
        # 2. Rectangle Math with smooth analytical SDF anti-aliasing
        if "rectangle" in tname:
            rx = getattr(shape, 'rx', 1.0)
            ry = getattr(shape, 'ry', 1.0)
            if tname == "rounded_rectangle":
                cr = min(rx, ry) * 0.25
                dx = torch.abs(xr) - (rx - cr)
                dy = torch.abs(yr) - (ry - cr)
                corner_dist = torch.sqrt(torch.clamp(dx, min=0.0)**2 + torch.clamp(dy, min=0.0)**2 + 1e-8) - cr
                edge_dist = torch.max(dx, dy)
                pixel_dist = torch.where((dx > 0) & (dy > 0), corner_dist, edge_dist)
                coverage = torch.clamp(0.5 - pixel_dist, 0.0, 1.0)
                mask = coverage * 255.0
            else:
                pixel_dist = torch.max(torch.abs(xr) - rx, torch.abs(yr) - ry)
                coverage = torch.clamp(0.5 - pixel_dist, 0.0, 1.0)
                mask = coverage * 255.0
                
        # 3. HalfEllipse Math with smooth boundaries
        elif tname == "half_ellipse":
            rx = getattr(shape, 'rx', 1.0)
            ry = getattr(shape, 'ry', 1.0)
            dx, dy = xr / max(rx, 1e-6), yr / max(ry, 1e-6)
            dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
            grad_x = xr / (max(rx, 1e-6) ** 2)
            grad_y = yr / (max(ry, 1e-6) ** 2)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) / dist
            pixel_dist = (dist - 1.0) / torch.clamp(grad_mag, min=1e-5)
            coverage = torch.clamp(0.5 - pixel_dist, 0.0, 1.0)
            mask = coverage * torch.clamp(yr + 0.5, 0.0, 1.0) * 255.0
            
        # 4. Standard Ellipse and Circle Math with smooth analytical anti-aliasing
        else:
            rx = getattr(shape, 'rx', getattr(shape, 'r', 1.0))
            ry = getattr(shape, 'ry', getattr(shape, 'r', 1.0))
            rx, ry = compensated_ellipse_size(rx, ry)
            
            dx, dy = xr / max(rx, 1e-6), yr / max(ry, 1e-6)
            dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
            grad_x = xr / (max(rx, 1e-6) ** 2)
            grad_y = yr / (max(ry, 1e-6) ** 2)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) / dist
            pixel_dist = (dist - 1.0) / torch.clamp(grad_mag, min=1e-5)
            coverage = torch.clamp(0.5 - pixel_dist, 0.0, 1.0)
            mask = coverage * 255.0
            
    return mask.to(torch.uint8), bbox

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

def estimate_optimal_color_gpu(
    mask_local: torch.Tensor,
    bbox: Tuple[int, int, int, int],
    canvas: torch.Tensor,
    target: torch.Tensor,
    shape_alpha: float,
    target_alpha_region: torch.Tensor | None = None
) -> Tuple[int, int, int]:
    """
    Compute optimal RGB, ignoring pixels where the target is transparent if target_alpha_region is given.
    """
    x0, y0, x1, y1 = bbox
    canvas_region = canvas[y0:y1, x0:x1, :3].float()
    target_region = target[y0:y1, x0:x1, :3].float()
    mask_f = (mask_local.float() / 255.0).unsqueeze(-1)
    
    if target_alpha_region is not None:
        t_alpha_f = target_alpha_region.float().unsqueeze(-1) / 255.0
        mask_f = mask_f * t_alpha_f
    
    sum_t_pm = torch.sum(target_region * mask_f, dim=(0, 1))
    sum_c_pm = torch.sum(canvas_region * mask_f, dim=(0, 1))
    n_pixels = torch.sum(mask_f, dim=(0, 1))
    
    a_s = shape_alpha / 255.0
    optimal_rgb = (sum_t_pm - (1.0 - a_s) * sum_c_pm) / torch.clamp(a_s * n_pixels, min=1e-6)
    
    # Non-synchronous branching: process fallback color value directly on GPU
    optimal_rgb = torch.where(n_pixels > 1e-5, optimal_rgb, torch.tensor(128.0, device=optimal_rgb.device))
    optimal_rgb = torch.clamp(optimal_rgb, 0.0, 255.0)
    
    return (int(optimal_rgb[0].item()), int(optimal_rgb[1].item()), int(optimal_rgb[2].item()))

def composite_gpu(canvas: torch.Tensor, shape: Shape, target: torch.Tensor, alpha_mask: torch.Tensor | None, recalculate_color: bool = True) -> Tuple[torch.Tensor, float]:
    w, h = canvas.shape[1], canvas.shape[0]
    mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
    x0, y0, x1, y1 = bbox
    if mask_local.numel() == 0: return canvas.clone(), _rms_error_gpu(canvas, target, alpha_mask)

    shape_alpha = float(shape.color[3]) if len(shape.color) >= 4 else 255.0
    if recalculate_color:
        # Pass target alpha region to avoid contamination from transparent pixels
        target_alpha_region = alpha_mask[y0:y1, x0:x1] if alpha_mask is not None else None
        rgb = estimate_optimal_color_gpu(mask_local, bbox, canvas, target, shape_alpha, target_alpha_region)
        shape.color = (*rgb, int(shape_alpha))
    else:
        rgb = shape.color[:3]
    
    shape_rgb = torch.as_tensor(rgb, dtype=torch.float32, device=canvas.device)
    canvas_new = canvas.clone()
    canvas_region = canvas_new[y0:y1, x0:x1].float()
    
    mask_f = (mask_local.float() / 255.0).unsqueeze(-1)
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
    
    # STRICT STICKER BOUNDARY CHECK – no leakage allowed
    if alpha_mask is not None:
        target_alpha_region = alpha_mask[y0:y1, x0:x1]
        # Reject if any pixel of the shape (soft mask > 0) touches transparent target (alpha < 15)
        if ((mask_local > 0) & (target_alpha_region < 15)).any():
            return float('inf'), (128, 128, 128, 255)
        weight_region = (target_alpha_region > 0).float().unsqueeze(-1)
    else:
        target_alpha_region = torch.full((y1-y0, x1-x0), 255, dtype=torch.uint8, device='cuda')
        weight_region = torch.ones((y1-y0, x1-x0, 1), dtype=torch.float32, device='cuda')
        
    shape_alpha = float(shape.color[3]) if len(shape.color) >= 4 else 255.0
    rgb = estimate_optimal_color_gpu(mask_local, bbox, canvas, target, shape_alpha, target_alpha_region)
    shape_rgb = torch.as_tensor(rgb, dtype=torch.float32, device=canvas.device)
    
    canvas_region, target_region = canvas[y0:y1, x0:x1].float(), target[y0:y1, x0:x1, :3].float()
    
    mask_f = (mask_local.float() / 255.0).unsqueeze(-1)
    a_src = (shape_alpha / 255.0) * mask_f
    
    blended_rgb = shape_rgb * a_src + canvas_region[..., :3] * (1.0 - a_src)
    blended_alpha = 255.0 * a_src + canvas_region[..., 3:4] * (1.0 - a_src)
    
    diff_in_rgb = blended_rgb - target_region
    diff_in_alpha = blended_alpha - target_alpha_region.float().unsqueeze(-1)
    
    diff_old_rgb = canvas_region[..., :3] - target_region
    diff_old_alpha = canvas_region[..., 3:4] - target_alpha_region.float().unsqueeze(-1)
    
    delta_sq_rgb = (diff_in_rgb ** 2) - (diff_old_rgb ** 2)
    delta_sq_alpha = (diff_in_alpha ** 2) - (diff_old_alpha ** 2)
    
    delta_sq_total = torch.sum(delta_sq_rgb, dim=-1, keepdim=True) + delta_sq_alpha
    
    # Fixed high threshold for severe degradation (original 400.0)
    severe_delta_threshold = 400.0
    
    weighted_delta = delta_sq_total * weight_region
    improvement = -torch.sum(torch.clamp(weighted_delta, max=0.0)).item()
    
    severe_degradation = torch.where(weighted_delta > severe_delta_threshold, weighted_delta, 0.0)
    degradation = torch.sum(severe_degradation).item()
    
    # Size-Adaptive Vandalism Firewall
    true_area = get_true_area_gpu(shape)
    shape_body = mask_local >= 128
    canvas_area = float(torch.sum(shape_body))
    
    if canvas_area < 250.0:
        max_allowed = max(5000.0, 0.40 * improvement)
    else:
        max_allowed = max(1000.0, 0.10 * improvement)
        
    if degradation > max_allowed:
        return float('inf'), (128, 128, 128, 255)
        
    region_delta_sq = torch.sum(weighted_delta)
    bleed_penalty = 0.0
    
    total_sq = current_diff_sq + region_delta_sq.item() + bleed_penalty
    return math.sqrt(max(0.0, total_sq) / max(n_px * 4.0, 1e-6)), (*rgb, int(shape_alpha))

def score_shape_gpu_batch(candidates, canvas, target, alpha_mask, current_diff_sq, n_px):
    scores, colors = [], []
    for s in candidates:
        rms, color = score_shape_gpu(s, canvas, target, alpha_mask, current_diff_sq, n_px)
        scores.append(rms); colors.append(color)
    return scores, colors