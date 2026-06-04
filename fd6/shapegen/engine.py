# --- START OF FILE engine.py ---

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import random
import time
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from fd6.shapegen.profile import Profile
from fd6.shapegen.shapes import Shape, random_shape


@dataclass
class EngineConfig:
    profile: Profile
    seed: int = 0


@dataclass
class EngineEvent:
    kind: str
    shape_count: int = 0
    rms: float = 0.0
    canvas: np.ndarray | None = None
    message: str = ""


class Engine:
    def _build_attention_map(self, target_rgb: np.ndarray, alpha_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import cv2
        except ImportError:
            return (np.ones((target_rgb.shape[0], target_rgb.shape[1]), dtype=np.float32)[..., None], 
                    np.zeros((target_rgb.shape[0], target_rgb.shape[1]), dtype=np.float32),
                    np.zeros((target_rgb.shape[0], target_rgb.shape[1]), dtype=np.float32))

        target_f32 = target_rgb.astype(np.float32)

        # 1. CLAHE for Local Contrast
        lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 2. Sobel Gradients
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Smooth gradients to prevent micro-aliasing angle noise on thin lines
        gy_smooth = cv2.GaussianBlur(gy, (5, 5), 0)
        gx_smooth = cv2.GaussianBlur(gx, (5, 5), 0)
        angle_map = (np.degrees(np.arctan2(gy_smooth, gx_smooth)) + 90.0) % 180.0

        grad_mass = cv2.GaussianBlur(grad_mag, (5, 5), 0)
        if grad_mass.max() > 0: grad_mass /= grad_mass.max()

        # 3. Laplacian
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_blur = cv2.GaussianBlur(laplacian, (3, 3), 0)
        if lap_blur.max() > 0: lap_blur /= lap_blur.max()

        # 4. Global Saliency
        blur_heavy = cv2.GaussianBlur(target_f32, (15, 15), 0)
        saliency = np.sum(np.abs(target_f32 - blur_heavy), axis=2)
        if saliency.max() > 0: saliency /= saliency.max()

        attention = 0.01 + (saliency * 0.2) + (grad_mass * 0.4) + (lap_blur * 0.5)

        # --- HIGH-PRECISION HOLE-AWARE SEGMENTATION LOGIC ---
        edges = (grad_mass > 0.04) | (lap_blur > 0.04)
        edges_u8 = (edges * 255).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_edges = cv2.morphologyEx(edges_u8, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        figure_mask = np.zeros_like(closed_edges)
        h, w = target_rgb.shape[:2]
        min_area_parent = (h * w) * 0.0005
        min_area_hole = (h * w) * 0.0002
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, contour in enumerate(contours):
                if hierarchy[idx][3] == -1:
                    if cv2.contourArea(contour) > min_area_parent:
                        cv2.drawContours(figure_mask, [contour], -1, 255, -1)
                        
                        child_idx = hierarchy[idx][2]
                        while child_idx != -1:
                            child_contour = contours[child_idx]
                            if cv2.contourArea(child_contour) > min_area_hole:
                                cv2.drawContours(figure_mask, [child_contour], -1, 0, -1)
                            child_idx = hierarchy[child_idx][0]
                
        figure_mask_smoothed = cv2.GaussianBlur(figure_mask.astype(np.float32), (9, 9), 0) / 255.0
        
        attention += figure_mask_smoothed * 0.04
        
        bg_attenuation = 0.65 + 0.35 * figure_mask_smoothed
        attention *= bg_attenuation
        
        attention = np.clip(attention, 0.005, 1.0)

        if alpha_mask is not None:
            interior = (alpha_mask > 128).astype(float)
            attention = attention * (0.1 + interior * 0.9)
            attention[alpha_mask < 15] = 0.0

        return attention.astype(np.float32)[..., None], angle_map, lap_blur

    def __init__(self, target_rgb: np.ndarray, config: EngineConfig, alpha_mask: np.ndarray | None = None) -> None:
        if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
            raise ValueError("target_rgb must be HxWx3 RGB uint8")
        
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FD6+ GPU rendering. No compatible GPU found.")

        self.config = config
        self.profile = config.profile
        self.h, self.w = target_rgb.shape[:2]
        
        if alpha_mask is None:
            alpha_mask = np.full((self.h, self.w), 255, dtype=np.uint8)

        self.attention_map_cpu, self.angle_map_cpu, self.lap_blur_map_cpu = self._build_attention_map(target_rgb, alpha_mask)

        self.alpha_mask_t = torch.from_numpy(alpha_mask.copy()).cuda()
        target_a = self.alpha_mask_t.float().unsqueeze(-1) / 255.0
        self.target = (torch.from_numpy(target_rgb.copy()).cuda().float() * target_a).to(torch.uint8)
        
        self.canvas = torch.zeros((self.h, self.w, 4), dtype=torch.uint8, device='cuda')

        self.shapes: list[Shape] = []
        
        from fd6.shapegen.gpu_scoring import _rms_error_gpu
        self.start_rms = max(_rms_error_gpu(self.canvas, self.target, self.alpha_mask_t), 1e-6)
        self.rms = self.start_rms
        
        self._stop = False
        self._pause = False
        self.rng = random.Random(config.seed or int(time.time() * 1000) & 0xFFFFFFFF)

    def _get_progressive_alpha_bounds(self, rms_ratio: float, local_priority: float) -> tuple[int, int]:
        min_alpha_start, min_alpha_end = 180, 30
        max_alpha_start, max_alpha_end = 255, 120
        
        r_factor = rms_ratio ** 1.2
        
        min_alpha_flat = min_alpha_end + (min_alpha_start - min_alpha_end) * r_factor
        max_alpha_flat = max_alpha_end + (max_alpha_start - max_alpha_end) * r_factor
        
        min_alpha_detail = 50 + 130 * r_factor
        max_alpha_detail = 255
        
        min_alpha = int(min_alpha_flat * (1.0 - local_priority) + min_alpha_detail * local_priority)
        max_alpha = int(max_alpha_flat * (1.0 - local_priority) + max_alpha_detail * local_priority)
        
        min_alpha = max(10, min(255, min_alpha))
        max_alpha = max(min_alpha + 10, min(255, max_alpha))
        
        return min_alpha, max_alpha

    def request_stop(self) -> None: 
        self._stop = True
        
    def set_pause(self, paused: bool) -> None: 
        self._pause = paused

    def seed_shapes(self, shapes: list[Shape]) -> None:
        from fd6.shapegen.gpu_scoring import composite_gpu
        for s in shapes:
            self.canvas, self.rms = composite_gpu(
                self.canvas, s, self.target, self.alpha_mask_t, recalculate_color=False
            )
            self.shapes.append(s)

    def _generate_candidate(self, types: list[str]) -> Shape:
        shape = random_shape(self.rng, self.w, self.h, types)
        rms_ratio = max(0.01, min(1.0, self.rms / self.start_rms))

        uniform_ratio = 0.2 + 0.5 * rms_ratio
        
        cx, cy = self.rng.randrange(self.w), self.rng.randrange(self.h)
        if self.rng.random() >= uniform_ratio:
            for _ in range(150):
                tx, ty = self.rng.randrange(self.w), self.rng.randrange(self.h)
                prob = self.attention_map_cpu[ty, tx, 0] / 3.0
                floor = (rms_ratio ** 2.0) * 0.5
                if self.rng.random() < (prob + floor):
                    cx, cy = tx, ty
                    break
                    
        shape.x, shape.y = float(cx), float(cy)

        iy = int(max(0, min(self.h - 1, cy)))
        ix = int(max(0, min(self.w - 1, cx)))
        local_priority = float(self.attention_map_cpu[iy, ix, 0])
        
        # Saliency edge guidance bias
        edge_weight = float(self.lap_blur_map_cpu[iy, ix])
        max_len = max(30.0, self.w * 0.80)  # Raised max length to 80% of width for lines

        if hasattr(shape, 'rx') and hasattr(shape, 'ry'):
            # Elongation selection bias on high-contrast lines
            if self.rng.random() < (0.60 + edge_weight * 0.30):
                shape.ry = float(self.rng.uniform(0.5, max(2.0, local_priority * 0.05)))
                shape.rx = float(self.rng.uniform(15.0, max_len))
                
                if hasattr(shape, 'angle'):
                    base_angle = float(self.angle_map_cpu[iy, ix])
                    # Reduce tangent jitter on verified sharp edges
                    jitter = 1.5 if edge_weight > 0.4 else 5.0
                    shape.angle = float((base_angle + self.rng.gauss(0, jitter)) % 180.0)
            else:
                scale_factor = self.rng.random() ** 2.0
                max_r_flat = self.w / 4.0
                max_r_detail = max(3.0, (self.w / 2.0) * (rms_ratio ** 1.5))
                local_max_r = max_r_flat * (1.0 - local_priority) + max_r_detail * local_priority
                shape.rx = float(1.0 + (local_max_r * scale_factor))
                shape.ry = float(1.0 + (local_max_r * scale_factor))
                
        elif hasattr(shape, 'r'):
            global_max_r = max(2.5, (self.w / 2.0) * (rms_ratio ** 1.5))
            size_multiplier = 1.0 - (0.85 * local_priority)
            local_max_r = max(2.5, global_max_r * size_multiplier)
            
            scale_factor = self.rng.random() ** 2.0
            shape.r = float(1.0 + (local_max_r * scale_factor))
                
        min_alpha, max_alpha = self._get_progressive_alpha_bounds(rms_ratio, local_priority)
        shape.color = (shape.color[0], shape.color[1], shape.color[2], self.rng.randint(min_alpha, max_alpha))
        return shape

    def _best_of_random(self, types: list[str], n: int, current_diff_sq: float, n_px: float) -> Shape:
        candidates = [self._generate_candidate(types) for _ in range(n)]
        
        from fd6.shapegen.gpu_scoring import score_shape_gpu_batch
        scores, colors = score_shape_gpu_batch(
            candidates, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px
        )
        
        best_idx = min(range(len(scores)), key=lambda i: scores[i])
        candidates[best_idx].color = colors[best_idx]
        return candidates[best_idx]

    def _hill_climb(self, shape: Shape, iterations: int, current_diff_sq: float, n_px: float) -> Shape:
        best = shape
        from fd6.shapegen.gpu_scoring import score_shape_gpu, score_shape_gpu_batch
        best_rms, best_color = score_shape_gpu(best, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
        best.color = best_color
        
        no_improve = 0
        batch_size = min(iterations, 32)
        rms_ratio = max(0.01, min(1.0, self.rms / self.start_rms))
        global_max_r = max(3.0, (self.w / 2.0) * (rms_ratio ** 1.5))
        
        cy = int(max(0, min(self.h - 1, best.y)))
        cx = int(max(0, min(self.w - 1, best.x)))
        local_priority = float(self.attention_map_cpu[cy, cx, 0])
        size_multiplier = 1.0 - (0.85 * local_priority)
        local_max_r = max(3.0, global_max_r * size_multiplier)
        
        is_elongated = getattr(best, 'rx', 0.0) > getattr(best, 'ry', 0.0) * 2.0
        
        # Lifted max boundary caps to allow long straight lines
        if is_elongated:
            max_rx_allowed = min(self.w * 0.85, getattr(best, 'rx', local_max_r) * 1.8 + 5.0)
        else:
            max_rx_allowed = min(local_max_r, getattr(best, 'rx', local_max_r) * 1.5 + 2.0)
            
        max_ry_allowed = min(local_max_r, getattr(best, 'ry', local_max_r) * 1.5 + 2.0)
        
        while iterations > 0 and not self._stop:
            mutations = []
            for _ in range(batch_size):
                from copy import copy as shallow_copy
                m = shallow_copy(best)
                r_val = self.rng.random()
                if r_val < 0.35: step_scale = 1.0
                elif r_val < 0.70: step_scale = 0.3
                else: step_scale = 0.05
                
                which = self.rng.randint(0, 2)
                if which == 0:
                    m.x = max(0.0, min(self.w - 1.0, m.x + self.rng.gauss(0, 16.0 * step_scale)))
                    m.y = max(0.0, min(self.h - 1.0, m.y + self.rng.gauss(0, 16.0 * step_scale)))
                elif which == 1:
                    if hasattr(m, 'rx'): m.rx = max(1.0, min(self.w, m.rx + self.rng.gauss(0, 16.0 * step_scale)))
                    if hasattr(m, 'ry'): m.ry = max(1.0, min(self.h, m.ry + self.rng.gauss(0, 16.0 * step_scale)))
                    if hasattr(m, 'r'): m.r = max(1.0, min(self.w, m.r + self.rng.gauss(0, 16.0 * step_scale)))
                else:
                    if hasattr(m, 'angle'): m.angle = (m.angle + self.rng.gauss(0, 25.0 * step_scale)) % 180.0
                
                m_cy = int(max(0, min(self.h - 1, m.y)))
                m_cx = int(max(0, min(self.w - 1, m.x)))
                m_local_priority = float(self.attention_map_cpu[m_cy, m_cx, 0])
                
                m_min_alpha, m_max_alpha = self._get_progressive_alpha_bounds(rms_ratio, m_local_priority)
                
                alpha_mutation = int(m.color[3] + self.rng.gauss(0, 15 * step_scale))
                alpha_clamped = max(m_min_alpha, min(m_max_alpha, alpha_mutation))
                m.color = (int(m.color[0]), int(m.color[1]), int(m.color[2]), alpha_clamped)

                if hasattr(m, 'rx'): m.rx = float(min(m.rx, max_rx_allowed))
                if hasattr(m, 'ry'): m.ry = float(min(m.ry, max_ry_allowed))
                if hasattr(m, 'r'): m.r = float(min(m.r, max_rx_allowed))
                
                if hasattr(m, 'rx') and hasattr(m, 'ry') and hasattr(m, 'angle'):
                    if m.rx > m.ry * 3.0: 
                        base_angle = float(self.angle_map_cpu[m_cy, m_cx])
                        diff = (m.angle - base_angle + 90.0) % 180.0 - 90.0
                        clamped_diff = max(-10.0, min(10.0, diff)) 
                        m.angle = float((base_angle + clamped_diff) % 180.0)
                mutations.append(m)
                
            scores, colors = score_shape_gpu_batch(mutations, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
            if scores[best_idx] < best_rms:
                best = mutations[best_idx]
                best.color = colors[best_idx]
                best_rms = scores[best_idx]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, batch_size // 4): break
            iterations -= batch_size
        return best

    def _hill_climb_on_background(self, shape: Shape, bg_canvas: torch.Tensor, iterations: int, current_diff_sq: float, n_px: float) -> Shape:
        best = shape
        from fd6.shapegen.gpu_scoring import score_shape_gpu, score_shape_gpu_batch
        best_rms, best_color = score_shape_gpu(best, bg_canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
        best.color = best_color
        
        no_improve = 0
        batch_size = min(iterations, 32)
        rms_ratio = max(0.01, min(1.0, best_rms / self.start_rms))
        global_max_r = max(3.0, (self.w / 2.0) * (rms_ratio ** 1.5))
        
        cy = int(max(0, min(self.h - 1, best.y)))
        cx = int(max(0, min(self.w - 1, best.x)))
        local_priority = float(self.attention_map_cpu[cy, cx, 0])
        size_multiplier = 1.0 - (0.85 * local_priority)
        local_max_r = max(3.0, global_max_r * size_multiplier)
        
        is_elongated = getattr(best, 'rx', 0.0) > getattr(best, 'ry', 0.0) * 2.0
        
        if is_elongated:
            max_rx_allowed = min(self.w * 0.85, getattr(best, 'rx', local_max_r) * 1.8 + 5.0)
        else:
            max_rx_allowed = min(local_max_r, getattr(best, 'rx', local_max_r) * 1.5 + 2.0)
            
        max_ry_allowed = min(local_max_r, getattr(best, 'ry', local_max_r) * 1.5 + 2.0)
        
        while iterations > 0 and not self._stop:
            mutations = []
            for _ in range(batch_size):
                from copy import copy as shallow_copy
                m = shallow_copy(best)
                r_val = self.rng.random()
                if r_val < 0.20: step_scale = 0.5
                elif r_val < 0.60: step_scale = 0.15
                else: step_scale = 0.03
                
                which = self.rng.randint(0, 2)
                if which == 0:
                    m.x = max(0.0, min(self.w - 1.0, m.x + self.rng.gauss(0, 16.0 * step_scale)))
                    m.y = max(0.0, min(self.h - 1.0, m.y + self.rng.gauss(0, 16.0 * step_scale)))
                elif which == 1:
                    if hasattr(m, 'rx'): m.rx = max(1.0, min(self.w, m.rx + self.rng.gauss(0, 16.0 * step_scale)))
                    if hasattr(m, 'ry'): m.ry = max(1.0, min(self.h, m.ry + self.rng.gauss(0, 16.0 * step_scale)))
                    if hasattr(m, 'r'): m.r = max(1.0, min(self.w, m.r + self.rng.gauss(0, 16.0 * step_scale)))
                else:
                    if hasattr(m, 'angle'): m.angle = (m.angle + self.rng.gauss(0, 25.0 * step_scale)) % 180.0
                
                m_cy = int(max(0, min(self.h - 1, m.y)))
                m_cx = int(max(0, min(self.w - 1, m.x)))
                m_local_priority = float(self.attention_map_cpu[m_cy, m_cx, 0])
                
                m_min_alpha, m_max_alpha = self._get_progressive_alpha_bounds(rms_ratio, m_local_priority)
                
                alpha_mutation = int(m.color[3] + self.rng.gauss(0, 10 * step_scale))
                alpha_clamped = max(m_min_alpha, min(m_max_alpha, alpha_mutation))
                m.color = (int(m.color[0]), int(m.color[1]), int(m.color[2]), alpha_clamped)

                if hasattr(m, 'rx'): m.rx = float(min(m.rx, max_rx_allowed))
                if hasattr(m, 'ry'): m.ry = float(min(m.ry, max_ry_allowed))
                if hasattr(m, 'r'): m.r = float(min(m.r, max_rx_allowed))
                
                if hasattr(m, 'rx') and hasattr(m, 'ry') and hasattr(m, 'angle'):
                    if m.rx > m.ry * 3.0: 
                        base_angle = float(self.angle_map_cpu[m_cy, m_cx])
                        diff = (m.angle - base_angle + 90.0) % 180.0 - 90.0
                        clamped_diff = max(-10.0, min(10.0, diff)) 
                        m.angle = float((base_angle + clamped_diff) % 180.0)
                mutations.append(m)
            scores, colors = score_shape_gpu_batch(mutations, bg_canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
            if scores[best_idx] < best_rms:
                best = mutations[best_idx]
                best.color = colors[best_idx]
                best_rms = scores[best_idx]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, batch_size // 4): break
            iterations -= batch_size
        return best

    def wiggle_and_prune(self, start_ratio: float = 0.0, iterations_per_shape: int = None, progress_callback=None) -> int:
        if not self.shapes: return 0
        if iterations_per_shape is None: iterations_per_shape = self.profile.mutated_samples
        from fd6.shapegen.gpu_scoring import composite_gpu, _rms_error_gpu, compute_diff_sq_sum_gpu
        import torch

        start_idx = int(len(self.shapes) * start_ratio)
        preserved_shapes = self.shapes[:start_idx]
        canvas_bg = torch.zeros((self.h, self.w, 4), dtype=torch.uint8, device='cuda')

        for i in range(start_idx):
            canvas_bg, _ = composite_gpu(canvas_bg, self.shapes[i], self.target, self.alpha_mask_t, recalculate_color=False)

        retained_shapes = list(preserved_shapes)
        num_to_optimize = len(self.shapes) - start_idx
        for k in range(start_idx, len(self.shapes)):
            if getattr(self, "_stop_check", lambda: False)(): break
            current_diff_sq, n_px = compute_diff_sq_sum_gpu(canvas_bg, self.target, self.alpha_mask_t)
            rms_before = _rms_error_gpu(canvas_bg, self.target, self.alpha_mask_t)
            old_shape = self.shapes[k]
            optimized_shape = self._hill_climb_on_background(old_shape, canvas_bg, iterations_per_shape, current_diff_sq, n_px)
            temp_canvas, rms_after = composite_gpu(canvas_bg, optimized_shape, self.target, self.alpha_mask_t, recalculate_color=False)
            reduction = rms_before - rms_after
            if optimized_shape.color[3] < 10 or reduction <= 0.0: continue
            retained_shapes.append(optimized_shape)
            canvas_bg = temp_canvas
            if progress_callback is not None and (k - start_idx) % 10 == 0:
                progress_callback(k - start_idx + 1, num_to_optimize, rms_before, self._get_numpy_canvas_from_state(canvas_bg))
        self.shapes = retained_shapes
        self.canvas = canvas_bg
        self.rms = _rms_error_gpu(self.canvas, self.target, self.alpha_mask_t)
        return len(self.shapes)

    def optimize_and_reclaim(self, progress_callback=None) -> int:
        if not self.shapes: return 0
        from fd6.shapegen.gpu_scoring import composite_gpu, _rms_error_gpu
        import torch
        canvas_bg = torch.zeros((self.h, self.w, 4), dtype=torch.uint8, device='cuda')
        for k in range(len(self.shapes)):
            if getattr(self, "_stop_check", lambda: False)(): break
            rms_before = _rms_error_gpu(canvas_bg, self.target, self.alpha_mask_t)
            canvas_orig, rms_orig = composite_gpu(canvas_bg, self.shapes[k], self.target, self.alpha_mask_t, recalculate_color=False)
            shape_promoted = self.shapes[k].with_color((*self.shapes[k].color[:3], 255))
            canvas_promoted, rms_promoted = composite_gpu(canvas_bg, shape_promoted, self.target, self.alpha_mask_t, recalculate_color=False)
            if rms_promoted <= rms_orig:
                self.shapes[k] = shape_promoted
                canvas_bg = canvas_promoted
            else: canvas_bg = canvas_orig
            if progress_callback is not None and k % 50 == 0:
                progress_callback(k, len(self.shapes), rms_before, self._get_numpy_canvas_from_state(canvas_bg))

        retained_shapes = []
        remaining_transparency = torch.ones((self.h, self.w), dtype=torch.float32, device='cuda')
        for k in reversed(range(len(self.shapes))):
            shape = self.shapes[k]
            mask_local, bbox = self._rasterize_shape_mask_only(shape)
            x0, y0, x1, y1 = bbox
            if mask_local.numel() == 0: continue
            region_trans = remaining_transparency[y0:y1, x0:x1]
            max_trans = region_trans[mask_local].max().item() if mask_local.any() else 0.0
            if max_trans < 0.02: continue
            retained_shapes.append(shape)
            alpha_val = shape.color[3] / 255.0
            region_trans[mask_local] *= (1.0 - alpha_val)
        retained_shapes.reverse()
        self.shapes = retained_shapes
        self.canvas = torch.zeros((self.h, self.w, 4), dtype=torch.uint8, device='cuda')
        for s in self.shapes:
            self.canvas, self.rms = composite_gpu(self.canvas, s, self.target, self.alpha_mask_t, recalculate_color=False)
        return len(self.shapes)

    def _rasterize_shape_mask_only(self, shape: Shape) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        from fd6.shapegen.gpu_scoring import _rasterize_mask_gpu
        return _rasterize_mask_gpu(shape, self.w, self.h)

    def _get_numpy_canvas_from_state(self, state_canvas: torch.Tensor) -> np.ndarray:
        return state_canvas.cpu().numpy().copy()

    def run(self) -> Iterable[EngineEvent]:
        from fd6.shapegen.gpu_scoring import compute_diff_sq_sum_gpu, composite_gpu
        p = self.profile
        types = [t for t in p.shape_types if t]
        if not types: types = ["rotated_ellipse"]
        save_at = set(p.save_at)
        consecutive_skips = 0
        try:
            while len(self.shapes) < p.stop_at and not self._stop:
                while self._pause and not self._stop: time.sleep(0.05)
                current_diff_sq, n_px = compute_diff_sq_sum_gpu(self.canvas, self.target, self.alpha_mask_t)
                candidate = self._best_of_random(types, max(1, p.random_samples), current_diff_sq, n_px)
                refined = self._hill_climb(candidate, max(1, p.mutated_samples), current_diff_sq, n_px)
                from fd6.shapegen.gpu_scoring import score_shape_gpu
                rscore, _ = score_shape_gpu(refined, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
                if rscore == float("inf") or rscore >= self.rms:
                    consecutive_skips += 1
                    if consecutive_skips >= 250:
                        yield EngineEvent(kind="done", shape_count=len(self.shapes), rms=self.rms, canvas=self._get_numpy_canvas(), message="Converged.")
                        return
                    continue
                consecutive_skips = 0
                self.canvas, self.rms = composite_gpu(self.canvas, refined, self.target, self.alpha_mask_t)
                self.shapes.append(refined)
                count = len(self.shapes)
                yield EngineEvent(kind="shape_committed", shape_count=count, rms=self.rms)
                if p.preview_every and (count % p.preview_every == 0):
                    yield EngineEvent(kind="preview", shape_count=count, rms=self.rms, canvas=self._get_numpy_canvas())
                if count in save_at or (p.save_every and count % p.save_every == 0):
                    yield EngineEvent(kind="checkpoint", shape_count=count, rms=self.rms)
            yield EngineEvent(kind="done", shape_count=len(self.shapes), rms=self.rms, canvas=self._get_numpy_canvas())
        except Exception as exc: yield EngineEvent(kind="error", message=f"{type(exc).__name__}: {exc}")

    def _get_numpy_canvas(self) -> np.ndarray | None:
        if self.canvas is None: return None
        return self.canvas.cpu().numpy().copy()


class PostOptimizeWorker(QObject):
    progress = Signal(int, int, float)
    preview = Signal(object)
    finished = Signal(str)
    error = Signal(str)
    status = Signal(str, str)

    def __init__(self, json_path: Path, image_path: Path, profile: Profile, sticker_mode: bool = True):
        super().__init__()
        self.json_path = json_path
        self.image_path = image_path
        self.profile = profile
        self.sticker_mode = sticker_mode
        self._stop = False
        self._paused = False

    def set_pause(self, paused: bool): self._paused = paused
    def stop(self): self._stop = True

    def run(self):
        try:
            from fd6.io.exporter import load_json
            self.status.emit("Loading JSON data...", "info")
            doc = load_json(str(self.json_path))
            shapes = doc.materialize_shapes()
            
            self.status.emit("Loading source target image...", "info")
            from PIL import Image
            img = Image.open(self.image_path)
            
            if not self.sticker_mode and img.size[0] != img.size[1]:
                side = max(img.size)
                square = Image.new("RGB", (side, side), (255, 255, 255))
                offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
                square.paste(img, offset)
                img = square
            
            if doc.image_size and doc.image_size[0] > 0:
                img = img.resize(doc.image_size, Image.LANCZOS)
            else:
                mr = self.profile.max_resolution
                if max(img.size) > mr:
                    scale = mr / max(img.size)
                    new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
                    img = img.resize(new_size, Image.LANCZOS)

            target_rgb = np.array(img.convert('RGB'))
            alpha_mask = None
            if img.mode == "RGBA" and self.sticker_mode:
                rgba = np.array(img)
                alpha_mask = rgba[..., 3]
            
            self.status.emit("Initializing GPU engine...", "info")
            engine = Engine(target_rgb, EngineConfig(profile=self.profile), alpha_mask=alpha_mask)
            engine.seed_shapes(shapes)
            engine._stop_check = lambda: self._stop
            
            self.status.emit(f"Wiggling and relaxing all {len(shapes)} shapes...", "info")
            self.progress.emit(0, len(shapes), engine.rms)
            self.preview.emit(engine._get_numpy_canvas())
            
            def on_wiggle_progress(count, total, current_rms, canvas_arr):
                self.progress.emit(count, total, current_rms)
                self.preview.emit(canvas_arr)
                self.status.emit(f"Wiggling and relaxing shape {count}/{total}... RMS={current_rms:.2f}", "info")
            
            retained_count = engine.optimize_and_reclaim(progress_callback=on_wiggle_progress)
            self.status.emit(f"Wiggled shapes: reclaimed {len(shapes) - retained_count} redundant shapes. Regenerating slots...", "info")
            
            target_count = len(shapes)
            engine.profile.stop_at = target_count
            engine.profile.wiggle_enabled = False
            for event in engine.run():
                if self._stop: break
                if event.kind == "shape_committed":
                    self.progress.emit(event.shape_count, target_count, event.rms)
                    self.status.emit(f"Regenerating shape slot {event.shape_count}/{target_count}... RMS={event.rms:.2f}", "info")
                elif event.kind == "preview": self.preview.emit(event.canvas)

            if not self._stop:
                out_path = self.json_path.parent / f"{self.json_path.stem}_wiggled.json"
                doc.shapes = [s.to_json() for s in engine.shapes]
                from fd6.io.exporter import save_json
                save_json(doc, str(out_path))
                self.finished.emit(str(out_path))
            else: self.status.emit("Cancelled.", "warning")
        except Exception as exc:
            self.status.emit(f"Error: {exc}", "error")
            self.error.emit(str(exc))

# --- END OF FILE ---