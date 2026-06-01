from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import random
import time
import math
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

        lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        angle_map = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0

        grad_mass = cv2.GaussianBlur(grad_mag, (5, 5), 0)
        if grad_mass.max() > 0: grad_mass /= grad_mass.max()

        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_blur = cv2.GaussianBlur(laplacian, (3, 3), 0)
        if lap_blur.max() > 0: lap_blur /= lap_blur.max()

        blur_heavy = cv2.GaussianBlur(target_f32, (15, 15), 0)
        saliency = np.sum(np.abs(target_f32 - blur_heavy), axis=2)
        if saliency.max() > 0: saliency /= saliency.max()

        attention = 0.01 + (saliency * 0.2) + (grad_mass * 0.4) + (lap_blur * 0.5)

        # HIGH-PRECISION HOLE-AWARE SEGMENTATION LOGIC
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
            attention[alpha_mask == 0] = 0.0

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
        
        self.attention_map_cpu, self.angle_map_cpu, self.lap_blur_map_cpu = self._build_attention_map(target_rgb, alpha_mask)

        self.target = torch.from_numpy(target_rgb.copy()).cuda()
        self.attention_map_t = torch.from_numpy(self.attention_map_cpu.copy()).cuda()
        
        if alpha_mask is not None:
            self.alpha_mask_t = torch.from_numpy(alpha_mask.copy()).cuda()
            self.canvas = torch.full((self.h, self.w, 3), 40, dtype=torch.uint8, device='cuda')
        else:
            self.alpha_mask_t = None
            avg = self.target.float().mean(dim=(0, 1)).to(torch.uint8)
            self.canvas = torch.tile(avg, (self.h, self.w, 1))

        self.shapes: list[Shape] = []
        
        from fd6.shapegen.gpu_scoring import _rms_error_gpu
        self.start_rms = max(_rms_error_gpu(self.canvas, self.target, self.alpha_mask_t), 1e-6)
        self.rms = self.start_rms
        
        self._stop = False
        self._pause = False
        self.rng = random.Random(config.seed or int(time.time() * 1000) & 0xFFFFFFFF)

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

        cx, cy = self.rng.randrange(self.w), self.rng.randrange(self.h)
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

        min_alpha = int(40 + 40 * local_priority)
        max_alpha = int(100 + 155 * local_priority)
        shape.color = (shape.color[0], shape.color[1], shape.color[2], self.rng.randint(min_alpha, max_alpha))

        if hasattr(shape, 'rx'):
            global_max_r = max(2.5, (self.w / 2.0) * (rms_ratio ** 1.5))
            size_multiplier = 1.0 - (0.85 * local_priority)
            local_max_r = max(2.5, global_max_r * size_multiplier)
            scale_factor = self.rng.random() ** 2.0
            shape.r = float(1.0 + (local_max_r * scale_factor))
            
            if hasattr(shape, 'ry'):
                shape.ry = float(1.0 + (local_max_r * scale_factor))

            if hasattr(shape, 'ry') and self.rng.random() > (rms_ratio + 0.15):
                shape.ry = float(self.rng.uniform(0.5, max(2.0, local_max_r * 0.1))) 
                shape.rx = float(self.rng.uniform(max(5.0, local_max_r * 0.5), max(10.0, local_max_r * 2.0)))
                if hasattr(shape, 'angle'):
                    base_angle = float(self.angle_map_cpu[cy, cx])
                    shape.angle = float((base_angle + self.rng.gauss(0, 5.0)) % 180.0)
            
            elif hasattr(shape, 'ry') and hasattr(shape, 'angle'):
                ratio = max(shape.rx / max(shape.ry, 1e-6), shape.ry / max(shape.rx, 1e-6))
                if ratio > 1.3:
                    base_angle = float(self.angle_map_cpu[cy, cx])
                    noise = 30.0 / ratio
                    shape.angle = float((base_angle + self.rng.gauss(0, noise)) % 180.0)
                    
        elif hasattr(shape, 'r'):
            global_max_r = max(2.5, (self.w / 2.0) * (rms_ratio ** 1.5))
            size_multiplier = 1.0 - (0.85 * local_priority)
            local_max_r = max(2.5, global_max_r * size_multiplier)
            scale_factor = self.rng.random() ** 2.0
            shape.r = float(1.0 + (local_max_r * scale_factor))
                
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
        max_rx_allowed = min(local_max_r, getattr(best, 'rx', local_max_r) * 1.5 + 2.0)
        max_ry_allowed = min(local_max_r, getattr(best, 'ry', local_max_r) * 1.5 + 2.0)
        
        while iterations > 0 and not self._stop:
            mutations = [best.mutate(self.rng, self.w, self.h) for _ in range(batch_size)]
            
            for m in mutations:
                m_cy = int(max(0, min(self.h - 1, m.y)))
                m_cx = int(max(0, min(self.w - 1, m.x)))
                m_local_priority = float(self.attention_map_cpu[m_cy, m_cx, 0])
                
                m_min_alpha = int(40 + 40 * m_local_priority)
                m_max_alpha = int(100 + 155 * m_local_priority)
                
                alpha_mutation = int(m.color[3] + self.rng.gauss(0, 15))
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
                
            scores, colors = score_shape_gpu_batch(mutations, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px)
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
            
            if scores[best_idx] < best_rms:
                best = mutations[best_idx]
                best.color = colors[best_idx]
                best_rms = scores[best_idx]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, batch_size // 4):
                    break
            iterations -= batch_size
            
        return best

    def wiggle_and_prune(self, progress_cb=None) -> int:
        """
        Runs a 100% exact forward sequential sweep.
        Evaluates shapes without subtraction using stable multipliers of top-layer occlusions.
        """
        if not self.shapes:
            return 0

        import torch
        import math
        from fd6.shapegen.gpu_scoring import _rasterize_mask_gpu, _rms_error_gpu, composite_gpu
        
        total = len(self.shapes)
        w, h = self.w, self.h
        target = self.target.float()
        
        if self.alpha_mask_t is not None:
            weight_global = (self.alpha_mask_t > 0).float().unsqueeze(-1)
            n_px = (weight_global.sum() * 3).item()
        else:
            n_px = float(target.numel())

        # Initialize caches
        cache = CanvasCache(self)
        occ_cache = OcclusionCache(self)
        
        # Setup forward canvas C_prev
        if self.alpha_mask_t is not None:
            C_prev = torch.full((h, w, 3), 40, dtype=torch.uint8, device='cuda')
        else:
            avg = self.target.float().mean(dim=(0, 1)).to(torch.uint8)
            C_prev = torch.tile(avg, (h, w, 1))
            
        retained_shapes = []

        def evaluate_shape_locally(shape, C_prev_local, target_local, A_local, x0, y0, x1, y1):
            mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
            m_x0, m_y0, m_x1, m_y1 = bbox
            if m_x1 <= m_x0 or m_y1 <= m_y0:
                return float('inf'), torch.zeros(3, device='cuda'), bbox
                
            w_m_local = torch.zeros((y1 - y0, x1 - x0, 1), dtype=torch.float32, device='cuda')
            
            # Clip the mutated bbox coordinates to the union bbox limits
            clip_x0 = max(x0, m_x0)
            clip_y0 = max(y0, m_y0)
            clip_x1 = min(x1, m_x1)
            clip_y1 = min(y1, m_y1)
            
            if clip_x1 > clip_x0 and clip_y1 > clip_y0:
                mask_slice = mask_local[(clip_y0 - m_y0):(clip_y1 - m_y0), (clip_x0 - m_x0):(clip_x1 - m_x0)]
                w_m_local[(clip_y0 - y0):(clip_y1 - y0), (clip_x0 - x0):(clip_x1 - x0), 0] = mask_slice.float() * (shape.color[3] / 255.0)
            
            # Analytical optimal color solver
            num = A_local * w_m_local * (target_local - (1.0 - w_m_local) * C_prev_local)
            den = A_local * w_m_local * w_m_local
            num_sum = num.sum(dim=(0, 1))
            den_sum = den.sum(dim=(0, 1))
            S_opt = torch.where(den_sum > 1e-6, num_sum / den_sum, torch.zeros_like(num_sum))
            S_opt = torch.clamp(S_opt, 0.0, 255.0)
            
            # Compute exact local visibility-weighted squared error
            C_cand_local = (1.0 - w_m_local) * C_prev_local + w_m_local * S_opt
            diff_cand_local = C_cand_local - target_local
            
            if self.alpha_mask_t is not None:
                weight_local = (self.alpha_mask_t[y0:y1, x0:x1] > 0).float().unsqueeze(-1)
                sq_local = A_local * weight_local * (diff_cand_local ** 2)
            else:
                sq_local = A_local * (diff_cand_local ** 2)
                
            return sq_local.sum().item(), S_opt, bbox

        # Sweep forward sequentially
        for k in range(total):
            if self._stop:
                break
                
            old_shape = self.shapes[k]
            
            # Fetch global occlusion map at k+1 backward
            A_global = occ_cache.get_occlusion(k + 1)
            C_prev_local_full = C_prev.float()
            
            bbox_old = old_shape.bbox(w, h)
            x0 = max(0, bbox_old[0] - 25)
            y0 = max(0, bbox_old[1] - 25)
            x1 = min(w, bbox_old[2] + 25)
            y1 = min(h, bbox_old[3] + 25)
            
            A_local = A_global[y0:y1, x0:x1]
            C_prev_local = C_prev_local_full[y0:y1, x0:x1]
            target_local = target[y0:y1, x0:x1]
            
            # Compute omission error (if shape k is deleted)
            diff_omit_local = C_prev_local - target_local
            if self.alpha_mask_t is not None:
                weight_local = (self.alpha_mask_t[y0:y1, x0:x1] > 0).float().unsqueeze(-1)
                sq_omit = (A_local * weight_local * (diff_omit_local ** 2)).sum().item()
            else:
                sq_omit = (A_local * (diff_omit_local ** 2)).sum().item()
                
            # Evaluate original shape error
            sq_orig, S_orig_opt, old_bbox = evaluate_shape_locally(old_shape, C_prev_local, target_local, A_local, x0, y0, x1, y1)
            
            if sq_omit <= sq_orig + 1e-4 or old_shape.color[3] < 15:
                continue
                
            best_shape = old_shape
            best_sq = sq_orig
            best_S = S_orig_opt
            
            iterations = self.profile.mutated_samples
            batch_size = 16
            
            rms_ratio = max(0.01, min(1.0, self.rms / self.start_rms))
            global_max_r = max(3.0, (self.w / 2.0) * (rms_ratio ** 1.5))
            cy = int(max(0, min(self.h - 1, old_shape.y)))
            cx = int(max(0, min(self.w - 1, old_shape.x)))
            local_priority = float(self.attention_map_cpu[cy, cx, 0])
            size_multiplier = 1.0 - (0.85 * local_priority)
            local_max_r = max(3.0, global_max_r * size_multiplier)
            max_rx_allowed = min(local_max_r, getattr(old_shape, 'rx', local_max_r) * 1.5 + 2.0)
            max_ry_allowed = min(local_max_r, getattr(old_shape, 'ry', local_max_r) * 1.5 + 2.0)
            
            for _ in range(0, iterations, batch_size):
                mutations = [best_shape.mutate(self.rng, self.w, self.h) for _ in range(batch_size)]
                
                for m in mutations:
                    if self.rng.random() < 0.80:
                        m.x = max(0.0, min(self.w - 1.0, best_shape.x + self.rng.gauss(0, 1.5)))
                        m.y = max(0.0, min(self.h - 1.0, best_shape.y + self.rng.gauss(0, 1.5)))
                        if hasattr(m, 'rx'):
                            m.rx = max(1.0, min(self.w, best_shape.rx + self.rng.gauss(0, 1.5)))
                            m.ry = max(1.0, min(self.h, best_shape.ry + self.rng.gauss(0, 1.5)))
                        if hasattr(m, 'r'):
                            m.r = max(1.0, min(self.w, best_shape.r + self.rng.gauss(0, 1.5)))
                        if hasattr(m, 'angle'):
                            m.angle = (best_shape.angle + self.rng.gauss(0, 3.0)) % 180.0
                    
                    m_cy = int(max(0, min(self.h - 1, m.y)))
                    m_cx = int(max(0, min(self.w - 1, m.x)))
                    
                    # Mutate alpha across the full range [15, 255] for precise masking edge-alignment
                    alpha_mutation = int(m.color[3] + self.rng.gauss(0, 20))
                    alpha_clamped = max(15, min(255, alpha_mutation))
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

                for m in mutations:
                    sq_cand, S_opt, bbox = evaluate_shape_locally(m, C_prev_local, target_local, A_local, x0, y0, x1, y1)
                    if sq_cand < best_sq:
                        best_sq = sq_cand
                        best_shape = m
                        best_shape.color = (int(S_opt[0].item()), int(S_opt[1].item()), int(S_opt[2].item()), m.color[3])
                        best_S = S_opt
            
            if best_shape.color[3] >= 15 and best_sq < sq_omit:
                retained_shapes.append(best_shape)
                C_prev, _ = composite_gpu(C_prev, best_shape, self.target, self.alpha_mask_t, recalculate_color=False)
            
            if progress_cb is not None and (k % 10 == 0 or k == total - 1):
                canvas_f = C_prev.clone()
                for s in self.shapes[(k + 1):]:
                    canvas_f, _ = composite_gpu(canvas_f, s, self.target, self.alpha_mask_t, recalculate_color=False)
                
                current_rms = _rms_error_gpu(canvas_f, self.target, self.alpha_mask_t)
                numpy_canvas = canvas_f.cpu().numpy()
                if self.alpha_mask_t is not None:
                    numpy_canvas = np.dstack([numpy_canvas, self.alpha_mask_t.cpu().numpy()])
                progress_cb(k + 1, total, current_rms, numpy_canvas)

        self.shapes = retained_shapes
        self.canvas = C_prev
        self.rms = _rms_error_gpu(self.canvas, self.target, self.alpha_mask_t)
        return len(self.shapes)

    def run(self) -> Iterable[EngineEvent]:
        from fd6.shapegen.gpu_scoring import compute_diff_sq_sum_gpu, composite_gpu
        p = self.profile
        types = [t for t in p.shape_types if t]
        if not types: types = ["rotated_ellipse"]
        save_at = set(p.save_at)
        
        consecutive_skips = 0
        try:
            while len(self.shapes) < p.stop_at and not self._stop:
                while self._pause and not self._stop:
                    time.sleep(0.05)
                
                current_diff_sq, n_px = compute_diff_sq_sum_gpu(self.canvas, self.target, self.alpha_mask_t)
                candidate = self._best_of_random(types, max(1, p.random_samples), current_diff_sq, n_px)
                refined = self._hill_climb(candidate, max(1, p.mutated_samples), current_diff_sq, n_px)
                
                from fd6.shapegen.gpu_scoring import score_shape_gpu
                rscore, _ = score_shape_gpu(
                    refined, self.canvas, self.target, self.alpha_mask_t, current_diff_sq, n_px
                )
                
                if rscore == float("inf") or rscore >= self.rms:
                    consecutive_skips += 1
                    if consecutive_skips >= 250:
                        yield EngineEvent(
                            kind="done", shape_count=len(self.shapes), rms=self.rms, 
                            canvas=self._get_numpy_canvas(), message="Converged."
                        )
                        return
                    continue
                    
                consecutive_skips = 0
                self.canvas, self.rms = composite_gpu(
                    self.canvas, refined, self.target, self.alpha_mask_t
                )
                self.shapes.append(refined)
                count = len(self.shapes)

                yield EngineEvent(kind="shape_committed", shape_count=count, rms=self.rms)
                
                if p.preview_every and (count % p.preview_every == 0):
                    yield EngineEvent(kind="preview", shape_count=count, rms=self.rms, canvas=self._get_numpy_canvas())
                if count in save_at or (p.save_every and count % p.save_every == 0):
                    yield EngineEvent(kind="checkpoint", shape_count=count, rms=self.rms)

            yield EngineEvent(kind="done", shape_count=len(self.shapes), rms=self.rms, canvas=self._get_numpy_canvas())
        except Exception as exc:
            yield EngineEvent(kind="error", message=f"{type(exc).__name__}: {exc}")

    def _get_numpy_canvas(self) -> np.ndarray | None:
        if self.canvas is None: 
            return None
        
        if self.alpha_mask_t is not None:
            c = self.canvas.cpu().numpy().copy()
            a = self.alpha_mask_t.cpu().numpy().copy()
            return np.dstack([c, a])
        return self.canvas.cpu().numpy().copy()


class CanvasCache:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.checkpoints = {}  # step -> CPU tensor
        self.step_size = 15
        self._build_checkpoints()
        
    def _build_checkpoints(self):
        h, w = self.engine.h, self.engine.w
        if self.engine.alpha_mask_t is not None:
            current = torch.full((h, w, 3), 40, dtype=torch.uint8, device='cuda')
        else:
            avg = self.engine.target.float().mean(dim=(0, 1)).to(torch.uint8)
            current = torch.tile(avg, (h, w, 1))
            
        self.checkpoints[0] = current.cpu()
        
        from fd6.shapegen.gpu_scoring import composite_gpu
        for i, shape in enumerate(self.engine.shapes):
            current, _ = composite_gpu(
                current, shape, self.engine.target, self.engine.alpha_mask_t, recalculate_color=False
            )
            step = i + 1
            if step % self.step_size == 0 and step < len(self.engine.shapes):
                self.checkpoints[step] = current.cpu()
                
    def get_canvas(self, step: int) -> torch.Tensor:
        nearest_step = (step // self.step_size) * self.step_size
        nearest_step = min(nearest_step, max(self.checkpoints.keys()))
        
        current = self.checkpoints[nearest_step].cuda()
        from fd6.shapegen.gpu_scoring import composite_gpu
        for i in range(nearest_step, step):
            current, _ = composite_gpu(
                current, self.engine.shapes[i], self.engine.target, self.engine.alpha_mask_t, recalculate_color=False
            )
        return current


class OcclusionCache:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.checkpoints = {}  # step -> CPU float32 tensor of shape [h, w, 1]
        self.step_size = 15
        self._build_checkpoints()
        
    def _build_checkpoints(self):
        h, w = self.engine.h, self.engine.w
        current = torch.ones((h, w, 1), dtype=torch.float32, device='cuda')
        
        total = len(self.engine.shapes)
        self.checkpoints[total] = current.cpu()
        
        from fd6.shapegen.gpu_scoring import _rasterize_mask_gpu
        for k in reversed(range(total)):
            shape = self.engine.shapes[k]
            mask_local, bbox = _rasterize_mask_gpu(shape, w, h)
            x0, y0, x1, y1 = bbox
            if x1 > x0 and y1 > y0:
                w_k = mask_local.float() * (shape.color[3] / 255.0)
                current[y0:y1, x0:x1, 0] *= (1.0 - w_k)
            
            step = k
            if step % self.step_size == 0 and step > 0:
                self.checkpoints[step] = current.cpu()
                
    def get_occlusion(self, step: int) -> torch.Tensor:
        total = len(self.engine.shapes)
        nearest_step = ((step + self.step_size - 1) // self.step_size) * self.step_size
        nearest_step = min(nearest_step, total)
        
        current = self.checkpoints[nearest_step].cuda()
        from fd6.shapegen.gpu_scoring import _rasterize_mask_gpu
        for k in reversed(range(step, nearest_step)):
            shape = self.engine.shapes[k]
            mask_local, bbox = _rasterize_mask_gpu(shape, self.engine.w, self.engine.h)
            x0, y0, x1, y1 = bbox
            if x1 > x0 and y1 > y0:
                w_k = mask_local.float() * (shape.color[3] / 255.0)
                current[y0:y1, x0:x1, 0] *= (1.0 - w_k)
        return current


class PostOptimizeWorker(QObject):
    progress = Signal(int, int, float)
    preview = Signal(object)
    finished = Signal(str)
    error = Signal(str)
    status = Signal(str, str)

    def __init__(self, json_path: Path, image_path: Path, profile: Profile, sticker_mode: bool = False):
        super().__init__()
        self.json_path = json_path
        self.image_path = image_path
        self.profile = profile
        self.sticker_mode = sticker_mode
        self._stop = False
        self._paused = False

    def set_pause(self, paused: bool):
        self._paused = paused

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from fd6.io.exporter import load_json
            self.status.emit("Loading JSON data...", "info")
            doc = load_json(str(self.json_path))
            shapes = doc.materialize_shapes()
            target_size = doc.image_size
            
            self.status.emit("Processing source target image...", "info")
            from PIL import Image
            img = Image.open(self.image_path)
            alpha_mask: np.ndarray | None = None
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            if has_alpha:
                rgba = img.convert("RGBA")
                if self.sticker_mode:
                    arr_rgba = np.asarray(rgba, dtype=np.uint8)
                    img = Image.fromarray(arr_rgba[:, :, :3], "RGB")
                    alpha_mask = arr_rgba[:, :, 3].copy()
                else:
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[3])
                    img = bg
            else:
                img = img.convert("RGB")
            
            if not self.sticker_mode and img.size[0] != img.size[1]:
                side = max(img.size)
                square = Image.new("RGB", (side, side), (255, 255, 255))
                offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
                square.paste(img, offset)
                img = square
            
            if target_size is not None:
                img = img.resize(target_size, Image.LANCZOS)
                if alpha_mask is not None:
                    am_img = Image.fromarray(alpha_mask, "L").resize(target_size, Image.LANCZOS)
                    alpha_mask = np.asarray(am_img, dtype=np.uint8)
            else:
                mr = self.profile.max_resolution
                if max(img.size) > mr:
                    scale = mr / max(img.size)
                    new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
                    img = img.resize(new_size, Image.LANCZOS)
                    if alpha_mask is not None:
                        am_img = Image.fromarray(alpha_mask, "L").resize(new_size, Image.LANCZOS)
                        alpha_mask = np.asarray(am_img, dtype=np.uint8)
            target_rgb = np.asarray(img, dtype=np.uint8)
            
            self.status.emit("Initializing GPU optimization engine...", "info")
            engine = Engine(target_rgb, EngineConfig(profile=self.profile), alpha_mask=alpha_mask)
            engine.seed_shapes(shapes)
            
            self.status.emit(f"Wiggling and relaxing all {len(shapes)} shapes forward...", "info")
            self.progress.emit(0, len(shapes), engine.rms)
            self.preview.emit(engine._get_numpy_canvas())
            
            def on_wiggle_progress(current_idx, total_shapes, current_rms, current_canvas):
                self.progress.emit(current_idx, total_shapes, current_rms)
                self.preview.emit(current_canvas)
                
            retained_count = engine.wiggle_and_prune(progress_cb=on_wiggle_progress)
            reclaimed_count = len(shapes) - retained_count
            
            self.status.emit(f"Wiggled shapes: reclaimed {reclaimed_count} redundant shapes. Regenerating slots...", "info")
            self.progress.emit(retained_count, len(shapes), engine.rms)
            self.preview.emit(engine._get_numpy_canvas())
            
            if self._stop:
                self.status.emit("Optimization cancelled.", "warning")
                return

            target_count = len(shapes)
            engine.profile.stop_at = target_count
            
            for event in engine.run():
                if self._stop:
                    break
                if event.kind == "shape_committed":
                    self.progress.emit(event.shape_count, target_count, event.rms)
                elif event.kind == "preview":
                    self.preview.emit(event.canvas)

            if self._stop:
                self.status.emit("Optimization cancelled.", "warning")
                return

            out_path = self.json_path.parent / f"{self.json_path.stem}_wiggled.json"
            from fd6.io.exporter import save_json
            doc.shapes = [s.to_json() for s in engine.shapes]
            save_json(doc, str(out_path))
            
            self.status.emit(f"Optimization complete. Saved to {out_path.name}", "success")
            self.finished.emit(str(out_path))
            
        except Exception as exc:
            self.status.emit(f"Error during optimization: {exc}", "error")
            self.error.emit(f"Wiggle failed: {type(exc).__name__}: {exc}")