from __future__ import annotations

import numpy as np

from fd6.shapegen.shapes.base import Shape


def rms_error(a: np.ndarray, b: np.ndarray, alpha_mask: np.ndarray | None = None) -> float:
    """RMS pixel error between two (H, W, 3) uint8 images. Lower is better.

    If `alpha_mask` (H, W) uint8 is given, only pixels where alpha>0 contribute; transparent
    pixels are ignored (sticker mode). The RMS is normalized by the count of contributing pixels.
    """
    diff = a.astype(np.int32) - b.astype(np.int32)
    sq = diff * diff
    if alpha_mask is None:
        return float(np.sqrt(sq.mean()))
    weight = (alpha_mask > 0)[:, :, None].astype(np.float32)
    total = float((sq * weight).sum())
    n = float(weight.sum() * 3)
    if n < 1:
        return 0.0
    return float(np.sqrt(total / n))


def compute_optimal_color(
    target: np.ndarray,
    current: np.ndarray,
    mask_local: np.ndarray,
    bbox: tuple[int, int, int, int],
    alpha: int,
) -> tuple[int, int, int, int]:
    """For a given shape mask and fixed alpha, compute the RGB color that minimizes RMS over the masked region.

    Closed-form: with `over` compositing `out = a*src + (1-a)*dst`, RMS is minimized when
    src = (target - (1-a)*dst) / a, averaged over the masked pixels.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return (0, 0, 0, alpha)
    tgt = target[y0:y1, x0:x1].astype(np.float32)
    cur = current[y0:y1, x0:x1].astype(np.float32)
    m = mask_local.astype(np.float32) / 255.0
    weight = m.sum()
    if weight < 0.5:
        return (0, 0, 0, alpha)
    a = alpha / 255.0
    if a < 1e-6:
        return (0, 0, 0, alpha)
    src = (tgt - (1.0 - a) * cur) / a
    src_masked = src * m[:, :, None]
    avg = src_masked.reshape(-1, 3).sum(axis=0) / weight
    avg = np.clip(avg, 0, 255).astype(np.int32)
    return (int(avg[0]), int(avg[1]), int(avg[2]), alpha)


def composite(
    current: np.ndarray,
    shape: Shape,
    target: np.ndarray,
    alpha_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Composite shape over current canvas with optimal color. Return (new_canvas, new_rms).

    In sticker mode (alpha_mask provided), the shape's per-pixel mask is AND-ed with the
    target's alpha mask so paint never lands in transparent areas — the dark-grey canvas
    background stays visible there, which is what the user expects from sticker mode.
    """
    h, w = current.shape[:2]
    mask_local, bbox = shape.rasterize_mask(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return current, rms_error(current, target, alpha_mask)
    # Combine shape mask with alpha mask if in sticker mode
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        # Element-wise min: paint only where both shape AND opaque
        effective_mask = np.minimum(mask_local, region_alpha)
    else:
        effective_mask = mask_local
    color = compute_optimal_color(target, current, effective_mask, bbox, shape.color[3])
    new = current.copy()
    a = color[3] / 255.0
    region_cur = new[y0:y1, x0:x1].astype(np.float32)
    region_tgt_color = np.array(color[:3], dtype=np.float32)
    m = (effective_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = m * (a * region_tgt_color + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
    new[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    shape.color = color
    return new, rms_error(new, target, alpha_mask)


# In sticker mode, virtually every "solid" pixel of a candidate shape must sit
# inside the opaque region. Anything less and the shape's body bleeds past the
# alpha edge in FH6 (no per-pixel alpha there → solid blob in transparent space).
# Counted against pixels where mask_local >= 128 (i.e., the shape's actual body,
# excluding anti-aliased fringe) so AA at the silhouette doesn't disqualify
# otherwise-clean shapes.
STICKER_OVERLAP_MIN = 0.995


def score_shape(
    shape: Shape,
    current: np.ndarray,
    target: np.ndarray,
    alpha_mask: np.ndarray | None = None,
) -> tuple[float, tuple[int, int, int, int]]:
    """Score a candidate without modifying the working canvas. Returns (rms_if_committed, optimal_color).

    Sticker-mode contract: a shape must sit ESSENTIALLY ENTIRELY inside the
    opaque region or it gets rejected with +inf. FH6 paints the full ellipse
    with no per-pixel alpha, so any shape that bleeds past the silhouette
    will render its body in what should be transparent space — exactly the
    'black outline artifacts' the user reported.
    """
    h, w = current.shape[:2]
    mask_local, bbox = shape.rasterize_mask(w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return float("inf"), shape.color
    effective_mask = mask_local
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        # Count only "solid" body pixels (alpha >=128) — ignores AA fringe so
        # antialiased silhouette edges don't artificially disqualify shapes.
        shape_body = mask_local >= 128
        body_total = float(shape_body.sum())
        if body_total < 1.0:
            return float("inf"), shape.color
        opaque_body = region_alpha >= 128
        if not opaque_body.any():
            return float("inf"), shape.color
        inside = float((shape_body & opaque_body).sum())
        if inside / body_total < STICKER_OVERLAP_MIN:
            return float("inf"), shape.color
        # AND-mask for color so the zeroed-out RGB of transparent pixels in
        # `target` can't drag the optimal color toward black.
        effective_mask = np.minimum(mask_local, region_alpha)
    color = compute_optimal_color(target, current, effective_mask, bbox, shape.color[3])
    a = color[3] / 255.0
    region_cur = current[y0:y1, x0:x1].astype(np.float32)
    region_tgt = target[y0:y1, x0:x1].astype(np.float32)
    src = np.array(color[:3], dtype=np.float32)
    m = (mask_local.astype(np.float32) / 255.0)[:, :, None]
    blended = m * (a * src + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
    diff_in = blended - region_tgt
    if alpha_mask is None:
        diff_out_squared_sum = float(((current.astype(np.int32) - target.astype(np.int32)) ** 2).sum())
        region_old_sq = float(((region_cur - region_tgt) ** 2).sum())
        region_new_sq = float((diff_in ** 2).sum())
        total_sq = diff_out_squared_sum - region_old_sq + region_new_sq
        n_px = current.shape[0] * current.shape[1] * 3
        return float(np.sqrt(max(0.0, total_sq) / n_px)), color
    # Sticker mode: weighted RMS, only opaque pixels contribute
    weight_full = (alpha_mask > 0)[:, :, None].astype(np.float32)
    weight_region = weight_full[y0:y1, x0:x1]
    diff_out_sq = ((current.astype(np.float32) - target.astype(np.float32)) ** 2) * weight_full
    diff_out_squared_sum = float(diff_out_sq.sum())
    region_old_sq = float((((region_cur - region_tgt) ** 2) * weight_region).sum())
    region_new_sq = float(((diff_in ** 2) * weight_region).sum())
    total_sq = diff_out_squared_sum - region_old_sq + region_new_sq
    n = float(weight_full.sum() * 3)
    if n < 1:
        return 0.0, color
    return float(np.sqrt(max(0.0, total_sq) / n)), color
