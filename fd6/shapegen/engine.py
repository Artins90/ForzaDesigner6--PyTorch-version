from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fd6.shapegen.profile import Profile
from fd6.shapegen.scoring import composite, rms_error, score_shape
from fd6.shapegen.shapes import Shape, random_shape


@dataclass
class EngineConfig:
    profile: Profile
    seed: int = 0  # 0 → time-based


@dataclass
class EngineEvent:
    """Event emitted at preview/save points. The worker translates these into Qt signals."""
    kind: str  # "shape_committed" | "checkpoint" | "preview" | "done" | "error"
    shape_count: int = 0
    rms: float = 0.0
    canvas: np.ndarray | None = None  # uint8 (H, W, 3); only set for preview/done
    message: str = ""


class Engine:
    """Image → shapes generator. Stateless w.r.t. external I/O — callers handle JSON/preview emission.

    Usage:
        engine = Engine(target_image_rgb, EngineConfig(profile=Profile(...)))
        for event in engine.run():
            if event.kind == "shape_committed": ...
            elif event.kind == "preview": ...
        # engine.shapes holds the final list
    """

    def __init__(self, target_rgb: np.ndarray, config: EngineConfig, alpha_mask: np.ndarray | None = None) -> None:
        if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
            raise ValueError("target_rgb must be HxWx3 RGB uint8")
        self.target = target_rgb.astype(np.uint8)
        self.config = config
        self.profile = config.profile
        self.h, self.w = self.target.shape[:2]
        # alpha_mask: H×W uint8, 0=transparent (ignore for scoring), 255=opaque
        # When provided, sticker mode is active — only opaque pixels contribute to the loss
        # function and we initialize the canvas to neutral grey (transparent areas stay grey,
        # which the user will discard when rendering the sticker on top of a vinyl group).
        self.alpha_mask = alpha_mask if alpha_mask is not None else None
        if self.alpha_mask is not None:
            # Sticker mode: zero the source RGB in transparent areas so any stray paint there
            # blends with black (not the garbage RGB that was under transparent pixels in the
            # source PNG — often bright green / magenta). And start canvas at neutral dark grey
            # so the user visually understands transparent regions don't get shapes.
            mask3 = (self.alpha_mask > 0)[:, :, None]
            self.target = self.target * mask3.astype(np.uint8)
            self.canvas = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        else:
            avg = self.target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            self.canvas = np.tile(avg, (self.h, self.w, 1)).astype(np.uint8)
        self.shapes: list[Shape] = []
        self.rms = rms_error(self.canvas, self.target, self.alpha_mask)
        self.start_rms = self.rms
        self._stop = False
        self._pause = False
        seed = config.seed or int(time.time() * 1000) & 0xFFFFFFFF
        self.rng = random.Random(seed)
        threads = self.profile.max_threads or os.cpu_count() or 1
        self._executor = ThreadPoolExecutor(max_workers=max(1, threads))

    def request_stop(self) -> None:
        self._stop = True

    def set_pause(self, paused: bool) -> None:
        self._pause = paused

    def seed_shapes(self, shapes: list[Shape]) -> None:
        """Resume mode: replay shapes onto the canvas before generation starts."""
        for s in shapes:
            self.canvas, self.rms = composite(self.canvas, s, self.target, self.alpha_mask)
            self.shapes.append(s)

    def _generate_candidate(self, types: list[str]) -> Shape:
        return random_shape(self.rng, self.w, self.h, types)

    def _best_of_random(self, types: list[str], n: int) -> Shape:
        """Sample n random shapes, return the one that would most reduce RMS."""
        candidates = [self._generate_candidate(types) for _ in range(n)]
        scored = list(self._executor.map(
            lambda s: (score_shape(s, self.canvas, self.target, self.alpha_mask), s),
            candidates,
        ))
        best = min(scored, key=lambda pair: pair[0][0])
        ((_rms, color), shape) = best
        shape.color = color
        return shape

    def _hill_climb(self, shape: Shape, iterations: int) -> Shape:
        """Greedy hill-climb mutation."""
        best = shape
        best_rms, best_color = score_shape(best, self.canvas, self.target, self.alpha_mask)
        best.color = best_color
        no_improve = 0
        for _ in range(iterations):
            if self._stop:
                break
            cand = best.mutate(self.rng, self.w, self.h)
            r, c = score_shape(cand, self.canvas, self.target, self.alpha_mask)
            if r < best_rms:
                best = cand
                best.color = c
                best_rms = r
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, iterations // 4):
                    break
        return best

    def run(self) -> Iterable[EngineEvent]:
        p = self.profile
        types = [t for t in p.shape_types if t]
        if not types:
            types = ["rotated_ellipse"]
        save_at = set(p.save_at)
        try:
            consecutive_skips = 0
            MAX_CONSECUTIVE_SKIPS = 80   # hard watchdog so the loop can never hang
            while len(self.shapes) < p.stop_at and not self._stop:
                while self._pause and not self._stop:
                    time.sleep(0.05)
                # Random search
                candidate = self._best_of_random(types, max(1, p.random_samples))
                # Hill climb
                refined = self._hill_climb(candidate, max(1, p.mutated_samples))
                # Sticker mode: a shape must fit ESSENTIALLY ENTIRELY inside
                # the opaque region (score_shape enforces STICKER_OVERLAP_MIN).
                # If not, retry up to 5 times then SKIP this iteration entirely
                # — committing a bad shape would inject as a solid blob in FH6.
                if self.alpha_mask is not None:
                    sticker_attempts = 0
                    while sticker_attempts < 5:
                        rscore, _ = score_shape(refined, self.canvas, self.target, self.alpha_mask)
                        if rscore != float("inf"):
                            break
                        candidate = self._best_of_random(types, max(1, p.random_samples))
                        refined = self._hill_climb(candidate, max(1, p.mutated_samples))
                        sticker_attempts += 1
                    else:
                        # All 5 attempts produced bleeding shapes — skip this iteration
                        consecutive_skips += 1
                        if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                            yield EngineEvent(
                                kind="done",
                                shape_count=len(self.shapes),
                                rms=self.rms,
                                canvas=self.canvas.copy(),
                                message=(
                                    f"Stopped early at {len(self.shapes)} shapes — couldn't "
                                    f"fit any more inside the opaque region after {MAX_CONSECUTIVE_SKIPS} "
                                    "consecutive attempts. Try increasing 'Random samples' or "
                                    "enabling smaller shape types."
                                ),
                            )
                            return
                        continue
                    consecutive_skips = 0
                # Commit
                new_canvas, new_rms = composite(self.canvas, refined, self.target, self.alpha_mask)
                if new_rms >= self.rms:
                    # No improvement found; emit a no-op shape_committed so callers can see progress.
                    # Still commit it — primitive does the same: every iteration adds a shape regardless.
                    pass
                self.canvas = new_canvas
                self.rms = new_rms
                self.shapes.append(refined)
                count = len(self.shapes)

                yield EngineEvent(kind="shape_committed", shape_count=count, rms=self.rms)

                if p.preview_every and (count % p.preview_every == 0):
                    yield EngineEvent(kind="preview", shape_count=count, rms=self.rms, canvas=self.canvas.copy())

                if count in save_at or (p.save_every and count % p.save_every == 0):
                    yield EngineEvent(kind="checkpoint", shape_count=count, rms=self.rms)

            yield EngineEvent(kind="done", shape_count=len(self.shapes), rms=self.rms, canvas=self.canvas.copy())
        except Exception as exc:  # surface to caller; engine still cleans up
            yield EngineEvent(kind="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            self._executor.shutdown(wait=False)
