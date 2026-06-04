from __future__ import annotations

import ctypes
import json
import struct
from ctypes import wintypes
from pathlib import Path

from fd6.inject import Injector, VinylGroupHandle, InjectResult
from fd6.inject.game_profiles import GameProfile, default_profile
from fd6.inject.patterns_io import DEFAULT_PATTERNS_PATH, load_patterns
from fd6.inject.rtti_locator import find_livery_group_candidates as rtti_find_candidates
from fd6.inject.win_process import ProcessHandle, find_process_id


PATTERNS_FILE = DEFAULT_PATTERNS_PATH
FH6_TARGET_BUILD = "364.933"

COUNT_OFF = 0x5A   # u16 layer count
TABLE_OFF = 0x78   # u64 pointer to layer table (array of u64 layer pointers, 8-byte stride)

# Layer struct offsets (within each Layer instance)
LAYER_POS_OFF = 0x18      # 2 x f32: x, y
LAYER_SCALE_OFF = 0x28    # 2 x f32: scale_x, scale_y
LAYER_ROT_OFF = 0x50      # f32: rotation degrees
LAYER_COLOR_OFF = 0x74    # 4 bytes: R, G, B, alpha (alpha must be 0 or 255)
LAYER_MASK_OFF = 0x78     # u8: mask flag (0 or 1)
LAYER_SHAPE_ID_OFF = 0x7A # u8: shape type id (102 = ellipse, 101 = other)

# Scale divisors (per bvzrays)
SCALE_DIVISOR_ELLIPSE = 63.0
SCALE_DIVISOR_OTHER = 127.0
SHAPE_ID_ELLIPSE = 102
SHAPE_ID_OTHER = 101

_OFFSET_FIELDS: dict[str, str] = {
    "livery_count_offset": "COUNT_OFF",
    "layer_table_offset": "TABLE_OFF",
    "layer_position_offset": "LAYER_POS_OFF",
    "layer_scale_offset": "LAYER_SCALE_OFF",
    "layer_rotation_offset": "LAYER_ROT_OFF",
    "layer_color_offset": "LAYER_COLOR_OFF",
    "layer_mask_offset": "LAYER_MASK_OFF",
    "layer_shape_id_offset": "LAYER_SHAPE_ID_OFF",
}
_DIVISOR_FIELDS = ("scale_divisor_ellipse", "scale_divisor_other",
                   "shape_id_ellipse", "shape_id_other")

def _coerce_int(v) -> int | None:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        try:
            return int(s, 0)
        except ValueError:
            return int(s, 16)
    except (TypeError, ValueError):
        return None

def _offset_override_paths() -> list[Path]:
    cands: list[Path] = [Path.cwd() / ".fd6_offsets.json"]
    try:
        import sys
        exe_dir = Path(sys.executable).resolve().parent
        cands.append(exe_dir / ".fd6_offsets.json")
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            cands.append(Path(sys.argv[0]).resolve().parent / ".fd6_offsets.json")
    except Exception:
        pass
    seen: set[str] = set()
    out: list[Path] = []
    for p in cands:
        k = str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out

def _load_offset_overrides(profile_key: str) -> dict:
    for path in _offset_override_paths():
        try:
            if not path.exists():
                continue
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        scoped = raw.get(profile_key)
        return scoped if isinstance(scoped, dict) else raw
    return {}

def _effective_profile(profile: GameProfile) -> tuple[GameProfile, dict]:
    import dataclasses
    overrides = _load_offset_overrides(profile.key)
    changes: dict = {}
    applied: dict = {}
    for field in _OFFSET_FIELDS:
        if field in overrides:
            iv = _coerce_int(overrides[field])
            if iv is not None:
                changes[field] = iv
                applied[field] = iv
    for field in _DIVISOR_FIELDS:
        if field in overrides:
            try:
                val = float(overrides[field]) if "divisor" in field else int(overrides[field])
            except (TypeError, ValueError):
                continue
            changes[field] = val
            applied[field] = val
    eff = dataclasses.replace(profile, **changes) if changes else profile
    _seed_module_offsets(eff)
    return eff, applied

def _seed_module_offsets(profile: GameProfile) -> None:
    g = globals()
    for field, gname in _OFFSET_FIELDS.items():
        g[gname] = getattr(profile, field)
    g["SCALE_DIVISOR_ELLIPSE"] = profile.scale_divisor_ellipse
    g["SCALE_DIVISOR_OTHER"] = profile.scale_divisor_other
    g["SHAPE_ID_ELLIPSE"] = profile.shape_id_ellipse
    g["SHAPE_ID_OTHER"] = profile.shape_id_other

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

SPHERE_FULL_TABLE_THRESHOLD = 0.85

def patterns_are_populated() -> bool:
    return True

def _get_module_base(pid: int, module_name: str) -> int | None:
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    LIST_MODULES_ALL = 0x03

    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleBaseNameW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleBaseNameW.restype = wintypes.DWORD

    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        modules = (ctypes.c_void_p * 1024)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(h, modules, ctypes.sizeof(modules), ctypes.byref(needed), LIST_MODULES_ALL):
            return None
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        target = module_name.lower()
        for i in range(count):
            mod = modules[i]
            if mod is None:
                continue
            buf = ctypes.create_unicode_buffer(260)
            n = psapi.GetModuleBaseNameW(h, mod, buf, 260)
            if n and buf.value.lower() == target:
                return int(mod)
        return None
    finally:
        k32.CloseHandle(h)

def _read_game_build(pid: int, process_names: tuple[str, ...]) -> str | None:
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ver = ctypes.WinDLL("version", use_last_error=True)
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not h:
            return None
        try:
            psapi.GetModuleFileNameExW.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
            psapi.GetModuleFileNameExW.restype = wintypes.DWORD
            buf = ctypes.create_unicode_buffer(1024)
            if not psapi.GetModuleFileNameExW(h, None, buf, 1024):
                return None
            exe_path = buf.value
        finally:
            k32.CloseHandle(h)
        if not exe_path:
            return None
        size = ver.GetFileVersionInfoSizeW(exe_path, None)
        if not size:
            return None
        data = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(exe_path, 0, size, data):
            return None
        block = ctypes.c_void_p()
        blen = wintypes.UINT()
        if not ver.VerQueryValueW(data, "\\", ctypes.byref(block), ctypes.byref(blen)):
            return None

        class VS_FIXEDFILEINFO(ctypes.Structure):
            _fields_ = [
                ("dwSignature", wintypes.DWORD), ("dwStrucVersion", wintypes.DWORD),
                ("dwFileVersionMS", wintypes.DWORD), ("dwFileVersionLS", wintypes.DWORD),
                ("dwProductVersionMS", wintypes.DWORD), ("dwProductVersionLS", wintypes.DWORD),
                ("dwFileFlagsMask", wintypes.DWORD), ("dwFileFlags", wintypes.DWORD),
                ("dwFileOS", wintypes.DWORD), ("dwFileType", wintypes.DWORD),
                ("dwFileSubtype", wintypes.DWORD), ("dwFileDateMS", wintypes.DWORD),
                ("dwFileDateLS", wintypes.DWORD),
            ]
        ffi = ctypes.cast(block, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
        ms, ls = ffi.dwFileVersionMS, ffi.dwFileVersionLS
        parts = (ms >> 16 & 0xFFFF, ms & 0xFFFF, ls >> 16 & 0xFFFF, ls & 0xFFFF)
        trimmed = list(parts)
        while len(trimmed) > 2 and trimmed[-1] == 0:
            trimmed.pop()
        return ".".join(str(p) for p in trimmed)
    except Exception:
        return None

def _is_user_ptr(val: int) -> bool:
    return 0x000001000000 < val < 0x800000000000

def _read_u64(proc: ProcessHandle, addr: int) -> int:
    b = proc.try_read(addr, 8)
    return struct.unpack('<Q', b)[0] if b and len(b) == 8 else 0

def _read_2f(proc: ProcessHandle, addr: int) -> tuple[float, float] | None:
    b = proc.try_read(addr, 8)
    return struct.unpack('<2f', b) if b and len(b) == 8 else None

def _score_layer(proc: ProcessHandle, lptr: int) -> int:
    if not _is_user_ptr(lptr):
        return 0
    score = 0
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos and all(_is_finite_float(v) and -8192.0 <= v <= 8192.0 for v in pos):
        score += 1
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale and all(_is_finite_float(v) and 0.0 < abs(v) <= 64.0 for v in scale):
        score += 1
    color = proc.try_read(lptr + LAYER_COLOR_OFF, 4)
    if color and len(color) == 4:
        score += 1
    shape = proc.try_read(lptr + LAYER_SHAPE_ID_OFF, 1)
    if shape and shape[0] in (101, 102):
        score += 1
    mask = proc.try_read(lptr + LAYER_MASK_OFF, 1)
    if mask and mask[0] in (0, 1):
        score += 1
    return score

def _is_finite_float(v: float) -> bool:
    import math
    return math.isfinite(v)

def _loose_validate_layer(proc: ProcessHandle, lptr: int) -> bool:
    if not _is_user_ptr(lptr):
        return False
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos is None or not all(_is_finite_float(v) for v in pos):
        return False
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale is None or not all(_is_finite_float(v) for v in scale):
        return False
    if proc.try_read(lptr + LAYER_COLOR_OFF, 4) is None:
        return False
    return True

def _count_loose_valid_layers(proc: ProcessHandle, table_addr: int, layer_count: int) -> int:
    valid = 0
    for k in range(layer_count):
        lptr = _read_u64(proc, table_addr + k * 8)
        if _loose_validate_layer(proc, lptr):
            valid += 1
    return valid

def locate_livery_group(
    proc: ProcessHandle, layer_count: int,
    progress_cb=None, max_candidates: int = 200000,
) -> tuple[int, int] | None:
    pattern = struct.pack('<H', layer_count)
    all_regions = []
    
    for r in proc.enumerate_regions():
        if r.readable and r.writable and not r.is_image:
            state = getattr(r, "state", None)
            m_type = getattr(r, "type", None)
            protect = getattr(r, "protect", None)

            if state is not None and state != 0x1000:
                continue
            if m_type is not None and m_type != 0x20000:
                continue
            if protect is not None and (protect & 0x100 or protect & 0x01):
                continue
                
            all_regions.append(r)

    small_regions = [r for r in all_regions if r.size <= 256 * 1024 * 1024]
    small_regions.sort(key=lambda r: r.base)  
    
    large_regions = [r for r in all_regions if r.size > 256 * 1024 * 1024]
    large_regions.sort(key=lambda r: r.size, reverse=True)  
    
    ordered_regions = small_regions + large_regions
    total = len(ordered_regions)
    candidates = 0
    perfect: list[tuple[int, int]] = []  
    
    for i, r in enumerate(ordered_regions):
        data = proc.try_read(r.base, r.size)
        if data is None:
            if progress_cb: progress_cb(i + 1, total, candidates)
            continue
        start = 0
        while True:
            pos = data.find(pattern, start)
            if pos < 0:
                break
            start = pos + 1
            candidates += 1
            if candidates > max_candidates:
                if progress_cb: progress_cb(i + 1, total, candidates)
                return _pick_best_perfect(proc, perfect, layer_count)
            count_addr = r.base + pos
            group_addr = count_addr - COUNT_OFF
            if group_addr < r.base:
                continue
            table_addr = _read_u64(proc, group_addr + TABLE_OFF)
            if not _is_user_ptr(table_addr):
                continue
            ok = True
            sample_n = min(layer_count, 16)
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
                if _score_layer(proc, lptr) < 5:
                    ok = False
                    break
            if ok:
                valid_full = _count_valid_layers(proc, table_addr, layer_count)
                if valid_full >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
                    if progress_cb:
                        progress_cb(i + 1, total, len(perfect) + 1)
                    return (group_addr, table_addr)
                perfect.append((group_addr, table_addr))
        if progress_cb: progress_cb(i + 1, total, len(perfect))
    return _pick_best_perfect(proc, perfect, layer_count)

def _pick_best_perfect(
    proc: ProcessHandle, perfect: list[tuple[int, int]], layer_count: int,
) -> tuple[int, int] | None:
    if not perfect:
        return None
    if len(perfect) == 1:
        group_addr, table_addr = perfect[0]
        valid_full = _count_valid_layers(proc, table_addr, layer_count)
        if valid_full >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
            return (group_addr, table_addr)
        return None
    scored: list[tuple[int, int, int]] = []
    for group_addr, table_addr in perfect:
        valid_full = _count_valid_layers(proc, table_addr, layer_count)
        scored.append((valid_full, group_addr, table_addr))
    scored.sort(reverse=True)
    best_valid, group_addr, table_addr = scored[0]
    if best_valid >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
        return (group_addr, table_addr)
    return None

def _count_valid_layers(proc: ProcessHandle, table_addr: int, layer_count: int) -> int:
    valid = 0
    for k in range(layer_count):
        lptr = _read_u64(proc, table_addr + k * 8)
        if _score_layer(proc, lptr) >= 5:
            valid += 1
    return valid

def _pack_color(shape_dict: dict, force_solid: bool = False) -> bytes:
    color = shape_dict.get("color")
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return bytes([255, 255, 255, 255])
    r = int(color[0]) & 0xFF
    g = int(color[1]) & 0xFF
    b = int(color[2]) & 0xFF
    a = 255 if force_solid else (int(color[3]) & 0xFF if len(color) > 3 else 255)
    return bytes([r, g, b, a])

class FH6Injector(Injector):
    def __init__(self, pid: int | None = None, patterns_path: Path | str = PATTERNS_FILE,
                 profile: GameProfile | None = None) -> None:
        self.pid = pid
        self.patterns_path = Path(patterns_path)
        base_profile = profile or default_profile()
        self.profile, self.offset_overrides = _effective_profile(base_profile)
        self._proc: ProcessHandle | None = None
        self._group_addr: int | None = None
        self._table_addr: int | None = None
        self._layer_count: int | None = None
        self.detected_build: str | None = None  

    @property
    def game_label(self) -> str:
        return self.profile.label

    def attach(self) -> None:
        if self.pid is None:
            for name in self.profile.process_names:
                self.pid = find_process_id(name)
                if self.pid is not None:
                    break
            if self.pid is None:
                names = " / ".join(self.profile.process_names)
                raise RuntimeError(
                    f"{self.profile.label} is not running, OR FD6 is running with lower "
                    f"privileges than the game. If the game IS open, close FD6 and "
                    f"re-launch it as Administrator (right-click FD6MultiSupport.exe → "
                    f"Run as administrator)."
                )
        self._proc = ProcessHandle(self.pid)
        self._proc.open()
        try:
            self.detected_build = _read_game_build(self.pid, self.profile.process_names)
        except Exception:
            self.detected_build = None

    def build_status(self) -> str:
        target = FH6_TARGET_BUILD if self.profile.key == "fh6" else "(profile defaults)"
        parts = [f"Offsets target build {target}."]
        if self.detected_build:
            parts.append(f"Attached game build: {self.detected_build}.")
            if self.profile.key == "fh6" and self.detected_build != FH6_TARGET_BUILD:
                parts.append("Build differs from target.")
        if self.offset_overrides:
            keys = ", ".join(sorted(self.offset_overrides))
            parts.append(f"Applied local offset overrides: {keys}.")
        return " ".join(parts)

    def detach(self) -> None:
        if self._proc:
            self._proc.close()
            self._proc = None

    def _try_rtti_locate(self, count_try: int, progress_cb=None, status_cb=None) -> tuple[int, int] | None:
        if self._proc is None or self.pid is None:
            return None
        proc = self._proc

        def _accept(group_addr: int, table_addr: int) -> bool:
            sample_n = min(count_try, 16)
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
                if not _loose_validate_layer(proc, lptr):
                    return False
            valid_full = _count_loose_valid_layers(proc, table_addr, count_try)
            return valid_full >= count_try * 0.95

        try:
            candidates = rtti_find_candidates(
                proc, self.pid, self.profile, count_try,
                progress_cb=(progress_cb if progress_cb else None),
                accept_cb=_accept,
                status_cb=(status_cb if status_cb else None),
            )
        except Exception:
            return None
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        scored: list[tuple[int, int, int]] = []
        for group_addr, table_addr in candidates:
            sample_n = min(count_try, 16)
            ok = True
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
                if not _loose_validate_layer(proc, lptr):
                    ok = False
                    break
            if not ok:
                continue
            valid_full = _count_loose_valid_layers(proc, table_addr, count_try)
            scored.append((valid_full, group_addr, table_addr))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_valid, group_addr, table_addr = scored[0]
        if best_valid >= count_try * 0.95:
            return (group_addr, table_addr)
        return None

    def find_active_vinyl_group(self, progress_cb=None, layer_count: int | None = None,
                                color_progress_cb=None, status_cb=None) -> VinylGroupHandle:
        if not self._proc:
            raise RuntimeError("Injector not attached. Call attach() first.")
        common = [500, 1500, 3000, 1000, 100, 50, 20, 10]
        if layer_count is not None:
            tries = [layer_count] + [c for c in common if c > layer_count]
        else:
            tries = common
        for count_try in tries:
            if count_try is None:
                continue
            result = locate_livery_group(self._proc, count_try, progress_cb=progress_cb)
            if result is None:
                if status_cb:
                    status_cb(f"Sphere-template scan found no fresh {count_try}-layer group. Checking RTTI.")
                result = self._try_rtti_locate(count_try, progress_cb=progress_cb, status_cb=status_cb)
            if result is not None:
                self._group_addr, self._table_addr = result
                self._layer_count = count_try
                table_bytes = self._proc.try_read(self._table_addr, count_try * 8)
                if table_bytes and len(table_bytes) == count_try * 8:
                    addrs = list(struct.unpack(f"<{count_try}Q", table_bytes))
                else:
                    addrs = [_read_u64(self._proc, self._table_addr + i * 8)
                             for i in range(count_try)]
                if status_cb:
                    status_cb(f"Located vinyl group with {count_try} layer slots. Writing...")
                return VinylGroupHandle(
                    base_addr=self._group_addr,
                    layer_count=count_try,
                    shape_array_addr=self._table_addr,
                    shape_stride=8,  
                    meta={
                        "group_addr": self._group_addr,
                        "table_addr": self._table_addr,
                        "layer_addrs": addrs,
                    },
                )
        raise RuntimeError(
            "No confident LiveryGroup match. Ensure your editor contains a raw, ungrouped template."
        )

    def inject(self, shapes: list, group: VinylGroupHandle, progress_cb=None,
               image_size: tuple[int, int] | None = None, coord_scale: float = 1.0) -> InjectResult:
        if not self._proc:
            raise RuntimeError("Injector not attached.")
        layer_addrs: list[int] = (group.meta or {}).get("layer_addrs") or []
        if not layer_addrs:
            return InjectResult(success=False, message="No layer addresses cached. Call find_active_vinyl_group first.")

        shape_dicts: list[dict] = []
        for s in shapes:
            if hasattr(s, "to_json"):
                shape_dicts.append(s.to_json())
            elif isinstance(s, dict):
                shape_dicts.append(s)
            else:
                raise TypeError(f"Unsupported shape type: {type(s)!r}")
        n = len(shape_dicts)
        if n > len(layer_addrs):
            return InjectResult(
                success=False, shapes_written=0,
                message=(f"Template has {len(layer_addrs)} layer slots, but JSON has {n} shapes. "
                         f"Load a larger template vinyl group."),
            )

        written = 0
        bytes_total = 0
        skipped = 0
        type_counts: dict[str, int] = {}
        for i, sd in enumerate(shape_dicts):
            lptr = layer_addrs[i]
            if not _is_user_ptr(lptr) or _score_layer(self._proc, lptr) < 5:
                skipped += 1
                if progress_cb:
                    progress_cb(written, n)
                continue
            shape_type = sd.get("type", "rotated_ellipse")
            is_ellipse = "ellipse" in shape_type or shape_type == "circle"
            scale_div = (
                self.profile.scale_divisor_ellipse if is_ellipse
                else self.profile.scale_divisor_other
            )

            try:
                x = float(sd.get("x", 0.0))
                y = float(sd.get("y", 0.0))
                self._proc.write(lptr + LAYER_POS_OFF, struct.pack('<2f', x, -y))
                bytes_total += 8

                rx = float(sd.get("rx", sd.get("hw", sd.get("r", 1.0))))
                ry = float(sd.get("ry", sd.get("hh", sd.get("r", 1.0))))

                if is_ellipse:
                    adj_w, adj_h = compensated_ellipse_size(rx, ry)
                    sx = adj_w / scale_div
                    sy = adj_h / scale_div
                else:
                    sx = (rx * 2.0) / scale_div
                    sy = (ry * 2.0) / scale_div

                self._proc.write(lptr + LAYER_SCALE_OFF, struct.pack('<2f', sx, sy))
                bytes_total += 8

                angle = float(sd.get("angle", 0.0)) % 360.0
                self._proc.write(lptr + LAYER_ROT_OFF, struct.pack('<f', (360.0 - angle) % 360.0))
                bytes_total += 4

                self._proc.write(lptr + LAYER_COLOR_OFF, _pack_color(sd, force_solid=False))
                bytes_total += 4

                self._proc.write(lptr + LAYER_MASK_OFF, bytes([0]))
                bytes_total += 1

                self._proc.write(lptr + LAYER_SHAPE_ID_OFF, bytes([
                    self.profile.shape_id_ellipse if is_ellipse else self.profile.shape_id_other
                ]))
                bytes_total += 1

                written += 1
                type_counts[shape_type] = type_counts.get(shape_type, 0) + 1
            except OSError:
                skipped += 1

            if progress_cb:
                progress_cb(written, n)

        msg = (f"Wrote {written}/{n} shapes ({bytes_total} bytes) via LiveryGroup layer table.")
        if type_counts:
            mix = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
            msg += f" Type mix written — {mix}."
        if skipped:
            msg += f" Skipped {skipped} unsafe layer(s) (failed revalidation)."
            if skipped >= max(1, n // 2):
                msg += " A high skip count usually means the layer-struct offsets shifted in this game build."
        msg += " " + self.build_status()
        return InjectResult(
            success=written > 0,
            shapes_written=written,
            message=msg,
        )