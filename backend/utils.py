
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import io, json, base64, math
from .config import THUMB_DIR, DATA_DIR, CACHE_DIR
from .grouping import load_config, resolve_existing_role_file
import hashlib
import os
import threading
import time

def safe_open_image(path: Path) -> Image.Image:
    """Open an image as a PIL Image.

    For `.fits` files, always use Astropy with ZScale normalization to avoid
    PIL's FITS plugin producing incorrect renders. For other formats, use PIL.
    """
    suffix = path.suffix.lower()
    if suffix == ".fits":
        # FITS support with ZScale via astropy
        try:
            from astropy.io import fits
            from astropy.visualization import ZScaleInterval
        except ImportError:
            raise ValueError(f"FITS file detected but astropy not available: {path}")

        try:
            with fits.open(path, memmap=False) as hdul:
                # Get the first HDU with 2D data
                data = None
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None and hdu.data.ndim >= 2:
                        data = hdu.data
                        break

                if data is None:
                    raise ValueError("No 2D data found in FITS file")

                # Reduce to 2D if needed
                if data.ndim > 2:
                    data = data[0]

                x = np.array(data, dtype=np.float64)
                # Clean data
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                # ZScale with contrast 0.1
                try:
                    interval = ZScaleInterval(contrast=0.1)
                    lo, hi = interval.get_limits(x)

                    if hi > lo:
                        y = (x - lo) / (hi - lo)
                        y = np.clip(y, 0.0, 1.0)
                    else:
                        # Fallback for flat images
                        y = np.zeros_like(x)
                except Exception:
                    # Fallback to percentile scaling
                    p2, p98 = np.percentile(x, [2, 98])
                    if p98 > p2:
                        y = (x - p2) / (p98 - p2)
                        y = np.clip(y, 0.0, 1.0)
                    else:
                        y = np.zeros_like(x)

                # Convert to 8-bit
                img8 = (y * 255.0).astype(np.uint8)

                # Create RGB (grayscale for astronomical data)
                rgb = np.stack([img8] * 3, axis=-1)
                return Image.fromarray(rgb, mode="RGB")

        except Exception as fits_error:
            raise ValueError(f"Failed to process FITS file {path}: {fits_error}")
    else:
        # Non-FITS handled by PIL
        return Image.open(path).convert("RGB")

def thumb_path_for(image_path: Path, size: int = 256) -> Path:
    # content-address thumbnails to avoid collisions
    h = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()
    return THUMB_DIR / f"{h}_{size}.jpg"

def get_or_make_thumb(image_path: Path, size: int = 256) -> Path:
    """Generate thumbnail from full-res PNG for better quality and consistency.
    
    This replaces the old JPEG thumbnail generation with PNG-based thumbnails.
    """
    tp = thumb_path_for(image_path, size)
    if tp.exists():
        return tp
    # Concurrency guard to avoid generating many thumbnails in parallel
    with _thumb_sem():
        if tp.exists():
            return tp
        # Use full-res PNG as source instead of processing FITS directly
        try:
            png_path = get_or_make_full_png(image_path)
            img = Image.open(png_path).convert("RGB")
        except Exception:
            # Fallback to direct processing if PNG generation fails
            img = safe_open_image(image_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size))
        tp.parent.mkdir(parents=True, exist_ok=True)
        img.save(tp, format="JPEG", quality=85)
        return tp


def ensure_placeholder_thumb(size: int = 256) -> Path:
    """Return path to a small gray placeholder thumbnail of given size."""
    ph = THUMB_DIR / f"placeholder_{size}.jpg"
    if ph.exists():
        return ph
    ph.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(32, 32, 32))
    img.save(ph, format="JPEG", quality=70)
    return ph

def triplet_thumb_path_for(base_key: str, size: int = 256) -> Path:
    h = hashlib.md5(base_key.encode("utf-8")).hexdigest()
    return THUMB_DIR / f"triplet_{h}_{size}.jpg"

def get_or_make_triplet_thumb(target: Optional[Path], ref: Optional[Path], diff: Optional[Path], size: int = 256) -> Path:
    """Create or retrieve a composite triplet thumbnail (target|ref|diff).

    The output is a horizontal 3-up image. Missing panels are left blank.
    Uses full-resolution PNGs as source for better quality.
    """
    parts = [str(p) if p else "" for p in [target, ref, diff]]
    base_key = "|".join(parts) + f"|{size}"
    tp = triplet_thumb_path_for(base_key, size)
    if tp.exists():
        return tp
    # Concurrency guard for triplet generation
    with _thumb_sem():
        if tp.exists():
            return tp

        panels: List[Image.Image] = []
        for p in [target, ref, diff]:
            if p and Path(p).exists():
                try:
                    # Use full-res PNG as source instead of processing FITS directly
                    png_path = get_or_make_full_png(Path(p))
                    img = Image.open(png_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (size, size), color=(0,0,0))
            else:
                img = Image.new("RGB", (size, size), color=(0,0,0))
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size))
            # place on square canvas size x size (letterbox) to align heights
            canvas = Image.new("RGB", (size, size), color=(0,0,0))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y))
            panels.append(canvas)

        # Concatenate horizontally
        out = Image.new("RGB", (size * 3, size), color=(0,0,0))
        for i, panel in enumerate(panels):
            out.paste(panel, (i * size, 0))

        tp.parent.mkdir(parents=True, exist_ok=True)
        out.save(tp, format="JPEG", quality=85)
        return tp


def composite_thumb_path_for(keys: List[str], size: int = 256) -> Path:
    key = "|".join(keys + [str(size)])
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return THUMB_DIR / f"composite_{h}_{size}.jpg"


def get_or_make_composite_thumb(parent: Path, base: str, size: int = 256) -> Path:
    """Generalized N-role composite thumbnail using grouping config.

    - Each role is rendered as a size x size panel on a horizontal strip.
    - Missing roles are rendered as black panels to preserve alignment.
    - Cache key includes resolved file paths and size.
    """
    cfg = load_config()
    role_paths: List[Optional[Path]] = []
    key_parts: List[str] = []
    for role in cfg.roles:
        rp = resolve_existing_role_file(parent, base, role, cfg)
        role_paths.append(rp)
        key_parts.append(str(rp) if rp else "")
    tp = composite_thumb_path_for(key_parts, size=size)
    if tp.exists():
        return tp
    with _thumb_sem():
        if tp.exists():
            return tp
        panels: List[Image.Image] = []
        for p in role_paths:
            if p and Path(p).exists():
                try:
                    # Use full-res PNG as source for better quality
                    png_path = get_or_make_full_png(Path(p))
                    img = Image.open(png_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (size, size), color=(0,0,0))
            else:
                img = Image.new("RGB", (size, size), color=(0,0,0))
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), color=(0,0,0))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y))
            panels.append(canvas)
        out = Image.new("RGB", (size * max(1, len(panels)), size), color=(0,0,0))
        for i, panel in enumerate(panels):
            out.paste(panel, (i * size, 0))
        tp.parent.mkdir(parents=True, exist_ok=True)
        out.save(tp, format="JPEG", quality=85)
        return tp

# ===== Full-resolution PNG cache (optional display/training source) =====
def _png_dir() -> Path:
    d = CACHE_DIR / "png"
    d.mkdir(parents=True, exist_ok=True)
    return d


def png_path_for(image_path: Path) -> Path:
    try:
        stat = image_path.stat()
        key = f"{str(image_path)}|{stat.st_mtime_ns}|fullpng"
    except Exception:
        key = f"{str(image_path)}|fullpng"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _png_dir() / f"{h}.png"


def get_or_make_full_png(image_path: Path) -> Path:
    """Render the source image to a full-resolution PNG and cache it.

    Note: This converts FITS via safe_open_image and preserves pixel dimensions
    of the rendered view (after stretching), not original FITS dtype range.
    """
    pp = png_path_for(image_path)
    if pp.exists():
        return pp
    with _thumb_sem():
        if pp.exists():
            return pp
        img = safe_open_image(image_path)
        img = ImageOps.exif_transpose(img)
        pp.parent.mkdir(parents=True, exist_ok=True)
        img.save(pp, format="PNG")
        return pp

# ===== Internal: bounded semaphore for generation =====
_SEM_OBJ = None


class _SemWrapper:
    def __init__(self, sem: threading.Semaphore):
        self._sem = sem

    def __enter__(self):
        self._sem.acquire()
        return self._sem

    def __exit__(self, exc_type, exc, tb):
        try:
            self._sem.release()
        except Exception:
            pass


def _thumb_sem():
    """Return a context manager that acquires a shared semaphore.

    Some Python environments may not support using threading.Semaphore
    directly as a context manager. This wrapper ensures reliability.
    """
    global _SEM_OBJ
    if _SEM_OBJ is None:
        try:
            workers = int(os.getenv("AL_THUMB_WORKERS", "2"))
            workers = max(1, min(workers, 8))
        except Exception:
            workers = 2
        _SEM_OBJ = threading.Semaphore(workers)
    return _SemWrapper(_SEM_OBJ)

def probs_to_margin(probs: List[float]) -> float:
    # margin = gap between top-2 probabilities (lower means more uncertain)
    if len(probs) < 2:
        return 1.0 - probs[0] if probs else 1.0
    s = sorted(probs, reverse=True)
    return float(s[0] - s[1])

def np_to_bytes(arr: np.ndarray) -> bytes:
    import io
    out = io.BytesIO()
    np.save(out, arr.astype(np.float32), allow_pickle=False)
    return out.getvalue()

def bytes_to_np(b: bytes) -> np.ndarray:
    import io
    return np.load(io.BytesIO(b), allow_pickle=False)

def topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(arr))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(-arr[idx])]

def farthest_first(X: np.ndarray, k: int) -> np.ndarray:
    # Greedy k-center: pick a random point, then iteratively add the point farthest from the set
    n = X.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(42)
    first = rng.integers(0, n)
    selected = [first]
    dist = np.full(n, np.inf, dtype=np.float64)
    dist = np.minimum(dist, np.linalg.norm(X - X[first], axis=1))
    while len(selected) < min(k, n):
        i = int(np.argmax(dist))
        selected.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    return np.array(selected, dtype=int)
