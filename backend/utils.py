
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import io, json, base64, math
from .grouping import load_config, resolve_existing_role_file, match_role_and_base
import hashlib
import os
import threading
import time

def safe_open_image(path: Path, use_zscale: bool = True, zscale_contrast: float = 0.1) -> Image.Image:
    """Open an image as a PIL Image.

    For `.fits` files, optionally use Astropy ZScale normalization with a configurable
    contrast. If `use_zscale` is False or ZScale fails, falls back to percentile scaling.
    For other formats, use PIL and convert to RGB.
    """
    suffix = path.suffix.lower()
    if suffix == ".fits":
        # FITS support with ZScale via astropy
        try:
            from astropy.io import fits
            # Import ZScaleInterval lazily/conditionally to avoid unused import when disabled
            if use_zscale:
                from astropy.visualization import ZScaleInterval  # type: ignore
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

                # Normalize using ZScale (optional) or percentile fallback
                y: np.ndarray
                if use_zscale:
                    try:
                        interval = ZScaleInterval(contrast=float(zscale_contrast))  # type: ignore[name-defined]
                        lo, hi = interval.get_limits(x)
                        if hi > lo:
                            y = (x - lo) / (hi - lo)
                            y = np.clip(y, 0.0, 1.0)
                        else:
                            y = np.zeros_like(x)
                    except Exception:
                        # Fallback to percentile scaling
                        p2, p98 = np.percentile(x, [2, 98])
                        if p98 > p2:
                            y = (x - p2) / (p98 - p2)
                            y = np.clip(y, 0.0, 1.0)
                        else:
                            y = np.zeros_like(x)
                else:
                    # Percentile scaling when ZScale explicitly disabled
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

# Legacy thumbnail path helpers removed

# Legacy thumbnail file generator removed


# Placeholder file helper removed (placeholders generated inline in routes)

# Legacy triplet/composite path helpers removed

# Legacy triplet generator removed


# Legacy composite path helper removed


# Legacy composite generator removed

# ===== Deletion helpers for thumbnails (legacy cleanup) =====

def _delete_file_if_exists(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        # Best-effort deletion; ignore errors
        pass
    return False


def delete_thumbnails_for_path(image_path: Path, sizes: List[int] | None = None) -> int:
    """Delete thumbnail files related to a source image path.

    Removes:
    - Single-image thumbnails for the exact path
    - Triplet thumbnails (target|ref|diff) if applicable
    - Composite group thumbnails according to current grouping config

    Returns the number of files deleted.
    """
    deleted = 0
    sizes = sizes or [256]
    # Single thumbs no longer created

    # Group-derived thumbs (triplet and composite) no longer created
    pass

    return deleted

# (Removed) Full-resolution PNG cache: all PNG caching logic deleted per design change

# ===== File content hashing =====
def compute_file_hash(path: Path, algo: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable content hash of a file by streaming bytes.

    Defaults to SHA-256 for collision resistance while staying reasonably fast.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ===== Backward-compatibility no-ops for removed thumbnail generators =====
def get_or_make_thumb(path: Path, size: int = 256):
    # Thumbnails are generated on-the-fly now; keep for backward compatibility
    return None


def get_or_make_full_png(path: Path):
    # Full PNG cache removed; keep for backward compatibility
    return None

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
