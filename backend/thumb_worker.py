from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import threading
import time

from .db import get_session, DatasetItem
"""
Deprecated: Thumbnail worker removed. This module is retained to avoid import errors but does nothing.
"""

def is_running() -> bool:
    return False

def start(*args, **kwargs):
    return {"queued": False, "position": 0}

def status():
    return {"running": False, "message": "idle"}

def cancel():
    return None
from .grouping import load_config, match_role_and_base

_state_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_cancel = threading.Event()

_status: Dict[str, object] = {
    "running": False,
    "mode": "triplet",
    "size": 256,
    "assets": "thumbs",  # thumbs only
    "unlabeled_only": True,
    "total": 0,
    "done": 0,
    "skipped": 0,
    "errors": 0,
    "eta_sec": None,
    "started_at": None,
    "message": "idle",
}


def _update(**kwargs):
    with _state_lock:
        _status.update(kwargs)


def _items_iter(unlabeled_only: bool) -> List[DatasetItem]:
    with next(get_session()) as session:
        if unlabeled_only:
            # Unlabeled = no label assigned and not explicitly skipped
            items = session.exec(
                DatasetItem.__table__.select().where(  # type: ignore[attr-defined]
                    (DatasetItem.label.is_(None)) & (DatasetItem.skipped == False)  # type: ignore[comparison-overlap]
                )
            ).all()
        else:
            items = session.exec(
                DatasetItem.__table__.select()  # type: ignore[attr-defined]
            ).all()
    return items


def _triplet_paths(p: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path], str]:
    # Backward-compatible helper used by existing callers; leave behavior unchanged
    name_lower = p.name.lower()
    base = None
    for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
        if name_lower.endswith(suf):
            base = p.name[: -len(suf)]
            break
    if base is None:
        base = p.stem
    expected = {
        "target": p.with_name(f"{base}_target.fits"),
        "ref": p.with_name(f"{base}_ref.fits"),
        "diff": p.with_name(f"{base}_diff.fits"),
    }
    target = expected["target"] if expected["target"].exists() else None
    ref = expected["ref"] if expected["ref"].exists() else None
    diff = expected["diff"] if expected["diff"].exists() else None
    base_key = "|".join([str(x) if x else "" for x in [target, ref, diff]])
    return target, ref, diff, base_key


def _worker(mode: str, size: int, only_missing: bool, limit: Optional[int], unlabeled_only: bool, assets: str):
    start_ts = time.time()
    _update(running=True, mode=mode, size=size, assets=assets, unlabeled_only=unlabeled_only, done=0, skipped=0, errors=0, eta_sec=None, started_at=start_ts, message="scanning")
    try:
        items = _items_iter(unlabeled_only=unlabeled_only)
        if limit is not None:
            items = items[: int(limit)]
        total = len(items)
        _update(total=total, message="building")

        last_update = start_ts
        for it in items:
            if _cancel.is_set():
                _update(message="cancelled")
                break
            p = Path(it.path)
            try:
                # Build thumbnails/composites if requested
                if assets in ("thumbs",):
                    if mode == "single":
                        get_or_make_thumb(p, size=size)
                    elif mode == "triplet":
                        target, ref, diff, _ = _triplet_paths(p)
                        if not any([target, ref, diff]):
                            get_or_make_thumb(p, size=size)
                        else:
                            get_or_make_triplet_thumb(target, ref, diff, size=size)
                    else:
                        # generic composite based on grouping config
                        cfg = load_config()
                        _, base = match_role_and_base(p, cfg)
                        get_or_make_composite_thumb(p.parent, base, size=size)
                done = int(_status.get("done", 0)) + 1
                now = time.time()
                if now - last_update > 0.5:
                    rate = done / max(1e-6, (now - start_ts))
                    remaining = max(0, total - done)
                    eta = remaining / max(1e-6, rate)
                    _update(done=done, eta_sec=int(eta))
                    last_update = now
                else:
                    _update(done=done)
            except Exception:
                _update(errors=int(_status.get("errors", 0)) + 1)
                continue

        if not _cancel.is_set():
            _update(message="completed")
    finally:
        _cancel.clear()
        _update(running=False)


def start(mode: str = "triplet", size: int = 256, only_missing: bool = True, limit: Optional[int] = None, unlabeled_only: bool = True, assets: str = "thumbs"):
    global _thread
    if is_running():
        raise RuntimeError("Worker already running")
    _thread = threading.Thread(target=_worker, args=(mode, size, only_missing, limit, unlabeled_only, assets), daemon=True)
    _thread.start()


def status() -> Dict[str, object]:
    with _state_lock:
        return dict(_status)


def is_running() -> bool:
    with _state_lock:
        return bool(_status.get("running", False))


def cancel():
    _cancel.set()


def shutdown(timeout: float = 5.0):
    """Gracefully stop the worker thread if running."""
    global _thread
    try:
        _cancel.set()
        t = _thread
        if t and t.is_alive():
            t.join(timeout=timeout)
    except Exception:
        pass
    finally:
        _update(running=False)

