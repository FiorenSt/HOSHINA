from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import threading
import time

from .db import get_session, DatasetItem
from .config import THUMB_DIR
from .utils import safe_open_image, get_or_make_full_png, get_or_make_thumb, get_or_make_composite_thumb
from .grouping import load_config as load_grouping, match_role_and_base, resolve_existing_role_file

_state_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_cancel = threading.Event()

_status: Dict[str, object] = {
    "running": False,
    "data_dir": None,
    "by_groups": False,
    "require_all_roles": None,
    "total": 0,
    "total_files": 0,
    "done": 0,
    "ingested": 0,
    "pngs": 0,
    "thumbs": 0,
    "skipped": 0,
    "errors": 0,
    "eta_sec": None,
    "started_at": None,
    "message": "idle",
    "db_ready_threshold": 120,
    "db_ready": False,
}

# Simple FIFO queue of pending ingest jobs (one-at-a-time processing)
_queue: List[Dict[str, object]] = []


def _update(**kwargs):
    with _state_lock:
        _status.update(kwargs)
        # keep queue size in status for UI
        _status["queue_size"] = len(_queue)

def _start_thread(params: Dict[str, object]) -> None:
    global _thread
    _thread = threading.Thread(
        target=_worker,
        args=(
            params["data_dir"],
            params["extensions"],
            params["by_groups"],
            params["max_groups"],
            params["generate_pngs"],
            params["make_thumbs"],
            params["require_all_roles"],
        ),
        daemon=True,
    )
    _thread.start()


def _gather_tasks_files(data_dir: Path, extensions: set[str]) -> List[Path]:
    files: List[Path] = []
    for f in data_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in extensions:
            continue
        files.append(f)
    return files


def _gather_tasks_groups(
    data_dir: Path,
    extensions: set[str],
    require_all_roles: bool,
    max_groups: Optional[int],
) -> List[Tuple[Path, Dict[str, Path]]]:
    cfg = load_grouping()
    files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
    # Build stable list of group keys
    keys: List[Tuple[Path, str]] = []
    seen = set()
    for f in files:
        _, base = match_role_and_base(f, cfg)
        key = (f.parent, base)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    keys.sort(key=lambda k: (str(k[0]), k[1]))

    def role_files_for(parent: Path, base: str) -> Dict[str, Path]:
        roles: Dict[str, Path] = {}
        for role in cfg.roles:
            rp = resolve_existing_role_file(parent, base, role, cfg)
            if rp is not None:
                roles[role] = rp
        return roles

    tasks: List[Tuple[Path, Dict[str, Path]]] = []
    with next(get_session()) as session:
        for parent, base in keys:
            roles = role_files_for(parent, base)
            if require_all_roles and not all(r in roles for r in cfg.roles):
                continue
            if not roles:
                continue

            # Only include groups with at least one new file
            has_new = False
            for rp in roles.values():
                canonical = str(Path(rp).resolve())
                existing = session.exec(
                    DatasetItem.__table__.select().where(DatasetItem.path == canonical)  # type: ignore[attr-defined]
                ).first()
                if not existing:
                    has_new = True
                    break
            if has_new:
                tasks.append((parent, roles))
            if max_groups is not None and len(tasks) >= int(max_groups):
                break
    return tasks


def _worker(
    data_dir: Path,
    extensions: set[str],
    by_groups: bool,
    max_groups: Optional[int],
    generate_pngs: bool,
    make_thumbs: bool,
    require_all_roles: Optional[bool],
):
    start_ts = time.time()
    _update(
        running=True,
        data_dir=str(data_dir),
        by_groups=by_groups,
        require_all_roles=require_all_roles,
        done=0,
        ingested=0,
        pngs=0,
        skipped=0,
        errors=0,
        eta_sec=None,
        started_at=start_ts,
        message="scanning",
        db_ready_threshold=120,
        db_ready=False,
    )

    try:
        if by_groups or (max_groups is not None):
            gcfg = load_grouping()
            must_have_all = gcfg.require_all_roles if require_all_roles is None else bool(require_all_roles)
            groups = _gather_tasks_groups(data_dir, extensions, must_have_all, max_groups)
            # total counts the number of triplet groups, not individual files
            total_groups = len(groups)
            total_files = sum(len(roles.values()) for _, roles in groups)
            _update(total=total_groups, total_files=total_files, message=f"ingesting {total_groups} triplet groups ({total_files} files)")
            last_update = start_ts
            target_ready_groups = max(1, min(120, total_groups))
            
            # Phase 1: Database ingestion only
            db_entries_created = 0
            groups_processed = 0
            png_queue = []  # Store paths for PNG generation
            thumb_groups_queue: List[tuple[Path, str]] = []  # (parent, base) for composite thumbs
            
            with next(get_session()) as session:
                for parent, roles in groups:
                    if _cancel.is_set():
                        _update(message="cancelled")
                        break
                        
                    for rp in roles.values():
                        try:
                            canonical = str(Path(rp).resolve())
                            existing = session.exec(
                                DatasetItem.__table__.select().where(DatasetItem.path == canonical)  # type: ignore[attr-defined]
                            ).first()
                            if existing:
                                if generate_pngs:
                                    png_queue.append(Path(canonical))
                                continue

                            img = safe_open_image(Path(rp))
                            w, h = img.size
                            it = DatasetItem(path=canonical, width=w, height=h)
                            session.add(it)
                            session.commit()
                            _update(ingested=int(_status.get("ingested", 0)) + 1)
                            db_entries_created += 1
                            
                            if generate_pngs:
                                png_queue.append(Path(canonical))
                        except Exception:
                            _update(errors=int(_status.get("errors", 0)) + 1)
                            continue
                    
                    # Update progress after processing each group (triplet)
                    groups_processed += 1
                    # Queue composite thumbnail generation for this group
                    if make_thumbs:
                        # Derive base using grouping config matching
                        any_file = next(iter(roles.values()))
                        _, base = match_role_and_base(any_file, gcfg)
                        if base:
                            thumb_groups_queue.append((parent, base))
                    
                    now = time.time()
                    if now - last_update > 0.5:
                        rate = groups_processed / max(1e-6, (now - start_ts))
                        remaining = max(0, total_groups - groups_processed)
                        eta = remaining / max(1e-6, rate)
                        _update(done=groups_processed, eta_sec=int(eta))
                        last_update = now
                    else:
                        _update(done=groups_processed)
            
            # Phase 2: Interleaved PNG & thumbnail generation (background)
            if ((generate_pngs and png_queue) or (make_thumbs and thumb_groups_queue)) and not _cancel.is_set():
                _update(message="generating PNGs and thumbnails in background")
                i_png = 0
                i_th = 0
                while not _cancel.is_set() and (i_png < len(png_queue) or i_th < len(thumb_groups_queue)):
                    if generate_pngs and i_png < len(png_queue):
                        png_path = png_queue[i_png]
                        i_png += 1
                        try:
                            get_or_make_full_png(png_path)
                            _update(pngs=int(_status.get("pngs", 0)) + 1)
                        except Exception:
                            pass
                    if make_thumbs and i_th < len(thumb_groups_queue):
                        parent, base = thumb_groups_queue[i_th]
                        i_th += 1
                        try:
                            get_or_make_composite_thumb(parent, base, size=256)
                            _update(thumbs=int(_status.get("thumbs", 0)) + 1)
                            # Compute readiness based on actual composite thumbs on disk
                            try:
                                ready = sum(1 for p in THUMB_DIR.glob("composite_*_256.jpg") if p.is_file())
                            except Exception:
                                ready = 0
                            if not _status.get("db_ready", False) and ready >= target_ready_groups:
                                _update(db_ready=True, message="thumbnails ready - you can start classifying!")
                        except Exception:
                            pass
                        
        else:
            files = _gather_tasks_files(data_dir, extensions)
            total = len(files)
            _update(total=total, message="ingesting database entries")
            last_update = start_ts
            
            # Phase 1: Database ingestion only
            db_entries_created = 0
            png_queue = []
            thumb_files_queue: List[Path] = []
            
            with next(get_session()) as session:
                for f in files:
                    if _cancel.is_set():
                        _update(message="cancelled")
                        break
                    try:
                        canonical = str(Path(f).resolve())
                        existing = session.exec(
                            DatasetItem.__table__.select().where(DatasetItem.path == canonical)  # type: ignore[attr-defined]
                        ).first()
                        if existing:
                            if generate_pngs:
                                png_queue.append(Path(canonical))
                            if make_thumbs:
                                thumb_files_queue.append(Path(canonical))
                            _update(done=int(_status.get("done", 0)) + 1)
                            continue
                        img = safe_open_image(Path(f))
                        w, h = img.size
                        it = DatasetItem(path=canonical, width=w, height=h)
                        session.add(it)
                        session.commit()
                        _update(ingested=int(_status.get("ingested", 0)) + 1)
                        db_entries_created += 1
                        
                        if generate_pngs:
                            png_queue.append(Path(canonical))
                        if make_thumbs:
                            thumb_files_queue.append(Path(canonical))
                        
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

            # Phase 2: Interleaved PNG & single thumbnail generation (background)
            if ((generate_pngs and png_queue) or (make_thumbs and thumb_files_queue)) and not _cancel.is_set():
                _update(message="generating PNGs and thumbnails in background")
                i_png = 0
                i_th = 0
                while not _cancel.is_set() and (i_png < len(png_queue) or i_th < len(thumb_files_queue)):
                    if generate_pngs and i_png < len(png_queue):
                        png_path = png_queue[i_png]
                        i_png += 1
                        try:
                            get_or_make_full_png(png_path)
                            _update(pngs=int(_status.get("pngs", 0)) + 1)
                        except Exception:
                            pass
                    if make_thumbs and i_th < len(thumb_files_queue):
                        fpath = thumb_files_queue[i_th]
                        i_th += 1
                        try:
                            get_or_make_thumb(fpath, size=256)
                            _update(thumbs=int(_status.get("thumbs", 0)) + 1)
                            # Compute readiness based on composite thumbs only (file-wise path may not reach threshold)
                            try:
                                ready = sum(1 for p in THUMB_DIR.glob("composite_*_256.jpg") if p.is_file())
                            except Exception:
                                ready = 0
                            if not _status.get("db_ready", False) and ready >= 120:
                                _update(db_ready=True, message="thumbnails ready - you can start classifying!")
                        except Exception:
                            pass

        if not _cancel.is_set():
            _update(message="completed")
    finally:
        _cancel.clear()
        # Mark current job done
        _update(running=False, message="completed")
        # Auto-start next job if queued
        try:
            with _state_lock:
                next_params = _queue.pop(0) if _queue else None
                _status["queue_size"] = len(_queue)
            if next_params and not _cancel.is_set():
                _update(message="starting next queued ingest")
                _start_thread(next_params)  # start next job
        except Exception:
            # If starting next job fails, leave queue as-is and report idle
            pass


def start(
    data_dir: Path,
    extensions: set[str],
    by_groups: bool,
    max_groups: Optional[int],
    generate_pngs: bool,
    make_thumbs: bool,
    require_all_roles: Optional[bool],
):
    params: Dict[str, object] = {
        "data_dir": data_dir,
        "extensions": extensions,
        "by_groups": by_groups,
        "max_groups": max_groups,
        "generate_pngs": generate_pngs,
        "make_thumbs": make_thumbs,
        "require_all_roles": require_all_roles,
    }
    if is_running():
        # Enqueue for later
        with _state_lock:
            _queue.append(params)
            _status["queue_size"] = len(_queue)
        return {"queued": True, "position": len(_queue)}
    _start_thread(params)
    return {"queued": False, "position": 0}


def status() -> Dict[str, object]:
    with _state_lock:
        return dict(_status)


def is_running() -> bool:
    with _state_lock:
        return bool(_status.get("running", False))


def cancel():
    _cancel.set()


def shutdown(timeout: float = 5.0):
    """Gracefully stop the worker thread if running.

    - Signals cancel
    - Joins the thread with a timeout to avoid blocking shutdown forever
    - Ensures running flag is cleared
    """
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

