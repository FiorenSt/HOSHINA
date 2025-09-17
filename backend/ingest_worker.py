from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import threading
import time

from .db import get_session, DatasetItem
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from .utils import safe_open_image, compute_file_hash
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


def _insert_items_batch(session, rows: List[Dict[str, object]], dedup_by_hash: bool) -> int:
    """Insert a batch of dataset items using INSERT OR IGNORE on path.

    - Uses SQLite upsert to avoid per-item existence checks
    - Optionally deduplicates by content_hash by skipping rows whose hash already exists
    - Returns number of rows actually inserted
    """
    if not rows:
        return 0

    to_insert: List[Dict[str, object]] = []
    if dedup_by_hash:
        # Build set of candidate hashes present in batch
        hashes = {r["content_hash"] for r in rows if r.get("content_hash")}
        existing: set[str] = set()
        if hashes:
            try:
                q = DatasetItem.__table__.select().where(DatasetItem.content_hash.in_(list(hashes)))  # type: ignore[attr-defined]
                found = session.exec(q).all()
                existing = {getattr(r, "content_hash") for r in found if getattr(r, "content_hash", None)}
            except Exception:
                existing = set()
        for r in rows:
            h = r.get("content_hash")
            if h and h in existing:
                continue
            to_insert.append(r)
    else:
        to_insert = list(rows)

    if not to_insert:
        return 0

    # Use SQLite INSERT OR IGNORE on path uniqueness
    stmt = sqlite_insert(DatasetItem.__table__).prefix_with("OR IGNORE")  # type: ignore[attr-defined]
    try:
        with session.begin():
            res = session.exec(stmt.values(to_insert))
        # SQLAlchemy may return None/-1 for rowcount on SQLite; treat as 0
        inserted = int(getattr(res, "rowcount", 0) or 0)
        if inserted <= 0:
            # Fallback to ORM bulk add if rowcount is unreliable
            try:
                objs = [DatasetItem(**r) for r in to_insert]
                with session.begin():
                    session.add_all(objs)
                inserted = len(objs)
            except Exception:
                inserted = 0
    except IntegrityError:
        # Fallback: try smaller chunks if a batch hits constraints
        inserted = 0
        chunk = max(50, len(to_insert) // 4)
        for i in range(0, len(to_insert), chunk):
            sub = to_insert[i:i + chunk]
            try:
                with session.begin():
                    res = session.exec(stmt.values(sub))
                inserted += int(getattr(res, "rowcount", 0) or 0)
            except Exception:
                # Skip problematic rows individually; fallback to ORM
                for r in sub:
                    try:
                        with session.begin():
                            res = session.exec(stmt.values(r))
                        inserted += int(getattr(res, "rowcount", 0) or 0)
                        if int(getattr(res, "rowcount", 0) or 0) <= 0:
                            with session.begin():
                                session.add(DatasetItem(**r))
                            inserted += 1
                    except Exception:
                        try:
                            with session.begin():
                                session.add(DatasetItem(**r))
                            inserted += 1
                        except Exception:
                            pass
    except Exception:
        # As a final fallback, try inserting individually
        inserted = 0
        for r in to_insert:
            try:
                with session.begin():
                    res = session.exec(stmt.values(r))
                rc = int(getattr(res, "rowcount", 0) or 0)
                if rc <= 0:
                    with session.begin():
                        session.add(DatasetItem(**r))
                    rc = 1
                inserted += rc
            except Exception:
                try:
                    with session.begin():
                        session.add(DatasetItem(**r))
                    inserted += 1
                except Exception:
                    pass

    return inserted

def _start_thread(params: Dict[str, object]) -> None:
    global _thread
    _thread = threading.Thread(
        target=_worker,
        args=(
            params["data_dir"],
            params["extensions"],
            params["by_groups"],
            params["max_groups"],
            params["make_thumbs"],
            params["require_all_roles"],
            params["batch_size"],
            params["skip_hash"],
            params["backfill_hashes"],
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
    make_thumbs: bool,
    require_all_roles: Optional[bool],
    batch_size: int,
    skip_hash: bool,
    backfill_hashes: bool,
):
    start_ts = time.time()
    _update(
        running=True,
        data_dir=str(data_dir),
        by_groups=by_groups,
        require_all_roles=require_all_roles,
        done=0,
        ingested=0,
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
            
            # Phase 1: Database ingestion only (SQLModel ORM add_all commits)
            db_entries_created = 0
            groups_processed = 0
            thumb_groups_queue: List[tuple[Path, str]] = []  # legacy; no longer used
            to_add: List[DatasetItem] = []
            
            with next(get_session()) as session:
                for parent, roles in groups:
                    if _cancel.is_set():
                        _update(message="cancelled")
                        break
                        
                    for rp in roles.values():
                        try:
                            canonical = str(Path(rp).resolve())
                            # Compute content hash if deduplication by hash is enabled
                            if skip_hash:
                                c_hash = None
                            else:
                                try:
                                    c_hash = compute_file_hash(Path(rp))
                                except Exception:
                                    c_hash = None

                            img = safe_open_image(Path(rp))
                            w, h = img.size
                            to_add.append(DatasetItem(path=canonical, width=w, height=h, content_hash=c_hash))

                            # Flush batch if needed
                            if len(to_add) >= int(max(1, batch_size)):
                                try:
                                    with session.begin():
                                        session.add_all(to_add)
                                    inserted = len(to_add)
                                except IntegrityError:
                                    inserted = 0
                                    # Insert individually on conflicts
                                    for obj in to_add:
                                        try:
                                            with session.begin():
                                                session.add(obj)
                                            inserted += 1
                                        except Exception:
                                            pass
                                db_entries_created += inserted
                                if inserted:
                                    _update(ingested=int(_status.get("ingested", 0)) + int(inserted))
                                to_add.clear()
                        except Exception:
                            _update(errors=int(_status.get("errors", 0)) + 1)
                            continue
                    
                    # Update progress after processing each group (triplet)
                    groups_processed += 1
                    # Thumbnail generation removed
                    
                    # Flush any remaining rows at group boundary (helps steady progress)
                    if to_add:
                        try:
                            try:
                                with session.begin():
                                    session.add_all(to_add)
                                inserted = len(to_add)
                            except IntegrityError:
                                inserted = 0
                                for obj in to_add:
                                    try:
                                        with session.begin():
                                            session.add(obj)
                                        inserted += 1
                                    except Exception:
                                        pass
                            db_entries_created += inserted
                            if inserted:
                                _update(ingested=int(_status.get("ingested", 0)) + int(inserted))
                        except Exception:
                            _update(errors=int(_status.get("errors", 0)) + 1)
                        finally:
                            to_add.clear()

                    now = time.time()
                    if now - last_update > 0.5:
                        rate = groups_processed / max(1e-6, (now - start_ts))
                        remaining = max(0, total_groups - groups_processed)
                        eta = remaining / max(1e-6, rate)
                        _update(done=groups_processed, eta_sec=int(eta))
                        last_update = now
                    else:
                        _update(done=groups_processed)
            
            # Phase 2 removed: thumbnails are generated on-the-fly
                        
        else:
            files = _gather_tasks_files(data_dir, extensions)
            total = len(files)
            _update(total=total, message="ingesting database entries")
            last_update = start_ts
            
            # Phase 1: Database ingestion only (SQLModel ORM add_all commits)
            db_entries_created = 0
            thumb_files_queue: List[Path] = []  # legacy; no longer used
            to_add: List[DatasetItem] = []
            
            with next(get_session()) as session:
                for f in files:
                    if _cancel.is_set():
                        _update(message="cancelled")
                        break
                    try:
                        canonical = str(Path(f).resolve())
                        # Compute content hash if deduplication by hash is enabled
                        if skip_hash:
                            c_hash = None
                        else:
                            try:
                                c_hash = compute_file_hash(Path(f))
                            except Exception:
                                c_hash = None

                        img = safe_open_image(Path(f))
                        w, h = img.size

                        to_add.append(DatasetItem(path=canonical, width=w, height=h, content_hash=c_hash))

                        # Thumbnail generation removed

                        # Flush if batch threshold reached
                        if len(to_add) >= int(max(1, batch_size)):
                            try:
                                with session.begin():
                                    session.add_all(to_add)
                                inserted = len(to_add)
                            except IntegrityError:
                                inserted = 0
                                for obj in to_add:
                                    try:
                                        with session.begin():
                                            session.add(obj)
                                        inserted += 1
                                    except Exception:
                                        pass
                            db_entries_created += inserted
                            if inserted:
                                _update(ingested=int(_status.get("ingested", 0)) + int(inserted))
                            to_add.clear()

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

            # Flush any remaining rows
            if to_add:
                try:
                    try:
                        with next(get_session()) as session_flush:
                            with session_flush.begin():
                                session_flush.add_all(to_add)
                        inserted = len(to_add)
                    except IntegrityError:
                        inserted = 0
                        with next(get_session()) as session_flush:
                            for obj in to_add:
                                try:
                                    with session_flush.begin():
                                        session_flush.add(obj)
                                    inserted += 1
                                except Exception:
                                    pass
                    db_entries_created += inserted
                    if inserted:
                        _update(ingested=int(_status.get("ingested", 0)) + int(inserted))
                except Exception:
                    _update(errors=int(_status.get("errors", 0)) + 1)
                finally:
                    to_add.clear()

            # Phase 2 removed: thumbnails are generated on-the-fly

        # Optional background backfill of missing content hashes
        if (skip_hash and backfill_hashes) and not _cancel.is_set():
            try:
                _update(message="backfilling content hashes")
                with next(get_session()) as session:
                    # Backfill in chunks to avoid long transactions
                    while not _cancel.is_set():
                        missing = session.exec(
                            select(DatasetItem).where(DatasetItem.content_hash.is_(None)).limit(100)
                        ).all()
                        if not missing:
                            break
                        for row in missing:
                            try:
                                row.content_hash = compute_file_hash(Path(row.path))
                                session.add(row)
                            except Exception:
                                pass
                        try:
                            session.commit()
                        except Exception:
                            session.rollback()
                            break
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
    make_thumbs: bool,
    require_all_roles: Optional[bool],
    batch_size: int = 500,
    skip_hash: bool = False,
    backfill_hashes: bool = False,
):
    params: Dict[str, object] = {
        "data_dir": data_dir,
        "extensions": extensions,
        "by_groups": by_groups,
        "max_groups": max_groups,
        "make_thumbs": make_thumbs,
        "require_all_roles": require_all_roles,
        "batch_size": int(max(1, batch_size)),
        "skip_hash": bool(skip_hash),
        "backfill_hashes": bool(backfill_hashes),
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

