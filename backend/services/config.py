from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from .. import config as cfg
from ..config import STORE_DIR
from ..db import DatasetItem
from ..grouping import load_config as load_grouping, match_role_and_base, resolve_existing_role_file
from ..utils import compute_file_hash, safe_open_image, get_or_make_full_png
from ..ingest_state import get_ingest_mode, set_ingest_mode


def get_config() -> Dict[str, str]:
    return {"data_dir": str(cfg.DATA_DIR), "store_dir": str(STORE_DIR)}


def get_grouping() -> Dict[str, Any]:
    from ..grouping import get_config_dict
    return {"ok": True, **get_config_dict()}


def set_grouping(payload: Dict[str, Any]) -> Dict[str, Any]:
    from ..grouping import set_config
    try:
        new_cfg = set_config(payload)
        return {"ok": True, **new_cfg}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def set_data_dir(
    *,
    payload: Dict[str, Any],
    session,
) -> Dict[str, Any]:
    new_dir = payload.get("data_dir", "").strip()
    if not new_dir:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="data_dir is required")

    p = cfg.set_data_dir(new_dir)

    do_ingest = bool(payload.get("ingest", False))
    exts_str = payload.get("extensions") or ".jpg,.jpeg,.png,.tif,.tiff,.fits,.fit,.fits.fz"
    extensions: Set[str] = set([e.strip().lower() for e in exts_str.split(",") if e.strip()])
    by_groups = bool(payload.get("by_groups", False))
    max_groups_raw = payload.get("max_groups")
    try:
        max_groups = int(max_groups_raw) if max_groups_raw is not None else None
    except Exception:
        max_groups = None
    generate_pngs = bool(payload.get("generate_pngs", False))
    req_all_override = payload.get("require_all_roles")

    ingested_count = 0
    if do_ingest:
        data_dir = p.resolve()
        # Enforce cross-run ingestion mode consistency
        previous_mode = get_ingest_mode()
        requested_grouped = bool(by_groups or (max_groups is not None))
        if previous_mode == "triplet" and not requested_grouped:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Ingestion mode locked to triplet/groups based on first run. Use grouped ingest (require_all_roles/max_groups) to continue.")

        if requested_grouped:
            from collections import defaultdict
            gcfg = load_grouping()
            require_all = gcfg.require_all_roles if req_all_override is None else bool(req_all_override)
            files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
            keys = []
            seen = set()
            for f in files:
                _, base = match_role_and_base(f, gcfg)
                key = (f.parent, base)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
            keys.sort(key=lambda k: (str(k[0]), k[1]))

            def role_files_for(parent, base):
                roles = {}
                for role in gcfg.roles:
                    rp = resolve_existing_role_file(parent, base, role, gcfg)
                    if rp is not None:
                        roles[role] = rp
                return roles

            eligible = []
            for parent, base in keys:
                roles = role_files_for(parent, base)
                if require_all and not all(r in roles for r in gcfg.roles):
                    continue
                if not roles:
                    continue
                has_new_files = False
                for rp in roles.values():
                    canonical_rp = str(Path(rp).resolve())
                    existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_rp)).first()
                    if not existing:
                        has_new_files = True
                        break
                if has_new_files:
                    eligible.append((parent, base, roles))

            if max_groups is not None:
                eligible = eligible[:max_groups]

            for parent, base, roles in eligible:
                for rp in roles.values():
                    canonical_rp = str(Path(rp).resolve())
                    existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_rp)).first()
                    if existing:
                        try:
                            if getattr(existing, "content_hash", None) is None:
                                existing.content_hash = compute_file_hash(Path(canonical_rp))
                                session.add(existing)
                                session.commit()
                        except Exception:
                            pass
                        continue
                    try:
                        c_hash = None
                        try:
                            c_hash = compute_file_hash(Path(rp))
                            dup = session.exec(select(DatasetItem).where(DatasetItem.content_hash == c_hash)).first()
                            if dup:
                                if generate_pngs:
                                    try:
                                        get_or_make_full_png(rp)
                                    except Exception:
                                        pass
                                continue
                        except Exception:
                            c_hash = None

                        img = safe_open_image(rp)
                        w, h = img.size
                    except Exception:
                        continue
                    it = DatasetItem(path=canonical_rp, width=w, height=h, content_hash=c_hash)
                    session.add(it)
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        continue
                    if generate_pngs:
                        try:
                            get_or_make_full_png(rp)
                        except Exception:
                            pass
                    ingested_count += 1

            total_groups_found = len(keys)
            new_groups_found = len(eligible)
            # Persist chosen mode as grouped/triplet
            set_ingest_mode("triplet")
        else:
            for f in data_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in extensions:
                    continue
                canonical_f = str(f.resolve())
                existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_f)).first()
                if existing:
                    try:
                        if getattr(existing, "content_hash", None) is None:
                            existing.content_hash = compute_file_hash(Path(canonical_f))
                            session.add(existing)
                            session.commit()
                    except Exception:
                        pass
                    continue
                try:
                    c_hash = None
                    try:
                        c_hash = compute_file_hash(Path(f))
                        dup = session.exec(select(DatasetItem).where(DatasetItem.content_hash == c_hash)).first()
                        if dup:
                            if generate_pngs:
                                try:
                                    get_or_make_full_png(f)
                                except Exception:
                                    pass
                            continue
                    except Exception:
                        c_hash = None

                    img = safe_open_image(f)
                    w, h = img.size
                except Exception:
                    continue
                it = DatasetItem(path=canonical_f, width=w, height=h, content_hash=c_hash)
                session.add(it)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    continue
                if generate_pngs:
                    try:
                        get_or_make_full_png(f)
                    except Exception:
                        pass
                ingested_count += 1
            # Persist chosen mode as files if nothing grouped was requested
            set_ingest_mode("files")

    response: Dict[str, Any] = {"ok": True, "data_dir": str(p), "ingested": ingested_count}
    if do_ingest and (by_groups or (max_groups is not None)):
        if 'total_groups_found' in locals():
            response.update({
                "total_groups_found": total_groups_found,
                "new_groups_processed": new_groups_found,
                "files_ingested": ingested_count
            })
    return response


