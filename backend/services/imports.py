from __future__ import annotations

from typing import Dict, Any
import io
import csv
from pathlib import Path
from sqlmodel import select

from ..db import DatasetItem
from ..utils import safe_open_image
from ..grouping import load_config as load_grouping, match_role_and_base


async def import_labels(file, session) -> Dict[str, Any]:
    if not file.filename.endswith('.csv'):
        return {"ok": False, "msg": "File must be a CSV file"}
    content = await file.read()
    try:
        s = io.StringIO(content.decode('utf-8'))
    except Exception:
        return {"ok": False, "msg": "Invalid encoding. Use UTF-8."}
    rdr = csv.DictReader(s)
    if set([c.lower() for c in rdr.fieldnames or []]) != set(["item_id","path","label","unsure","skipped","ingested_at"]):
        return {"ok": False, "msg": "CSV must have columns: item_id,path,label,unsure,skipped,ingested_at"}
    updated = 0
    for row in rdr:
        try:
            iid = int(row["item_id"]) if row.get("item_id") else None
        except Exception:
            iid = None
        if iid:
            it = session.get(DatasetItem, iid)
            if not it:
                continue
        else:
            path = row.get("path") or ""
            if not path:
                continue
            it = None
        if it:
            it.label = (row.get("label") or None) or None
            try:
                it.unsure = bool(int(row.get("unsure") or 0))
                it.skipped = bool(int(row.get("skipped") or 0))
            except Exception:
                pass
            updated += 1
    session.commit()
    return {"ok": True, "updated": updated}


def import_folder_labeled(*, folder_path: str, class_name: str, make_thumbs: bool, group_require_all, max_groups, session) -> Dict[str, Any]:
    from ..utils import get_or_make_thumb
    p = Path(folder_path).expanduser()
    if not p.exists() or not p.is_dir():
        return {"ok": False, "msg": f"Folder not found: {folder_path}"}
    allowed = {".jpg",".jpeg",".png",".tif",".tiff",".fits",".fit",".fits.fz"}
    ingested = labeled = triplet_built = skipped_groups = processed_groups = 0
    grouped_mode = group_require_all is not None or max_groups is not None
    if grouped_mode:
        from collections import defaultdict
        groups: dict[tuple[Path, str], dict[str, Path]] = defaultdict(dict)
        gcfg = load_grouping()
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in allowed:
                role, base = match_role_and_base(f, gcfg)
                if role and base:
                    groups[(f.parent, base)][role] = f
        for (parent, base), roles in groups.items():
            if max_groups is not None and processed_groups >= int(max_groups):
                break
            must_have_all = gcfg.require_all_roles if group_require_all is None else bool(group_require_all)
            if must_have_all and not all(r in roles for r in gcfg.roles):
                skipped_groups += 1
                continue
            files = [roles[r] for r in gcfg.roles if r in roles] if gcfg.roles else list(roles.values())
            for f in files:
                canonical = str(Path(f).resolve())
                existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical)).first()
                if existing:
                    if existing.label != class_name:
                        existing.label = class_name
                        session.add(existing)
                        labeled += 1
                    if make_thumbs:
                        try:
                            get_or_make_thumb(Path(canonical), size=256)
                        except Exception:
                            pass
                    continue
                try:
                    img = safe_open_image(f)
                    w, h = img.size
                except Exception:
                    continue
                it = DatasetItem(path=canonical, width=w, height=h, label=class_name)
                session.add(it)
                session.commit()
                ingested += 1
                labeled += 1
                if make_thumbs:
                    try:
                        get_or_make_thumb(Path(canonical), size=256)
                    except Exception:
                        pass
            processed_groups += 1
    else:
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in allowed:
                continue
            canonical = str(Path(f).resolve())
            existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical)).first()
            if existing:
                if existing.label != class_name:
                    existing.label = class_name
                    session.add(existing)
                    labeled += 1
                if make_thumbs:
                    try:
                        get_or_make_thumb(Path(canonical), size=256)
                    except Exception:
                        pass
                continue
            try:
                img = safe_open_image(f)
                w, h = img.size
            except Exception:
                continue
            it = DatasetItem(path=canonical, width=w, height=h, label=class_name)
            session.add(it)
            session.commit()
            ingested += 1
            labeled += 1
            if make_thumbs:
                try:
                    get_or_make_thumb(Path(canonical), size=256)
                except Exception:
                    pass
    session.commit()
    return {"ok": True, "ingested": ingested, "labeled": labeled, "triplet_groups": triplet_built, "skipped_groups": skipped_groups, "processed_groups": processed_groups}


