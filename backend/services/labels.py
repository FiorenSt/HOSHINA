from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from ..db import DatasetItem, LabelEvent
from ..utils import delete_thumbnails_for_path


def set_label(*, session, item_ids: List[int], label: Optional[str], unsure: bool, skip: bool, user: Optional[str]) -> Dict[str, Any]:
    deleted = 0
    for iid in item_ids:
        it = session.get(DatasetItem, iid)
        if not it:
            continue
        prev = it.label
        it.label = None if skip else label
        it.unsure = unsure
        it.skipped = skip
        ev = LabelEvent(item_id=iid, prev_label=prev, new_label=it.label, unsure=unsure, skipped=skip, user=user)
        session.add(ev)
        session.add(it)
        try:
            deleted += delete_thumbnails_for_path(Path(it.path), sizes=[256])
        except Exception:
            pass
    session.commit()
    return {"ok": True, "deleted_thumbs": int(deleted)}


def batch_label_items(*, session, item_ids: List[int], label: Optional[str], unsure: bool, skip: bool, user: Optional[str]) -> Dict[str, Any]:
    if not item_ids:
        return {"ok": False, "msg": "No items provided"}
    success_count = 0
    deleted = 0
    for iid in item_ids:
        it = session.get(DatasetItem, iid)
        if not it:
            continue
        prev = it.label
        it.label = None if skip else label
        it.unsure = unsure
        it.skipped = skip
        ev = LabelEvent(item_id=iid, prev_label=prev, new_label=it.label, unsure=unsure, skipped=skip, user=user)
        session.add(ev)
        session.add(it)
        success_count += 1
        try:
            deleted += delete_thumbnails_for_path(Path(it.path), sizes=[256])
        except Exception:
            pass
    session.commit()
    return {"ok": True, "labeled_count": success_count, "deleted_thumbs": int(deleted)}


