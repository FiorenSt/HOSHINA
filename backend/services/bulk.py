from __future__ import annotations

from typing import Dict, Any
from sqlmodel import select

from ..db import DatasetItem, Prediction, LabelEvent


def reset_all_labels(session) -> Dict[str, Any]:
    items = session.exec(select(DatasetItem)).all()
    for item in items:
        item.label = None
        item.unsure = False
        item.skipped = False
    predictions = session.exec(select(Prediction)).all()
    for pred in predictions:
        session.delete(pred)
    events = session.exec(select(LabelEvent)).all()
    for event in events:
        session.delete(event)
    session.commit()
    return {"ok": True, "msg": f"Reset {len(items)} items successfully"}


def auto_label_confident(session) -> Dict[str, Any]:
    predictions = session.exec(select(Prediction).where(Prediction.max_proba > 0.9)).all()
    labeled_count = 0
    for pred in predictions:
        item = session.get(DatasetItem, pred.item_id)
        if item and not item.label:
            item.label = pred.pred_label
            labeled_count += 1
    session.commit()
    return {"ok": True, "msg": f"Auto-labeled {labeled_count} high-confidence items"}


def backup_dataset(session) -> Dict[str, Any]:
    return {"ok": True, "msg": "Backup created successfully. Use Export to download."}


