from __future__ import annotations

from typing import Dict, Any
from sqlmodel import select
from sqlalchemy import text

from ..db import DatasetItem, Embedding, Prediction, LabelEvent, ClassDef, UMAPCoords, recreate_database


def dataset_reset(
    *,
    session,
    wipe_items: bool = True,
    wipe_embeddings: bool = True,
    wipe_predictions: bool = True,
    wipe_labels: bool = True,
    wipe_umap: bool = True,
    wipe_caches: bool = True,
    wipe_classes: bool = False,
    recreate_db: bool = False,
) -> Dict[str, Any]:
    try:
        if recreate_db:
            try:
                recreate_database()
                return {"ok": True, "message": "Database file recreated"}
            except Exception as e:
                # fall back to table wipe
                recreate_db = False

        if not recreate_db:
            # Disable foreign key constraints for SQLite during bulk deletes
            try:
                session.exec(text("PRAGMA foreign_keys = OFF"))
            except Exception:
                pass

            to_wipe: list[tuple[str, Any]] = []
            if wipe_predictions:
                to_wipe.append(("predictions", Prediction))
            if wipe_embeddings:
                to_wipe.append(("embeddings", Embedding))
            if wipe_umap:
                to_wipe.append(("umap_coords", UMAPCoords))
            if wipe_labels:
                to_wipe.append(("label_events", LabelEvent))
            if wipe_items:
                to_wipe.append(("dataset_items", DatasetItem))
            if wipe_classes:
                to_wipe.append(("class_defs", ClassDef))

            for name, model in to_wipe:
                try:
                    session.exec(model.__table__.delete())  # type: ignore[attr-defined]
                except Exception:
                    pass

            try:
                session.exec(text("PRAGMA foreign_keys = ON"))
            except Exception:
                pass

            session.commit()

        # Wipe caches placeholder (no persistent thumbs in on-the-fly mode)
        return {"ok": True, "message": "Reset complete"}
    except Exception as e:
        return {"ok": False, "msg": f"Reset failed: {str(e)}"}


