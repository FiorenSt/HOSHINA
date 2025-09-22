from __future__ import annotations

from typing import Any, Dict
import io
import json
import zipfile
import datetime as dt
from sqlmodel import select
from fastapi.responses import StreamingResponse

from ..db import DatasetItem, ClassDef
from ..config import STORE_DIR


def export_all(session) -> StreamingResponse:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        rows = session.exec(select(DatasetItem)).all()
        csv_bytes = io.StringIO()
        import csv
        w = csv.writer(csv_bytes)
        w.writerow(["item_id","path","label","unsure","skipped","ingested_at"])
        for r in rows:
            w.writerow([r.id, r.path, r.label or "", int(r.unsure), int(r.skipped), r.ingested_at.isoformat()])
        z.writestr("labels.csv", csv_bytes.getvalue())

        clz = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
        z.writestr("classes.json", json.dumps([{"name": c.name, "key": c.key, "order": c.order} for c in clz], indent=2))

        manifest = {
            "exported_at": dt.datetime.utcnow().isoformat() + "Z",
            "store_dir": str(STORE_DIR),
            "db_path": "store/app.db",
            "embedding_backend": "auto",
            "version": 1
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    mem.seek(0)
    headers = {"Content-Disposition": "attachment; filename=dataset_export.zip"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)


def export_labels(session) -> StreamingResponse:
    rows = session.exec(select(DatasetItem)).all()
    csv_bytes = io.StringIO()
    import csv
    w = csv.writer(csv_bytes)
    w.writerow(["item_id","path","label","unsure","skipped","ingested_at"])
    for r in rows:
        w.writerow([r.id, r.path, r.label or "", int(r.unsure), int(r.skipped), r.ingested_at.isoformat()])
    csv_content = csv_bytes.getvalue()
    headers = {"Content-Disposition": "attachment; filename=labels.csv"}
    return StreamingResponse(io.BytesIO(csv_content.encode('utf-8')), media_type="text/csv", headers=headers)


def export_model(session) -> StreamingResponse:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        model_dir = STORE_DIR / "model"
        if model_dir.exists():
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    z.write(file_path, file_path.relative_to(STORE_DIR))
        clz = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
        z.writestr("classes.json", json.dumps([{"name": c.name, "key": c.key, "order": c.order} for c in clz], indent=2))
        manifest = {
            "exported_at": dt.datetime.utcnow().isoformat() + "Z",
            "store_dir": str(STORE_DIR),
            "model_dir": str(model_dir),
            "version": 1
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    mem.seek(0)
    headers = {"Content-Disposition": "attachment; filename=model_artifacts.zip"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)


