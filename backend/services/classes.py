from __future__ import annotations

from typing import List, Dict, Any
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from sqlmodel import select
from fastapi.responses import JSONResponse

from ..db import ClassDef


def get_classes(session) -> List[Dict[str, Any]]:
    rows = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
    return [{"id": r.id, "name": r.name, "key": r.key, "order": r.order} for r in rows]


def set_classes(classes: List[Dict[str, Any]], session):
    try:
        session.exec(text('DELETE FROM classdef'))
        session.commit()
        for i, c in enumerate(classes):
            name = (c.get("name") or "").strip()
            if not name:
                continue
            cd = ClassDef(name=name, key=c.get("key"), order=i)
            session.add(cd)
        session.commit()
        return {"ok": True}
    except IntegrityError:
        session.rollback()
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Duplicate class names are not allowed"})
    except Exception as e:
        session.rollback()
        raise e


