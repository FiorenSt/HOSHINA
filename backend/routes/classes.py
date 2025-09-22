from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from ..db import get_session

router = APIRouter(tags=["classes"])


@router.get("/classes")
def get_classes(session=Depends(get_session)):
    from ..services.classes import get_classes as _svc_get_classes
    return _svc_get_classes(session)


@router.post("/classes")
def set_classes(classes: List[Dict[str, Any]], session=Depends(get_session)):
    from ..services.classes import set_classes as _svc_set_classes
    return _svc_set_classes(classes, session)


