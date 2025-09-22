from fastapi import APIRouter, Body, Depends
from typing import Any, Dict
from ..db import get_session

router = APIRouter(tags=["config"])


@router.get("/config")
def get_config():
    from ..services.config import get_config as _svc_get
    return _svc_get()


@router.get("/grouping")
def get_grouping():
    from ..services.config import get_grouping as _svc_grouping
    return _svc_grouping()


@router.post("/grouping")
def set_grouping(payload: Dict[str, Any] = Body(...)):
    from ..services.config import set_grouping as _svc_set_grouping
    return _svc_set_grouping(payload)


@router.post("/set-data-dir")
def set_data_dir(payload: Dict[str, Any] = Body(...), session=Depends(get_session)):
    from ..services.config import set_data_dir as _svc_set_data_dir
    return _svc_set_data_dir(payload=payload, session=session)


