from fastapi import APIRouter, Body, Depends
from typing import List, Optional
from ..db import get_session

router = APIRouter(tags=["labels"])


@router.post("/label")
def set_label(
    item_ids: List[int] = Body(..., embed=True),
    label: Optional[str] = Body(None),
    unsure: bool = Body(False),
    skip: bool = Body(False),
    user: Optional[str] = Body(None),
    session=Depends(get_session)
):
    from ..services.labels import set_label as _svc_set_label
    return _svc_set_label(session=session, item_ids=item_ids, label=label, unsure=unsure, skip=skip, user=user)


@router.post("/batch-label")
def batch_label_items(
    item_ids: List[int] = Body(..., embed=True),
    label: Optional[str] = Body(None),
    unsure: bool = Body(False),
    skip: bool = Body(False),
    user: Optional[str] = Body(None),
    session=Depends(get_session)
):
    from ..services.labels import batch_label_items as _svc_batch
    return _svc_batch(session=session, item_ids=item_ids, label=label, unsure=unsure, skip=skip, user=user)


