from fastapi import APIRouter, Depends
from ..db import get_session

router = APIRouter(tags=["media"])


@router.get("/thumb/{item_id}")
def get_thumb(item_id: int, size: int = 256, session=Depends(get_session)):
    from ..services.media import get_thumb as _svc_get_thumb
    return _svc_get_thumb(item_id=item_id, size=size, session=session)


@router.get("/triplet-thumb/{item_id}")
def get_triplet_thumb(item_id: int, size: int = 256, session=Depends(get_session)):
    from ..services.media import get_triplet_thumb as _svc_get_triplet_thumb
    return _svc_get_triplet_thumb(item_id=item_id, size=size, session=session)


@router.get("/group-thumb/{item_id}")
def get_group_thumb(item_id: int, size: int = 256, session=Depends(get_session)):
    from ..services.media import get_group_thumb as _svc_get_group_thumb
    return _svc_get_group_thumb(item_id=item_id, size=size, session=session)


@router.get("/file/{item_id}")
def get_file(item_id: int, session=Depends(get_session)):
    from ..services.media import get_file as _svc_get_file
    return _svc_get_file(item_id=item_id, session=session)


@router.get("/triplet/{item_id}")
def get_triplet(item_id: int, session=Depends(get_session)):
    from ..services.media import get_triplet_details as _svc_triplet
    return _svc_triplet(item_id=item_id, session=session)


