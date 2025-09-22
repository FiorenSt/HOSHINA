from fastapi import APIRouter, Depends
from ..db import get_session

router = APIRouter(tags=["exports"])


@router.get("/export")
def export_all(session=Depends(get_session)):
    from ..services.exports import export_all as _exp
    return _exp(session)


@router.get("/export-labels")
def export_labels(session=Depends(get_session)):
    from ..services.exports import export_labels as _labels
    return _labels(session)


@router.get("/export-model")
def export_model(session=Depends(get_session)):
    from ..services.exports import export_model as _model
    return _model(session)


