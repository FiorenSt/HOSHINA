from fastapi import APIRouter, Body, Depends
from ..db import get_session

router = APIRouter(tags=["admin"])


@router.post("/dataset/reset")
def dataset_reset(
    wipe_items: bool = Body(True, embed=True),
    wipe_embeddings: bool = Body(True, embed=True),
    wipe_predictions: bool = Body(True, embed=True),
    wipe_labels: bool = Body(True, embed=True),
    wipe_umap: bool = Body(True, embed=True),
    wipe_caches: bool = Body(True, embed=True),
    wipe_classes: bool = Body(False, embed=True),
    recreate_db: bool = Body(False, embed=True),
    session=Depends(get_session)
):
    from ..services.admin import dataset_reset as _reset
    return _reset(
        session=session,
        wipe_items=wipe_items,
        wipe_embeddings=wipe_embeddings,
        wipe_predictions=wipe_predictions,
        wipe_labels=wipe_labels,
        wipe_umap=wipe_umap,
        wipe_caches=wipe_caches,
        wipe_classes=wipe_classes,
        recreate_db=recreate_db,
    )


