from fastapi import APIRouter, Depends, Query, Body
from typing import Optional
from ..db import get_session

router = APIRouter(tags=["analytics"])


@router.get("/predictions/histogram")
def predictions_histogram(
    bins: int = Query(20, ge=5, le=100),
    class_name: str = Query("", description="If provided, class probability; else max_proba"),
    session=Depends(get_session)
):
    from ..services.analytics import predictions_histogram as _hist
    return _hist(session=session, bins=bins, class_name=class_name)


@router.get("/predictions/test-histogram")
def predictions_test_histogram(
    bins: int = Query(20, ge=5, le=100),
    class_name: str = Query("", description="If provided, histogram of this class' probability"),
    session=Depends(get_session)
):
    from ..services.analytics import predictions_test_histogram as _th
    return _th(session=session, bins=bins, class_name=class_name)


@router.get("/predictions/summary")
def predictions_summary(session=Depends(get_session)):
    from ..services.analytics import predictions_summary as _sum
    return _sum(session=session)


@router.post("/predictions/apply-threshold")
def predictions_apply_threshold(
    class_name: str = Body(...),
    negative_class: Optional[str] = Body(None),
    threshold: float = Body(0.5),
    unlabeled_only: bool = Body(True),
    session=Depends(get_session)
):
    from ..services.analytics import apply_threshold as _apply
    return _apply(session=session, class_name=class_name, negative_class=negative_class, threshold=threshold, unlabeled_only=unlabeled_only)


@router.get("/stats")
def get_labeling_stats(session=Depends(get_session)):
    from ..services.stats import get_labeling_stats as _stats
    return _stats(session)


