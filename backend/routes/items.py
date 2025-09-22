from fastapi import APIRouter, Depends, Query
from typing import Optional
from ..db import get_session
from ..config import MAX_PAGE_SIZE

router = APIRouter(tags=["items"])


@router.get("/items")
def list_items(
    queue: str = Query("uncertain", enum=["uncertain","diverse","odd","band","all","certain"]),
    page: int = 1,
    page_size: int = 50,
    prob_low: float = 0.3,
    prob_high: float = 0.8,
    class_name: str = Query("", description="For certain queue: filter by predicted class name (optional)"),
    certain_thr: float = Query(0.9, description="Minimum probability for 'certain' queue"),
    unlabeled_only: bool = True,
    search: str = Query("", description="Search by filename"),
    label_filter: str = Query("", enum=["", "labeled", "unlabeled", "unsure", "skipped"]),
    simple: bool = Query(False, description="If true, skip heavy computations and just paginate"),
    only_ready: bool = Query(False, description="If true, include only groups with composite thumbs ready (size 256)"),
    seed: Optional[int] = Query(None, description="Stable seed for deterministic random order in 'all' queue"),
    sort_pred: str = Query("", enum=["", "asc", "desc"], description="Global sort by predicted value: asc or desc"),
    pos_class: str = Query("", description="Optional positive class name to score by p(class)"),
    session=Depends(get_session)
):
    from ..services.items import list_items as _svc_list_items
    return _svc_list_items(
        session=session,
        queue=queue,
        page=page,
        page_size=page_size,
        prob_low=prob_low,
        prob_high=prob_high,
        class_name=class_name,
        certain_thr=certain_thr,
        unlabeled_only=unlabeled_only,
        search=search,
        label_filter=label_filter,
        simple=simple,
        only_ready=only_ready,
        seed=seed,
        sort_pred=sort_pred,
        pos_class=pos_class,
        MAX_PAGE_SIZE=MAX_PAGE_SIZE,
    )


@router.get("/similar/{item_id}")
def similar_items(item_id: int, k: int = 12, session=Depends(get_session)):
    from ..services.items import similar_items as _svc_similar
    return _svc_similar(item_id=item_id, k=k, session=session)


@router.get("/map")
def get_map_coordinates(session=Depends(get_session)):
    from ..services.items import get_map_coordinates as _svc_map
    return _svc_map(session=session)


