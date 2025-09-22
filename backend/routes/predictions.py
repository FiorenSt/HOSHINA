from fastapi import APIRouter, Body, Depends
from typing import Any, Dict
from ..db import get_session

router = APIRouter(tags=["predictions"])


@router.post("/predictions/run/start")
def predictions_run_start(
    repredict_all: bool = Body(False),
    batch_size: int = Body(200),
    session=Depends(get_session)
):
    from ..services.predictions import start_prediction_run
    return start_prediction_run(session=session, repredict_all=repredict_all, batch_size=batch_size)


@router.get("/predictions/run/status")
def predictions_run_status() -> Dict[str, Any]:
    from ..services.predictions import get_prediction_run_status
    return get_prediction_run_status()


@router.post("/predictions/run/cancel")
def predictions_run_cancel():
    from ..services.predictions import cancel_prediction_run
    cancel_prediction_run()
    return {"ok": True}


@router.post("/predictions/run")
def predictions_run(
    repredict_all: bool = Body(False),
    limit: Optional[int] = Body(None),
    session=Depends(get_session)
):
    from ..services.predictions import run_predictions_once
    return run_predictions_once(session=session, repredict_all=repredict_all, limit=limit)


