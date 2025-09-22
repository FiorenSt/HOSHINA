from fastapi import APIRouter, Body, Depends
from typing import Any, Dict, Optional
from ..db import get_session

router = APIRouter(tags=["train"])


@router.post("/train")
def train_now(session=Depends(get_session)):
    from ..services.train import train_classic
    return train_classic(session)


@router.get("/train/options")
def train_options():
    from ..services.train import train_options as _opts
    return _opts()


@router.post("/train/start")
def train_start(
    model: str = Body("mobilenet_v2"),
    epochs: int = Body(3),
    batch_size: int = Body(32),
    augment: bool = Body(False),
    input_mode: str = Body("single"),
    class_map: Optional[Dict[str, int]] = Body(default=None),
    split_pct: int = Body(85),
    split_strategy: str = Body("natural"),
    single_role: str = Body("target", description="For single mode: one of target/ref/diff"),
    loss: str = Body("sparse_categorical_crossentropy"),
    class_weight: Optional[Dict[str, float]] = Body(default=None),
):
    from ..services.train import train_start as _start
    return _start(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        augment=augment,
        input_mode=input_mode,
        class_map=class_map,
        split_pct=split_pct,
        split_strategy=split_strategy,
        single_role=single_role,
        loss=loss,
        class_weight=class_weight,
    )


@router.get("/train/status")
def train_status():
    from ..services.train import train_status as _status
    return _status()


@router.post("/train/cancel")
def train_cancel():
    from ..services.train import train_cancel as _cancel
    return _cancel()


@router.get("/model-builder/graph")
def get_model_graph():
    from ..services.train import model_graph_get
    return model_graph_get()


@router.post("/model-builder/graph")
def save_model_graph(graph: Dict[str, Any] = Body(...)):
    from ..services.train import model_graph_save
    return model_graph_save(graph)


