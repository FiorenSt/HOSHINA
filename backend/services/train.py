from __future__ import annotations

from typing import Any, Dict, Optional
import json
import numpy as np
from sqlmodel import select

from ..db import DatasetItem, Prediction, ClassDef
from ..utils import bytes_to_np
from ..training import Trainer
from ..config import STORE_DIR
from .. import tf_training


def train_classic(session) -> Dict[str, Any]:
    labeled = session.exec(select(DatasetItem).where(DatasetItem.label.is_not(None))).all()
    if not labeled:
        return {"ok": False, "msg": "No labeled items yet."}
    emb = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Prediction.__table__)).all()}  # placeholder to avoid import cycle
    from ..db import Embedding
    emb = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
    ids = [it.id for it in labeled if it.id in emb]
    if len(ids) < 2:
        return {"ok": False, "msg": f"Need at least 2 labeled items with embeddings. Found {len(ids)}."}
    X = np.stack([emb[i] for i in ids], axis=0)
    y = [next(it for it in labeled if it.id==i).label for i in ids]
    if len(set(y)) < 2:
        return {"ok": False, "msg": f"Need at least 2 different classes. Found: {set(y)}"}
    trainer = Trainer(STORE_DIR)
    art = trainer.train(X, y)

    all_items = session.exec(select(DatasetItem)).all()
    all_ids = [it.id for it in all_items if it.id in emb]
    Xall = np.stack([emb[i] for i in all_ids], axis=0) if all_ids else np.zeros((0, 16), dtype=np.float32)
    if len(all_ids):
        probs = trainer.predict_proba(Xall)
        classes = art.classes
        for iid, pr in zip(all_ids, probs):
            prob_map = {cls: float(p) for cls, p in zip(classes, pr)}
            pv = np.array(list(prob_map.values()))
            order = np.argsort(-pv)
            pred_lbl = classes[int(order[0])] if len(classes)>0 else None
            if len(pv) >= 2:
                sorted_pv = np.sort(pv)
                maxp = float(sorted_pv[-1])
                margin = float(sorted_pv[-1] - sorted_pv[-2])
            elif len(pv) == 1:
                maxp = float(pv[0])
                margin = float(1.0 - pv[0])
            else:
                maxp = None
                margin = None
            p = Prediction(item_id=iid, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp)
            existing = session.get(Prediction, iid)
            if existing:
                existing.proba_json = p.proba_json
                existing.pred_label = p.pred_label
                existing.margin = p.margin
                existing.max_proba = p.max_proba
                existing.updated_at = p.updated_at
                session.add(existing)
            else:
                session.add(p)
        session.commit()
    return {"ok": True, "classes": art.classes, "labeled": len(ids), "predicted": len(all_ids)}


def train_options() -> Dict[str, Any]:
    return {"ok": True, **tf_training.list_options()}


def train_start(*, model: str, epochs: int, batch_size: int, augment: bool, input_mode: str, class_map: Optional[Dict[str, int]], split_pct: int, split_strategy: str, single_role: str, loss: str, class_weight: Optional[Dict[str, float]]):
    tf_training.start_training(
        model_id=model,
        epochs=epochs,
        batch_size=batch_size,
        augment=augment,
        input_mode=input_mode,
        class_map=class_map or None,
        split_pct=split_pct,
        split_strategy=split_strategy,
        single_role=single_role,
        loss=loss,
        class_weight=class_weight or None,
    )
    return {"ok": True}


def train_status() -> Dict[str, Any]:
    st = tf_training.get_status()
    device = st.device
    gpu = st.gpu
    try:
        import tensorflow as tf  # type: ignore
        if not device:
            device = tf.test.gpu_device_name() or "CPU"
        if not gpu:
            gpus = tf.config.list_physical_devices('GPU')
            gpu = gpus[0].name if gpus else None
    except Exception:
        pass
    return {
        "running": st.running,
        "stage": st.stage,
        "epoch": st.epoch,
        "total_epochs": st.total_epochs,
        "loss": st.loss,
        "acc": st.acc,
        "val_loss": st.val_loss,
        "val_acc": st.val_acc,
        "message": st.message,
        "model": st.model_name,
        "params": st.params,
        "device": device,
        "gpu": gpu,
        "history": st.history,
    }


def train_cancel() -> Dict[str, Any]:
    tf_training.cancel()
    return {"ok": True}


def model_graph_get() -> Dict[str, Any]:
    g = tf_training.get_saved_custom_graph()
    return {"ok": True, "graph": g or {"nodes": [], "edges": [], "input_shape": [224,224,3]}}


def model_graph_save(graph: dict) -> Dict[str, Any]:
    try:
        tf_training.build_model_from_graph(graph, num_classes=2)
    except Exception as ve:
        return {"ok": False, "msg": f"Invalid graph: {ve}"}
    tf_training.save_custom_graph(graph)
    return {"ok": True}


