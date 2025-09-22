from __future__ import annotations

from dataclasses import dataclass
from threading import Thread, Lock, Event
from pathlib import Path
from typing import Optional, Dict, List, Any
import json
import datetime as dt
import ast

import numpy as _np
from sqlmodel import select

from ..db import DatasetItem, Embedding, Prediction, ClassDef
from ..utils import bytes_to_np, safe_open_image
from ..training import Trainer
from .. import tf_training
from ..config import STORE_DIR


@dataclass
class _PredStatus:
    running: bool = False
    total_groups: int = 0
    processed_groups: int = 0
    message: str = ""
    repredict_all: bool = False
    batch_size: int = 200
    current_batch: int = 0
    total_batches: int = 0


_pred_status = _PredStatus()
_pred_lock = Lock()
_pred_worker: Optional[Thread] = None
_pred_cancel_evt: Event = Event()


def _set_pred_status(**kwargs):
    global _pred_status
    with _pred_lock:
        for k, v in kwargs.items():
            setattr(_pred_status, k, v)


def _get_pred_status() -> _PredStatus:
    with _pred_lock:
        return _PredStatus(**_pred_status.__dict__)


def start_prediction_run(session, repredict_all: bool = False, batch_size: int = 200) -> Dict[str, Any]:
    global _pred_worker, _pred_cancel_evt
    with _pred_lock:
        if _pred_status.running:
            return {"ok": False, "msg": "Prediction run already in progress"}
        _pred_cancel_evt.clear()
        _pred_status = _PredStatus(
            running=True,
            total_groups=0,
            processed_groups=0,
            message="Starting...",
            repredict_all=bool(repredict_all),
            batch_size=int(max(1, batch_size)),
        )

    def work():
        try:
            # Build groups
            from collections import defaultdict
            items = session.exec(select(DatasetItem)).all()
            pred_rows = session.exec(select(Prediction)).all()
            already_ids = {p.item_id for p in pred_rows}

            groups: dict[tuple[Path, str], list[DatasetItem]] = defaultdict(list)
            for it in items:
                p = Path(it.path)
                name_lower = p.name.lower()
                base = None
                for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                    if name_lower.endswith(suf):
                        base = p.name[: -len(suf)]
                        break
                if base is None:
                    continue
                groups[(p.parent, base)].append(it)

            group_keys = list(groups.keys())
            group_keys.sort(key=lambda k: (str(k[0]), k[1]))
            # Only groups not fully predicted unless repredict_all
            if not _get_pred_status().repredict_all:
                group_keys = [k for k in group_keys if not all(m.id in already_ids for m in groups[k])]

            total = len(group_keys)
            bs = max(1, _get_pred_status().batch_size)
            try:
                import math as _math
                total_batches = int(_math.ceil(total / float(bs))) if total > 0 else 0
            except Exception:
                total_batches = (total + bs - 1) // bs if total > 0 else 0
            _set_pred_status(total_groups=int(total), processed_groups=0, message="Prepared", current_batch=0, total_batches=int(total_batches))
            if total == 0:
                _set_pred_status(running=False, message="No groups to predict")
                return

            # Try classic trainer first
            trainer = Trainer(STORE_DIR)
            trainer.load()
            use_classic = trainer.clf is not None

            # TF fallback setup (lazily load)
            tf_model = None
            preprocess = None
            input_size = (224, 224)
            class_to_idx: Dict[str, int] = {}
            remap: Dict[int, int] = {}

            def ensure_tf_loaded():
                nonlocal tf_model, preprocess, input_size, class_to_idx, remap
                if tf_model is not None:
                    return
                import tensorflow as tf  # type: ignore
                model_path = STORE_DIR / "model" / "tf" / "model.keras"
                if not model_path.exists():
                    raise RuntimeError("No TF model found for fallback")
                st = tf_training.get_status()
                model_id = st.model_name or "mobilenet_v2"
                params = st.params or {}
                input_mode = (params.get("input_mode") or "triplet").lower()
                single_role = params.get("single_role", "target")
                class_map_raw = params.get("class_map")
                try:
                    class_to_idx = ast.literal_eval(class_map_raw) if class_map_raw else None
                except Exception:
                    class_to_idx = None
                if not class_to_idx:
                    names = [c.name for c in session.exec(select(ClassDef).order_by(ClassDef.order)).all()]
                    class_to_idx = {name: i for i, name in enumerate(names)}

                def _preprocess_for_model(mid: str):
                    mid = (mid or "").lower()
                    if mid == "efficientnet_b0":
                        from tensorflow.keras.applications import efficientnet as app  # type: ignore
                        return app.preprocess_input
                    if mid == "resnet50":
                        from tensorflow.keras.applications import resnet50 as app  # type: ignore
                        return app.preprocess_input
                    from tensorflow.keras.applications import mobilenet_v2 as app  # type: ignore
                    return app.preprocess_input

                preprocess = _preprocess_for_model(model_id)
                tf_model = tf.keras.models.load_model(model_path)
                try:
                    ish = tf_model.input_shape
                    input_size = (int(ish[1]), int(ish[2])) if isinstance(ish, (list, tuple)) and len(ish) >= 3 else (224, 224)
                except Exception:
                    input_size = (224, 224)
                # remap targets to dense indices
                unique_targets = sorted(set(int(v) for v in class_to_idx.values()))
                remap = {t: i for i, t in enumerate(unique_targets)}
                # store back convenience fields on status
                return input_mode, single_role

            # Embeddings for classic
            emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()} if use_classic else {}

            bs = max(1, _get_pred_status().batch_size)
            processed = 0
            idx = 0
            batch_index = 0
            while idx < total and not _pred_cancel_evt.is_set():
                batch_keys = group_keys[idx: idx + bs]
                idx += len(batch_keys)
                batch_index += 1
                try:
                    if use_classic:
                        # representatives with embeddings
                        def role_rank(path_str: str) -> int:
                            s = path_str.lower()
                            if s.endswith("_target.fits"): return 0
                            if s.endswith("_ref.fits"): return 1
                            if s.endswith("_diff.fits"): return 2
                            return 3

                        rep_ids: List[int] = []
                        rep_to_members: List[tuple[int, List[DatasetItem]]] = []
                        for key in batch_keys:
                            members = groups[key]
                            members_sorted = sorted(members, key=lambda m: role_rank(m.path))
                            rep = next((m for m in members_sorted if m.id in emb_map), None)
                            if rep is None:
                                continue
                            rep_ids.append(rep.id)
                            rep_to_members.append((rep.id, members))
                        if rep_ids:
                            X = _np.stack([emb_map[i] for i in rep_ids], axis=0)
                            probs = trainer.predict_proba(X)
                            classes = trainer.classes_ or []
                            for (rep_iid, members), pr in zip(rep_to_members, probs):
                                prob_map = {cls: float(p) for cls, p in zip(classes, pr)}
                                pv = _np.array(list(prob_map.values()))
                                order = _np.argsort(-pv) if len(pv) else []
                                pred_lbl = classes[int(order[0])] if len(order) else None
                                if len(pv) >= 2:
                                    sorted_pv = _np.sort(pv)
                                    maxp = float(sorted_pv[-1])
                                    margin = float(sorted_pv[-1] - sorted_pv[-2])
                                elif len(pv) == 1:
                                    maxp = float(pv[0])
                                    margin = float(1.0 - pv[0])
                                else:
                                    maxp = None
                                    margin = None
                                for m in members:
                                    existing = session.get(Prediction, m.id)
                                    if existing:
                                        existing.proba_json = json.dumps(prob_map)
                                        existing.pred_label = pred_lbl
                                        existing.margin = margin
                                        existing.max_proba = maxp
                                        existing.updated_at = dt.datetime.utcnow()
                                        session.add(existing)
                                    else:
                                        session.add(Prediction(item_id=m.id, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp))
                            session.commit()
                            processed += len(rep_ids)
                    else:
                        # TF fallback
                        in_mode, single_role = ensure_tf_loaded()
                        from ..utils import safe_open_image as _safe_open_image  # type: ignore
                        for parent, base in batch_keys:
                            if _pred_cancel_evt.is_set():
                                break
                            members = groups[(parent, base)]
                            # choose candidate
                            candidate = None
                            for role_suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                                cp = parent / f"{base}{role_suf}"
                                if cp.exists():
                                    candidate = cp
                                    break
                            if candidate is None:
                                continue
                            if (in_mode or "triplet") == "triplet":
                                arr = tf_training._compose_triplet_array(candidate, input_size)  # type: ignore[attr-defined]
                            else:
                                role = (single_role or "target").lower()
                                rp = parent / f"{base}_{role}.fits"
                                if rp.exists():
                                    img = _safe_open_image(rp).convert("L").resize(input_size)
                                    g = _np.asarray(img, dtype=_np.float32)
                                    arr = _np.stack([g, g, g], axis=-1)
                                else:
                                    img = _safe_open_image(candidate).convert("RGB").resize(input_size)
                                    arr = _np.asarray(img, dtype=_np.float32)
                            x = preprocess(arr)
                            x = _np.expand_dims(x, axis=0)
                            pr = tf_model.predict(x, verbose=0)[0]
                            # build prob map via class_to_idx and remap
                            prob_map: Dict[str, float] = {}
                            for name, tgt in class_to_idx.items():
                                ridx = remap.get(int(tgt))
                                if ridx is not None and 0 <= ridx < len(pr):
                                    prob_map[name] = float(pr[ridx])
                            pv = _np.array(list(prob_map.values()))
                            if len(pv) > 0:
                                order = _np.argsort(-pv)
                                pred_lbl = list(prob_map.keys())[int(order[0])]
                                margin = float(pv[order[0]] - pv[order[1]]) if len(pv) > 1 else float(1.0 - pv[0])
                                maxp = float(pv[order[0]])
                            else:
                                pred_lbl, margin, maxp = None, None, None
                            for m in members:
                                existing = session.get(Prediction, m.id)
                                if existing:
                                    existing.proba_json = json.dumps(prob_map)
                                    existing.pred_label = pred_lbl
                                    existing.margin = margin
                                    existing.max_proba = maxp
                                    existing.updated_at = dt.datetime.utcnow()
                                    session.add(existing)
                                else:
                                    session.add(Prediction(item_id=m.id, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp))
                            session.commit()
                            processed += 1
                except Exception as e:
                    _set_pred_status(message=f"Batch error: {str(e)}")
                finally:
                    # Update status with batch counters and processed groups
                    _set_pred_status(processed_groups=int(processed), current_batch=int(batch_index), message=f"Batch {batch_index}/{total_batches} • Processed {processed}/{total}")

            if _pred_cancel_evt.is_set():
                _set_pred_status(running=False, message="Cancelled")
            else:
                _set_pred_status(running=False, message="Done")
        except Exception as e:
            _set_pred_status(running=False, message=f"Error: {str(e)}")

    _pred_worker = Thread(target=work, daemon=True)
    _pred_worker.start()
    return {"ok": True}


def get_prediction_run_status() -> Dict[str, Any]:
    st = _get_pred_status()
    return {
        "running": st.running,
        "total_groups": int(st.total_groups),
        "processed_groups": int(st.processed_groups),
        "message": st.message,
        "batch_size": int(st.batch_size),
        "repredict_all": bool(st.repredict_all),
        "current_batch": int(st.current_batch),
        "total_batches": int(st.total_batches),
    }


def cancel_prediction_run() -> None:
    global _pred_cancel_evt
    _pred_cancel_evt.set()


def run_predictions_once(*, session, repredict_all: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    try:
        trainer = Trainer(STORE_DIR)
        trainer.load()
        if trainer.clf is None:
            import tensorflow as tf  # type: ignore
            model_path = STORE_DIR / "model" / "tf" / "model.keras"
            if not model_path.exists():
                return {"ok": False, "msg": "No classic model available and no TF model found. Train a model first."}

            st = tf_training.get_status()
            model_id = st.model_name or "mobilenet_v2"
            params = st.params or {}
            input_mode = params.get("input_mode", "triplet")
            single_role = params.get("single_role", "target")
            class_map_raw = params.get("class_map")
            class_to_idx = None
            if class_map_raw:
                try:
                    class_to_idx = ast.literal_eval(class_map_raw)
                except Exception:
                    class_to_idx = None
            if class_to_idx is None:
                names = [c.name for c in session.exec(select(ClassDef).order_by(ClassDef.order)).all()]
                class_to_idx = {name: i for i, name in enumerate(names)}

            def _preprocess_for_model(mid: str):
                mid = (mid or "").lower()
                if mid == "efficientnet_b0":
                    from tensorflow.keras.applications import efficientnet as app  # type: ignore
                    return app.preprocess_input
                if mid == "resnet50":
                    from tensorflow.keras.applications import resnet50 as app  # type: ignore
                    return app.preprocess_input
                from tensorflow.keras.applications import mobilenet_v2 as app  # type: ignore
                return app.preprocess_input

            preprocess = _preprocess_for_model(model_id)
            model = tf.keras.models.load_model(model_path)
            try:
                ish = model.input_shape
                input_size = (int(ish[1]), int(ish[2])) if isinstance(ish, (list, tuple)) and len(ish) >= 3 else (224, 224)
            except Exception:
                input_size = (224, 224)

            from collections import defaultdict
            items = session.exec(select(DatasetItem)).all()
            pred_rows = session.exec(select(Prediction)).all()
            already_ids = {p.item_id for p in pred_rows}
            groups: dict[tuple[Path, str], list[DatasetItem]] = defaultdict(list)
            for it in items:
                p = Path(it.path)
                name_lower = p.name.lower()
                base = None
                for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                    if name_lower.endswith(suf):
                        base = p.name[: -len(suf)]
                        break
                if base is None:
                    continue
                groups[(p.parent, base)].append(it)
            group_keys = list(groups.keys())
            group_keys.sort(key=lambda k: (str(k[0]), k[1]))
            if not repredict_all:
                group_keys = [k for k in group_keys if not all(m.id in already_ids for m in groups[k])]
            if limit is not None and limit > 0:
                group_keys = group_keys[:limit]
            if not group_keys:
                return {"ok": True, "predicted": 0}

            unique_targets = sorted(set(int(v) for v in class_to_idx.values()))
            remap = {t: i for i, t in enumerate(unique_targets)}

            total = 0
            for parent, base in group_keys:
                candidate = None
                for role_suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                    cp = parent / f"{base}{role_suf}"
                    if cp.exists():
                        candidate = cp
                        break
                if candidate is None:
                    continue
                if (input_mode or "triplet") == "triplet":
                    arr = tf_training._compose_triplet_array(candidate, input_size)  # type: ignore[attr-defined]
                else:
                    role = (single_role or "target").lower()
                    rp = parent / f"{base}_{role}.fits"
                    if rp.exists():
                        img = safe_open_image(rp).convert("L").resize(input_size)
                        g = _np.asarray(img, dtype=_np.float32)
                        arr = _np.stack([g, g, g], axis=-1)
                    else:
                        img = safe_open_image(candidate).convert("RGB").resize(input_size)
                        arr = _np.asarray(img, dtype=_np.float32)
                x = preprocess(arr)
                x = _np.expand_dims(x, axis=0)
                pr = model.predict(x, verbose=0)[0]
                prob_map: Dict[str, float] = {}
                for name, tgt in class_to_idx.items():
                    ridx = remap.get(int(tgt))
                    if ridx is not None and 0 <= ridx < len(pr):
                        prob_map[name] = float(pr[ridx])
                pv = _np.array(list(prob_map.values()))
                if len(pv) > 0:
                    order = _np.argsort(-pv)
                    pred_lbl = list(prob_map.keys())[int(order[0])]
                    margin = float(pv[order[0]] - pv[order[1]]) if len(pv) > 1 else float(1.0 - pv[0])
                    maxp = float(pv[order[0]])
                else:
                    pred_lbl, margin, maxp = None, None, None
                members = groups[(parent, base)]
                for m in members:
                    existing = session.get(Prediction, m.id)
                    if existing:
                        existing.proba_json = json.dumps(prob_map)
                        existing.pred_label = pred_lbl
                        existing.margin = margin
                        existing.max_proba = maxp
                        existing.updated_at = dt.datetime.utcnow()
                        session.add(existing)
                    else:
                        session.add(Prediction(item_id=m.id, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp))
                session.commit()
                total += 1
            return {"ok": True, "predicted": int(total)}
        else:
            # Classic trainer path
            from collections import defaultdict
            items = session.exec(select(DatasetItem)).all()
            pred_rows = session.exec(select(Prediction)).all()
            already_ids = {p.item_id for p in pred_rows}
            groups: dict[tuple[Path, str], list[DatasetItem]] = defaultdict(list)
            for it in items:
                p = Path(it.path)
                name_lower = p.name.lower()
                base = None
                for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                    if name_lower.endswith(suf):
                        base = p.name[: -len(suf)]
                        break
                if base is None:
                    continue
                groups[(p.parent, base)].append(it)
            group_keys = list(groups.keys())
            group_keys.sort(key=lambda k: (str(k[0]), k[1]))
            if not repredict_all:
                group_keys = [k for k in group_keys if not all(m.id in already_ids for m in groups[k])]
            if limit is not None and limit > 0:
                group_keys = group_keys[:limit]
            if not group_keys:
                return {"ok": True, "predicted": 0}
            emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
            total = 0
            for key in group_keys:
                members = groups[key]
                members_sorted = sorted(members, key=lambda m: m.path)
                rep = next((m for m in members_sorted if m.id in emb_map), None)
                if rep is None:
                    continue
                X = _np.stack([emb_map[rep.id]], axis=0)
                probs = trainer.predict_proba(X)[0]
                classes = trainer.classes_ or []
                prob_map = {cls: float(p) for cls, p in zip(classes, probs)}
                pv = _np.array(list(prob_map.values()))
                order = _np.argsort(-pv) if len(pv) else []
                pred_lbl = classes[int(order[0])] if len(order) else None
                if len(pv) >= 2:
                    sorted_pv = _np.sort(pv)
                    maxp = float(sorted_pv[-1])
                    margin = float(sorted_pv[-1] - sorted_pv[-2])
                elif len(pv) == 1:
                    maxp = float(pv[0])
                    margin = float(1.0 - pv[0])
                else:
                    maxp = None
                    margin = None
                for m in members:
                    existing = session.get(Prediction, m.id)
                    if existing:
                        existing.proba_json = json.dumps(prob_map)
                        existing.pred_label = pred_lbl
                        existing.margin = margin
                        existing.max_proba = maxp
                        existing.updated_at = dt.datetime.utcnow()
                        session.add(existing)
                    else:
                        session.add(Prediction(item_id=m.id, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp))
                session.commit()
                total += 1
            return {"ok": True, "predicted": int(total)}
    except Exception as e:
        return {"ok": False, "msg": f"Prediction failed: {str(e)}"}


