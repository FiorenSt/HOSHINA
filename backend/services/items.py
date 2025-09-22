from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np

from sqlmodel import select

from ..db import DatasetItem, Embedding, Prediction, UMAPCoords
from ..utils import bytes_to_np, farthest_first
from ..grouping import load_config, match_role_and_base
import datetime as dt


def list_items(
    *,
    session,
    queue: str,
    page: int,
    page_size: int,
    prob_low: float,
    prob_high: float,
    class_name: str,
    certain_thr: float,
    unlabeled_only: bool,
    search: str,
    label_filter: str,
    simple: bool,
    only_ready: bool,
    seed: Optional[int],
    sort_pred: str,
    pos_class: str,
    MAX_PAGE_SIZE: int,
) -> Dict[str, Any]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    stmt = select(DatasetItem)
    if search:
        stmt = stmt.where(DatasetItem.path.contains(search))
    items = session.exec(stmt).all()
    from ..grouping import group_items
    groups = group_items(items, get_path=lambda x: Path(x.path))
    if not groups:
        return {"total": 0, "page": int(page), "page_size": int(page_size), "items": []}

    cfg = load_config()

    def role_rank(path_str: str) -> int:
        pth = Path(path_str)
        role, _ = match_role_and_base(pth, cfg)
        if role is None:
            return 999
        try:
            return cfg.roles.index(role)
        except ValueError:
            return 999

    group_keys = list(groups.keys())
    group_keys.sort(key=lambda k: (str(k[0]), k[1]))

    rep_ids: list[int] = []
    rep_to_members: dict[int, list[DatasetItem]] = {}
    rep_to_groupkey: dict[int, tuple[Path, str]] = {}
    for key in group_keys:
        members = groups[key]
        if unlabeled_only and not label_filter:
            if any(m.label is not None or m.skipped for m in members):
                continue
        elif label_filter:
            if label_filter == "labeled":
                if not all(m.label is not None for m in members):
                    continue
            elif label_filter == "unlabeled":
                if not all(m.label is None and not m.skipped and not m.unsure for m in members):
                    continue
            elif label_filter == "unsure":
                if not any(m.unsure for m in members):
                    continue
            elif label_filter == "skipped":
                if not any(m.skipped for m in members):
                    continue
        members_sorted = sorted(members, key=lambda m: role_rank(m.path))
        rep = members_sorted[0]
        rep_ids.append(rep.id)
        rep_to_members[rep.id] = members_sorted
        rep_to_groupkey[rep.id] = key

        if only_ready:
            pass

    emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
    preds_map = {}
    for p in session.exec(select(Prediction)).all():
        preds_map[p.item_id] = json.loads(p.proba_json), p.pred_label, p.margin, p.max_proba

    ordered_rep_ids: list[int] = rep_ids[:]
    try:
        if sort_pred in ("asc", "desc"):
            def score_for_rep(rid: int) -> float:
                try:
                    probs_map, pred_lbl, _, maxp = preds_map.get(rid, ({}, None, None, None))
                    if pos_class:
                        try:
                            v = probs_map.get(pos_class)
                            if v is not None:
                                return float(v)
                        except Exception:
                            pass
                    if maxp is not None:
                        return float(maxp)
                    if pred_lbl is not None and isinstance(probs_map, dict):
                        v = probs_map.get(pred_lbl)
                        if v is not None:
                            return float(v)
                except Exception:
                    pass
                return float("nan")

            def key_fn(rid: int) -> float:
                s = score_for_rep(rid)
                if sort_pred == "asc":
                    return s if np.isfinite(s) else float("inf")
                else:
                    return (-s) if np.isfinite(s) else float("inf")

            ordered_rep_ids = sorted(rep_ids, key=key_fn)
        elif queue == "all":
            if seed is not None:
                import hashlib
                def order_key(rid: int) -> int:
                    parent, base = rep_to_groupkey.get(rid, (Path(""), ""))
                    s = f"{parent}|{base}|{seed}".encode("utf-8")
                    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")
                ordered_rep_ids = sorted(rep_ids, key=order_key)
            else:
                if len(rep_ids) > 1:
                    perm = np.random.permutation(len(rep_ids))
                    ordered_rep_ids = [rep_ids[i] for i in perm]
                else:
                    ordered_rep_ids = rep_ids[:]
        elif queue == "uncertain":
            margins: list[float] = []
            for i in rep_ids:
                if i in preds_map:
                    probs, _, m, _ = preds_map[i]
                    if m is None:
                        pv = np.array(list(probs.values()))
                        m = float(np.sort(pv)[-1] - np.sort(pv)[-2]) if len(pv) >= 2 else 1.0
                else:
                    m = 1.0
                margins.append(m)
            order = np.argsort(margins)
            ordered_rep_ids = [rep_ids[i] for i in order]
        elif queue == "diverse":
            valid_ids = [i for i in rep_ids if i in emb_map]
            if valid_ids:
                embs = [emb_map[i] for i in valid_ids]
                shapes = [e.shape for e in embs]
                if len(set(shapes)) == 1:
                    X = np.stack(embs, axis=0)
                    ids_for_X = valid_ids
                else:
                    from collections import Counter
                    shape_counts = Counter(shapes)
                    target_shape = shape_counts.most_common(1)[0][0]
                    ids_for_X, X_list = [], []
                    for i, e in zip(valid_ids, embs):
                        if e.shape == target_shape:
                            ids_for_X.append(i)
                            X_list.append(e)
                    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, target_shape[0]), dtype=np.float32)
                if len(ids_for_X):
                    idx = farthest_first(X, k=min(len(ids_for_X), max(page_size*page, 1)))
                    diverse_ids = [ids_for_X[i] for i in idx]
                    tail = [i for i in rep_ids if i not in diverse_ids]
                    ordered_rep_ids = diverse_ids + tail
        elif queue == "odd":
            valid_ids = [i for i in rep_ids if i in emb_map]
            if len(valid_ids) > 0:
                embs = np.stack([emb_map[i] for i in valid_ids], axis=0)
                from sklearn.neighbors import LocalOutlierFactor
                lof = LocalOutlierFactor(n_neighbors=min(20, len(valid_ids)-1), novelty=False)
                lof.fit(embs)
                scores = -lof.negative_outlier_factor_
                order = np.argsort(-scores)
                odd_ids = [valid_ids[i] for i in order]
                tail = [i for i in rep_ids if i not in odd_ids]
                ordered_rep_ids = odd_ids + tail
        elif queue == "band":
            band_ids: list[int] = []
            band_scores: list[float] = []
            for i in rep_ids:
                if i in preds_map and preds_map[i][3] is not None:
                    maxp = float(preds_map[i][3])
                    if prob_low <= maxp <= prob_high:
                        band_ids.append(i)
                        band_scores.append(abs(0.5 - maxp))
            order = np.argsort(band_scores) if band_scores else []
            ordered_rep_ids = [band_ids[i] for i in order] + [i for i in rep_ids if i not in band_ids]
        elif queue == "certain":
            high_ids: list[int] = []
            high_scores: list[float] = []
            low_tail: list[int] = []
            for i in rep_ids:
                if i in preds_map:
                    probs_map, pred_lbl, _, maxp = preds_map[i]
                    if maxp is not None:
                        if class_name:
                            try:
                                pmap = probs_map if isinstance(probs_map, dict) else {}
                                pv = float(pmap.get(class_name, -1.0)) if class_name in pmap else -1.0
                            except Exception:
                                pv = -1.0
                            if pv >= float(certain_thr):
                                high_ids.append(i)
                                high_scores.append(-pv)
                            else:
                                low_tail.append(i)
                        else:
                            if float(maxp) >= float(certain_thr):
                                high_ids.append(i)
                                high_scores.append(-float(maxp))
                            else:
                                low_tail.append(i)
                    else:
                        low_tail.append(i)
                else:
                    low_tail.append(i)
            order = np.argsort(high_scores) if high_scores else []
            ordered_rep_ids = [high_ids[i] for i in order] + low_tail
    except Exception:
        ordered_rep_ids = rep_ids[:]

    total_groups = len(ordered_rep_ids)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_rep_ids = ordered_rep_ids[start:end]

    payload: list[dict[str, Any]] = []
    for rep_id in page_rep_ids:
        members = rep_to_members.get(rep_id, [])
        for it in members:
            probs, pred_lbl, m, maxp = preds_map.get(it.id, ({}, None, None, None))
            payload.append({
                "id": it.id,
                "path": it.path,
                "thumb": f"/api/group-thumb/{it.id}",
                "label": it.label,
                "unsure": it.unsure,
                "pred_label": pred_lbl,
                "probs": probs,
                "margin": m,
                "max_proba": maxp,
            })

    return {"total": int(total_groups), "page": int(page), "page_size": int(page_size), "items": payload}


def similar_items(*, item_id: int, k: int, session) -> Dict[str, Any]:
    e = session.get(Embedding, item_id)
    if not e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Embedding not found for item")
    vec = bytes_to_np(e.vector).reshape(1, -1)
    all_emb = session.exec(select(Embedding)).all()
    ids = [em.item_id for em in all_emb]
    X = np.stack([bytes_to_np(em.vector) for em in all_emb])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(ids)))
    nn.fit(X)
    d, idx = nn.kneighbors(vec, n_neighbors=min(k, len(ids)))
    neigh_ids = [ids[i] for i in idx[0]]
    items = session.exec(select(DatasetItem).where(DatasetItem.id.in_(neigh_ids))).all()
    items_map = {it.id: it for it in items}
    out = []
    for i in neigh_ids:
        it = items_map.get(i)
        if it:
            out.append({"id": i, "thumb": f"/api/group-thumb/{i}", "label": it.label, "path": it.path})
    return {"neighbors": out}


def get_map_coordinates(*, session) -> Dict[str, Any]:
    try:
        items = session.exec(select(DatasetItem)).all()
        embeddings = session.exec(select(Embedding)).all()
        emb_map = {e.item_id: bytes_to_np(e.vector) for e in embeddings}
        valid_items = [it for it in items if it.id in emb_map]
        if len(valid_items) < 2:
            return {"points": [], "msg": "Need at least 2 items with embeddings"}
        cached_coords = session.exec(select(UMAPCoords)).all()
        cached_map = {c.item_id: (c.x, c.y) for c in cached_coords}
        valid_ids = {it.id for it in valid_items}
        cached_ids = set(cached_map.keys())
        if valid_ids == cached_ids and len(cached_coords) > 0:
            points = []
            for it in valid_items:
                if it.id in cached_map:
                    x, y = cached_map[it.id]
                    points.append({"id": it.id, "x": x, "y": y, "label": it.label, "path": it.path})
            return {"points": points}
        import umap
        ids = [it.id for it in valid_items]
        X = np.stack([emb_map[i] for i in ids], axis=0)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)
        for i, (item_id, (x, y)) in enumerate(zip(ids, coords)):
            existing = session.get(UMAPCoords, item_id)
            if existing:
                existing.x = float(x)
                existing.y = float(y)
                existing.updated_at = dt.datetime.utcnow()
                session.add(existing)
            else:
                session.add(UMAPCoords(item_id=item_id, x=float(x), y=float(y)))
        session.commit()
        points = []
        for it in valid_items:
            idx = ids.index(it.id)
            x, y = coords[idx]
            points.append({"id": it.id, "x": float(x), "y": float(y), "label": it.label, "path": it.path})
        return {"points": points}
    except Exception as e:
        return {"points": [], "error": f"UMAP computation failed: {str(e)}"}


