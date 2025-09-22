from __future__ import annotations

from typing import Any, Dict, Optional
import json
import numpy as np
from pathlib import Path
from sqlmodel import select

from ..db import DatasetItem, Prediction


def predictions_histogram(*, session, bins: int, class_name: str) -> Dict[str, Any]:
    from collections import defaultdict
    items = session.exec(select(DatasetItem)).all()
    pred_rows = session.exec(select(Prediction)).all()
    pred_map = {pr.item_id: pr for pr in pred_rows}

    def role_rank(path_str: str) -> int:
        s = path_str.lower()
        if s.endswith("_target.fits"): return 0
        if s.endswith("_ref.fits"): return 1
        if s.endswith("_diff.fits"): return 2
        return 3

    groups: dict[tuple[Path, str], list[DatasetItem]] = defaultdict(list)
    for it in items:
        pth = Path(it.path)
        name_lower = pth.name.lower()
        base = None
        for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
            if name_lower.endswith(suf):
                base = pth.name[: -len(suf)]
                break
        if base is None:
            continue
        groups[(pth.parent, base)].append(it)

    values: list[float] = []
    for key, members in groups.items():
        members_sorted = sorted(members, key=lambda m: role_rank(m.path))
        rep = next((m for m in members_sorted if m.id in pred_map), None)
        if rep is None:
            continue
        p = pred_map.get(rep.id)
        if not p:
            continue
        v: Optional[float] = None
        if class_name:
            try:
                mp = json.loads(p.proba_json)
                if class_name in mp:
                    v = float(mp[class_name])
            except Exception:
                v = None
        else:
            if p.max_proba is not None:
                v = float(p.max_proba)
        if v is None or not (0.0 <= v <= 1.0):
            continue
        values.append(v)

    if not values:
        edges = np.linspace(0.0, 1.0, num=bins+1)
        counts = np.zeros(bins, dtype=int)
        return {
            "bins": int(bins),
            "edges": [float(x) for x in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
            "total": 0
        }
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {
        "bins": int(bins),
        "edges": [float(x) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "total": len(values)
    }


def predictions_test_histogram(*, session, bins: int, class_name: str) -> Dict[str, Any]:
    try:
        from collections import defaultdict
        items = session.exec(select(DatasetItem)).all()
        pred_rows = session.exec(select(Prediction)).all()
        pred_map = {pr.item_id: pr for pr in pred_rows}

        def role_rank(path_str: str) -> int:
            s = path_str.lower()
            if s.endswith("_target.fits"): return 0
            if s.endswith("_ref.fits"): return 1
            if s.endswith("_diff.fits"): return 2
            return 3

        groups: dict[tuple[Path, str], list[DatasetItem]] = defaultdict(list)
        for it in items:
            pth = Path(it.path)
            name_lower = pth.name.lower()
            base = None
            for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                if name_lower.endswith(suf):
                    base = pth.name[: -len(suf)]
                    break
            if base is None:
                continue
            groups[(pth.parent, base)].append(it)

        ordered_labels: list[str] = []
        try:
            from ..db import ClassDef
            labels = [c.name for c in session.exec(select(ClassDef).order_by(ClassDef.order)).all()]
            ordered_labels = labels
        except Exception:
            ordered_labels = []
        values_by_label: dict[str, list[float]] = {lbl: [] for lbl in ordered_labels}

        for key, members in groups.items():
            members_sorted = sorted(members, key=lambda m: role_rank(m.path))
            lbl_item = next((m for m in members_sorted if m.label), None)
            if not lbl_item or not lbl_item.label:
                continue
            rep_item = next((m for m in members_sorted if m.id in pred_map), None)
            if not rep_item:
                continue
            p = pred_map.get(rep_item.id)
            if not p:
                continue
            v: Optional[float] = None
            if class_name:
                try:
                    mp = json.loads(p.proba_json)
                    if class_name in mp:
                        v = float(mp[class_name])
                except Exception:
                    v = None
            else:
                if p.max_proba is not None:
                    v = float(p.max_proba)
            if v is None or not (0.0 <= v <= 1.0):
                continue
            lbl = lbl_item.label
            if lbl not in values_by_label:
                values_by_label[lbl] = []
                ordered_labels.append(lbl)
            values_by_label[lbl].append(v)

        edges = np.linspace(0.0, 1.0, num=bins + 1)
        counts_by_label: dict[str, list[int]] = {}
        totals_by_label: dict[str, int] = {}
        for lbl in ordered_labels:
            vals = values_by_label.get(lbl, [])
            if vals:
                counts, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
                counts_by_label[lbl] = [int(c) for c in counts.tolist()]
                totals_by_label[lbl] = int(len(vals))
            else:
                counts_by_label[lbl] = [0] * int(bins)
                totals_by_label[lbl] = 0

        total = int(sum(totals_by_label.values()))
        return {
            "bins": int(bins),
            "edges": [float(x) for x in edges.tolist()],
            "labels": ordered_labels,
            "counts_by_label": counts_by_label,
            "totals_by_label": totals_by_label,
            "total": total,
        }
    except Exception as e:
        return {
            "bins": int(bins),
            "edges": [],
            "labels": [],
            "counts_by_label": {},
            "totals_by_label": {},
            "total": 0,
            "error": str(e),
        }


def predictions_summary(*, session) -> Dict[str, Any]:
    from collections import defaultdict
    items = session.exec(select(DatasetItem)).all()
    groups = defaultdict(list)
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
    total_groups = len(groups)
    if total_groups == 0:
        return {"total_items": 0, "predicted": 0, "remaining": 0}
    pred_rows = session.exec(select(Prediction)).all()
    predicted_ids = {pr.item_id for pr in pred_rows}
    predicted_groups = 0
    for _, members in groups.items():
        if members and all(m.id in predicted_ids for m in members):
            predicted_groups += 1
    remaining = max(0, total_groups - predicted_groups)
    return {
        "total_items": int(total_groups),
        "predicted": int(predicted_groups),
        "remaining": int(remaining),
    }


def apply_threshold(*, session, class_name: str, negative_class: Optional[str], threshold: float, unlabeled_only: bool) -> Dict[str, Any]:
    if not class_name:
        return {"ok": False, "msg": "class_name is required"}
    updated = 0
    preds = session.exec(select(Prediction)).all()
    for p in preds:
        it = session.get(DatasetItem, p.item_id)
        if not it:
            continue
        if unlabeled_only and (it.label is not None):
            continue
        try:
            mp = json.loads(p.proba_json)
        except Exception:
            continue
        if class_name not in mp:
            continue
        v = float(mp[class_name])
        if v >= threshold:
            if it.label != class_name:
                it.label = class_name
                session.add(it)
                updated += 1
        elif negative_class:
            if it.label != negative_class:
                it.label = negative_class
                session.add(it)
                updated += 1
    session.commit()
    return {"ok": True, "updated": updated}


