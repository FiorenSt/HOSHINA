from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
from sqlmodel import select

from ..db import DatasetItem, Embedding, Prediction
from ..grouping import group_items


def get_labeling_stats(session) -> Dict[str, Any]:
    total_items = session.exec(select(DatasetItem)).all()
    total_count = len(total_items)
    labeled_items = session.exec(select(DatasetItem).where(DatasetItem.label.is_not(None))).all()
    labeled_count = len(labeled_items)
    skipped_items = session.exec(select(DatasetItem).where(DatasetItem.skipped == True)).all()
    skipped_count = len(skipped_items)
    unsure_items = session.exec(select(DatasetItem).where(DatasetItem.unsure == True)).all()
    unsure_count = len(unsure_items)
    class_counts = {}
    for item in labeled_items:
        if item.label:
            class_counts[item.label] = class_counts.get(item.label, 0) + 1
    embeddings = session.exec(select(Embedding)).all()
    embedding_count = len(embeddings)
    predictions = session.exec(select(Prediction)).all()
    prediction_count = len(predictions)
    groups = group_items(total_items, get_path=lambda x: Path(x.path))
    total_groups = len(groups)
    class_distribution_groups: dict[str, int] = {}
    labeled_groups = 0
    for members in groups.values():
        lbls = {m.label for m in members if m.label}
        if len(lbls) == 1:
            lbl = next(iter(lbls))
            class_distribution_groups[lbl] = class_distribution_groups.get(lbl, 0) + 1
            labeled_groups += 1
        elif len(lbls) > 1:
            class_distribution_groups["(mixed)"] = class_distribution_groups.get("(mixed)", 0) + 1
            labeled_groups += 1
    predictions = session.exec(select(Prediction)).all()
    confidence_high = 0
    confidence_medium = 0
    confidence_low = 0
    for pred in predictions:
        if pred.max_proba is not None:
            if pred.max_proba > 0.8:
                confidence_high += 1
            elif pred.max_proba > 0.5:
                confidence_medium += 1
            else:
                confidence_low += 1
    total_predictions = confidence_high + confidence_medium + confidence_low
    confidence_high_pct = round((confidence_high / total_predictions * 100) if total_predictions > 0 else 0, 1)
    confidence_medium_pct = round((confidence_medium / total_predictions * 100) if total_predictions > 0 else 0, 1)
    confidence_low_pct = round((confidence_low / total_predictions * 100) if total_predictions > 0 else 0, 1)
    return {
        "total_items": total_count,
        "labeled_items": labeled_count,
        "total_groups": int(total_groups),
        "labeled_groups": int(labeled_groups),
        "skipped_items": skipped_count,
        "unsure_items": unsure_count,
        "unlabeled_items": total_count - labeled_count - skipped_count,
        "embedding_count": embedding_count,
        "prediction_count": prediction_count,
        "class_distribution": class_counts,
        "class_distribution_groups": class_distribution_groups,
        "progress_percentage": round((labeled_count / total_count * 100) if total_count > 0 else 0, 1),
        "confidence_high": confidence_high_pct,
        "confidence_medium": confidence_medium_pct,
        "confidence_low": confidence_low_pct
    }


