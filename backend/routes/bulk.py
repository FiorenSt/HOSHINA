from fastapi import APIRouter, Depends
from ..db import get_session

router = APIRouter(tags=["bulk"])


@router.post("/bulk-reset-all-labels")
def bulk_reset_all_labels(session=Depends(get_session)):
    from ..services.bulk import reset_all_labels
    return reset_all_labels(session)


@router.post("/bulk-auto-label-confident")
def bulk_auto_label_confident(session=Depends(get_session)):
    from ..services.bulk import auto_label_confident
    return auto_label_confident(session)


@router.post("/bulk-backup-dataset")
def bulk_backup_dataset(session=Depends(get_session)):
    from ..services.bulk import backup_dataset
    return backup_dataset(session)


