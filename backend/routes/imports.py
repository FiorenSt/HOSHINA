from fastapi import APIRouter, Depends, UploadFile, File, Body
from typing import Optional
from ..db import get_session

router = APIRouter(tags=["imports"])


@router.post("/import-labels")
async def import_labels(file: UploadFile = File(...), session=Depends(get_session)):
    from ..services.imports import import_labels as _import_labels
    return await _import_labels(file, session)


@router.post("/import-folder-labeled")
def import_folder_labeled(
    folder_path: str = Body(...),
    class_name: str = Body(...),
    make_thumbs: bool = Body(False),
    group_require_all: Optional[bool] = Body(None),
    max_groups: Optional[int] = Body(None),
    session=Depends(get_session)
):
    from ..services.imports import import_folder_labeled as _import_folder
    return _import_folder(folder_path=folder_path, class_name=class_name, make_thumbs=make_thumbs, group_require_all=group_require_all, max_groups=max_groups, session=session)


