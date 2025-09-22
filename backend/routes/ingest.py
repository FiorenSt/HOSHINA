from fastapi import APIRouter, Body
from typing import Optional

router = APIRouter(tags=["ingest"])


@router.post("/ingest/start")
def ingest_start(
    data_dir: Optional[str] = Body(None, embed=True),
    by_groups: bool = Body(False, embed=True),
    max_groups: Optional[int] = Body(None, embed=True),
    extensions: Optional[str] = Body(None, embed=True),
    make_thumbs: bool = Body(True, embed=True),
    require_all_roles: Optional[bool] = Body(None, embed=True),
    batch_size: int = Body(500, embed=True),
    skip_hash: bool = Body(False, embed=True),
    backfill_hashes: bool = Body(False, embed=True),
    use_zscale: bool = Body(True, embed=True),
    zscale_contrast: float = Body(0.1, embed=True),
):
    from ..services.ingest import start_ingest
    try:
        return start_ingest(
            data_dir=data_dir,
            by_groups=by_groups,
            max_groups=max_groups,
            extensions=extensions,
            make_thumbs=make_thumbs,
            require_all_roles=require_all_roles,
            batch_size=batch_size,
            skip_hash=skip_hash,
            backfill_hashes=backfill_hashes,
            use_zscale=use_zscale,
            zscale_contrast=zscale_contrast,
        )
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.get("/ingest/status")
def ingest_status():
    from ..services.ingest import ingest_status as _status
    try:
        return _status()
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.post("/ingest/cancel")
def ingest_cancel():
    from ..services.ingest import ingest_cancel as _cancel
    try:
        return _cancel()
    except Exception as e:
        return {"ok": False, "msg": str(e)}


