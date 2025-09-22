from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Set

from .. import ingest_worker
from ..ingest_state import get_ingest_mode, set_ingest_mode


def start_ingest(
    *,
    data_dir: Optional[str],
    by_groups: bool,
    max_groups: Optional[int],
    extensions: Optional[str],
    make_thumbs: bool,
    require_all_roles: Optional[bool],
    batch_size: int,
    skip_hash: bool,
    backfill_hashes: bool,
    use_zscale: bool = True,
    zscale_contrast: float = 0.1,
) -> Dict[str, Any]:
    base = Path(data_dir).resolve() if data_dir else None
    if base is None or not Path(base).exists():
        return {"ok": False, "msg": f"Data directory not found: {base}"}

    exts_str = extensions or ".jpg,.jpeg,.png,.tif,.tiff,.fits,.fit,.fits.fz"
    exts: Set[str] = set([e.strip().lower() for e in exts_str.split(",") if e.strip()])

    # Enforce ingestion mode consistency across runs
    previous_mode = get_ingest_mode()
    requested_grouped = bool(by_groups or (max_groups is not None))
    if previous_mode == "triplet" and not requested_grouped:
        return {"ok": False, "msg": "Ingestion mode locked to triplet/groups based on first run. Start ingest with groups (by_groups=true or max_groups set)."}

    try:
        r = ingest_worker.start(
            data_dir=Path(base),
            extensions=exts,
            by_groups=requested_grouped,
            max_groups=max_groups,
            make_thumbs=make_thumbs,
            require_all_roles=require_all_roles,
            batch_size=int(max(1, batch_size)),
            skip_hash=bool(skip_hash),
            backfill_hashes=bool(backfill_hashes),
            use_zscale=bool(use_zscale),
            zscale_contrast=float(zscale_contrast),
        )
    except TypeError as e:
        # Backward-compat: older workers may not accept zscale kwargs yet
        if "unexpected keyword" in str(e) or "use_zscale" in str(e):
            r = ingest_worker.start(
                data_dir=Path(base),
                extensions=exts,
                by_groups=requested_grouped,
                max_groups=max_groups,
                make_thumbs=make_thumbs,
                require_all_roles=require_all_roles,
                batch_size=int(max(1, batch_size)),
                skip_hash=bool(skip_hash),
                backfill_hashes=bool(backfill_hashes),
            )
        else:
            raise
    if isinstance(r, dict) and r.get("queued"):
        # Persist requested mode immediately; queue ensures same params will be used
        set_ingest_mode("triplet" if requested_grouped else "files")
        return {"ok": True, "queued": True, "position": int(r.get("position", 0))}
    # Not queued: started immediately. Persist mode as well.
    set_ingest_mode("triplet" if requested_grouped else "files")
    return {"ok": True, "queued": False}


def ingest_status() -> Dict[str, Any]:
    st = ingest_worker.status()
    return {"ok": True, **st}


def ingest_cancel() -> Dict[str, Any]:
    ingest_worker.cancel()
    return {"ok": True}


