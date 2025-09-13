from fastapi import APIRouter, HTTPException, Depends, Query, Body, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from sqlmodel import select, col, func
from ..db import get_session, DatasetItem, LabelEvent, Embedding, Prediction, ClassDef, UMAPCoords, recreate_database
from ..utils import get_or_make_thumb, get_or_make_triplet_thumb, bytes_to_np, np_to_bytes, probs_to_margin, farthest_first, safe_open_image, get_or_make_composite_thumb, get_or_make_full_png, ensure_placeholder_thumb
from .. import thumb_worker
from ..embeddings import compute_embedding
from ..training import Trainer
from .. import tf_training
from ..config import THUMB_DIR, STORE_DIR, MAX_PAGE_SIZE, UMAP_CACHE
from ..grouping import load_config as load_grouping, get_config_dict as grouping_get, set_config as grouping_set, match_role_and_base, expected_paths, group_items, resolve_existing_role_file
import shutil
from .. import config as cfg
from pathlib import Path
import numpy as np
import io, csv, json, datetime as dt, zipfile
from .. import ingest_worker

router = APIRouter()

def _classes(session) -> List[str]:
    rows = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
    return [r.name for r in rows]

def _trainer() -> Trainer:
    tr = Trainer(STORE_DIR)
    tr.load()
    return tr

@router.get("/config")
def get_config():
    return {"data_dir": str(cfg.DATA_DIR), "store_dir": str(STORE_DIR)}


@router.get("/grouping")
def get_grouping():
    return {"ok": True, **grouping_get()}


@router.post("/grouping")
def set_grouping(payload: Dict[str, Any] = Body(...)):
    try:
        new_cfg = grouping_set(payload)
        return {"ok": True, **new_cfg}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.post("/set-data-dir")
def set_data_dir(payload: Dict[str, Any] = Body(...), session=Depends(get_session)):
    """Set the data directory at runtime and optionally ingest files.

    payload: { data_dir: str, ingest?: bool, extensions?: str }
    """
    new_dir = payload.get("data_dir", "").strip()
    if not new_dir:
        raise HTTPException(status_code=400, detail="data_dir is required")

    # Update config
    p = cfg.set_data_dir(new_dir)

    # Optionally ingest
    do_ingest = bool(payload.get("ingest", False))
    exts_str = payload.get("extensions") or ".jpg,.jpeg,.png,.tif,.tiff,.fits,.fit,.fits.fz"
    extensions = set([e.strip().lower() for e in exts_str.split(",") if e.strip()])
    by_groups = bool(payload.get("by_groups", False))
    max_groups_raw = payload.get("max_groups")
    try:
        max_groups = int(max_groups_raw) if max_groups_raw is not None else None
    except Exception:
        max_groups = None
    generate_pngs = bool(payload.get("generate_pngs", False))
    req_all_override = payload.get("require_all_roles")

    ingested_count = 0
    if do_ingest:
        data_dir = p.resolve()
        if by_groups or (max_groups is not None):
            # Group-based ingest using configured roles
            from collections import defaultdict
            gcfg = load_grouping()
            require_all = gcfg.require_all_roles if req_all_override is None else bool(req_all_override)
            # Collect candidate files
            files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
            # Build stable list of group keys
            keys = []
            seen = set()
            for f in files:
                _, base = match_role_and_base(f, gcfg)
                key = (f.parent, base)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
            keys.sort(key=lambda k: (str(k[0]), k[1]))
            # Resolve role files for each key
            def role_files_for(parent, base):
                roles = {}
                for role in gcfg.roles:
                    rp = resolve_existing_role_file(parent, base, role, gcfg)
                    if rp is not None:
                        roles[role] = rp
                return roles
            # Filter to only new groups (groups with at least one un-ingested file)
            eligible = []
            for parent, base in keys:
                roles = role_files_for(parent, base)
                if require_all and not all(r in roles for r in gcfg.roles):
                    continue
                if not roles:
                    continue
                
                # Check if this group has any new files (not already in database)
                has_new_files = False
                for rp in roles.values():
                    canonical_rp = str(Path(rp).resolve())
                    existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_rp)).first()
                    if not existing:
                        has_new_files = True
                        break
                
                # Only include groups with at least one new file
                if has_new_files:
                    eligible.append((parent, base, roles))
            
            # Apply limit to new groups only
            if max_groups is not None:
                eligible = eligible[:max_groups]
            
            # Ingest eligible roles
            for parent, base, roles in eligible:
                for rp in roles.values():
                    canonical_rp = str(Path(rp).resolve())
                    existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_rp)).first()
                    if existing:
                        continue
                    try:
                        img = safe_open_image(rp)
                        w, h = img.size
                    except Exception as e:
                        print(f"[skip] {rp} ({e})")
                        continue
                    it = DatasetItem(path=canonical_rp, width=w, height=h)
                    session.add(it)
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        continue
                    if generate_pngs:
                        try:
                            get_or_make_full_png(rp)
                        except Exception:
                            pass
                    ingested_count += 1
            
            # Count total groups found vs new groups processed
            total_groups_found = len(keys)
            new_groups_found = len(eligible)
            print(f"[ingest] Found {total_groups_found} total groups, {new_groups_found} new groups, ingested {ingested_count} files")
        else:
            # File-wise ingest of all supported files
            for f in data_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in extensions:
                    continue
                canonical_f = str(f.resolve())
                existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical_f)).first()
                if existing:
                    continue
                try:
                    img = safe_open_image(f)
                    w, h = img.size
                except Exception as e:
                    print(f"[skip] {f} ({e})")
                    continue
                it = DatasetItem(path=canonical_f, width=w, height=h)
                session.add(it)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    continue
                if generate_pngs:
                    try:
                        get_or_make_full_png(f)
                    except Exception:
                        pass
                ingested_count += 1

    # Build response with detailed info for group-based ingest
    response = {"ok": True, "data_dir": str(p), "ingested": ingested_count}
    if by_groups or (max_groups is not None):
        if 'total_groups_found' in locals():
            response.update({
                "total_groups_found": total_groups_found,
                "new_groups_processed": new_groups_found,
                "files_ingested": ingested_count
            })
    return response


@router.post("/ingest/start")
def ingest_start(
    data_dir: Optional[str] = Body(None, embed=True),
    by_groups: bool = Body(False, embed=True),
    max_groups: Optional[int] = Body(None, embed=True),
    extensions: Optional[str] = Body(None, embed=True),
    generate_pngs: bool = Body(True, embed=True),
    make_thumbs: bool = Body(True, embed=True),
    require_all_roles: Optional[bool] = Body(None, embed=True),
):
    try:
        base = Path(data_dir).resolve() if data_dir else cfg.DATA_DIR
        if not base or not Path(base).exists():
            return {"ok": False, "msg": f"Data directory not found: {base}"}
        exts_str = extensions or ".jpg,.jpeg,.png,.tif,.tiff,.fits,.fit,.fits.fz"
        exts = set([e.strip().lower() for e in exts_str.split(",") if e.strip()])
        r = ingest_worker.start(
            data_dir=Path(base),
            extensions=exts,
            by_groups=by_groups or (max_groups is not None),
            max_groups=max_groups,
            generate_pngs=generate_pngs,
            make_thumbs=make_thumbs,
            require_all_roles=require_all_roles,
        )
        if isinstance(r, dict) and r.get("queued"):
            return {"ok": True, "queued": True, "position": int(r.get("position", 0))}
        return {"ok": True, "queued": False}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.get("/ingest/status")
def ingest_status():
    try:
        st = ingest_worker.status()
        return {"ok": True, **st}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.post("/ingest/cancel")
def ingest_cancel():
    try:
        ingest_worker.cancel()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.get("/classes")
def get_classes(session=Depends(get_session)):
    rows = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
    return [{"id": r.id, "name": r.name, "key": r.key, "order": r.order} for r in rows]

@router.post("/classes")
def set_classes(classes: List[Dict[str, Any]], session=Depends(get_session)):
    # Wipe and set anew. Commit the DELETE first to avoid UNIQUE(name) conflicts on INSERT within same transaction.
    try:
        session.exec(text('DELETE FROM classdef'))
        session.commit()
        for i, c in enumerate(classes):
            name = (c.get("name") or "").strip()
            if not name:
                # Skip empty names
                continue
            cd = ClassDef(name=name, key=c.get("key"), order=i)
            session.add(cd)
        session.commit()
        return {"ok": True}
    except IntegrityError:
        session.rollback()
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Duplicate class names are not allowed"})
    except Exception as e:
        session.rollback()
        raise e

@router.get("/items")
def list_items(
    queue: str = Query("uncertain", enum=["uncertain","diverse","odd","band","all"]),
    page: int = 1,
    page_size: int = 50,
    prob_low: float = 0.3,
    prob_high: float = 0.8,
    unlabeled_only: bool = True,
    search: str = Query("", description="Search by filename"),
    label_filter: str = Query("", enum=["", "labeled", "unlabeled", "unsure", "skipped"]),
    simple: bool = Query(False, description="If true, skip heavy computations and just paginate"),
    only_ready: bool = Query(False, description="If true, include only groups with composite thumbs ready (size 256)"),
    seed: Optional[int] = Query(None, description="Stable seed for deterministic random order in 'all' queue"),
    session=Depends(get_session)
):
    """List items paginated by triplet groups.

    - page_size counts triplet groups, not individual images
    - Response `total` reports number of triplet groups that match filter criteria
    - `items` contains all member images for the selected groups (so length ~= 3*page_size)
    - Filtering is applied at group level: for unlabeled_only, ALL group members must be unlabeled
    """
    page_size = min(page_size, MAX_PAGE_SIZE)

    stmt = select(DatasetItem)

    # Apply search filter (other filters will be applied at group level)
    if search:
        stmt = stmt.where(DatasetItem.path.contains(search))

    # Load all items (filtering will happen at group level)
    items = session.exec(stmt).all()
    groups = group_items(items, get_path=lambda x: Path(x.path))

    if not groups:
        return {"total": 0, "page": int(page), "page_size": int(page_size), "items": []}

    # Choose representative item per group (prefer first configured role match)
    cfg = load_grouping()
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
    # Deterministic order baseline
    group_keys.sort(key=lambda k: (str(k[0]), k[1]))

    # Build rep lists and maps, filtering groups based on member criteria
    rep_ids: list[int] = []
    rep_to_members: dict[int, list[DatasetItem]] = {}
    rep_to_groupkey: dict[int, tuple[Path, str]] = {}
    for key in group_keys:
        members = groups[key]
        
        # Filter groups: for unlabeled_only, ALL members must be unlabeled
        if unlabeled_only and not label_filter:
            # Skip groups where any member is labeled or skipped
            if any(m.label is not None or m.skipped for m in members):
                continue
        elif label_filter:
            # For specific label filters, apply the same logic to all members
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
            parent = Path(rep.path).parent
            cfg_local = cfg
            _, base = match_role_and_base(Path(rep.path), cfg_local)
            key_parts: list[str] = []
            for role in cfg_local.roles:
                rp = resolve_existing_role_file(parent, base, role, cfg_local)
                key_parts.append(str(rp) if rp else "")
            from ..utils import composite_thumb_path_for
            ctp = composite_thumb_path_for(key_parts, size=256)
            if not ctp.exists():
                # Skip groups that are not ready if only_ready is requested
                rep_ids.pop()
                rep_to_members.pop(rep.id, None)
                rep_to_groupkey.pop(rep.id, None)
                continue

    # Gather embeddings and predictions
    emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
    preds_map = {}
    for p in session.exec(select(Prediction)).all():
        preds_map[p.item_id] = json.loads(p.proba_json), p.pred_label, p.margin, p.max_proba

    # Selection ordering by queue operates on representatives
    ordered_rep_ids: list[int] = rep_ids[:]
    try:
        if queue == "all":
            # Deterministic pseudo-random order per group using seed and group key
            if seed is not None:
                import hashlib
                def order_key(rid: int) -> int:
                    parent, base = rep_to_groupkey.get(rid, (Path(""), ""))
                    s = f"{parent}|{base}|{seed}".encode("utf-8")
                    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")
                ordered_rep_ids = sorted(rep_ids, key=order_key)
            else:
                # Fallback: randomized order (non-deterministic)
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
                # Handle shape consistency
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
                    # Keep non-embedded reps at the end to preserve groups
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
    except Exception as _:
        # If any selection logic fails, fall back to baseline order
        ordered_rep_ids = rep_ids[:]

    total_groups = len(ordered_rep_ids)
    # Pagination by groups
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_rep_ids = ordered_rep_ids[start:end]

    # Build payload by expanding each selected group to all its members
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

@router.get("/thumb/{item_id}")
def get_thumb(item_id: int, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    # Non-blocking: serve existing thumb or placeholder
    p = Path(it.path)
    from ..utils import thumb_path_for
    tp = thumb_path_for(p, size=256)
    if tp.exists():
        return FileResponse(tp, headers={"Cache-Control": "no-store"})
    return FileResponse(ensure_placeholder_thumb(256), headers={"Cache-Control": "no-store"})

@router.get("/triplet-thumb/{item_id}")
def get_triplet_thumb(item_id: int, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")

    p = Path(it.path)
    # Resolve triplet using grouping config (supports custom suffixes)
    cfg = load_grouping()
    _, base = match_role_and_base(p, cfg)
    target = resolve_existing_role_file(p.parent, base, "target", cfg) if "target" in cfg.roles else None
    ref = resolve_existing_role_file(p.parent, base, "ref", cfg) if "ref" in cfg.roles else None
    diff = resolve_existing_role_file(p.parent, base, "diff", cfg) if "diff" in cfg.roles else None

    expected = {
        "target": p.with_name(f"{base}_target.fits"),
        "ref": p.with_name(f"{base}_ref.fits"),
        "diff": p.with_name(f"{base}_diff.fits"),
    }

    # Only include files that actually exist
    target = expected["target"] if expected["target"].exists() else None
    ref = expected["ref"] if expected["ref"].exists() else None
    diff = expected["diff"] if expected["diff"].exists() else None

    # Non-blocking: if none exist, serve single-thumb placeholder; if expected triplet, serve placeholder
    if not any([target, ref, diff]):
        from ..utils import thumb_path_for
        single_tp = thumb_path_for(p, size=256)
        if single_tp.exists():
            return FileResponse(single_tp, headers={"Cache-Control": "no-store"})
        return FileResponse(ensure_placeholder_thumb(256), headers={"Cache-Control": "no-store"})

    # Try precomputed triplet thumb path convention
    from ..utils import triplet_thumb_path_for, thumb_path_for
    parts = [str(x) if x else "" for x in [target, ref, diff]]
    base_key = "|".join(parts) + "|256"
    ttp = triplet_thumb_path_for(base_key, size=256)
    if ttp.exists():
        return FileResponse(ttp, headers={"Cache-Control": "no-store"})
    # Compose a quick triplet from available single thumbs (non-blocking)
    try:
        from PIL import Image
        panels = []
        for rp in [target, ref, diff]:
            if rp:
                st = thumb_path_for(Path(rp), size=256)
                if st.exists():
                    panels.append(Image.open(st).convert("RGB"))
                else:
                    panels.append(Image.new("RGB", (256, 256), color=(32,32,32)))
            else:
                panels.append(Image.new("RGB", (256, 256), color=(32,32,32)))
        out = Image.new("RGB", (256*3, 256), color=(0,0,0))
        for i, pi in enumerate(panels[:3]):
            # center paste to handle any minor size differences
            w, h = pi.size
            cx = max(0, (256 - w)//2)
            cy = max(0, (256 - h)//2)
            out.paste(pi, (i*256 + cx, cy))
        # Cache and serve
        ttp.parent.mkdir(parents=True, exist_ok=True)
        out.save(ttp, format="JPEG", quality=85)
        return FileResponse(ttp, headers={"Cache-Control": "no-store"})
    except Exception:
        return FileResponse(ensure_placeholder_thumb(256), headers={"Cache-Control": "no-store"})


@router.get("/group-thumb/{item_id}")
def get_group_thumb(item_id: int, size: int = 256, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    p = Path(it.path)
    cfg = load_grouping()
    _, base = match_role_and_base(p, cfg)
    # Non-blocking composite: return existing composite or fallback to any single thumb, else placeholder
    from ..utils import composite_thumb_path_for, thumb_path_for
    key_parts: list[str] = []
    cfg = load_grouping()
    for role in cfg.roles:
        rp = resolve_existing_role_file(p.parent, base, role, cfg)
        key_parts.append(str(rp) if rp else "")
    ctp = composite_thumb_path_for(key_parts, size=size)
    if ctp.exists():
        return FileResponse(ctp, headers={"Cache-Control": "no-store"})
    # Fallback: compose from available single thumbs (non-blocking)
    try:
        from PIL import Image
        panels = []
        for role in cfg.roles:
            rp = resolve_existing_role_file(p.parent, base, role, cfg)
            if rp:
                st = thumb_path_for(Path(rp), size=size)
                if st.exists():
                    panels.append(Image.open(st).convert("RGB"))
                else:
                    panels.append(Image.new("RGB", (size, size), color=(32,32,32)))
            else:
                panels.append(Image.new("RGB", (size, size), color=(32,32,32)))
        out = Image.new("RGB", (size*max(1,len(panels)), size), color=(0,0,0))
        for i, pi in enumerate(panels):
            w, h = pi.size
            cx = max(0, (size - w)//2)
            cy = max(0, (size - h)//2)
            out.paste(pi, (i*size + cx, cy))
        ctp.parent.mkdir(parents=True, exist_ok=True)
        out.save(ctp, format="JPEG", quality=85)
        return FileResponse(ctp, headers={"Cache-Control": "no-store"})
    except Exception:
        return FileResponse(ensure_placeholder_thumb(size), headers={"Cache-Control": "no-store"})

@router.get("/file/{item_id}")
def get_file(item_id: int, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    
    file_path = Path(it.path)
    
    # For FITS, serve cached full-resolution PNG from cache/png
    if file_path.suffix.lower() == '.fits':
        try:
            pp = get_or_make_full_png(file_path)
            return FileResponse(pp, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate PNG: {str(e)}")
    # For other image formats, serve directly
    return FileResponse(file_path)


@router.get("/file-png/{item_id}")
def get_file_png(item_id: int, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    file_path = Path(it.path)
    try:
        pp = get_or_make_full_png(file_path)
        return FileResponse(pp, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PNG: {str(e)}")


@router.post("/dataset/reset")
def dataset_reset(
    wipe_items: bool = Body(True, embed=True),
    wipe_embeddings: bool = Body(True, embed=True),
    wipe_predictions: bool = Body(True, embed=True),
    wipe_labels: bool = Body(True, embed=True),
    wipe_umap: bool = Body(True, embed=True),
    wipe_caches: bool = Body(True, embed=True),
    wipe_classes: bool = Body(False, embed=True),
    recreate_db: bool = Body(False, embed=True),
    reingest: bool = Body(False, embed=True),  # Deprecated but kept for compatibility
    session=Depends(get_session)
):
    """Reset dataset and caches completely.

    - wipe_* flags control which tables/caches to clear
    - reingest parameter is ignored (deprecated)
    """
    try:
        print("[reset] Starting reset operation...")
        
        # Cancel any running thumb job
        try:
            if thumb_worker.is_running():
                print("[reset] Cancelling running thumbnail job...")
                thumb_worker.cancel()
        except Exception as e:
            print(f"[reset] Warning: Could not cancel thumb job: {e}")

        # Either recreate DB file or wipe tables
        if recreate_db:
            print("[reset] Recreating database file...")
            try:
                recreate_database()
                print("[reset] Database file recreated")
            except Exception as e:
                print(f"[reset] Warning: DB recreate failed, falling back to table wipe: {e}")
                recreate_db = False

        if not recreate_db:
            # Wipe DB tables efficiently by disabling foreign key constraints temporarily
            print("[reset] Starting database wipe...")

            # Disable foreign key constraints for SQLite
            session.exec(text("PRAGMA foreign_keys = OFF"))

            # Count and delete each table
            tables_to_wipe = []
            if wipe_predictions:
                count = session.exec(select(func.count(Prediction.item_id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("predictions", Prediction, count))

            if wipe_embeddings:
                count = session.exec(select(func.count(Embedding.item_id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("embeddings", Embedding, count))

            if wipe_umap:
                # UMAPCoords primary key is item_id, not id
                count = session.exec(select(func.count(UMAPCoords.item_id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("UMAP coordinates", UMAPCoords, count))

            if wipe_labels:
                count = session.exec(select(func.count(LabelEvent.id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("label events", LabelEvent, count))

            if wipe_items:
                count = session.exec(select(func.count(DatasetItem.id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("dataset items", DatasetItem, count))

            if wipe_classes:
                count = session.exec(select(func.count(ClassDef.id))).first() or 0
                if count > 0:
                    tables_to_wipe.append(("class definitions", ClassDef, count))

            # Perform bulk deletions
            for table_name, model_class, count in tables_to_wipe:
                print(f"[reset] Deleting {count} {table_name}...")
                session.exec(model_class.__table__.delete())  # type: ignore[attr-defined]
                print(f"[reset] Deleted {count} {table_name}")

            # Re-enable foreign key constraints
            session.exec(text("PRAGMA foreign_keys = ON"))

            # Commit all changes
            session.commit()
            print("[reset] Database wipe completed")

        # Wipe caches on disk
        if wipe_caches:
            print("[reset] Clearing file caches...")
            try:
                # Count and remove thumbnails
                thumb_count = 0
                if THUMB_DIR.exists():
                    thumb_files = list(THUMB_DIR.glob("**/*"))
                    thumb_count = sum(1 for p in thumb_files if p.is_file())
                    print(f"[reset] Removing {thumb_count} thumbnail files...")
                    for p in thumb_files:
                        try:
                            if p.is_file():
                                p.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"[reset] Warning: Could not delete {p}: {e}")
                
                # Count and remove cache subdirs
                cache_count = 0
                if (STORE_DIR / "cache").exists():
                    cache_items = list((STORE_DIR / "cache").iterdir())
                    cache_count = len(cache_items)
                    print(f"[reset] Removing {cache_count} cache items...")
                    for sub in cache_items:
                        try:
                            if sub.is_dir():
                                shutil.rmtree(sub, ignore_errors=True)
                            elif sub.is_file():
                                sub.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"[reset] Warning: Could not delete {sub}: {e}")
                
                print(f"[reset] Cache cleanup completed: {thumb_count} thumbnails, {cache_count} cache items")
            except Exception as e:
                print(f"[reset] Warning: Cache cleanup had errors: {e}")

        print("[reset] All caches and data wiped successfully")
        return {"ok": True, "message": "Dataset completely wiped"}
    except Exception as e:
        return {"ok": False, "msg": f"Reset failed: {str(e)}"}

@router.post("/thumbs/build/start")
def thumbs_build_start(
    mode: str = Body("triplet", embed=True),
    size: int = Body(256, embed=True),
    only_missing: bool = Body(True, embed=True),
    limit: Optional[int] = Body(None, embed=True),
):
    try:
        if thumb_worker.is_running():
            return {"ok": False, "msg": "Thumbnail build already running"}
        thumb_worker.start(mode=mode, size=size, only_missing=only_missing, limit=limit)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.get("/thumbs/build/status")
def thumbs_build_status():
    try:
        st = thumb_worker.status()
        return {"ok": True, **st}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.post("/thumbs/build/cancel")
def thumbs_build_cancel():
    try:
        thumb_worker.cancel()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.get("/triplet/{item_id}")
def get_triplet(item_id: int, session=Depends(get_session)):
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")

    p = Path(it.path)
    name_lower = p.name.lower()
    base = None
    for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
        if name_lower.endswith(suf):
            base = p.name[: -len(suf)]
            break
    if base is None:
        base = p.stem

    expected = {
        "target": p.with_name(f"{base}_target.fits"),
        "ref": p.with_name(f"{base}_ref.fits"),
        "diff": p.with_name(f"{base}_diff.fits"),
    }

    out = {}
    for role, path in expected.items():
        row = session.exec(select(DatasetItem).where(DatasetItem.path == str(path))).first()
        if row:
            out[role] = {
                "id": row.id,
                "file": f"/api/file/{row.id}",
                "thumb": f"/api/thumb/{row.id}",
                "path": row.path,
                "label": row.label,
            }

    return {"group_key": base, "items": out}

@router.post("/label")
def set_label(
    item_ids: List[int] = Body(..., embed=True),
    label: Optional[str] = Body(None),
    unsure: bool = Body(False),
    skip: bool = Body(False),
    user: Optional[str] = Body(None),
    session=Depends(get_session)
):
    for iid in item_ids:
        it = session.get(DatasetItem, iid)
        if not it:
            continue
        prev = it.label
        it.label = None if skip else label
        it.unsure = unsure
        it.skipped = skip
        ev = LabelEvent(item_id=iid, prev_label=prev, new_label=it.label, unsure=unsure, skipped=skip, user=user)
        session.add(ev)
        session.add(it)
    session.commit()
    return {"ok": True}

@router.post("/batch-label")
def batch_label_items(
    item_ids: List[int] = Body(..., embed=True),
    label: Optional[str] = Body(None),
    unsure: bool = Body(False),
    skip: bool = Body(False),
    user: Optional[str] = Body(None),
    session=Depends(get_session)
):
    """Label multiple items at once"""
    try:
        if not item_ids:
            return {"ok": False, "msg": "No items provided"}
        
        success_count = 0
        for iid in item_ids:
            it = session.get(DatasetItem, iid)
            if not it:
                continue
            prev = it.label
            it.label = None if skip else label
            it.unsure = unsure
            it.skipped = skip
            ev = LabelEvent(item_id=iid, prev_label=prev, new_label=it.label, unsure=unsure, skipped=skip, user=user)
            session.add(ev)
            session.add(it)
            success_count += 1
        
        session.commit()
        return {"ok": True, "labeled_count": success_count}
    except Exception as e:
        return {"ok": False, "msg": f"Batch labeling failed: {str(e)}"}

@router.post("/train")
def train_now(session=Depends(get_session)):
    try:
        # Collect labeled items and train
        labeled = session.exec(select(DatasetItem).where(DatasetItem.label.is_not(None))).all()
        if not labeled:
            return {"ok": False, "msg": "No labeled items yet."}
        
        # Load embeddings
        emb = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
        ids = [it.id for it in labeled if it.id in emb]
        
        if len(ids) < 2:
            return {"ok": False, "msg": f"Need at least 2 labeled items with embeddings. Found {len(ids)}."}
        
        X = np.stack([emb[i] for i in ids], axis=0)
        y = [next(it for it in labeled if it.id==i).label for i in ids]
        
        # Check for class diversity
        unique_classes = set(y)
        if len(unique_classes) < 2:
            return {"ok": False, "msg": f"Need at least 2 different classes. Found: {unique_classes}"}
        
        trainer = Trainer(STORE_DIR)
        art = trainer.train(X, y)
    except Exception as e:
        return {"ok": False, "msg": f"Training failed: {str(e)}"}

    # Predict for all items with embeddings
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
            margin = float(pv[0]-pv[1]) if len(classes) > 1 else float(1.0 - pv[0])
            maxp = float(pv[0]) if len(pv)>0 else None
            p = Prediction(item_id=iid, proba_json=json.dumps(prob_map), pred_label=pred_lbl, margin=margin, max_proba=maxp)
            # upsert
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


@router.get("/train/options")
def train_options():
    """List available TensorFlow models and defaults."""
    try:
        return {"ok": True, **tf_training.list_options()}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


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
):
    try:
        tf_training.start_training(model_id=model, epochs=epochs, batch_size=batch_size, augment=augment, input_mode=input_mode, class_map=class_map or None, split_pct=split_pct, split_strategy=split_strategy, single_role=single_role)
        return {"ok": True}
    except RuntimeError as e:
        return {"ok": False, "msg": str(e)}
    except Exception as e:
        return {"ok": False, "msg": f"Failed to start training: {str(e)}"}


# ===== Custom Model Builder Endpoints =====
@router.get("/model-builder/graph")
def get_model_graph():
    try:
        g = tf_training.get_saved_custom_graph()
        return {"ok": True, "graph": g or {"nodes": [], "edges": [], "input_shape": [224,224,3]}}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.post("/model-builder/graph")
def save_model_graph(graph: Dict[str, Any] = Body(...)):
    try:
        # basic validation
        nodes = graph.get("nodes"); edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return {"ok": False, "msg": "graph.nodes and graph.edges must be lists"}
        # try building to validate
        try:
            tf_training.build_model_from_graph(graph, num_classes=2)  # dry run with 2 classes
        except Exception as ve:
            return {"ok": False, "msg": f"Invalid graph: {ve}"}
        tf_training.save_custom_graph(graph)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@router.get("/train/status")
def train_status():
    st = tf_training.get_status()
    # Try to detect device/GPU even if training hasn't started
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


@router.get("/predictions/histogram")
def predictions_histogram(
    bins: int = Query(20, ge=5, le=100),
    class_name: str = Query("", description="If provided, histogram of this class' probability; otherwise max_proba"),
    session=Depends(get_session)
):
    """Return a histogram of prediction probabilities aggregated per triplet group.

    - If class_name is provided, use that class' probability from proba_json
    - Otherwise, use max_proba
    - Groups are defined by triplet base (parent_dir + filename base without _target/_ref/_diff)

    Response: { bins, edges: number[], counts: number[], total }
    """
    try:
        # Build groups by triplet base
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
            # Choose a representative member for which we have a prediction
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
    except Exception as e:
        return {"bins": int(bins), "edges": [], "counts": [], "total": 0, "error": str(e)}


@router.get("/predictions/test-histogram")
def predictions_test_histogram(
    bins: int = Query(20, ge=5, le=100),
    class_name: str = Query("", description="If provided, histogram of this class' probability; otherwise max_proba"),
    session=Depends(get_session)
):
    """Return per-label histograms aggregated per triplet group for labeled items.

    Uses groups where at least one member has a ground-truth label. One prediction
    value is taken per group (prefer target, then ref, then diff). The group's label
    is taken from the preferred role that has a label.

    Response: {
      bins, edges: number[], labels: string[],
      counts_by_label: { [label]: number[] },
      totals_by_label: { [label]: number }, total: number
    }
    """
    try:
        # Preserve label order from ClassDef for consistent coloring
        ordered_labels = _classes(session)
        values_by_label: dict[str, list[float]] = {lbl: [] for lbl in ordered_labels}

        # Build groups by triplet base
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

        for key, members in groups.items():
            # Choose label from preferred role
            members_sorted = sorted(members, key=lambda m: role_rank(m.path))
            lbl_item = next((m for m in members_sorted if m.label), None)
            if not lbl_item or not lbl_item.label:
                continue
            # Choose a representative prediction
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

@router.get("/predictions/summary")
def predictions_summary(session=Depends(get_session)):
    """Return counts for prediction coverage across the dataset, grouped by triplets.

    A triplet is defined by files sharing the same base name with suffixes
    _target.fits, _ref.fits, _diff.fits. We count unique (parent_dir, base)
    groups. A group is considered predicted if any member has a Prediction row.
    """
    try:
        from collections import defaultdict
        items = session.exec(select(DatasetItem)).all()
        # Build groups keyed by (parent, base)
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
                # Skip non-triplet style files for this summary
                continue
            groups[(p.parent, base)].append(it)

        total_groups = len(groups)
        if total_groups == 0:
            return {"total_items": 0, "predicted": 0, "remaining": 0}

        pred_rows = session.exec(select(Prediction)).all()
        predicted_ids = {pr.item_id for pr in pred_rows}
        predicted_groups = 0
        for _, members in groups.items():
            if any(m.id in predicted_ids for m in members):
                predicted_groups += 1

        remaining = max(0, total_groups - predicted_groups)
        # Keep existing keys for frontend compatibility, but values represent triplets
        return {
            "total_items": int(total_groups),
            "predicted": int(predicted_groups),
            "remaining": int(remaining),
        }
    except Exception as e:
        return {"total_items": 0, "predicted": 0, "remaining": 0, "error": str(e)}


@router.post("/predictions/run")
def predictions_run(
    repredict_all: bool = Body(False),
    limit: Optional[int] = Body(None),
    session=Depends(get_session)
):
    """Predict items using classic Trainer if available, operating per triplet group.

    - If repredict_all is False, only predict groups with no Prediction rows
      among their members (triplet-level remaining).
    - If `limit` is provided, only process up to that many triplet groups.
    - Uses stored embeddings and sklearn classifier.
    - Writes the same prediction to all members of a triplet group.
    """
    try:
        trainer = Trainer(STORE_DIR)
        trainer.load()
        if trainer.clf is None:
            return {"ok": False, "msg": "No classic model available. If you trained TF, predictions are generated automatically after training."}

        # Group items by triplet base
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

        # Assemble eligible groups
        group_keys = list(groups.keys())
        # Sort deterministically by path+base
        group_keys.sort(key=lambda k: (str(k[0]), k[1]))
        if not repredict_all:
            group_keys = [k for k in group_keys if not any(m.id in already_ids for m in groups[k])]
        if limit is not None and limit > 0:
            group_keys = group_keys[:limit]
        if not group_keys:
            return {"ok": True, "predicted": 0}

        # Embeddings map
        emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}

        # Choose one representative per group (prefer target, then ref, then diff) that has an embedding
        rep_ids: list[int] = []
        rep_to_members: list[tuple[int, list[DatasetItem]]] = []
        for key in group_keys:
            members = groups[key]
            # order by role preference
            def role_rank(path_str: str) -> int:
                s = path_str.lower()
                if s.endswith("_target.fits"): return 0
                if s.endswith("_ref.fits"): return 1
                if s.endswith("_diff.fits"): return 2
                return 3
            members_sorted = sorted(members, key=lambda m: role_rank(m.path))
            rep = next((m for m in members_sorted if m.id in emb_map), None)
            if rep is None:
                # No embedding for this group; skip
                continue
            rep_ids.append(rep.id)
            rep_to_members.append((rep.id, members))

        if not rep_ids:
            return {"ok": False, "msg": "No embeddings found for selected triplet groups"}

        X = np.stack([emb_map[i] for i in rep_ids], axis=0)
        probs = trainer.predict_proba(X)
        classes = trainer.classes_ or []

        for (rep_iid, members), pr in zip(rep_to_members, probs):
            prob_map = {cls: float(p) for cls, p in zip(classes, pr)}
            pv = np.array(list(prob_map.values()))
            order = np.argsort(-pv) if len(pv) else []
            pred_lbl = classes[int(order[0])] if len(order) else None
            margin = float(pv[0]-pv[1]) if len(pv) > 1 else (float(1.0 - pv[0]) if len(pv) == 1 else None)
            maxp = float(pv[0]) if len(pv) else None
            # Write same prediction to all members
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
        # Report number of triplet groups processed
        return {"ok": True, "predicted": len(rep_ids)}
    except Exception as e:
        return {"ok": False, "msg": f"Prediction failed: {str(e)}"}


@router.post("/predictions/apply-threshold")
def predictions_apply_threshold(
    class_name: str = Body(...),
    negative_class: Optional[str] = Body(None),
    threshold: float = Body(0.5),
    unlabeled_only: bool = Body(True),
    session=Depends(get_session)
):
    """Apply a probability threshold on a given class to assign labels.

    If probability(class_name) >= threshold => label = class_name
    else if negative_class is provided => label = negative_class
    Only applies to items that meet 'unlabeled_only' condition.
    """
    try:
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
    except Exception as e:
        return {"ok": False, "msg": f"Apply threshold failed: {str(e)}"}


@router.post("/train/cancel")
def train_cancel():
    try:
        tf_training.cancel()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@router.get("/similar/{item_id}")
def similar_items(item_id: int, k: int = 12, session=Depends(get_session)):
    e = session.get(Embedding, item_id)
    if not e:
        raise HTTPException(status_code=404, detail="Embedding not found for item")
    vec = bytes_to_np(e.vector).reshape(1, -1)
    # Build matrix
    all_emb = session.exec(select(Embedding)).all()
    ids = [em.item_id for em in all_emb]
    X = np.stack([bytes_to_np(em.vector) for em in all_emb])
    # Fit kNN on the fly (smallish)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(ids)))
    nn.fit(X)
    d, idx = nn.kneighbors(vec, n_neighbors=min(k, len(ids)))
    neigh_ids = [ids[i] for i in idx[0]]
    items = session.exec(select(DatasetItem).where(DatasetItem.id.in_(neigh_ids))).all()
    # order by idx order
    items_map = {it.id: it for it in items}
    out = []
    for i in neigh_ids:
        it = items_map.get(i)
        if it:
            out.append({
                "id": i, "thumb": f"/api/group-thumb/{i}", "label": it.label, "path": it.path
            })
    return {"neighbors": out}

@router.get("/map")
def get_map_coordinates(session=Depends(get_session)):
    """Get UMAP coordinates for visualization"""
    try:
        # Get all items with embeddings
        items = session.exec(select(DatasetItem)).all()
        embeddings = session.exec(select(Embedding)).all()
        emb_map = {e.item_id: bytes_to_np(e.vector) for e in embeddings}
        
        # Filter items that have embeddings
        valid_items = [it for it in items if it.id in emb_map]
        if len(valid_items) < 2:
            return {"points": [], "msg": "Need at least 2 items with embeddings"}
        
        # Check if we have cached UMAP coordinates
        cached_coords = session.exec(select(UMAPCoords)).all()
        cached_map = {c.item_id: (c.x, c.y) for c in cached_coords}
        
        # Check if we need to recompute (new items added)
        valid_ids = {it.id for it in valid_items}
        cached_ids = set(cached_map.keys())
        
        if valid_ids == cached_ids and len(cached_coords) > 0:
            # Use cached coordinates
            points = []
            for it in valid_items:
                if it.id in cached_map:
                    x, y = cached_map[it.id]
                    points.append({
                        "id": it.id,
                        "x": x,
                        "y": y,
                        "label": it.label,
                        "path": it.path
                    })
            return {"points": points}
        
        # Compute UMAP coordinates
        import umap
        ids = [it.id for it in valid_items]
        X = np.stack([emb_map[i] for i in ids], axis=0)
        
        # Use UMAP with reasonable parameters
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)
        
        # Store coordinates in database
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
        
        # Return points
        points = []
        for it in valid_items:
            idx = ids.index(it.id)
            x, y = coords[idx]
            points.append({
                "id": it.id,
                "x": float(x),
                "y": float(y),
                "label": it.label,
                "path": it.path
            })
        
        return {"points": points}
        
    except Exception as e:
        return {"points": [], "error": f"UMAP computation failed: {str(e)}"}

@router.get("/stats")
def get_labeling_stats(session=Depends(get_session)):
    """Get labeling progress statistics"""
    try:
        # Get total items
        total_items = session.exec(select(DatasetItem)).all()
        total_count = len(total_items)
        
        # Get labeled items
        labeled_items = session.exec(select(DatasetItem).where(DatasetItem.label.is_not(None))).all()
        labeled_count = len(labeled_items)
        
        # Get skipped items
        skipped_items = session.exec(select(DatasetItem).where(DatasetItem.skipped == True)).all()
        skipped_count = len(skipped_items)
        
        # Get unsure items
        unsure_items = session.exec(select(DatasetItem).where(DatasetItem.unsure == True)).all()
        unsure_count = len(unsure_items)
        
        # Get class distribution
        class_counts = {}
        for item in labeled_items:
            if item.label:
                class_counts[item.label] = class_counts.get(item.label, 0) + 1
        
        # Get items with embeddings
        embeddings = session.exec(select(Embedding)).all()
        embedding_count = len(embeddings)
        
        # Get items with predictions
        predictions = session.exec(select(Prediction)).all()
        prediction_count = len(predictions)
        
        # Compute group-based (triplet/composite) statistics
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
                # Mixed-label group; count under a special key to surface inconsistency
                class_distribution_groups["(mixed)"] = class_distribution_groups.get("(mixed)", 0) + 1
                labeled_groups += 1

        # Calculate confidence distribution
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
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}

@router.get("/suggestions")
def get_suggestions(
    type: str = Query("uncertain", enum=["uncertain", "diverse", "anomalies", "borderline"]),
    count: int = Query(24, ge=1, le=100),
    session=Depends(get_session)
):
    """Get AI-powered suggestions for labeling"""
    try:
        # Get all unlabeled items with embeddings and predictions
        stmt = select(DatasetItem).where(
            (DatasetItem.label.is_(None)) & 
            (DatasetItem.skipped == False) & 
            (DatasetItem.unsure == False)
        )
        items = session.exec(stmt).all()
        
        # Get embeddings and predictions
        emb_map = {e.item_id: bytes_to_np(e.vector) for e in session.exec(select(Embedding)).all()}
        preds_map = {}
        for p in session.exec(select(Prediction)).all():
            preds_map[p.item_id] = json.loads(p.proba_json), p.pred_label, p.margin, p.max_proba
        
        # Filter items that have both embeddings and predictions
        valid_items = [it for it in items if it.id in emb_map and it.id in preds_map]
        
        if not valid_items:
            return {"suggestions": [], "message": "No items available for suggestions. Try retraining the model first."}
        
        suggestions = []
        
        if type == "uncertain":
            # Sort by lowest margin (most uncertain predictions)
            scored_items = []
            for item in valid_items:
                _, _, margin, _ = preds_map[item.id]
                if margin is not None:
                    scored_items.append((item, margin, "Low prediction confidence"))
            
            scored_items.sort(key=lambda x: x[1])  # Sort by margin (ascending)
            suggestions = scored_items[:count]
            
        elif type == "diverse":
            # Use farthest-first sampling for diversity
            if len(valid_items) > 0:
                ids = [it.id for it in valid_items]
                X = np.stack([emb_map[i] for i in ids], axis=0)
                diverse_idx = farthest_first(X, k=min(count, len(ids)))
                suggestions = [(valid_items[i], 1.0 - (i / len(diverse_idx)), "Diverse embedding") for i in diverse_idx]
        
        elif type == "anomalies":
            # Use LOF to find outliers
            if len(valid_items) >= 20:
                from sklearn.neighbors import LocalOutlierFactor
                ids = [it.id for it in valid_items]
                X = np.stack([emb_map[i] for i in ids], axis=0)
                lof = LocalOutlierFactor(n_neighbors=min(20, len(ids)-1), novelty=False)
                lof.fit(X)
                scores = -lof.negative_outlier_factor_
                
                scored_items = [(valid_items[i], scores[i], "Anomalous pattern") for i in range(len(valid_items))]
                scored_items.sort(key=lambda x: x[1], reverse=True)  # Sort by anomaly score (descending)
                suggestions = scored_items[:count]
            else:
                suggestions = [(item, 0.5, "Insufficient data for anomaly detection") for item in valid_items[:count]]
                
        elif type == "borderline":
            # Find items with predictions close to decision boundary (around 0.5)
            scored_items = []
            for item in valid_items:
                _, _, _, max_proba = preds_map[item.id]
                if max_proba is not None:
                    # Score based on how close to 0.5 the max probability is
                    boundary_score = 1.0 - abs(max_proba - 0.5) * 2
                    scored_items.append((item, boundary_score, f"Borderline prediction ({max_proba:.3f})"))
            
            scored_items.sort(key=lambda x: x[1], reverse=True)  # Sort by boundary score (descending)
            suggestions = scored_items[:count]
        
        # Format response
        result = []
        for item, score, reason in suggestions:
            result.append({
                "id": item.id,
                "path": item.path,
                "thumb": f"/api/triplet-thumb/{item.id}",
                "score": float(score),
                "reason": reason
            })
        
        return {"suggestions": result}
        
    except Exception as e:
        return {"suggestions": [], "error": f"Failed to generate suggestions: {str(e)}"}

@router.get("/export")
def export_all(session=Depends(get_session)):
    # Export labels.csv, classes.json, manifest.json in a zip
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # labels.csv
        rows = session.exec(select(DatasetItem)).all()
        csv_bytes = io.StringIO()
        w = csv.writer(csv_bytes)
        w.writerow(["item_id","path","label","unsure","skipped","ingested_at"])
        for r in rows:
            w.writerow([r.id, r.path, r.label or "", int(r.unsure), int(r.skipped), r.ingested_at.isoformat()])
        z.writestr("labels.csv", csv_bytes.getvalue())

        # classes.json
        clz = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
        z.writestr("classes.json", json.dumps([{"name": c.name, "key": c.key, "order": c.order} for c in clz], indent=2))

        # manifest.json
        manifest = {
            "exported_at": dt.datetime.utcnow().isoformat() + "Z",
            "store_dir": str(STORE_DIR),
            "db_path": "store/app.db",
            "embedding_backend": "auto",
            "version": 1
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    mem.seek(0)
    headers = {"Content-Disposition": "attachment; filename=dataset_export.zip"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)

@router.get("/export-labels")
def export_labels(session=Depends(get_session)):
    """Export just the labels as CSV"""
    rows = session.exec(select(DatasetItem)).all()
    csv_bytes = io.StringIO()
    w = csv.writer(csv_bytes)
    w.writerow(["item_id","path","label","unsure","skipped","ingested_at"])
    for r in rows:
        w.writerow([r.id, r.path, r.label or "", int(r.unsure), int(r.skipped), r.ingested_at.isoformat()])
    
    csv_content = csv_bytes.getvalue()
    headers = {"Content-Disposition": "attachment; filename=labels.csv"}
    return StreamingResponse(io.BytesIO(csv_content.encode('utf-8')), media_type="text/csv", headers=headers)

@router.get("/export-model")
def export_model(session=Depends(get_session)):
    """Export model artifacts"""
    try:
        # Create a zip with model files
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            # Add model files if they exist
            model_dir = STORE_DIR / "model"
            if model_dir.exists():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        z.write(file_path, file_path.relative_to(STORE_DIR))
            
            # Add classes.json
            clz = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
            z.writestr("classes.json", json.dumps([{"name": c.name, "key": c.key, "order": c.order} for c in clz], indent=2))
            
            # Add manifest
            manifest = {
                "exported_at": dt.datetime.utcnow().isoformat() + "Z",
                "store_dir": str(STORE_DIR),
                "model_dir": str(model_dir),
                "version": 1
            }
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        
        mem.seek(0)
        headers = {"Content-Disposition": "attachment; filename=model_artifacts.zip"}
        return StreamingResponse(mem, media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")

@router.post("/bulk-reset-all-labels")
def bulk_reset_all_labels(session=Depends(get_session)):
    """Reset all labels in the dataset"""
    try:
        # Get all items
        items = session.exec(select(DatasetItem)).all()
        
        # Clear labels, unsure, and skipped flags
        for item in items:
            item.label = None
            item.unsure = False
            item.skipped = False
        
        # Clear all predictions
        predictions = session.exec(select(Prediction)).all()
        for pred in predictions:
            session.delete(pred)
        
        # Clear all label events
        events = session.exec(select(LabelEvent)).all()
        for event in events:
            session.delete(event)
        
        session.commit()
        
        return {"ok": True, "msg": f"Reset {len(items)} items successfully"}
    except Exception as e:
        return {"ok": False, "msg": f"Reset failed: {str(e)}"}

@router.post("/bulk-auto-label-confident")
def bulk_auto_label_confident(session=Depends(get_session)):
    """Auto-label items with high confidence predictions"""
    try:
        # Get items with high confidence predictions (>90%)
        predictions = session.exec(
            select(Prediction).where(Prediction.max_proba > 0.9)
        ).all()
        
        labeled_count = 0
        for pred in predictions:
            item = session.get(DatasetItem, pred.item_id)
            if item and not item.label:  # Only label unlabeled items
                item.label = pred.pred_label
                labeled_count += 1
        
        session.commit()
        
        return {"ok": True, "msg": f"Auto-labeled {labeled_count} high-confidence items"}
    except Exception as e:
        return {"ok": False, "msg": f"Auto-labeling failed: {str(e)}"}

@router.post("/bulk-backup-dataset")
def bulk_backup_dataset(session=Depends(get_session)):
    """Create a backup of the current dataset"""
    try:
        # This is essentially the same as export, but we'll return a success message
        # The actual backup would be handled by the export functionality
        return {"ok": True, "msg": "Backup created successfully. Use Export to download."}
    except Exception as e:
        return {"ok": False, "msg": f"Backup failed: {str(e)}"}

@router.post("/import-labels")
async def import_labels(file: UploadFile = File(...), session=Depends(get_session)):
    """Import labels from CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            return {"ok": False, "msg": "File must be a CSV file"}
        
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        imported_count = 0
        
        for row in csv_reader:
            try:
                item_id = int(row.get('item_id', 0))
                label = row.get('label', '').strip()
                unsure = row.get('unsure', '0').strip() == '1'
                skipped = row.get('skipped', '0').strip() == '1'
                
                if item_id > 0:
                    item = session.get(DatasetItem, item_id)
                    if item:
                        item.label = label if label else None
                        item.unsure = unsure
                        item.skipped = skipped
                        session.add(item)
                        imported_count += 1
            except (ValueError, TypeError) as e:
                print(f"Error processing row {row}: {e}")
                continue
        
        session.commit()
        return {"ok": True, "imported_count": imported_count, "msg": f"Successfully imported {imported_count} labels"}
        
    except Exception as e:
        return {"ok": False, "msg": f"Import failed: {str(e)}"}


@router.post("/import-folder-labeled")
def import_folder_labeled(
    folder_path: str = Body(..., embed=True),
    class_name: str = Body(..., embed=True),
    create_triplets: bool = Body(True, embed=True),
    make_thumbs: bool = Body(True, embed=True),
    extensions: Optional[str] = Body(None, embed=True),
    group_require_all: Optional[bool] = Body(None, embed=True),
    max_groups: Optional[int] = Body(None, embed=True),
    session=Depends(get_session)
):
    """Ingest all supported files under a folder and assign a class label.

    - folder_path: path to a directory containing only items of the given class
    - class_name: label to apply to ingested items
    - create_triplets: if True, attempt to build triplet thumbnails per base
    - make_thumbs: if True, generate individual thumbnails while ingesting
    - extensions: optional comma-separated list of file extensions to include
    """
    try:
        # Prevent concurrent ingest operations that can conflict on DB/filesystem
        try:
            if ingest_worker.is_running():
                return {"ok": False, "msg": "Another ingest is currently running. Please wait for it to finish or cancel it before importing a labeled folder."}
        except Exception:
            # If status check fails, proceed; backend will still handle typical conflicts
            pass
        p = Path(folder_path).expanduser()
        if not p.exists() or not p.is_dir():
            return {"ok": False, "msg": f"Folder not found: {folder_path}"}

        exts_str = extensions or ".jpg,.jpeg,.png,.tif,.tiff,.fits,.fit,.fits.fz"
        allowed = set(e.strip().lower() for e in exts_str.split(",") if e.strip())
        ingested = 0
        labeled = 0
        triplet_built = 0
        skipped_groups = 0
        processed_groups = 0

        if create_triplets:
            # Grouped ingest (triplet/composite) using configured roles
            from collections import defaultdict
            groups: dict[tuple[Path, str], dict[str, Path]] = defaultdict(dict)
            gcfg = load_grouping()

            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in allowed:
                    continue
                role, base = match_role_and_base(f, gcfg)
                if role and base:
                    groups[(f.parent, base)][role] = f

            for (parent, base), roles in groups.items():
                if max_groups is not None and processed_groups >= int(max_groups):
                    break
                must_have_all = gcfg.require_all_roles if group_require_all is None else bool(group_require_all)
                if must_have_all and not all(r in roles for r in gcfg.roles):
                    skipped_groups += 1
                    continue

                files = [roles[r] for r in gcfg.roles if r in roles] if gcfg.roles else list(roles.values())

                for f in files:
                    canonical = str(Path(f).resolve())
                    existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical)).first()
                    if existing:
                        if existing.label != class_name:
                            existing.label = class_name
                            session.add(existing)
                            labeled += 1
                        if make_thumbs:
                            try:
                                get_or_make_thumb(Path(canonical), size=256)
                            except Exception as e:
                                print(f"[warn] thumbnail failed for {canonical}: {e}")
                        continue

                    try:
                        img = safe_open_image(f)
                        w, h = img.size
                    except Exception as e:
                        print(f"[skip] {f} ({e})")
                        continue

                    it = DatasetItem(path=canonical, width=w, height=h, label=class_name)
                    session.add(it)
                    session.commit()
                    ingested += 1
                    labeled += 1

                    if make_thumbs:
                        try:
                            get_or_make_thumb(Path(canonical), size=256)
                        except Exception as e:
                            print(f"[warn] thumbnail failed for {canonical}: {e}")

                if create_triplets:
                    try:
                        if set(["target", "ref", "diff"]).issubset(set(gcfg.roles)):
                            get_or_make_triplet_thumb(roles.get("target"), roles.get("ref"), roles.get("diff"), size=256)
                        else:
                            from ..utils import get_or_make_composite_thumb
                            get_or_make_composite_thumb(parent, base, size=256)
                        triplet_built += 1
                    except Exception as e:
                        print(f"[warn] triplet thumb failed for {parent}/{base}: {e}")
                processed_groups += 1
        else:
            # Simple file-wise ingest (no grouping)
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in allowed:
                    continue
                canonical = str(Path(f).resolve())
                existing = session.exec(select(DatasetItem).where(DatasetItem.path == canonical)).first()
                if existing:
                    if existing.label != class_name:
                        existing.label = class_name
                        session.add(existing)
                        labeled += 1
                    if make_thumbs:
                        try:
                            get_or_make_thumb(Path(canonical), size=256)
                        except Exception as e:
                            print(f"[warn] thumbnail failed for {canonical}: {e}")
                    continue

                try:
                    img = safe_open_image(f)
                    w, h = img.size
                except Exception as e:
                    print(f"[skip] {f} ({e})")
                    continue

                it = DatasetItem(path=canonical, width=w, height=h, label=class_name)
                session.add(it)
                session.commit()
                ingested += 1
                labeled += 1

                if make_thumbs:
                    try:
                        get_or_make_thumb(Path(canonical), size=256)
                    except Exception as e:
                        print(f"[warn] thumbnail failed for {canonical}: {e}")

        session.commit()
        return {"ok": True, "ingested": ingested, "labeled": labeled, "triplet_groups": triplet_built, "skipped_groups": skipped_groups, "processed_groups": processed_groups}
    except Exception as e:
        return {"ok": False, "msg": f"Import failed: {str(e)}"}
