from __future__ import annotations

from pathlib import Path
import io
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from ..db import DatasetItem
from ..utils import safe_open_image
from ..grouping import load_config as load_grouping, match_role_and_base, resolve_existing_role_file
from .. import config as cfg


def get_triplet_details(item_id: int, session):
    it = session.get(DatasetItem, item_id)
    if not it:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Item not found")
    p = Path(it.path)
    gcfg = load_grouping()
    _, base = match_role_and_base(p, gcfg)

    out = {}
    from sqlmodel import select
    for role in gcfg.roles:
        rp = resolve_existing_role_file(p.parent, base, role, gcfg)
        # Fallback: if suffix map doesn't encode extensions or uses different ones,
        # search for any file matching base + suffix (with or without extension)
        if rp is None:
            suf_list = gcfg.suffix_map.get(role, []) or [f"_{role}"]
            found: Path | None = None
            for suf in suf_list:
                cand = p.parent / f"{base}{suf}"
                if cand.exists():
                    found = cand
                    break
                try:
                    # search any extension variant
                    for q in p.parent.glob(f"{base}{suf}.*"):
                        if q.is_file():
                            found = q
                            break
                    if found:
                        break
                except Exception:
                    pass
            rp = found
        if rp is None:
            # Last-chance generic search: look for files beginning with base_rolename.*
            try:
                for q in p.parent.glob(f"{base}_{role}.*"):
                    if q.is_file():
                        rp = q
                        break
            except Exception:
                pass
        if rp is None:
            continue
        canonical = str(Path(rp).resolve())
        row = session.exec(select(DatasetItem).where(DatasetItem.path == canonical)).first()
        if row:
            out[role] = {
                "id": row.id,
                "file": f"/api/file/{row.id}",
                "thumb": f"/api/thumb/{row.id}",
                "path": row.path,
                "label": row.label,
            }

    return {"group_key": base, "items": out}

def get_thumb(item_id: int, size: int, session) -> StreamingResponse:
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    p = Path(it.path)
    try:
        img = safe_open_image(p).convert("RGB")
        from PIL import ImageOps, Image
        img = ImageOps.exif_transpose(img)
        s = int(size)
        # Add a small margin so thumbnails don't fully touch edges
        pad = max(1, int(0.06 * s))
        box = max(1, s - 2 * pad)
        # Resize to fit inside box while preserving aspect ratio (upscale if needed)
        w, h = img.width, img.height
        if w <= 0 or h <= 0:
            w, h = 1, 1
        scale = min(box / w, box / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (s, s), color=(0, 0, 0))
        x = pad + (box - img.width) // 2
        y = pad + (box - img.height) // 2
        canvas.paste(img, (x, y))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception:
        from PIL import Image
        s = int(size)
        ph = Image.new("RGB", (s, s), color=(32, 32, 32))
        buf = io.BytesIO()
        ph.save(buf, format="JPEG", quality=70)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def get_triplet_thumb(item_id: int, size: int, session) -> StreamingResponse:
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")

    p = Path(it.path)
    cfg_g = load_grouping()
    _, base = match_role_and_base(p, cfg_g)
    target = resolve_existing_role_file(p.parent, base, "target", cfg_g) if "target" in cfg_g.roles else None
    ref = resolve_existing_role_file(p.parent, base, "ref", cfg_g) if "ref" in cfg_g.roles else None
    diff = resolve_existing_role_file(p.parent, base, "diff", cfg_g) if "diff" in cfg_g.roles else None

    s = int(size)
    from PIL import Image, ImageOps
    panels = []
    for rp in [target, ref, diff]:
        try:
            if rp and Path(rp).exists():
                img = safe_open_image(Path(rp)).convert("RGB")
            else:
                img = Image.new("RGB", (s, s), color=(0,0,0))
        except Exception:
            img = Image.new("RGB", (s, s), color=(0,0,0))
        img = ImageOps.exif_transpose(img)
        pad = max(1, int(0.06 * s))
        box = max(1, s - 2 * pad)
        w, h = img.width, img.height
        if w <= 0 or h <= 0:
            w, h = 1, 1
        scale = min(box / w, box / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (s, s), color=(0, 0, 0))
        x = pad + (box - img.width) // 2
        y = pad + (box - img.height) // 2
        canvas.paste(img, (x, y))
        panels.append(canvas)
    out = Image.new("RGB", (s * 3, s), color=(0,0,0))
    for i, panel in enumerate(panels[:3]):
        out.paste(panel, (i * s, 0))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def get_group_thumb(item_id: int, size: int, session) -> StreamingResponse:
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    p = Path(it.path)
    cfg_g = load_grouping()
    _, base = match_role_and_base(p, cfg_g)
    from PIL import Image, ImageOps
    s = int(size)
    panels = []
    for role in cfg_g.roles:
        rp = resolve_existing_role_file(p.parent, base, role, cfg_g)
        try:
            if rp and Path(rp).exists():
                img = safe_open_image(Path(rp)).convert("RGB")
            else:
                img = Image.new("RGB", (s, s), color=(0,0,0))
        except Exception:
            img = Image.new("RGB", (s, s), color=(0,0,0))
        img = ImageOps.exif_transpose(img)
        pad = max(1, int(0.06 * s))
        box = max(1, s - 2 * pad)
        w, h = img.width, img.height
        if w <= 0 or h <= 0:
            w, h = 1, 1
        scale = min(box / w, box / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (s, s), color=(0, 0, 0))
        x = pad + (box - img.width) // 2
        y = pad + (box - img.height) // 2
        canvas.paste(img, (x, y))
        panels.append(canvas)
    out = Image.new("RGB", (s * max(1, len(panels)), s), color=(0,0,0))
    for i, pi in enumerate(panels):
        out.paste(pi, (i * s, 0))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def get_file(item_id: int, session) -> FileResponse | StreamingResponse:
    it = session.get(DatasetItem, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    file_path = Path(it.path)
    try:
        rp = file_path.resolve(strict=True)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")

    # Access check relaxed: item_id comes from DB, so the path is trusted.
    # We no longer restrict to DATA_DIR for full-resolution viewing.

    if rp.suffix.lower() == '.fits':
        try:
            from PIL import Image
            img = safe_open_image(rp).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render FITS: {str(e)}")
    return FileResponse(str(rp))


