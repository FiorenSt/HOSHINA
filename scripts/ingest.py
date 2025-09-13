
import argparse, os, sys, json
from pathlib import Path

# Add the parent directory to the path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from sqlmodel import select
from backend.db import get_session, DatasetItem, init_db
from backend.utils import safe_open_image, get_or_make_thumb, get_or_make_full_png
from backend.config import DB_PATH, STORE_DIR

def is_image(p: Path, exts):
    return p.suffix.lower() in exts

def main():
    parser = argparse.ArgumentParser(description="Ingest images into the HOSHINA DB")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing images (recursively scanned)")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.tif,.tiff,.fits", help="Comma-separated list")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    exts = set([e.strip().lower() for e in args.extensions.split(",") if e.strip()])

    print(f"[ingest] DB: {DB_PATH}")
    print(f"[ingest] Scanning: {data_dir}")
    print(f"[ingest] Extensions: {exts}")

    init_db()
    session = get_session()

    # Recursively scan
    count = 0
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if not is_image(p, exts):
            continue
        # Add if not present
        existing = session.exec(select(DatasetItem).where(DatasetItem.path == str(p))).first()
        if existing:
            continue
        try:
            img = safe_open_image(p)
            w, h = img.size
        except Exception as e:
            print(f"[skip] {p} ({e})")
            continue
        it = DatasetItem(path=str(p), width=w, height=h)
        session.add(it)
        session.commit()
        # build full-res PNG (thumbnails will be generated from this)
        try:
            get_or_make_full_png(p)
        except Exception as e:
            print(f"[warn] PNG generation failed for {p}: {e}")
        count += 1
        if count % 100 == 0:
            print(f"[ingest] added {count} items")

    print(f"[done] Ingested {count} items")

if __name__ == "__main__":
    main()
