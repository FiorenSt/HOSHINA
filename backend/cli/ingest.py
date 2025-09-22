import argparse
from pathlib import Path

from PIL import Image  # noqa: F401  # Imported for potential side effects/compat
from sqlmodel import select

from backend.db import get_session, DatasetItem, init_db
from backend.utils import safe_open_image, get_or_make_full_png
from backend.config import DB_PATH


def is_image(path: Path, allowed_exts: set[str]) -> bool:
    return path.suffix.lower() in allowed_exts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest images into the HOSHINA DB")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing images (recursively scanned)")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.tif,.tiff,.fits",
        help="Comma-separated list of file extensions",
    )
    parser.add_argument("--use-zscale", dest="use_zscale", action="store_true", help="Enable Astropy ZScale for FITS (default)")
    parser.add_argument("--no-zscale", dest="use_zscale", action="store_false", help="Disable ZScale; use percentile scaling")
    parser.set_defaults(use_zscale=True)
    parser.add_argument("--zscale-contrast", type=float, default=0.1, help="ZScale contrast (e.g., 0.1). Only used if --use-zscale.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}

    print(f"[ingest] DB: {DB_PATH}")
    print(f"[ingest] Scanning: {data_dir}")
    print(f"[ingest] Extensions: {exts}")

    init_db()
    session = get_session()

    count = 0
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if not is_image(p, exts):
            continue
        existing = session.exec(select(DatasetItem).where(DatasetItem.path == str(p))).first()
        if existing:
            continue
        try:
            img = safe_open_image(p, use_zscale=args.use_zscale, zscale_contrast=args.zscale_contrast)
            w, h = img.size
        except Exception as e:
            print(f"[skip] {p} ({e})")
            continue
        it = DatasetItem(path=str(p), width=w, height=h)
        session.add(it)
        session.commit()
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


