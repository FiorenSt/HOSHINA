
import os
from pathlib import Path

# Root directory of project (this repo)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage (DB, thumbnails, models)
STORE_DIR = Path(os.getenv("AL_STORE_DIR", PROJECT_ROOT / "store"))
STORE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("AL_DB_PATH", STORE_DIR / "app.db"))

# Data directory where images live
DATA_DIR = Path(os.getenv("AL_DATA_DIR", ""))  # set during ingest; can override here
if DATA_DIR and not Path(DATA_DIR).exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Thumbnails directory
THUMB_DIR = STORE_DIR / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings / models cache
CACHE_DIR = STORE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Which embedding backend to use: 'auto' | 'tf' | 'hog'
EMBEDDING_BACKEND = os.getenv("AL_EMBEDDING_BACKEND", "auto")

# UMAP cache file
UMAP_CACHE = STORE_DIR / "umap_cache.npz"

# Max items per page to avoid overloading the browser
MAX_PAGE_SIZE = 300

# Runtime setter to update the data directory from the server/UI
def set_data_dir(new_dir: str) -> Path:
    """Update the global DATA_DIR at runtime and ensure it exists.

    Returns the resolved Path to the configured data directory.
    """
    global DATA_DIR
    p = Path(new_dir)
    if p and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    DATA_DIR = p
    return DATA_DIR
