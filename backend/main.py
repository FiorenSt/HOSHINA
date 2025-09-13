
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pathlib import Path
from .config import PROJECT_ROOT
from .db import init_db
from . import thumb_worker  # ensure worker module is importable
from . import ingest_worker  # ensure worker module is importable
from .routes.api import router as api_router

app = FastAPI(title="HOSHINA", version="0.1.0")

# CORS (allow browser clients elsewhere if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
init_db()

# API routes
app.include_router(api_router, prefix="/api")

# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

# Serve thumbnails as static files for direct access
from .config import THUMB_DIR
app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")

@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    return FileResponse(index)

@app.get("/healthz")
def health():
    return {"ok": True}

# Graceful shutdown of background workers
@app.on_event("shutdown")
def shutdown_event():
    try:
        thumb_worker.shutdown(timeout=3.0)
    except Exception:
        pass
    try:
        ingest_worker.shutdown(timeout=3.0)
    except Exception:
        pass
