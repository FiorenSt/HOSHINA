
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pathlib import Path
import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from .config import PROJECT_ROOT
from .db import init_db
from . import thumb_worker  # ensure worker module is importable
from . import ingest_worker  # ensure worker module is importable
# Monolithic router kept only as a fallback import path inside try/except below

# Structured logging with rotating file handler
_handlers = [logging.StreamHandler()]
try:
    _handlers.append(RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5))
except Exception:
    # If filesystem not writable, continue with stdout only
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        init_db()
    except Exception as e:
        logger.exception("DB init failed: %s", e)
    yield
    # Shutdown
    try:
        thumb_worker.shutdown(timeout=3.0)
    except Exception:
        pass
    try:
        ingest_worker.shutdown(timeout=3.0)
    except Exception:
        pass


app = FastAPI(title="HOSHINA", version="0.1.0", lifespan=lifespan)

# CORS (restrict in production)
environment = os.getenv("ENVIRONMENT", "development").lower()
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if environment == "production":
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or []
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
try:
    from .routes import router as modular_router
    app.include_router(modular_router, prefix="/api")
except Exception:
    # Fallback to monolithic if modular fails
    try:
        from .routes.api import router as api_router
        app.include_router(api_router, prefix="/api")
    except Exception:
        pass

# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

# On-the-fly mode: no static thumbnails mount

@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    return FileResponse(index)

@app.get("/healthz")
def health():
    return {"ok": True}

# Shutdown handled via lifespan
