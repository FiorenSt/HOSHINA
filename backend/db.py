
from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy.pool import NullPool
from typing import Optional, List, Dict
from pathlib import Path
import datetime as dt
from .config import DB_PATH

# SQLite DB engine
# SQLite engine with safer defaults for web apps
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)

class DatasetItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    width: Optional[int] = None
    height: Optional[int] = None
    meta_json: Optional[str] = None
    ingested_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    # current label (denormalized for convenience)
    label: Optional[str] = Field(default=None, index=True)
    unsure: bool = Field(default=False, index=True)
    skipped: bool = Field(default=False, index=True)

class LabelEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(index=True, foreign_key="datasetitem.id")
    prev_label: Optional[str] = None
    new_label: Optional[str] = None
    user: Optional[str] = None
    unsure: bool = Field(default=False)
    skipped: bool = Field(default=False)
    ts: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())

class Embedding(SQLModel, table=True):
    item_id: int = Field(primary_key=True, foreign_key="datasetitem.id")
    # numpy serialized to bytes
    vector: bytes

class Prediction(SQLModel, table=True):
    item_id: int = Field(primary_key=True, foreign_key="datasetitem.id")
    proba_json: str  # {"class": prob}
    pred_label: Optional[str] = None
    margin: Optional[float] = None  # uncertainty metric
    max_proba: Optional[float] = None
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())

class ClassDef(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    key: Optional[str] = None  # keyboard shortcut (1..9)
    order: int = Field(default=0)

class UMAPCoords(SQLModel, table=True):
    item_id: int = Field(primary_key=True, foreign_key="datasetitem.id")
    x: float
    y: float
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    # Dependency that yields a session and ensures it is closed after the request
    with Session(engine) as session:
        yield session


def recreate_database() -> None:
    """Recreate a fresh SQLite database file and schema.

    - Disposes existing engine connections
    - Removes main DB file and any WAL/SHM sidecar files
    - Recreates empty schema
    """
    # Ensure no pooled/open connections remain
    try:
        engine.dispose()
    except Exception:
        pass

    db_main = Path(DB_PATH)

    # Remove database file and sidecars if present
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(db_main) + suffix)
        try:
            if p.exists():
                p.unlink()
        except Exception:
            # Ignore removal errors to avoid blocking reset
            pass

    # Ensure parent directory exists
    try:
        db_main.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Recreate empty schema
    init_db()
