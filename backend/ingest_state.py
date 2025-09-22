from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import json

from .config import STORE_DIR

_STATE_PATH = STORE_DIR / "ingest_state.json"


def _read_state() -> dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _write_state(data: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        # Do not raise on persistence issues
        pass


def get_ingest_mode() -> Optional[Literal["triplet", "files"]]:
    """Return the saved ingestion mode if any.

    Modes:
    - "triplet": ingestion was done by groups (triplets or configured grouping)
    - "files": ingestion was done as individual files
    """
    st = _read_state()
    mode = st.get("mode")
    if mode in {"triplet", "files"}:
        return mode  # type: ignore[return-value]
    return None


def set_ingest_mode(mode: Literal["triplet", "files"]) -> None:
    st = _read_state()
    st["mode"] = mode
    _write_state(st)


