# empty to mark package

# Initialize singleton worker state on import (best-effort)
try:
    from . import thumb_worker  # noqa: F401
except Exception:
    pass