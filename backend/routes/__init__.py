from fastapi import APIRouter

router = APIRouter()

try:
    from .predictions import router as predictions_router
    router.include_router(predictions_router)
except Exception:
    # During progressive refactor, missing modules are tolerated
    pass

try:
    from .ingest import router as ingest_router
    router.include_router(ingest_router)
except Exception:
    pass

try:
    from .media import router as media_router
    router.include_router(media_router)
except Exception:
    pass

try:
    from .classes import router as classes_router
    router.include_router(classes_router)
except Exception:
    pass

try:
    from .items import router as items_router
    router.include_router(items_router)
except Exception:
    pass

try:
    from .labels import router as labels_router
    router.include_router(labels_router)
except Exception:
    pass

try:
    from .config import router as config_router
    router.include_router(config_router)
except Exception:
    pass

try:
    from .train import router as train_router
    router.include_router(train_router)
except Exception:
    pass

try:
    from .analytics import router as analytics_router
    router.include_router(analytics_router)
except Exception:
    pass

try:
    from .exports import router as exports_router
    router.include_router(exports_router)
except Exception:
    pass

try:
    from .bulk import router as bulk_router
    router.include_router(bulk_router)
except Exception:
    pass

try:
    from .imports import router as imports_router
    router.include_router(imports_router)
except Exception:
    pass

try:
    from .admin import router as admin_router
    router.include_router(admin_router)
except Exception:
    pass


