"""API routers."""
from fastapi import APIRouter

from . import captures, config, explain, health, models

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(explain.models_router)   # /models/{id}/layers
api_router.include_router(config.router)
# Explain first: its /captures/{id}/explain/... paths must not be swallowed by
# captures' /captures/{id}/files/{name} and /captures/{id} catch-alls.
api_router.include_router(explain.router)
api_router.include_router(captures.router)
