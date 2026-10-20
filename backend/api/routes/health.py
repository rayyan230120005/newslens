"""
routes/health.py

Just answers "is the server up." No dependencies on models, cache, or
anything else - this exists so we can confirm the backend boots and
responds before adding a single line of actual pipeline logic.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
