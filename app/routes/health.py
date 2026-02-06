from __future__ import annotations

import os

from fastapi import APIRouter

from app.store.session_store import PERSISTENT_SESSION_STORE

router = APIRouter()


def _get_commit_sha() -> str | None:
    for key in (
        # Railway
        "RAILWAY_GIT_COMMIT_SHA",
        "RAILWAY_GIT_COMMIT",
        # Common CI providers
        "GITHUB_SHA",
        # Vercel (if ever deployed there)
        "VERCEL_GIT_COMMIT_SHA",
        # Generic fallbacks
        "COMMIT_SHA",
        "GIT_SHA",
    ):
        value = os.getenv(key)
        if value:
            return value
    return None


@router.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "pivota-glow-agent",
        "commit_sha": _get_commit_sha(),
        "environment": os.getenv("RAILWAY_ENVIRONMENT_NAME") or os.getenv("ENVIRONMENT"),
        "persistent_session_store_backend": PERSISTENT_SESSION_STORE.backend_kind,
    }
