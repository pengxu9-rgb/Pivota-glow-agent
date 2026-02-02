from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.v1 import router as v1_router

DEFAULT_CORS_ORIGINS = [
    # Production chatbox custom domain (Vercel).
    "https://aurora.pivota.cc",
    # Default Vercel domain (allows preview via allow_origin_regex when present).
    "https://pivota-aurora-chatbox.vercel.app",
]


def _parse_cors_origins(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _build_allow_origin_regex(origins: list[str]) -> Optional[str]:
    patterns: list[str] = []
    for origin in origins:
        try:
            parsed = urlparse(origin)
        except Exception:
            continue

        if not parsed.scheme or not parsed.hostname:
            continue

        host = parsed.hostname
        if not host.endswith(".vercel.app"):
            continue

        base = host[: -len(".vercel.app")]
        if not base:
            continue

        # Allow Vercel preview URLs for the same project, e.g.
        # - https://<project>.vercel.app
        # - https://<project>-git-main-<team>.vercel.app
        # - https://<project>-<hash>.vercel.app
        patterns.append(rf"{re.escape(parsed.scheme)}://{re.escape(base)}(-.*)?\.vercel\.app")

    if not patterns:
        return None

    return rf"^(?:{'|'.join(patterns)})$"


def _setup_logging() -> None:
    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_app() -> FastAPI:
    _setup_logging()
    app = FastAPI(title="Pivota Glow Agent", version="0.1.0")

    origins = _parse_cors_origins(os.getenv("CORS_ORIGINS"))
    allow_all = "*" in origins
    if not allow_all:
        for origin in DEFAULT_CORS_ORIGINS:
            if origin not in origins:
                origins.append(origin)
    allow_origin_regex = None if allow_all else _build_allow_origin_regex(origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origin_regex=allow_origin_regex,
        max_age=86400,
    )

    app.include_router(health_router)
    app.include_router(v1_router, prefix="/v1")

    return app


app = create_app()
