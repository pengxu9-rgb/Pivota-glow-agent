from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.v1 import router as v1_router


def _parse_cors_origins(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    app.include_router(health_router)
    app.include_router(v1_router, prefix="/v1")

    return app


app = create_app()
