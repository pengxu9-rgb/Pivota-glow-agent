from __future__ import annotations

import asyncio
import time
from typing import Any, Optional


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    async def get(self, brief_id: str) -> dict[str, Any]:
        async with self._lock:
            return dict(self._sessions.get(brief_id, {}))

    async def upsert(self, brief_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            current = self._sessions.get(brief_id) or {}
            merged = _deep_merge(current, patch)
            merged["_updated_at_ms"] = int(time.time() * 1000)
            self._sessions[brief_id] = merged
            return dict(merged)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and key in {"photos", "selected_offers", "product_selections"}
        ):
            merged[key] = {**(merged[key] or {}), **value}
        else:
            merged[key] = value
    return merged


SESSION_STORE = SessionStore()

