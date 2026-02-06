from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import Any, Mapping, Optional, Protocol, Union, cast

from pydantic import BaseModel, ConfigDict, Field

try:
    import redis.asyncio as aioredis  # type: ignore[import-not-found]
    from redis.exceptions import RedisError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency in some envs
    aioredis = None  # type: ignore[assignment]

    class RedisError(Exception):
        pass


logger = logging.getLogger("pivota-glow-agent.session-store")

SCHEMA_VERSION = "0.1"


GoalsType = Union[str, list[str], dict[str, Any]]
SensitivitiesType = Union[str, list[str], dict[str, Any]]


class SessionData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION)
    profile: Optional[dict[str, Any]] = None
    goals: Optional[GoalsType] = None
    sensitivities: Optional[SensitivitiesType] = None
    current_products: Optional[list[dict[str, Any]]] = None
    minimal_plan: Optional[dict[str, Any]] = None
    last_state: Optional[str] = None
    last_seen_at: Optional[datetime] = None
    last_checkin_at: Optional[datetime] = None


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION)
    goal_summary: Optional[str] = None
    plan_summary: Optional[str] = None
    sensitivities_summary: Optional[str] = None
    days_since_last: Optional[int] = None


class SessionDataPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile: Optional[dict[str, Any]] = None
    goals: Optional[GoalsType] = None
    sensitivities: Optional[SensitivitiesType] = None
    current_products: Optional[list[dict[str, Any]]] = None
    minimal_plan: Optional[dict[str, Any]] = None
    last_state: Optional[str] = None
    last_seen_at: Optional[datetime] = None
    last_checkin_at: Optional[datetime] = None


class SessionStore(Protocol):
    async def get(self, uid: str) -> Optional[SessionData]: ...

    async def set(self, uid: str, session: SessionData, *, ttl_days: Optional[float] = None) -> None: ...

    async def patch(self, uid: str, partial_update: SessionDataPatch | Mapping[str, Any], *, ttl_days: Optional[float] = None) -> SessionData: ...

    async def close(self) -> None: ...


_FORBIDDEN_PII_KEYS = {
    "email",
    "e-mail",
    "phone",
    "phone_number",
    "phonenumber",
    "mobile",
    "tel",
    "telephone",
}


def _strip_pii(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.strip().lower() in _FORBIDDEN_PII_KEYS:
                continue
            cleaned[k] = _strip_pii(v)
        return cleaned
    if isinstance(value, list):
        return [_strip_pii(v) for v in value]
    return value


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = dict(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged.get(key), value)
            else:
                merged[key] = value
        return merged
    return patch


def _coerce_ttl_seconds(ttl_days: Optional[float], default_ttl_days: float) -> float:
    days = default_ttl_days if ttl_days is None else float(ttl_days)
    if days <= 0:
        return 0.0
    return days * 86400.0


def build_session_summary(session: SessionData, *, now: Optional[datetime] = None) -> SessionSummary:
    now_dt = now or datetime.now(timezone.utc)
    days_since_last: Optional[int] = None
    if session.last_seen_at:
        last_seen = session.last_seen_at
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        else:
            last_seen = last_seen.astimezone(timezone.utc)
        delta = now_dt - last_seen
        days_since_last = max(0, int(delta.total_seconds() // 86400))

    return SessionSummary(
        goal_summary=_summarize_field(session.goals),
        plan_summary=_summarize_plan(session.minimal_plan),
        sensitivities_summary=_summarize_field(session.sensitivities),
        days_since_last=days_since_last,
    )


def _summarize_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(items)[:500] if items else None
    if isinstance(value, dict):
        if isinstance(value.get("summary"), str):
            s = value["summary"].strip()
            return s[:500] if s else None
        if isinstance(value.get("items"), list):
            items = [str(v).strip() for v in value.get("items") if str(v).strip()]
            return ", ".join(items)[:500] if items else None
    return str(value).strip()[:500] or None


def _summarize_plan(plan: Any) -> Optional[str]:
    if plan is None:
        return None
    if isinstance(plan, str):
        return plan.strip()[:800] or None
    if isinstance(plan, dict):
        if isinstance(plan.get("summary"), str):
            s = plan["summary"].strip()
            return s[:800] if s else None
        parts: list[str] = []
        for key in ("am", "pm", "morning", "night"):
            v = plan.get(key)
            if isinstance(v, list) and v:
                steps = [str(x).strip() for x in v if str(x).strip()]
                if steps:
                    parts.append(f"{key.upper()}: " + "; ".join(steps)[:400])
        return " | ".join(parts)[:800] or None
    return str(plan).strip()[:800] or None


def _normalize_uid(uid: str) -> str:
    if not isinstance(uid, str):
        raise TypeError("uid must be a string")
    normalized = uid.strip()
    if not normalized:
        raise ValueError("uid must be non-empty")
    if len(normalized) > 200:
        raise ValueError("uid too long")
    return normalized



def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _session_to_json_dict(session: SessionData) -> dict[str, Any]:
    data = session.model_dump(mode="json")
    data["schema_version"] = SCHEMA_VERSION
    return cast(dict[str, Any], _strip_pii(data))


def _patch_to_json_dict(partial_update: SessionDataPatch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(partial_update, SessionDataPatch):
        data = partial_update.model_dump(exclude_unset=True, mode="json")
    else:
        data = SessionDataPatch.model_validate(partial_update).model_dump(exclude_unset=True, mode="json")
    data.pop("schema_version", None)
    return cast(dict[str, Any], _strip_pii(data))


class InMemorySessionStore(SessionStore):
    def __init__(self, *, default_ttl_days: float = 30.0) -> None:
        self._default_ttl_days = default_ttl_days
        self._lock = asyncio.Lock()
        self._items: dict[str, tuple[dict[str, Any], Optional[float]]] = {}

    async def get(self, uid: str) -> Optional[SessionData]:
        key = _normalize_uid(uid)
        async with self._lock:
            record = self._items.get(key)
            if not record:
                return None
            data, expires_at = record
            if expires_at is not None and time.monotonic() >= expires_at:
                self._items.pop(key, None)
                return None
            return SessionData.model_validate(data)

    async def set(self, uid: str, session: SessionData, *, ttl_days: Optional[float] = None) -> None:
        key = _normalize_uid(uid)
        ttl_seconds = _coerce_ttl_seconds(ttl_days, self._default_ttl_days)
        expires_at = None if ttl_seconds <= 0 else (time.monotonic() + ttl_seconds)
        data = _session_to_json_dict(session)
        async with self._lock:
            self._items[key] = (data, expires_at)

    async def patch(self, uid: str, partial_update: SessionDataPatch | Mapping[str, Any], *, ttl_days: Optional[float] = None) -> SessionData:
        key = _normalize_uid(uid)
        ttl_seconds = _coerce_ttl_seconds(ttl_days, self._default_ttl_days)
        expires_at = None if ttl_seconds <= 0 else (time.monotonic() + ttl_seconds)
        patch_dict = _patch_to_json_dict(partial_update)

        async with self._lock:
            base: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
            record = self._items.get(key)
            if record:
                existing, existing_expires_at = record
                if existing_expires_at is None or time.monotonic() < existing_expires_at:
                    base = dict(existing)

            merged = cast(dict[str, Any], _deep_merge(base, patch_dict))
            merged["schema_version"] = SCHEMA_VERSION
            self._items[key] = (merged, expires_at)

        return SessionData.model_validate(merged)

    async def close(self) -> None:
        return None


class RedisSessionStore(SessionStore):
    def __init__(
        self,
        *,
        redis_url: str,
        default_ttl_days: float = 30.0,
        connect_timeout_s: float = 1.0,
        socket_timeout_s: float = 1.0,
        key_prefix: str = "aurora_session",
    ) -> None:
        if aioredis is None:
            raise RuntimeError("redis dependency not available")
        self._redis_url = redis_url
        self._default_ttl_days = default_ttl_days
        self._connect_timeout_s = connect_timeout_s
        self._socket_timeout_s = socket_timeout_s
        self._key_prefix = key_prefix.strip(":") or "aurora_session"
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=self._connect_timeout_s,
            socket_timeout=self._socket_timeout_s,
        )

    async def ping(self) -> None:
        await self._redis.ping()

    def _key(self, uid: str) -> str:
        return f"{self._key_prefix}:{_normalize_uid(uid)}"

    async def get(self, uid: str) -> Optional[SessionData]:
        raw = await self._redis.get(self._key(uid))
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except Exception:
            logger.warning("redis_session_parse_failed uid=%s", uid)
            return None
        if not isinstance(obj, dict):
            return None
        return SessionData.model_validate(obj)

    async def set(self, uid: str, session: SessionData, *, ttl_days: Optional[float] = None) -> None:
        ttl_seconds = _coerce_ttl_seconds(ttl_days, self._default_ttl_days)
        ttl_seconds_int = int(max(1.0, ttl_seconds)) if ttl_seconds > 0 else 0
        data = _session_to_json_dict(session)
        value = _json_dumps(data)
        if ttl_seconds_int > 0:
            await self._redis.set(self._key(uid), value, ex=ttl_seconds_int)
        else:
            await self._redis.set(self._key(uid), value)

    async def patch(self, uid: str, partial_update: SessionDataPatch | Mapping[str, Any], *, ttl_days: Optional[float] = None) -> SessionData:
        key = self._key(uid)
        existing_raw = await self._redis.get(key)
        base: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        if existing_raw:
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, dict):
                    base = parsed
            except Exception:
                pass

        patch_dict = _patch_to_json_dict(partial_update)
        merged = cast(dict[str, Any], _deep_merge(base, patch_dict))
        merged["schema_version"] = SCHEMA_VERSION

        ttl_seconds = _coerce_ttl_seconds(ttl_days, self._default_ttl_days)
        ttl_seconds_int = int(max(1.0, ttl_seconds)) if ttl_seconds > 0 else 0
        value = _json_dumps(merged)

        if ttl_seconds_int > 0:
            await self._redis.set(key, value, ex=ttl_seconds_int)
        else:
            await self._redis.set(key, value)

        return SessionData.model_validate(merged)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:
            pass


class PersistentSessionStore(SessionStore):
    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        default_ttl_days: Optional[float] = None,
        connect_timeout_s: float = 1.0,
        socket_timeout_s: float = 1.0,
        key_prefix: str = "aurora_session",
    ) -> None:
        self._redis_url = redis_url
        self._default_ttl_days = default_ttl_days
        self._connect_timeout_s = connect_timeout_s
        self._socket_timeout_s = socket_timeout_s
        self._key_prefix = key_prefix

        ttl_days = self._default_ttl_days if self._default_ttl_days is not None else _env_float("SESSION_TTL_DAYS", 30.0)
        self._backend: SessionStore = InMemorySessionStore(default_ttl_days=ttl_days)
        self._backend_kind = "memory"

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    async def initialize(self) -> None:
        redis_url = (self._redis_url or os.getenv("REDIS_URL") or "").strip() or None
        ttl_days = self._default_ttl_days if self._default_ttl_days is not None else _env_float("SESSION_TTL_DAYS", 30.0)

        if not redis_url:
            self._backend = InMemorySessionStore(default_ttl_days=ttl_days)
            self._backend_kind = "memory"
            logger.info("persistent_session_store_backend=memory reason=missing_REDIS_URL")
            return

        if aioredis is None:
            self._backend = InMemorySessionStore(default_ttl_days=ttl_days)
            self._backend_kind = "memory"
            logger.warning("persistent_session_store_backend=memory reason=redis_dependency_missing")
            return

        try:
            redis_backend = RedisSessionStore(
                redis_url=redis_url,
                default_ttl_days=ttl_days,
                connect_timeout_s=self._connect_timeout_s,
                socket_timeout_s=self._socket_timeout_s,
                key_prefix=self._key_prefix,
            )
            await redis_backend.ping()
        except Exception as exc:
            self._backend = InMemorySessionStore(default_ttl_days=ttl_days)
            self._backend_kind = "memory"
            logger.warning(
                "persistent_session_store_backend=memory reason=redis_unavailable err=%s",
                getattr(exc, "message", str(exc)),
            )
            return

        self._backend = redis_backend
        self._backend_kind = "redis"
        logger.info("persistent_session_store_backend=redis")

    async def get(self, uid: str) -> Optional[SessionData]:
        try:
            return await self._backend.get(uid)
        except RedisError as exc:
            logger.warning("persistent_session_store_get_failed backend=%s err=%s", self._backend_kind, getattr(exc, "message", str(exc)))
            await self._fallback_to_memory(reason="redis_error")
            return None

    async def set(self, uid: str, session: SessionData, *, ttl_days: Optional[float] = None) -> None:
        try:
            await self._backend.set(uid, session, ttl_days=ttl_days)
        except RedisError as exc:
            logger.warning("persistent_session_store_set_failed backend=%s err=%s", self._backend_kind, getattr(exc, "message", str(exc)))
            await self._fallback_to_memory(reason="redis_error")

    async def patch(self, uid: str, partial_update: SessionDataPatch | Mapping[str, Any], *, ttl_days: Optional[float] = None) -> SessionData:
        try:
            return await self._backend.patch(uid, partial_update, ttl_days=ttl_days)
        except RedisError as exc:
            logger.warning("persistent_session_store_patch_failed backend=%s err=%s", self._backend_kind, getattr(exc, "message", str(exc)))
            await self._fallback_to_memory(reason="redis_error")
            return await self._backend.patch(uid, partial_update, ttl_days=ttl_days)

    async def _fallback_to_memory(self, *, reason: str) -> None:
        if self._backend_kind == "memory":
            return
        try:
            await self._backend.close()
        except Exception:
            pass
        ttl_days = self._default_ttl_days if self._default_ttl_days is not None else _env_float("SESSION_TTL_DAYS", 30.0)
        self._backend = InMemorySessionStore(default_ttl_days=ttl_days)
        self._backend_kind = "memory"
        logger.warning("persistent_session_store_backend=memory reason=%s", reason)

    async def close(self) -> None:
        await self._backend.close()


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


PERSISTENT_SESSION_STORE = PersistentSessionStore()
