from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.store.session_store import InMemorySessionStore, PersistentSessionStore, SessionData, SessionDataPatch


class TestInMemorySessionStore(unittest.IsolatedAsyncioTestCase):
    async def test_set_get_roundtrip(self) -> None:
        store = InMemorySessionStore(default_ttl_days=30.0)
        await store.set("uid_123", SessionData(profile={"skin_type": "oily"}, goals=["acne"]))
        loaded = await store.get("uid_123")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.profile, {"skin_type": "oily"})
        self.assertEqual(loaded.goals, ["acne"])

    async def test_patch_deep_merge_and_strip_pii(self) -> None:
        store = InMemorySessionStore(default_ttl_days=30.0)
        await store.set(
            "uid_123",
            SessionData(profile={"skin_type": "oily", "nested": {"a": 1}}),
        )
        await store.patch(
            "uid_123",
            SessionDataPatch(profile={"email": "test@example.com", "nested": {"b": 2}}),
        )
        loaded = await store.get("uid_123")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.profile.get("nested"), {"a": 1, "b": 2})
        self.assertNotIn("email", loaded.profile)

    async def test_ttl_expires(self) -> None:
        store = InMemorySessionStore(default_ttl_days=30.0)
        await store.set("uid_123", SessionData(profile={"k": "v"}), ttl_days=1.0 / 86400.0)
        await asyncio.sleep(1.1)
        loaded = await store.get("uid_123")
        self.assertIsNone(loaded)


class TestPersistentSessionStore(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_falls_back_when_redis_unavailable(self) -> None:
        store = PersistentSessionStore(
            redis_url="redis://localhost:6390/0",
            connect_timeout_s=0.05,
            socket_timeout_s=0.05,
            default_ttl_days=30.0,
        )
        await store.initialize()
        self.assertEqual(store.backend_kind, "memory")

        await store.patch("uid_1", {"profile": {"skin_type": "dry"}})
        loaded = await store.get("uid_1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.profile, {"skin_type": "dry"})

