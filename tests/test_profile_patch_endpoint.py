from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import uuid

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app


class TestSessionProfilePatchEndpoint(unittest.TestCase):
    def test_patch_requires_uid(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        with TestClient(app) as client:
            res = client.post("/v1/session/profile/patch", json={"goal_primary": "breakouts"})

        self.assertEqual(res.status_code, 400)

    def test_patch_persists_and_bootstrap_reflects_goal_primary(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        uid = f"uid_test_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            patch_res = client.post(
                "/v1/session/profile/patch",
                headers={"X-Aurora-Uid": uid},
                json={"goal_primary": "breakouts", "skin_feel": "oily"},
            )
            self.assertEqual(patch_res.status_code, 200)
            patch_data = patch_res.json()
            self.assertTrue(patch_data["ok"])
            self.assertEqual(patch_data["session"]["profile"]["goal_primary"], "breakouts")

            boot_res = client.get(
                "/v1/session/bootstrap",
                headers={"X-Aurora-Uid": uid, "X-Aurora-Lang": "en"},
            )

        self.assertEqual(boot_res.status_code, 200)
        data = boot_res.json()
        self.assertTrue(data["is_returning"])
        self.assertEqual(data["aurora_uid"], uid)
        self.assertEqual(data["lang"], "en")
        self.assertEqual(data["summary"]["goal_primary"], "breakouts")

