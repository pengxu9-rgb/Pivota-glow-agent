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


class TestSessionBootstrapEndpoint(unittest.TestCase):
    def test_bootstrap_no_uid_returns_not_returning(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        with TestClient(app) as client:
            res = client.get("/v1/session/bootstrap")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_returning"])
        self.assertEqual(data["aurora_uid"], "")
        self.assertEqual(data["lang"], "en")
        self.assertEqual(
            data["artifacts_present"],
            {"has_profile": False, "has_products": False, "has_plan": False},
        )
        self.assertEqual(data["summary"]["plan_am_short"], [])
        self.assertEqual(data["summary"]["plan_pm_short"], [])
        self.assertEqual(data["summary"]["sensitivities"], [])

    def test_bootstrap_uid_with_no_data_returns_not_returning(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        uid = f"uid_test_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            res = client.get(
                "/v1/session/bootstrap",
                headers={"X-Aurora-Uid": uid, "X-Aurora-Lang": "cn"},
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_returning"])
        self.assertEqual(data["aurora_uid"], uid)
        self.assertEqual(data["lang"], "cn")

    def test_bootstrap_with_data_returns_summary(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        uid = f"uid_test_{uuid.uuid4().hex}"
        brief_id = f"brief_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            diagnosis_res = client.post(
                "/v1/diagnosis",
                headers={"X-Brief-ID": brief_id, "X-Aurora-Uid": uid},
                json={
                    "skinType": "oily",
                    "concerns": ["acne"],
                    "currentRoutine": "basic",
                    "barrierStatus": "healthy",
                },
            )
            self.assertEqual(diagnosis_res.status_code, 200)

            res = client.get(
                "/v1/session/bootstrap",
                headers={"X-Aurora-Uid": uid, "X-Aurora-Lang": "cn"},
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_returning"])
        self.assertEqual(data["aurora_uid"], uid)
        self.assertEqual(data["lang"], "cn")

        self.assertEqual(
            data["artifacts_present"],
            {"has_profile": True, "has_products": False, "has_plan": False},
        )

        self.assertEqual(data["summary"]["goal_primary"], "acne")
        self.assertEqual(data["summary"]["plan_am_short"], [])
        self.assertEqual(data["summary"]["plan_pm_short"], [])
        self.assertEqual(data["summary"]["sensitivities"], [])
        self.assertIsNotNone(data["summary"]["last_seen_at"])
        self.assertEqual(data["summary"]["days_since_last"], 0)
        self.assertFalse(data["summary"]["checkin_due"])

