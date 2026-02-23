from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app


async def _fake_aurora_chat_analysis(**kwargs):
    _ = kwargs
    return {
        "answer": (
            '{"features":[{"observation":"Barrier may be stressed; prioritize gentle support.",'
            '"confidence":"somewhat_sure"}],'
            '"strategy":"Keep a minimal routine for 7 days. What cleanser/SPF are you using now?",'
            '"needs_risk_check":false}'
        ),
        "context": {"detected": {"barrier_impaired": True}},
    }


async def _fake_aurora_chat_chat(**kwargs):
    _ = kwargs
    return {
        "answer": "Use a gentle AM/PM baseline and refresh based on your recent check-ins.",
        "intent": "answer",
        "clarification": None,
        "context": {"detected": {"barrier_impaired": False}},
    }


class TestChatboxContractEndpoints(unittest.TestCase):
    def test_analysis_skin_returns_envelope_and_analysis_meta(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        brief_id = f"brief_{uuid.uuid4().hex}"

        with patch("app.routes.v1.aurora_chat", side_effect=_fake_aurora_chat_analysis):
            with TestClient(app) as client:
                diagnosis_res = client.post(
                    "/v1/diagnosis",
                    headers={"X-Brief-ID": brief_id},
                    json={
                        "skinType": "sensitive",
                        "concerns": ["redness"],
                        "currentRoutine": "basic",
                        "barrierStatus": "impaired",
                    },
                )
                self.assertEqual(diagnosis_res.status_code, 200)

                res = client.post(
                    "/v1/analysis/skin",
                    headers={"X-Brief-ID": brief_id, "X-Trace-ID": "trace_analysis_contract"},
                    json={"use_photo": False},
                )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data.get("request_id"), str)
        self.assertEqual(data.get("trace_id"), "trace_analysis_contract")
        self.assertIsInstance(data.get("cards"), list)
        self.assertTrue(any((c or {}).get("type") == "analysis_summary" for c in data["cards"]))
        self.assertIsInstance(data.get("analysis_meta"), dict)
        self.assertIn("llm_report_called", data["analysis_meta"])
        self.assertIn("artifact_usable", data["analysis_meta"])
        self.assertEqual((data.get("session") or {}).get("next_state"), "S5_ANALYSIS_SUMMARY")

    def test_tracker_log_returns_refresh_hint_and_envelope(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        brief_id = f"brief_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            res = client.post(
                "/v1/tracker/log",
                headers={"X-Brief-ID": brief_id, "X-Trace-ID": "trace_tracker_contract", "X-Aurora-Lang": "en"},
                json={"redness": 3, "acne": 2, "hydration": 1, "notes": "travel week"},
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data.get("request_id"), str)
        self.assertEqual(data.get("trace_id"), "trace_tracker_contract")
        hint = data.get("reco_refresh_hint")
        self.assertIsInstance(hint, dict)
        self.assertTrue(hint.get("should_refresh"))
        self.assertIsInstance(hint.get("reason"), str)
        self.assertIsInstance(data.get("session_patch"), dict)
        self.assertIn("recent_logs", data.get("session_patch") or {})

    def test_chat_returns_recommendation_meta_with_recent_logs_and_itinerary(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        brief_id = f"brief_{uuid.uuid4().hex}"

        with patch("app.routes.v1.aurora_chat", side_effect=_fake_aurora_chat_chat):
            with TestClient(app) as client:
                _ = client.post(
                    "/v1/tracker/log",
                    headers={"X-Brief-ID": brief_id},
                    json={"redness": 1, "acne": 1, "hydration": 4, "notes": "steady"},
                )

                res = client.post(
                    "/v1/chat",
                    headers={"X-Brief-ID": brief_id, "X-Trace-ID": "trace_chat_contract"},
                    json={
                        "message": "refresh my routine for travel",
                        "itinerary": "3 days in dry climate, long-haul flight",
                    },
                )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data.get("request_id"), str)
        self.assertEqual(data.get("trace_id"), "trace_chat_contract")
        self.assertIsInstance(data.get("assistant_message"), dict)
        self.assertIsInstance(data.get("recommendation_meta"), dict)
        reco_meta = data["recommendation_meta"]
        self.assertTrue(reco_meta.get("used_recent_logs"))
        self.assertTrue(reco_meta.get("used_itinerary"))
        self.assertIn("used_safety_flags", reco_meta)

