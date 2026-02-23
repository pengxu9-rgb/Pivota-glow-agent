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
from app.routes import v1 as v1_routes


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
    def test_qc_harmonize_passed_overrides_pending_advice(self) -> None:
        qc_status, qc_advice = v1_routes._harmonize_qc_result(
            qc_status="passed",
            qc_advice={
                "summary": "QC is pending.",
                "suggestions": ["Processing your photo…"],
                "retryable": False,
            },
            tips={"general": ["Clean lens"]},
        )
        self.assertEqual(qc_status, "passed")
        self.assertIsInstance(qc_advice, dict)
        self.assertIn("passed", str(qc_advice.get("summary") or "").lower())
        self.assertNotIn("pending", str(qc_advice.get("summary") or "").lower())

    def test_qc_harmonize_pending_has_default_advice(self) -> None:
        qc_status, qc_advice = v1_routes._harmonize_qc_result(
            qc_status=None,
            qc_advice=None,
            tips=None,
        )
        self.assertEqual(qc_status, "pending")
        self.assertIsInstance(qc_advice, dict)
        self.assertIn("processing", str(qc_advice.get("summary") or "").lower())

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

    def test_auth_verify_returns_auth_session_card(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        brief_id = f"brief_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            res = client.post(
                "/v1/auth/verify",
                headers={"X-Brief-ID": brief_id},
                json={"email": "test@example.com", "code": "123456"},
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        cards = data.get("cards")
        self.assertIsInstance(cards, list)
        auth_card = next((c for c in cards if isinstance(c, dict) and c.get("type") == "auth_session"), None)
        self.assertIsInstance(auth_card, dict)
        payload = auth_card.get("payload") if isinstance(auth_card, dict) else None
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload.get("token"))

    def test_product_and_dupe_routes_return_contract_cards(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        brief_id = f"brief_{uuid.uuid4().hex}"

        with TestClient(app) as client:
            parse_res = client.post(
                "/v1/product/parse",
                headers={"X-Brief-ID": brief_id},
                json={"text": "La Roche Posay cleanser"},
            )
            self.assertEqual(parse_res.status_code, 200)
            parse_cards = parse_res.json().get("cards") or []
            self.assertTrue(any((c or {}).get("type") == "product_parse" for c in parse_cards))

            analyze_res = client.post(
                "/v1/product/analyze",
                headers={"X-Brief-ID": brief_id},
                json={"name": "La Roche Posay cleanser"},
            )
            self.assertEqual(analyze_res.status_code, 200)
            analyze_cards = analyze_res.json().get("cards") or []
            self.assertTrue(any((c or {}).get("type") == "product_analysis" for c in analyze_cards))

            dupe_res = client.post(
                "/v1/dupe/suggest",
                headers={"X-Brief-ID": brief_id},
                json={"original_text": "SK-II essence"},
            )
            self.assertEqual(dupe_res.status_code, 200)
            dupe_cards = dupe_res.json().get("cards") or []
            self.assertTrue(any((c or {}).get("type") == "dupe_suggest" for c in dupe_cards))
