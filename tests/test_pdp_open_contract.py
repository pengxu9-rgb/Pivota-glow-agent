from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app


class TestPdpOpenContract(unittest.TestCase):
    def test_known_internal_product_the_ordinary_uses_subject_direct(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        payload = {
            "language": "EN",
            "card": {
                "product": {
                    "brand": "The Ordinary",
                    "name": "Niacinamide 10% + Zinc 1%",
                    "subject": {
                        "kind": "product_group",
                        "product_group_id": "pg:merch_the_ordinary:prod_niacinamide",
                    },
                },
            },
        }

        with TestClient(app) as client:
            res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["pdp_open_path"], "subject_direct")
        self.assertEqual(data["reason_code"], "subject_direct")
        self.assertEqual(data["resolve_attempt_count"], 0)
        self.assertTrue(data["internal_candidate_found"])
        self.assertIsNotNone(data["pdp_target"])
        self.assertEqual(data["pdp_target"]["schema_version"], "pdp_target.v1")
        self.assertEqual(data["pdp_target"]["subject"]["kind"], "product_group")

    def test_known_internal_product_winona_uses_canonical_ref_direct(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        payload = {
            "language": "EN",
            "card": {
                "product": {
                    "brand": "Winona",
                    "name": "Soothing Repair Serum",
                    "canonical_product_ref": {
                        "merchant_id": "merch_winona",
                        "product_id": "prod_winona_soothing_repair_serum",
                    },
                },
            },
        }

        with TestClient(app) as client:
            res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pdp_open_path"], "canonical_ref_direct")
        self.assertEqual(data["reason_code"], "canonical_ref_direct")
        self.assertEqual(data["resolve_attempt_count"], 0)
        self.assertTrue(data["internal_candidate_found"])
        self.assertIsNotNone(data["pdp_target"])
        self.assertEqual(data["pdp_target"]["subject"]["kind"], "canonical_product")

    def test_uuid_only_sku_never_populates_hints_product_ref(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        captured_hints: dict[str, object] = {}

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, brief_id, trace_id, aurora_uid, request_id)
            captured_hints.update(hints or {})
            return {"resolved": False, "reason_code": "NO_CANDIDATES"}

        payload = {
            "language": "EN",
            "card": {
                "product": {
                    "brand": "Legacy Brand",
                    "name": "Legacy Name",
                    "sku_id": "c231aaaa-8b00-4145-a704-684931049303",
                    "canonical_product_ref": {
                        "merchant_id": "merch_legacy",
                        "product_id": "c231aaaa-8b00-4145-a704-684931049303",
                    },
                },
            },
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pdp_open_path"], "fallback_external")
        self.assertEqual(data["resolve_attempt_count"], 1)
        self.assertNotIn("product_ref", captured_hints)

    def test_resolve_failure_returns_fallback_external_with_explicit_reason(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            return {"resolved": False, "reason_code": "UPSTREAM_TIMEOUT"}

        payload = {
            "language": "EN",
            "card": {
                "product": {
                    "brand": "Unknown Brand",
                    "name": "Unknown Item",
                },
            },
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pdp_open_path"], "fallback_external")
        self.assertEqual(data["reason_code"], "upstream_timeout")
        self.assertEqual(data["fallback_reason_code"], "upstream_timeout")
        self.assertTrue(str(data.get("external_url") or "").startswith("https://www.google.com/search?"))

    def test_resolve_called_at_most_once(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        counter = {"count": 0}

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            counter["count"] += 1
            return {"resolved": False, "reason_code": "NO_CANDIDATES"}

        payload = {
            "language": "EN",
            "card": {"product": {"brand": "Legacy Brand", "name": "Legacy Item"}},
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(counter["count"], 1)
        self.assertEqual(data["resolve_attempt_count"], 1)

    def test_card_click_prefers_subject_direct_when_subject_exists(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        payload = {
            "language": "EN",
            "card": {
                "subject": {"kind": "product_group", "product_group_id": "pg:merch_demo:prod_demo"},
                "display": {"title": "Demo Product"},
            },
        }

        with TestClient(app) as client:
            res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pdp_open_path"], "subject_direct")
        self.assertEqual(data["reason_code"], "subject_direct")
        self.assertEqual(data["resolve_attempt_count"], 0)

    def test_legacy_only_card_works_via_single_resolve_attempt(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()
        counter = {"count": 0}

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            counter["count"] += 1
            return {
                "resolved": True,
                "canonical_product_ref": {
                    "merchant_id": "merch_to",
                    "product_id": "prod_to_niacinamide",
                },
            }

        payload = {
            "language": "EN",
            "card": {
                "brand": "The Ordinary",
                "name": "Niacinamide 10% + Zinc 1%",
            },
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(counter["count"], 1)
        self.assertEqual(data["pdp_open_path"], "resolve_once")
        self.assertEqual(data["reason_code"], "mapped_hit")
        self.assertTrue(data["internal_candidate_found"])
        self.assertIsNotNone(data["pdp_target"])

    def test_no_name_guessing_path_is_used(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            return {"resolved": False, "reason_code": "NO_CANDIDATES"}

        payload = {
            "language": "EN",
            "card": {"product": {"brand": "Some Brand", "name": "Some Product"}},
        }

        with patch("app.routes.v1._best_product_match", side_effect=AssertionError("must not call name guessing")) as guess_spy:
            with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
                with TestClient(app) as client:
                    res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        self.assertEqual(guess_spy.call_count, 0)

    def test_never_returns_success_with_empty_reason_code(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            return {}

        payload = {
            "language": "EN",
            "card": {"product": {"brand": "Unknown", "name": "Unknown"}},
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(str(data.get("reason_code") or "").strip())
        self.assertIn(
            data["reason_code"],
            {
                "mapped_hit",
                "subject_direct",
                "canonical_ref_direct",
                "no_candidates",
                "db_timeout",
                "upstream_timeout",
                "invalid_id",
                "fallback_external",
            },
        )

    def test_click_path_never_asks_user_for_product_name(self) -> None:
        os.environ["REDIS_URL"] = ""
        app = create_app()

        async def fake_resolve_once_for_pdp(
            *,
            query: str,
            lang: str,
            hints: dict[str, object] | None,
            brief_id: str | None,
            trace_id: str | None,
            aurora_uid: str | None,
            request_id: str | None,
        ) -> dict[str, object]:
            _ = (query, lang, hints, brief_id, trace_id, aurora_uid, request_id)
            return {"resolved": False, "reason_code": "NO_CANDIDATES"}

        payload = {
            "language": "EN",
            "card": {"product": {"brand": "Unknown", "name": "Unknown"}},
        }

        with patch("app.routes.v1._resolve_products_once_for_pdp", new=fake_resolve_once_for_pdp):
            with TestClient(app) as client:
                res = client.post("/v1/pdp/open-contract", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertNotIn("clarification", data)
        self.assertNotIn("answer", data)
        self.assertEqual(data["pdp_open_path"], "fallback_external")
