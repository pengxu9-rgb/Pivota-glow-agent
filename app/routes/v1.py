from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.services.aurora import aurora_chat, extract_json_object
from app.services.session_store import SESSION_STORE


router = APIRouter()

logger = logging.getLogger("pivota-glow-agent.v1")


PIVOTA_AGENT_GATEWAY_BASE_URL = (os.getenv("PIVOTA_AGENT_GATEWAY_BASE_URL") or "https://pivota-agent-production.up.railway.app").rstrip("/")
PIVOTA_AGENT_API_KEY = (os.getenv("PIVOTA_AGENT_API_KEY") or "").strip() or None
AURORA_DECISION_BASE_URL = (os.getenv("AURORA_DECISION_BASE_URL") or "https://aurora-beauty-decision-system.vercel.app").rstrip("/")

DEFAULT_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S") or "10")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _mk_offer_id(prefix: str, sku_id: str) -> str:
    safe = sku_id.replace(" ", "_")[:80]
    return f"{prefix}_{safe}"


def _map_purchase_route(p: dict[str, Any]) -> Literal["internal_checkout", "affiliate_outbound"]:
    if p.get("external_redirect_url") or p.get("external_url"):
        return "affiliate_outbound"
    return "internal_checkout"


def _map_product(category: str, p: dict[str, Any]) -> dict[str, Any]:
    sku_id = str(p.get("product_id") or p.get("id") or "")
    if not sku_id:
        sku_id = f"sku_{uuid.uuid4().hex}"

    name = str(p.get("title") or p.get("name") or "Unknown Product")
    brand = str(p.get("brand") or p.get("vendor") or "Unknown")
    description = str(p.get("description") or "")
    image_url = ""
    if isinstance(p.get("image_url"), str) and p["image_url"]:
        image_url = p["image_url"]
    else:
        imgs = p.get("image_urls")
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
            image_url = imgs[0]

    return {
        "sku_id": sku_id,
        "name": name,
        "brand": brand,
        "category": category,
        "description": description[:2000],
        "image_url": image_url or "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
        "size": "1 unit",
    }


def _map_offer(p: dict[str, Any], *, sku_id: str) -> dict[str, Any]:
    price = p.get("price")
    try:
        price_f = float(price) if price is not None else 0.0
    except Exception:
        price_f = 0.0

    currency = str(p.get("currency") or "USD")
    seller = str(p.get("merchant_name") or p.get("merchant_id") or p.get("brand") or "Retailer")

    purchase_route = _map_purchase_route(p)
    affiliate_url = None
    if purchase_route == "affiliate_outbound":
        affiliate_url = p.get("external_redirect_url") or p.get("external_url")

    return {
        "offer_id": _mk_offer_id("offer", sku_id),
        "seller": seller,
        "price": round(price_f, 2),
        "currency": currency,
        "original_price": None,
        "shipping_days": 5,
        "returns_policy": "Standard returns",
        "reliability_score": 80,
        "badges": ["best_price"],
        "in_stock": bool(p.get("in_stock", True)),
        "purchase_route": purchase_route,
        "affiliate_url": affiliate_url,
    }


async def _agent_invoke(operation: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    url = f"{PIVOTA_AGENT_GATEWAY_BASE_URL}/agent/shop/v1/invoke"
    headers = {"Content-Type": "application/json"}
    if PIVOTA_AGENT_API_KEY:
        headers["X-Api-Key"] = PIVOTA_AGENT_API_KEY

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        res = await client.post(
            url,
            headers=headers,
            json={
                "operation": operation,
                "payload": payload,
                "metadata": {"source": "pivota-glow-agent", "trace_id": f"glow-agent:{uuid.uuid4()}"},
            },
        )

    try:
        data = res.json()
    except Exception:
        data = {"raw": res.text}

    if res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"upstream": "pivota-agent", "status": res.status_code, "body": data},
        )

    return data if isinstance(data, dict) else {"data": data}


async def _find_one_product(query: str, *, limit: int = 5, timeout_s: float) -> Optional[dict[str, Any]]:
    try:
        result = await _agent_invoke(
            "find_products_multi",
            {
                "search": {"query": query, "page": 1, "limit": max(1, min(limit, 20)), "in_stock_only": False},
                "metadata": {"source": "pivota-glow-agent"},
            },
            timeout_s=timeout_s,
        )
    except Exception:
        return None

    products = result.get("products")
    if not isinstance(products, list) or not products:
        return None
    first = products[0]
    return first if isinstance(first, dict) else None


def _analysis_from_diagnosis(diagnosis: Optional[dict[str, Any]]) -> dict[str, Any]:
    concerns = diagnosis.get("concerns") if isinstance(diagnosis, dict) else []
    concerns = concerns if isinstance(concerns, list) else []
    cset = {str(c) for c in concerns}

    needs_risk = any(c in cset for c in {"acne", "dark_spots", "wrinkles"})
    features: list[dict[str, Any]] = []

    if "acne" in cset:
        features.append({"observation": "Clogging risk around T-zone", "confidence": "pretty_sure"})
    if "dark_spots" in cset:
        features.append({"observation": "Uneven tone / hyperpigmentation signals", "confidence": "somewhat_sure"})
    if "redness" in cset or "sensitive" in cset:
        features.append({"observation": "Barrier looks reactive (possible redness/sensitivity)", "confidence": "somewhat_sure"})
    if not features:
        features = [
            {"observation": "Skin looks generally balanced", "confidence": "somewhat_sure"},
            {"observation": "Hydration could be improved", "confidence": "not_sure"},
        ]

    strategy = "Build a simple routine: cleanse → treat (if needed) → moisturize → SPF."
    if needs_risk:
        strategy = "Prioritize gentle actives with barrier-first support to hit your goals without irritation."

    return {"features": features, "strategy": strategy, "needs_risk_check": needs_risk}


def _require_brief_id(x_brief_id: Optional[str]) -> str:
    if not x_brief_id:
        raise HTTPException(status_code=400, detail="Missing X-Brief-ID")
    return x_brief_id


def _coerce_analysis(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    features_raw = raw.get("features")
    if not isinstance(features_raw, list):
        return None
    features: list[dict[str, Any]] = []
    for item in features_raw:
        if not isinstance(item, dict):
            continue
        obs = item.get("observation")
        conf = item.get("confidence")
        if not isinstance(obs, str) or not obs.strip():
            continue
        if conf not in {"pretty_sure", "somewhat_sure", "not_sure"}:
            conf = "somewhat_sure"
        features.append({"observation": obs.strip()[:240], "confidence": conf})

    strategy = raw.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        strategy = "Build a simple routine: cleanse → treat (if needed) → moisturize → SPF."

    needs_risk_check = raw.get("needs_risk_check")
    needs_risk_check = bool(needs_risk_check) if isinstance(needs_risk_check, bool) else False

    if not features:
        return None
    return {"features": features[:6], "strategy": strategy.strip()[:600], "needs_risk_check": needs_risk_check}


def _build_aurora_analysis_query(context: dict[str, Any]) -> str:
    return (
        "You are Aurora Beauty Decision System v4.0.\n"
        "Given the user context below, produce a concise skin analysis.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no backticks, no extra text) with this exact shape:\n"
        "{\n"
        '  "analysis": {\n'
        '    "features": [{"observation": string, "confidence": "pretty_sure"|"somewhat_sure"|"not_sure"}],\n'
        '    "strategy": string,\n'
        '    "needs_risk_check": boolean\n'
        "  }\n"
        "}\n"
        "\n"
        "Rules:\n"
        "- If fields are missing, make reasonable assumptions and proceed; do NOT ask clarification questions.\n"
        "- Keep it safe and conservative. No medical diagnosis.\n"
        "\n"
        f"Context JSON:\n{context}\n"
    )


def _build_aurora_routine_query(context: dict[str, Any]) -> str:
    return (
        "You are Aurora Beauty Decision System v4.0.\n"
        "Create an AM/PM routine blueprint and search queries for products.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no backticks, no extra text) with this exact shape:\n"
        "{\n"
        '  "am": [{"category": "cleanser"|"treatment"|"moisturizer"|"sunscreen", "premium_query": string, "dupe_query": string}],\n'
        '  "pm": [{"category": "cleanser"|"treatment"|"moisturizer"|"sunscreen", "premium_query": string, "dupe_query": string}],\n'
        '  "conflicts": [string]\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- If context is missing, make reasonable assumptions; do NOT ask clarification questions.\n"
        "- Keep the routine minimal and realistic (3-4 steps AM, 2-3 steps PM).\n"
        '- Use broad, shopper-friendly queries (e.g., "CeraVe hydrating cleanser").\n'
        "\n"
        f"Context JSON:\n{context}\n"
    )


@router.post("/diagnosis")
async def diagnosis(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    skipped = bool(body.get("skipped", False))
    diagnosis_payload = {
        "skinType": body.get("skinType") or body.get("skin_type"),
        "concerns": body.get("concerns") or [],
        "currentRoutine": body.get("currentRoutine") or body.get("current_routine") or "basic",
    }

    patch: dict[str, Any] = {
        "diagnosis": diagnosis_payload,
        "intent_id": body.get("intent_id"),
        "intent_text": body.get("intent_text"),
        "market": body.get("market"),
        "budget_tier": body.get("budget_tier"),
        "next_state": "S3_PHOTO_OPTION",
    }

    if skipped:
        patch["next_state"] = "S3_PHOTO_OPTION"

    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            **patch,
        },
    )
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/photos")
async def photos_upload(
    daylight: Optional[UploadFile] = File(default=None),
    indoor_white: Optional[UploadFile] = File(default=None),
    trace_id: Optional[str] = None,
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    # MVP: accept uploads but do not persist; front-end keeps local previews.
    photos_patch: dict[str, Any] = {}
    if daylight is not None:
        photos_patch["daylight"] = {"qc_status": "passed", "retry_count": 0}
    if indoor_white is not None:
        photos_patch["indoor_white"] = {"qc_status": "passed", "retry_count": 0}

    patch = {"photos": photos_patch, "next_state": "S4_ANALYSIS_LOADING"}
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "photos": photos_patch,
        },
    )
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/photos/sample")
async def photos_sample(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    sample_set_id = str(body.get("sample_set_id") or "")
    # Mirror the frontend sample IDs (sample_set_A/B/C).
    sample_sets: dict[str, dict[str, str]] = {
        "sample_set_A": {
            "daylight": "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=400&h=400&fit=crop",
            "indoor_white": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&h=400&fit=crop",
        },
        "sample_set_B": {
            "daylight": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
            "indoor_white": "https://images.unsplash.com/photo-1549351236-caca0f174515?w=400&h=400&fit=crop",
        },
        "sample_set_C": {
            "daylight": "https://images.unsplash.com/photo-1594824476967-48c8b964273f?w=400&h=400&fit=crop",
            "indoor_white": "https://images.unsplash.com/photo-1595959183082-7b570b7e08f4?w=400&h=400&fit=crop",
        },
    }

    urls = sample_sets.get(sample_set_id) or sample_sets["sample_set_A"]

    patch = {
        "sample_photo_set_id": sample_set_id or "sample_set_A",
        "photos": {
            "daylight": {"preview_url": urls["daylight"], "qc_status": "passed", "retry_count": 0},
            "indoor_white": {"preview_url": urls["indoor_white"], "qc_status": "passed", "retry_count": 0},
        },
        "next_state": "S4_ANALYSIS_LOADING",
    }
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "sample_photo_set_id": patch["sample_photo_set_id"],
            "photos": patch["photos"],
        },
    )
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/analysis")
async def analysis(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    stored = await SESSION_STORE.get(brief_id)

    diagnosis_payload = stored.get("diagnosis") if isinstance(stored.get("diagnosis"), dict) else None
    if diagnosis_payload is None and isinstance(body.get("diagnosis"), dict):
        diagnosis_payload = body.get("diagnosis")

    intent_id = stored.get("intent_id") or body.get("intent_id") or "routine"
    market = stored.get("market") or body.get("market") or "US"
    budget_tier = stored.get("budget_tier") or body.get("budget_tier") or "$$"
    photos = stored.get("photos") if isinstance(stored.get("photos"), dict) else {}
    photos_present = False
    if isinstance(photos, dict):
        photos_present = any(isinstance(v, dict) and (v.get("preview_url") or v.get("qc_status")) for v in photos.values())

    context = {
        "intent_id": intent_id,
        "market": market,
        "budget_tier": budget_tier,
        "diagnosis": diagnosis_payload,
        "photos_present": photos_present,
    }

    analysis_result: dict[str, Any]
    try:
        query = _build_aurora_analysis_query(context)
        aurora_payload = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=query,
            timeout_s=DEFAULT_TIMEOUT_S,
        )
        aurora_answer = aurora_payload.get("answer") if isinstance(aurora_payload, dict) else None
        parsed = extract_json_object(aurora_answer) if isinstance(aurora_answer, str) else None
        coerced = _coerce_analysis((parsed or {}).get("analysis") if isinstance(parsed, dict) else None)
        analysis_result = coerced or _analysis_from_diagnosis(diagnosis_payload)
    except Exception as exc:
        logger.warning("Aurora analysis failed; falling back. err=%s", exc)
        analysis_result = _analysis_from_diagnosis(diagnosis_payload)

    patch = {"analysis": analysis_result, "next_state": "S5_ANALYSIS_SUMMARY"}
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "analysis": analysis_result,
        },
    )
    return {"session": patch, "next_state": patch["next_state"], "analysis": analysis_result}


@router.post("/analysis/risk")
async def analysis_risk(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    answer = str(body.get("answer") or "skip")
    using_actives = answer == "yes"
    patch = {
        "analysis": {"risk_answered": True, "using_actives": using_actives},
        "next_state": "S6_BUDGET",
    }
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "analysis": patch["analysis"],
        },
    )
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/routine/reorder")
async def routine_reorder(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    stored = await SESSION_STORE.get(brief_id)
    preference = str(body.get("preference") or stored.get("preference") or "keep")
    market = stored.get("market") or body.get("market") or "US"
    budget_tier = stored.get("budget_tier") or body.get("budget_tier") or "$$"
    intent_id = stored.get("intent_id") or body.get("intent_id") or "routine"
    diagnosis_payload = stored.get("diagnosis") if isinstance(stored.get("diagnosis"), dict) else None
    analysis_payload = stored.get("analysis") if isinstance(stored.get("analysis"), dict) else None

    # Default queries (fallback only)
    queries: dict[str, dict[str, str]] = {
        "cleanser": {"premium": "Sulwhasoo cleansing foam", "dupe": "CeraVe hydrating cleanser"},
        "treatment": {"premium": "SkinCeuticals vitamin c serum", "dupe": "The Ordinary niacinamide serum"},
        "moisturizer": {"premium": "La Mer moisturizer", "dupe": "CeraVe moisturizing cream"},
        "sunscreen": {"premium": "Shiseido sunscreen", "dupe": "La Roche-Posay sunscreen"},
    }

    categories_am = ["cleanser", "treatment", "moisturizer", "sunscreen"]
    categories_pm = ["cleanser", "treatment", "moisturizer"]
    conflicts: list[str] = []

    # Ask Aurora to propose category order + search queries.
    try:
        context = {
            "intent_id": intent_id,
            "market": market,
            "budget_tier": budget_tier,
            "preference": preference,
            "diagnosis": diagnosis_payload,
            "analysis": analysis_payload,
        }
        aurora_payload = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=_build_aurora_routine_query(context),
            timeout_s=DEFAULT_TIMEOUT_S,
        )
        aurora_answer = aurora_payload.get("answer") if isinstance(aurora_payload, dict) else None
        parsed = extract_json_object(aurora_answer) if isinstance(aurora_answer, str) else None
        if isinstance(parsed, dict):
            def _read_steps(key: str) -> list[dict[str, Any]]:
                v = parsed.get(key)
                return v if isinstance(v, list) else []

            def _norm_category(cat: Any) -> Optional[str]:
                if not isinstance(cat, str):
                    return None
                c = cat.strip().lower()
                return c if c in {"cleanser", "treatment", "moisturizer", "sunscreen"} else None

            am_steps = _read_steps("am")
            pm_steps = _read_steps("pm")
            if am_steps:
                categories_am = [c for c in (_norm_category(s.get("category")) for s in am_steps) if c] or categories_am
            if pm_steps:
                categories_pm = [c for c in (_norm_category(s.get("category")) for s in pm_steps) if c] or categories_pm

            for step in am_steps + pm_steps:
                cat = _norm_category(step.get("category"))
                if not cat:
                    continue
                premium_q = step.get("premium_query")
                dupe_q = step.get("dupe_query")
                if isinstance(premium_q, str) and premium_q.strip():
                    queries.setdefault(cat, {})["premium"] = premium_q.strip()[:120]
                if isinstance(dupe_q, str) and dupe_q.strip():
                    queries.setdefault(cat, {})["dupe"] = dupe_q.strip()[:120]

            conflicts_raw = parsed.get("conflicts")
            if isinstance(conflicts_raw, list):
                conflicts = [str(x)[:200] for x in conflicts_raw if x]
    except Exception as exc:
        logger.warning("Aurora routine planning failed; using fallback queries. err=%s", exc)

    async def get_pair(category: str) -> dict[str, Any]:
        q = queries[category]
        premium_q = q["premium"]
        dupe_q = q["dupe"]

        premium_raw, dupe_raw = await asyncio.gather(
            _find_one_product(premium_q, limit=5, timeout_s=DEFAULT_TIMEOUT_S),
            _find_one_product(dupe_q, limit=5, timeout_s=DEFAULT_TIMEOUT_S),
            return_exceptions=False,
        )

        if not premium_raw or not dupe_raw:
            # Fallback: minimal, but keeps UI functional.
            premium_raw = premium_raw or {
                "product_id": f"demo_{category}_premium",
                "title": f"{category.title()} Premium",
                "brand": "Premium",
                "price": 55,
                "currency": "USD",
                "external_url": "https://example.com",
            }
            dupe_raw = dupe_raw or {
                "product_id": f"demo_{category}_dupe",
                "title": f"{category.title()} Dupe",
                "brand": "Dupe",
                "price": 18,
                "currency": "USD",
                "external_url": "https://example.com",
            }

        premium_product = _map_product(category, premium_raw)
        dupe_product = _map_product(category, dupe_raw)

        return {
            "category": category,
            "premium": {
                "product": premium_product,
                "offers": [_map_offer(premium_raw, sku_id=premium_product["sku_id"])],
            },
            "dupe": {
                "product": dupe_product,
                "offers": [_map_offer(dupe_raw, sku_id=dupe_product["sku_id"])],
            },
        }

    # Fetch all unique categories concurrently.
    all_categories = sorted(set(categories_am + categories_pm))
    pairs_by_cat_list = await asyncio.gather(*(get_pair(cat) for cat in all_categories))
    pairs_by_cat = {p["category"]: p for p in pairs_by_cat_list}

    am_pairs = [pairs_by_cat[c] for c in categories_am if c in pairs_by_cat]
    pm_pairs = [pairs_by_cat[c] for c in categories_pm if c in pairs_by_cat]

    selected_offers: dict[str, str] = {}
    for p in pairs_by_cat_list:
        for side in ("premium", "dupe"):
            prod = p[side]["product"]
            offer = p[side]["offers"][0]
            selected_offers[prod["sku_id"]] = offer["offer_id"]

    patch = {
        "productPairs": {"am": am_pairs, "pm": pm_pairs},
        "selected_offers": selected_offers,
        "routine_conflicts": conflicts,
        "next_state": "S7_PRODUCT_RECO",
    }
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "productPairs": patch["productPairs"],
            "selected_offers": patch["selected_offers"],
            "routine_conflicts": conflicts,
            "preference": preference,
            "market": market,
            "budget_tier": budget_tier,
        },
    )
    return {"session": patch, "next_state": patch["next_state"], "productPairs": patch["productPairs"]}


@router.patch("/routine/selection")
async def routine_selection(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    selection = body.get("selection") if isinstance(body.get("selection"), dict) else {}
    key = str(selection.get("key") or selection.get("category") or "unknown")
    sel_type = str(selection.get("type") or "dupe")
    offer_id = selection.get("offer_id")

    patch: dict[str, Any] = {
        "product_selections": {key: {"type": sel_type, "offerId": offer_id}},
    }
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "product_selections": patch["product_selections"],
        },
    )
    return {"session": patch}


@router.post("/checkout")
async def checkout(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    # MVP: treat as success. Internal checkout can be wired to pivota-agent `create_order` later.
    order_id = f"ORD-{_now_ms()}"
    result = {
        "success": True,
        "order_id": order_id,
        "total": 0,
        "currency": "USD",
        "eta": "3-5 business days",
    }
    patch = {"checkout_result": result, "next_state": "S9_SUCCESS"}
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "checkout_result": result,
        },
    )
    return {"session": patch, "next_state": patch["next_state"], "checkout_result": result}


@router.post("/affiliate/outcome")
async def affiliate_outcome(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    _ = _require_brief_id(x_brief_id)
    # Store/reporting hook can be added later.
    return {"ok": True}
