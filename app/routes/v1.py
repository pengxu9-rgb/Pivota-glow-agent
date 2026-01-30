from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Literal

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile


router = APIRouter()


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


async def _find_one_product(query: str, *, limit: int = 5, timeout_s: float) -> dict[str, Any] | None:
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


def _analysis_from_diagnosis(diagnosis: dict[str, Any] | None) -> dict[str, Any]:
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


@router.post("/diagnosis")
async def diagnosis(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
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

    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/photos")
async def photos_upload(
    daylight: UploadFile | None = File(default=None),
    indoor_white: UploadFile | None = File(default=None),
    trace_id: str | None = None,
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    # MVP: accept uploads but do not persist; front-end keeps local previews.
    photos_patch: dict[str, Any] = {}
    if daylight is not None:
        photos_patch["daylight"] = {"qc_status": "passed", "retry_count": 0}
    if indoor_white is not None:
        photos_patch["indoor_white"] = {"qc_status": "passed", "retry_count": 0}

    patch = {"photos": photos_patch, "next_state": "S4_ANALYSIS_LOADING"}
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/photos/sample")
async def photos_sample(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
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
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/analysis")
async def analysis(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    diagnosis_payload = body.get("diagnosis") if isinstance(body.get("diagnosis"), dict) else None
    analysis_result = _analysis_from_diagnosis(diagnosis_payload)
    patch = {"analysis": analysis_result, "next_state": "S5_ANALYSIS_SUMMARY"}
    return {"session": patch, "next_state": patch["next_state"], "analysis": analysis_result}


@router.post("/analysis/risk")
async def analysis_risk(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    answer = str(body.get("answer") or "skip")
    using_actives = answer == "yes"
    patch = {
        "analysis": {"risk_answered": True, "using_actives": using_actives},
        "next_state": "S6_BUDGET",
    }
    return {"session": patch, "next_state": patch["next_state"]}


@router.post("/routine/reorder")
async def routine_reorder(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    _ = str(body.get("preference") or "keep")

    # Basic query set; keep fast and deterministic. If the upstream is slow, we fall back to static demo products.
    queries = {
        "cleanser": {"premium": "Sulwhasoo cleansing foam", "dupe": "CeraVe hydrating cleanser"},
        "treatment": {"premium": "SkinCeuticals vitamin c serum", "dupe": "The Ordinary niacinamide serum"},
        "moisturizer": {"premium": "La Mer moisturizer", "dupe": "CeraVe moisturizing cream"},
        "sunscreen": {"premium": "Shiseido sunscreen", "dupe": "La Roche-Posay sunscreen"},
    }

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

    categories_am = ["cleanser", "treatment", "moisturizer", "sunscreen"]
    categories_pm = ["cleanser", "treatment", "moisturizer"]

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
        "next_state": "S7_PRODUCT_RECO",
    }
    return {"session": patch, "next_state": patch["next_state"], "productPairs": patch["productPairs"]}


@router.patch("/routine/selection")
async def routine_selection(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    selection = body.get("selection") if isinstance(body.get("selection"), dict) else {}
    key = str(selection.get("key") or selection.get("category") or "unknown")
    sel_type = str(selection.get("type") or "dupe")
    offer_id = selection.get("offer_id")

    patch: dict[str, Any] = {
        "product_selections": {key: {"type": sel_type, "offerId": offer_id}},
    }
    return {"session": patch}


@router.post("/checkout")
async def checkout(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
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
    return {"session": patch, "next_state": patch["next_state"], "checkout_result": result}


@router.post("/affiliate/outcome")
async def affiliate_outcome(
    body: dict[str, Any],
    x_brief_id: str | None = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    # Store/reporting hook can be added later.
    return {"ok": True}
