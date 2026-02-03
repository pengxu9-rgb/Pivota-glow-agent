from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import difflib
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid
import urllib.parse
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Request, UploadFile

from app.services.aurora import aurora_chat, extract_json_object
from app.services.session_store import SESSION_STORE


router = APIRouter()

logger = logging.getLogger("pivota-glow-agent.v1")


PIVOTA_AGENT_GATEWAY_BASE_URL = (os.getenv("PIVOTA_AGENT_GATEWAY_BASE_URL") or "https://pivota-agent-production.up.railway.app").rstrip("/")
PIVOTA_AGENT_API_KEY = (os.getenv("PIVOTA_AGENT_API_KEY") or "").strip() or None
AURORA_DECISION_BASE_URL = (os.getenv("AURORA_DECISION_BASE_URL") or "https://aurora-beauty-decision-system.vercel.app").rstrip("/")
GLOW_SYSTEM_PROMPT = (os.getenv("GLOW_SYSTEM_PROMPT") or "").strip()
USE_PIVOTA_AGENT_SEARCH = (os.getenv("USE_PIVOTA_AGENT_SEARCH") or "").strip().lower() in {"1", "true", "yes", "y"}

DEFAULT_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S") or "10")
OFFERS_RESOLVE_TIMEOUT_S = float(os.getenv("OFFERS_RESOLVE_TIMEOUT_S") or "55")

# Optional analytics forwarding.
POSTHOG_API_KEY = (os.getenv("POSTHOG_API_KEY") or "").strip() or None
POSTHOG_HOST = (os.getenv("POSTHOG_HOST") or os.getenv("POSTHOG_URL") or "").strip().rstrip("/") or None
POSTHOG_TIMEOUT_S = float(os.getenv("POSTHOG_TIMEOUT_S") or "3")
EVENTS_JSONL_SINK_DIR = (os.getenv("EVENTS_JSONL_SINK_DIR") or "").strip() or None
EVENTS_INCLUDE_CLIENT_IP = (os.getenv("EVENTS_INCLUDE_CLIENT_IP") or "").strip().lower() in {"1", "true", "yes", "y"}

# Best-effort cache for product lookups so repeated checkout clicks are fast.
_PRODUCT_SEARCH_TTL_MS = int(float(os.getenv("PRODUCT_SEARCH_CACHE_TTL_SECONDS") or "3600") * 1000)
_PRODUCT_SEARCH_CACHE: dict[tuple[str, str], tuple[int, list[dict[str, Any]]]] = {}
_PRODUCT_SEARCH_LOCK = asyncio.Lock()


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

async def _photos_api_json(
    method: str,
    path: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    url = f"{PIVOTA_AGENT_GATEWAY_BASE_URL}{path}"
    headers: dict[str, str] = {}
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    if PIVOTA_AGENT_API_KEY:
        headers["X-Api-Key"] = PIVOTA_AGENT_API_KEY

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        res = await client.request(method, url, headers=headers, json=json_body, params=params)

    try:
        data = res.json()
    except Exception:
        data = {"raw": res.text}

    if res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"upstream": "pivota-agent", "path": path, "status": res.status_code, "body": data},
        )

    return data if isinstance(data, dict) else {"data": data}


async def _upload_photo_via_pivota(
    *,
    blob: bytes,
    content_type: str,
    file_name: Optional[str],
    user_id: str,
    consent: bool,
) -> dict[str, Any]:
    presign = await _photos_api_json(
        "POST",
        "/photos/presign",
        json_body={
            "content_type": content_type,
            "file_name": file_name,
            "byte_size": len(blob),
            "consent": consent,
            "user_id": user_id,
        },
        timeout_s=DEFAULT_TIMEOUT_S,
    )
    upload_id = str(presign.get("upload_id") or "")
    upload = presign.get("upload") if isinstance(presign.get("upload"), dict) else {}
    method = str(upload.get("method") or "PUT").upper()
    upload_url = str(upload.get("url") or "")
    upload_headers = upload.get("headers") if isinstance(upload.get("headers"), dict) else {}
    if not upload_id or not upload_url:
        raise HTTPException(status_code=502, detail={"error": "UPLOAD_PRESIGN_INVALID", "body": presign})
    if method != "PUT":
        raise HTTPException(status_code=502, detail={"error": "UPLOAD_METHOD_UNSUPPORTED", "method": method})

    put_headers = {**{k: str(v) for k, v in upload_headers.items()}, "Content-Type": content_type}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
        put_res = await client.put(upload_url, headers=put_headers, content=blob)
    if put_res.status_code not in {200, 201, 204}:
        raise HTTPException(
            status_code=502,
            detail={"error": "UPLOAD_FAILED", "status": put_res.status_code, "body": put_res.text[:500]},
        )

    await _photos_api_json(
        "POST",
        "/photos/confirm",
        json_body={"upload_id": upload_id, "byte_size": len(blob)},
        timeout_s=DEFAULT_TIMEOUT_S,
    )

    qc_status: Optional[str] = None
    qc_advice: Optional[dict[str, Any]] = None
    # GET /photos/qc lazily computes QC if needed; retry briefly for robustness.
    for _ in range(6):
        qc = await _photos_api_json("GET", "/photos/qc", params={"upload_id": upload_id}, timeout_s=DEFAULT_TIMEOUT_S)
        qc_status = qc.get("qc_status") or (qc.get("qc") or {}).get("qc_status")
        qc_advice = (qc.get("qc") or {}).get("advice") if isinstance(qc.get("qc"), dict) else None
        if qc_status:
            break
        await asyncio.sleep(0.4)

    if not qc_status:
        qc_status = "pending"
    if not qc_advice and qc_status == "pending":
        tips = presign.get("tips") if isinstance(presign.get("tips"), dict) else {}
        qc_advice = {
            "summary": "Photo is processing.",
            "suggestions": ["Wait a moment and retry.", "If it keeps failing, try re-uploading in better light."],
            "tips": tips,
            "retryable": True,
        }

    return {"upload_id": upload_id, "qc_status": qc_status, "qc_advice": qc_advice}


def _guess_category_from_query(query: str) -> str:
    ql = query.lower()
    if any(k in ql for k in ["cleanser", "face wash", "cleansing"]):
        return "cleanser"
    if any(k in ql for k in ["moisturizer", "moisturising", "cream", "lotion"]):
        return "moisturizer"
    if any(k in ql for k in ["sunscreen", "spf", "uv"]):
        return "sunscreen"
    return "treatment"


def _score_product_for_category(category: str, product: dict[str, Any]) -> int:
    title = str(product.get("title") or product.get("name") or "")
    desc = str(product.get("description") or "")
    text = f"{title}\n{desc}".lower()

    positives = {
        "cleanser": ["cleanser", "face wash", "cleansing", "foam", "gel cleanser"],
        "moisturizer": ["moisturizer", "moisturising", "cream", "lotion", "hydrating cream"],
        "sunscreen": ["sunscreen", "sun screen", "spf", "uv"],
        "treatment": ["serum", "retinol", "niacinamide", "vitamin c", "aha", "bha", "acid", "treatment", "ampoule"],
    }.get(category, [])

    negatives = [
        "brush",
        "makeup",
        "foundation",
        "concealer",
        "mascara",
        "lipstick",
        "lip gloss",
        "eyeshadow",
        "blush",
        "highlighter",
        "palette",
    ]

    s = 0
    for kw in positives:
        if kw in text:
            s += 2
    for kw in negatives:
        if kw in text:
            s -= 3
    return s


def _sort_by_price(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _price(p: dict[str, Any]) -> float:
        try:
            return float(p.get("price") or 0)
        except Exception:
            return 0.0

    return sorted(products, key=_price)


async def _find_products(query: str, *, limit: int, timeout_s: float) -> list[dict[str, Any]]:
    return await _find_products_with_category(query, category=None, limit=limit, timeout_s=timeout_s)


async def _find_products_with_category(
    query: str,
    *,
    category: Optional[str],
    limit: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    cat_hint = (category or "").strip().lower()
    cache_key = (cat_hint, q.lower())
    now_ms = _now_ms()

    async with _PRODUCT_SEARCH_LOCK:
        cached = _PRODUCT_SEARCH_CACHE.get(cache_key)
        if cached and now_ms - cached[0] <= _PRODUCT_SEARCH_TTL_MS:
            return cached[1][:limit]

    try:
        search_payload: dict[str, Any] = {
            "query": q,
            "page": 1,
            "limit": max(1, min(limit, 50)),
            "in_stock_only": False,
        }

        result = await _agent_invoke(
            "find_products_multi",
            {
                "search": search_payload,
                "metadata": {"source": "pivota-glow-agent"},
            },
            timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.warning("Product search failed. query=%r err=%r", query, exc)
        return []

    products = result.get("products")
    if not isinstance(products, list) or not products:
        return []

    category_guess = cat_hint or _guess_category_from_query(q)
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        scored.append((_score_product_for_category(category_guess, p), p))

    # Prefer category-looking hits; if none, return raw list to allow price-based picking.
    scored.sort(key=lambda x: x[0], reverse=True)
    positive = [p for s, p in scored if s > 0]
    if positive:
        # Keep positives first, but preserve the full candidate set so exact title/brand matches
        # are still possible even when category scoring is imperfect.
        selected = positive + [p for s, p in scored if s <= 0]
    else:
        selected = [p for _, p in scored]

    async with _PRODUCT_SEARCH_LOCK:
        _PRODUCT_SEARCH_CACHE[cache_key] = (now_ms, selected)

    return selected[:limit]


async def _find_one_product(query: str, *, limit: int = 5, timeout_s: float) -> Optional[dict[str, Any]]:
    try:
        products = await _find_products(query, limit=limit, timeout_s=timeout_s)
    except Exception:
        return None

    if not products:
        return None
    first = products[0]
    return first if isinstance(first, dict) else None


def _normalize_match_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _best_product_match(
    desired: str,
    *,
    category: str,
    candidates: list[dict[str, Any]],
    brand_hint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    desired_norm = _normalize_match_text(desired)
    if not desired_norm:
        return None

    brand_norm = _normalize_match_text(brand_hint or "")
    brand_required = brand_norm and brand_norm not in {"unknown", "aurora", "premium", "dupe"}
    best: tuple[float, dict[str, Any]] | None = None

    for cand in candidates:
        title = str(cand.get("title") or cand.get("name") or "")
        if not title.strip():
            continue

        title_norm = _normalize_match_text(title)
        if not title_norm:
            continue

        if brand_required and brand_norm not in title_norm:
            continue

        ratio = difflib.SequenceMatcher(a=desired_norm, b=title_norm).ratio()
        cat_score = float(_score_product_for_category(category, cand))
        brand_bonus = 0.2 if brand_norm and brand_norm in title_norm else 0.0

        score = ratio + (cat_score * 0.06) + brand_bonus
        if best is None or score > best[0]:
            best = (score, cand)

    if not best:
        return None

    score, chosen = best
    # Require a minimum similarity so we don't show clearly unrelated products.
    if score < 0.35:
        return None
    return chosen


def _analysis_from_diagnosis(diagnosis: Optional[dict[str, Any]]) -> dict[str, Any]:
    diag = diagnosis if isinstance(diagnosis, dict) else {}

    skin_type_raw = diag.get("skinType") or diag.get("skin_type")
    skin_type = str(skin_type_raw).strip().lower() if skin_type_raw else ""

    barrier_raw = diag.get("barrierStatus") or diag.get("barrier_status") or diag.get("barrier")
    barrier = str(barrier_raw).strip().lower() if barrier_raw is not None else "unknown"

    regimen_raw = diag.get("currentRoutine") or diag.get("current_routine") or diag.get("current_regimen")
    regimen = str(regimen_raw).strip().lower() if regimen_raw else "basic"

    concerns_raw = diag.get("concerns") if isinstance(diag.get("concerns"), list) else []
    concerns = [str(c).strip().lower() for c in concerns_raw if c]
    cset = set(concerns)

    needs_risk = any(c in cset for c in {"acne", "dark_spots", "wrinkles", "pores"})
    features: list[dict[str, Any]] = []

    # 1) Barrier framing (safe default).
    if barrier in {"impaired", "reactive", "stressed", "sensitive"}:
        features.append(
            {
                "observation": "Barrier likely stressed → prioritize repair and avoid stacking strong actives until stinging/redness calms.",
                "confidence": "pretty_sure",
            }
        )
    elif barrier in {"healthy", "stable", "ok", "good"}:
        features.append(
            {
                "observation": "Barrier seems stable → you can introduce actives gradually (one at a time, low frequency).",
                "confidence": "somewhat_sure",
            }
        )
    else:
        features.append(
            {
                "observation": "Barrier status is unclear → start gentle and treat skin as sensitive until proven otherwise.",
                "confidence": "somewhat_sure",
            }
        )

    # 2) Skin-type context (self-reported).
    skin_type_map: dict[str, str] = {
        "oily": "Self-reported oily skin → higher sebum/clogging tendency; avoid over-stripping (oily skin can still be dehydrated).",
        "dry": "Self-reported dry skin → prioritize barrier support and longer-lasting hydration.",
        "combination": "Self-reported combination skin → balance: lighter layers on T‑zone, richer support on drier areas.",
        "normal": "Self-reported normal skin → keep it minimal and consistent; avoid unnecessary actives.",
        "sensitive": "Self-reported sensitive skin → minimize fragrance/alcohol and introduce actives slowly.",
    }
    if skin_type in skin_type_map:
        features.append({"observation": skin_type_map[skin_type], "confidence": "pretty_sure"})

    # 3) Goals / concerns (self-reported).
    concern_map: dict[str, str] = {
        "acne": "Acne/clogged pores goal → change one thing at a time; if using actives, start 2–3 nights/week and titrate only if irritation stays low.",
        "pores": "Pores/texture goal → gentle exfoliation can help, but avoid combining multiple strong actives in the same night.",
        "dark_spots": "Dark spots goal → daily SPF is the biggest lever; add ONE brightening active only after tolerance is confirmed.",
        "wrinkles": "Fine lines goal → sunscreen + hydration first; consider a retinoid only after the barrier feels stable (start 2 nights/week).",
        "redness": "Redness/sensitivity goal → reduce triggers and prioritize calm/repair before adding new actives.",
        "dehydration": "Dehydration goal → focus on hydration + repair and avoid harsh cleansing that can worsen tightness.",
        "dullness": "Dullness goal → often improves with consistent SPF + hydration; consider gentle exfoliation later if skin tolerates it.",
    }

    for key in ["acne", "dark_spots", "wrinkles", "pores", "redness", "dehydration", "dullness"]:
        if key in cset and key in concern_map:
            features.append({"observation": concern_map[key], "confidence": "pretty_sure"})
        if len(features) >= 5:
            break

    # 4) Routine complexity (helps explain “why minimal”).
    if regimen in {"full", "complex", "many"} and len(features) < 6:
        features.append(
            {
                "observation": "A more complex regimen increases conflict/irritation risk → simplify first, then re‑introduce actives stepwise.",
                "confidence": "somewhat_sure",
            }
        )
    elif regimen in {"none", "no", "start", "from scratch"} and len(features) < 6:
        features.append(
            {
                "observation": "No current regimen → build a stable baseline first (cleanse + moisturize + SPF) before adding actives.",
                "confidence": "pretty_sure",
            }
        )

    # Ensure the card feels informative even when the user provides only 1 concern.
    if len(features) < 4:
        features.append(
            {
                "observation": "Key unknowns (current products + irritation history) can change the plan — share your cleanser/actives/moisturizer/SPF for a safer, tighter recommendation.",
                "confidence": "not_sure",
            }
        )

    strategy = _strategy_from_profile(diagnosis, None)

    return {"features": features[:6], "strategy": strategy, "needs_risk_check": needs_risk}


def _strategy_from_profile(
    diagnosis: Optional[dict[str, Any]],
    detected: Optional[dict[str, Any]] = None,
) -> str:
    concerns_raw = diagnosis.get("concerns") if isinstance(diagnosis, dict) else []
    concerns = {str(c).strip().lower() for c in concerns_raw if c} if isinstance(concerns_raw, list) else set()

    barrier = None
    if isinstance(diagnosis, dict):
        barrier = diagnosis.get("barrierStatus") or diagnosis.get("barrier_status") or diagnosis.get("barrier")
    barrier = str(barrier).strip().lower() if barrier is not None else "unknown"

    detected = detected if isinstance(detected, dict) else {}
    oily_acne = bool(detected.get("oily_acne")) or ("acne" in concerns)
    barrier_impaired = bool(detected.get("barrier_impaired")) or barrier == "impaired"
    barrier_unknown = barrier in {"unknown", "unsure", "not sure", "n/a", ""}
    sensitive = bool(detected.get("sensitive_skin")) or ("redness" in concerns) or barrier_impaired or barrier_unknown
    dark_spots = "dark_spots" in concerns
    anti_aging = "wrinkles" in concerns

    lines: list[str] = []

    if sensitive:
        lines.append(
            "Barrier-first for 7–10 days: avoid stacking strong actives (retinoids, acids, benzoyl peroxide). If you get stinging/peeling, pause actives and focus on repair."
        )
        lines.append("Minimal baseline: AM rinse (or gentle cleanse) → moisturizer → SPF; PM gentle cleanse → moisturizer.")
        lines.append("Re‑introduce only ONE active at a time once skin feels calm for 3+ days (start 2 nights/week).")

    if oily_acne:
        lines.append(
            "For oil + breakouts: keep cleansing gentle (avoid over‑stripping) and introduce ONE acne active at a time (e.g., BHA) 2–3 nights/week."
        )
        lines.append("Avoid combining multiple strong actives in the same night; prioritize consistency + sunscreen.")

    if dark_spots:
        lines.append(
            "For dark spots: daily SPF is non‑negotiable; add one brightening active only after tolerance is confirmed (irritation can worsen pigmentation)."
        )

    if anti_aging:
        lines.append(
            "For fine lines: sunscreen + hydration first; consider a retinoid only after the barrier feels stable (start 2 nights/week and increase slowly)."
        )
        lines.append("Timeline: expect visible texture/line changes in ~8–12 weeks; faster changes are usually irritation, not improvement.")

    if not lines:
        lines.append("Keep it simple: AM gentle cleanse → moisturizer → SPF; PM cleanse → moisturizer.")

    lines.append(
        "Quick question: what cleanser/actives/moisturizer/SPF are you using right now, and do you ever sting/burn after products?"
    )

    # Keep it short for the UI.
    return " ".join(lines[:6]).strip()


def _ensure_strategy_has_question(*, strategy: str, lang_code: Literal["EN", "CN"]) -> str:
    """
    The analysis "strategy" must end with a direct question so users know what to do next.
    Avoid adding fake precision; ask for high-signal missing inputs instead.
    """

    base = (strategy or "").strip()
    if not base:
        base = ""

    # If there's already a question (English or CJK), keep it.
    if re.search(r"[?？]", base):
        return base[:900]

    question = (
        "Quick question: what cleanser/actives/moisturizer/SPF are you using right now, and do you ever sting/burn after products?"
        if lang_code == "EN"
        else "快速确认一下：你现在用的洁面/活性（酸/维A/过氧化苯甲酰等）/保湿/防晒分别是什么？用完会刺痛或发红吗？"
    )

    if not base:
        return question[:900]

    combined = f"{base.rstrip()} {question}"
    if len(combined) <= 900:
        return combined

    max_base_len = 900 - len(question) - 1
    if max_base_len <= 0:
        return question[:900]

    trimmed_base = base[:max_base_len].rstrip()
    return f"{trimmed_base} {question}"


def _normalize_analysis_from_llm(obj: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    raw_features = obj.get("features")
    features: list[dict[str, Any]] = []
    if isinstance(raw_features, list):
        for item in raw_features:
            if not isinstance(item, dict):
                continue
            observation = item.get("observation") or item.get("finding") or item.get("text")
            if not isinstance(observation, str) or not observation.strip():
                continue
            confidence = item.get("confidence")
            if not isinstance(confidence, str):
                confidence = "somewhat_sure"
            confidence = confidence.strip()
            if confidence not in {"pretty_sure", "somewhat_sure", "not_sure"}:
                confidence = "somewhat_sure"
            features.append({"observation": observation.strip()[:220], "confidence": confidence})

    strategy = obj.get("strategy")
    if isinstance(strategy, list):
        strategy = "\n".join(str(x) for x in strategy if x)
    if not isinstance(strategy, str):
        strategy = ""

    needs_risk_check = obj.get("needs_risk_check")
    if needs_risk_check is None:
        needs_risk_check = obj.get("needsRiskCheck")
    if isinstance(needs_risk_check, str):
        needs_risk_check = needs_risk_check.strip().lower() in {"1", "true", "yes", "y"}
    if not isinstance(needs_risk_check, bool):
        needs_risk_check = False

    if not (features or strategy.strip()):
        return None

    trimmed_features = features[:6]

    # Even when Aurora replies, it may return only 2–3 features.
    # Ensure the UI has enough substance without introducing fake precision.
    if 0 < len(trimmed_features) < 4:
        trimmed_features.append(
            {
                "observation": "Key unknowns (current products + irritation history) can change the plan — share your cleanser/actives/moisturizer/SPF for a safer, tighter recommendation.",
                "confidence": "not_sure",
            }
        )

    return {
        "features": trimmed_features,
        "strategy": strategy.strip()[:900],
        "needs_risk_check": needs_risk_check,
    }

def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _analysis_contains_cjk(obj: dict[str, Any]) -> bool:
    if _contains_cjk(str(obj.get("strategy") or "")):
        return True
    feats = obj.get("features")
    if isinstance(feats, list):
        for f in feats:
            if isinstance(f, dict) and _contains_cjk(str(f.get("observation") or "")):
                return True
    return False


def _analysis_violates_guardrails(*, analysis_obj: dict[str, Any], photos_provided: bool) -> bool:
    """
    Protect against "fake precision" and visual claims when no photos are provided.
    If violated, we fall back to rule-based analysis.
    """

    pieces: list[str] = [str(analysis_obj.get("strategy") or "")]
    feats = analysis_obj.get("features")
    if isinstance(feats, list):
        for f in feats:
            if isinstance(f, dict):
                pieces.append(str(f.get("observation") or ""))
    joined = " ".join(pieces).strip()
    lowered = joined.lower()

    # Avoid fake numeric scoring.
    if "match score" in lowered or "%" in joined:
        return True

    # Avoid visual claims when we have no photos.
    if not photos_provided:
        visual_markers = [
            "looks",
            "i see",
            "in the photo",
            "in the photos",
            "photo shows",
            "appears to be",
            "looks like",
        ]
        if any(m in lowered for m in visual_markers):
            return True

    return False


async def _translate_analysis_to_english(
    *,
    analysis_obj: dict[str, Any],
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> dict[str, Any]:
    """
    Aurora sometimes replies in Chinese even when asked for English. For analysis JSON,
    translate values while keeping the same schema.
    """

    try:
        source = json.dumps(analysis_obj, ensure_ascii=False)
    except Exception:
        return analysis_obj

    try:
        translation = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=(
                "Translate the following JSON into English.\n"
                "Rules:\n"
                "- Return ONLY valid JSON.\n"
                "- Keep the same keys and structure.\n"
                "- Keep `confidence` values as one of: pretty_sure | somewhat_sure | not_sure.\n"
                "- Do NOT add any numeric scores/percentages.\n"
                "IMPORTANT: Reply ONLY in English. Do not use Chinese.\n\n"
                f"JSON:\n{source}"
            ),
            timeout_s=DEFAULT_TIMEOUT_S,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        answer = translation.get("answer") if isinstance(translation, dict) else None
        parsed = extract_json_object(answer or "")
        normalized = _normalize_analysis_from_llm(parsed)
        return normalized or analysis_obj
    except Exception as exc:
        logger.warning("Analysis translation failed; keeping original. err=%s", exc)
        return analysis_obj


async def _translate_analysis_to_chinese(
    *,
    analysis_obj: dict[str, Any],
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> dict[str, Any]:
    """
    Aurora sometimes replies in English even when asked for Chinese. For analysis JSON,
    translate values while keeping the same schema.
    """

    try:
        source = json.dumps(analysis_obj, ensure_ascii=False)
    except Exception:
        return analysis_obj

    try:
        translation = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=(
                "Translate the following JSON into Simplified Chinese.\n"
                "Rules:\n"
                "- Return ONLY valid JSON.\n"
                "- Keep the same keys and structure.\n"
                "- Keep `confidence` values as one of: pretty_sure | somewhat_sure | not_sure.\n"
                "- Do NOT add any numeric scores/percentages.\n"
                "IMPORTANT: Reply ONLY in Simplified Chinese. Do not use English.\n\n"
                f"JSON:\n{source}"
            ),
            timeout_s=DEFAULT_TIMEOUT_S,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        answer = translation.get("answer") if isinstance(translation, dict) else None
        parsed = extract_json_object(answer or "")
        normalized = _normalize_analysis_from_llm(parsed)
        return normalized or analysis_obj
    except Exception as exc:
        logger.warning("Analysis translation failed; keeping original. err=%s", exc)
        return analysis_obj


def _budget_tier_to_aurora_budget(budget_tier: Any) -> str:
    mapping = {"$": "¥200", "$$": "¥500", "$$$": "¥1000+"}
    if isinstance(budget_tier, str):
        v = budget_tier.strip()
        if v in mapping:
            return mapping[v]
        if v.startswith("¥"):
            return v
        if v.isdigit():
            return f"¥{v}"
    return "¥500"


def _aurora_profile_line(
    *,
    diagnosis: Optional[dict[str, Any]],
    market: str,
    budget: str,
) -> str:
    skin_type = None
    concerns: list[str] = []
    current_routine = None
    barrier_status = None

    if isinstance(diagnosis, dict):
        skin_type = diagnosis.get("skinType") or diagnosis.get("skin_type")
        concerns_raw = diagnosis.get("concerns")
        if isinstance(concerns_raw, list):
            concerns = [str(c) for c in concerns_raw if c]
        current_routine = diagnosis.get("currentRoutine") or diagnosis.get("current_routine")
        barrier_status = diagnosis.get("barrierStatus") or diagnosis.get("barrier_status") or diagnosis.get("barrier")

    concerns_str = ", ".join(concerns) if concerns else "none"
    skin_str = str(skin_type or "unknown")
    routine_str = str(current_routine or "basic")
    barrier_str = str(barrier_status or "unknown")

    return (
        # NOTE: Avoid the substring "routine" in this line because Aurora's /api/chat
        # intent heuristics treat any mention of "routine" as a routine request.
        f"skin_type={skin_str}; barrier_status={barrier_str}; concerns={concerns_str}; region={market}; budget={budget}; current_regimen={routine_str}."
    )

def _aurora_profile_sentence(
    *,
    diagnosis: Optional[dict[str, Any]],
    market: str,
    budget: str,
) -> str:
    skin_type = None
    concerns: list[str] = []
    barrier_status = None

    if isinstance(diagnosis, dict):
        skin_type = diagnosis.get("skinType") or diagnosis.get("skin_type")
        concerns_raw = diagnosis.get("concerns")
        if isinstance(concerns_raw, list):
            concerns = [str(c) for c in concerns_raw if c]
        barrier_status = diagnosis.get("barrierStatus") or diagnosis.get("barrier_status") or diagnosis.get("barrier")

    # Map front-end concern IDs to keywords Aurora's routine planner reliably detects.
    # (Aurora's current clarify logic is keyword-based; include bilingual hints for robustness.)
    alias_map = {
        "acne": "acne (痘痘)",
        "dark_spots": "dark spots / hyperpigmentation (淡斑/痘印)",
        "dullness": "brightening (提亮/美白)",
        "wrinkles": "anti-aging (抗老/细纹)",
        "aging": "anti-aging (抗老/细纹)",
        "pores": "closed comedones / rough texture (闭口/黑头/粗糙)",
        "redness": "redness / sensitive skin (泛红敏感)",
        "dehydration": "hydration + repair (补水修护)",
        "repair": "barrier repair (屏障修护)",
        "barrier": "barrier repair (屏障修护)",
    }

    normalized: list[str] = []
    for c in concerns:
        key = c.strip().lower()
        normalized.append(alias_map.get(key, c))
    # Preserve order but remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for c in normalized:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)

    concerns_str = ", ".join(deduped) if deduped else "none"
    skin_str = str(skin_type or "unknown")
    barrier_norm = str(barrier_status or "").strip().lower()
    if barrier_norm in {"healthy", "stable", "ok", "good"}:
        barrier_str = "Stable barrier (no stinging/redness) / 屏障稳定"
    elif barrier_norm in {"impaired", "sensitive", "reactive"}:
        barrier_str = "Impaired barrier (stinging/redness) / 刺痛泛红，屏障受损"
    elif barrier_norm in {"unknown", "unsure", "not sure", "n/a"}:
        barrier_str = "Barrier status unknown (not sure) / 不确定屏障状态"
    else:
        barrier_str = str(barrier_status or "unknown")

    return f"User profile: skin type {skin_str}; barrier status: {barrier_str}; concerns: {concerns_str}; region: {market}; budget: {budget}."

def _normalize_clarification(clarification: Any, *, language: Literal["EN", "CN"]) -> Any:
    if not isinstance(clarification, dict):
        return clarification

    questions = clarification.get("questions")
    if not isinstance(questions, list) or not questions:
        return clarification

    templates: dict[str, dict[str, Any]] = {}
    if language == "CN":
        templates = {
            "anchor": {
                "question": "你想评估的具体产品是？",
                "options": ["发产品名（品牌 + 名称）", "粘贴购买链接", "上传产品照片/成分表"],
            },
            "skin_type": {
                "question": "你的肤质更接近哪一种？",
                "options": ["油皮", "干皮", "混合皮", "敏感肌", "不确定"],
            },
            "barrier_status": {
                "question": "你最近是否有刺痛/泛红/爆皮（可能屏障受损）？",
                "options": ["没有", "轻微", "明显（刺痛/泛红）", "不确定"],
            },
            "concerns": {
                "question": "你最想优先解决的 1-2 个问题是？",
                "options": ["闭口/黑头", "痘痘", "暗沉/美白", "泛红敏感", "抗老", "补水修护"],
            },
        }
    else:
        templates = {
            "anchor": {
                "question": "Which exact product do you want to evaluate?",
                "options": ["Send the product name (brand + name)", "Paste a product link", "Upload a product photo/ingredients"],
            },
            "skin_type": {
                "question": "Which skin type fits you best?",
                "options": ["Oily", "Dry", "Combination", "Sensitive", "Not sure"],
            },
            "barrier_status": {
                "question": "Lately, do you get stinging/redness/flaking (possible barrier stress)?",
                "options": ["No", "Mild", "Yes (stinging/redness)", "Not sure"],
            },
            "concerns": {
                "question": "What are your top 1–2 priorities right now?",
                "options": ["Closed comedones/blackheads", "Acne", "Dark spots/brightening", "Redness/sensitivity", "Anti-aging", "Hydration/repair"],
            },
        }

    normalized: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        tpl = templates.get(qid)
        if tpl:
            normalized.append({**q, **tpl, "id": qid})
        else:
            normalized.append(q)

    if not normalized:
        return clarification
    return {**clarification, "questions": normalized}


def _analysis_from_aurora_context(
    diagnosis: Optional[dict[str, Any]],
    aurora_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    base = _analysis_from_diagnosis(diagnosis)
    detected = aurora_context.get("detected") if isinstance(aurora_context, dict) else None
    if not isinstance(detected, dict):
        base["strategy"] = _strategy_from_profile(diagnosis, None)
        return base

    features: list[dict[str, Any]] = []
    if detected.get("oily_acne") is True:
        features.extend(
            [
                {
                    "observation": "Oily/acne‑prone tendency detected → higher clogging risk; keep cleansing gentle and avoid over‑stripping.",
                    "confidence": "somewhat_sure",
                },
                {
                    "observation": "If you’re breakout‑prone, introduce actives stepwise (one at a time) to avoid irritation‑driven flares.",
                    "confidence": "somewhat_sure",
                },
            ]
        )
    if detected.get("sensitive_skin") is True:
        features.extend(
            [
                {
                    "observation": "Sensitive/reactive tendency detected → minimize irritants (fragrance/alcohol) and patch‑test new actives.",
                    "confidence": "somewhat_sure",
                },
            ]
        )
    if detected.get("barrier_impaired") is True:
        features.extend(
            [
                {
                    "observation": "Barrier‑stress signal detected → prioritize repair first; avoid stacking strong actives until irritation settles.",
                    "confidence": "somewhat_sure",
                },
            ]
        )

    merged_features = features + base.get("features", [])
    if merged_features:
        base["features"] = merged_features[:6]

    # Upgrade the default strategy into something more actionable even when we don't have
    # full CV signals; keep it product-agnostic and barrier-safe.
    base["strategy"] = _strategy_from_profile(diagnosis, detected)

    return base


def _looks_like_routine_request(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False

    # Avoid over-triggering on single-product questions (handled via anchor flow).
    product_q = any(k in t for k in ["this product", "this skincare", "this one", "is it good", "is it ok"]) or any(
        k in text for k in ["这款", "这个产品", "适合吗", "能用吗"]
    )
    if product_q:
        return False

    return any(
        k in t
        for k in [
            "routine",
            "regimen",
            "am/pm",
            "morning and night",
            "skincare steps",
            "skin care steps",
            "what should i use",
            "what should i buy",
            "build me",
            "recommend a routine",
            "simple routine",
            "simple regimen",
        ]
    ) or any(k in text for k in ["护肤流程", "护肤步骤", "早晚", "一套护肤", "搭配", "步骤", "护肤方案"])


def _looks_like_current_products_review_request(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False

    return any(
        k in t
        for k in [
            "review my products",
            "review my skincare",
            "check my products",
            "check my skincare",
            "analyze my products",
            "analyse my products",
            "what i'm using",
            "what i am using",
            "current products",
            "current regimen",
            "existing products",
        ]
    ) or any(k in text for k in ["评估我现在用", "分析我现在用", "看看我现在用", "我现在用的护肤品", "现有产品", "现在用的产品"])


def _looks_like_product_list_input(text: str, lang_code: Literal["EN", "CN"]) -> bool:
    """
    Heuristic: detect when the user is pasting their current products (often as a list),
    especially after we asked "what are you using now?" in analysis.
    """

    raw = (text or "").strip()
    if len(raw) < 30:
        return False

    lower = raw.lower()

    # List-like formatting is a strong signal.
    if "\n" in raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            bullet_like = sum(
                1
                for ln in lines
                if ln.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "am", "pm", "早", "晚"))
            )
            if bullet_like >= 1:
                return True

    keywords_en = [
        "am",
        "pm",
        "morning",
        "night",
        "cleanser",
        "face wash",
        "toner",
        "serum",
        "essence",
        "moisturizer",
        "cream",
        "spf",
        "sunscreen",
        "retinol",
        "tretinoin",
        "vitamin c",
        "aha",
        "bha",
        "benzoyl",
        "niacinamide",
    ]
    keywords_cn = [
        "早",
        "晚",
        "晨",
        "夜",
        "洁面",
        "洗面奶",
        "爽肤水",
        "精华",
        "面霜",
        "保湿",
        "防晒",
        "a酸",
        "维a",
        "果酸",
        "水杨酸",
        "过氧化苯甲酰",
        "烟酰胺",
    ]

    keywords = keywords_cn if lang_code == "CN" else keywords_en
    hits = sum(1 for k in keywords if k in lower)

    # Require multiple signals to avoid false positives on normal chat.
    return hits >= 2


def _is_no_products_reply(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False

    if any(k in t for k in ["none", "no products", "no routine", "start fresh", "from scratch"]):
        return True
    return any(k in text for k in ["没有", "无", "从零开始", "不用", "没用"])


def _normalize_skin_type_answer(text: str, lang_code: Literal["EN", "CN"]) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None

    # English
    if any(k in t for k in ["oily", "oil"]):
        return "oily"
    if any(k in t for k in ["dry", "dehydrated"]):
        return "dry"
    if "combination" in t or "combo" in t:
        return "combination"
    if any(k in t for k in ["sensitive", "reactive"]):
        return "sensitive"
    if any(k in t for k in ["not sure", "unsure", "unknown", "idk", "don't know"]):
        return "unknown"

    # Chinese
    if any(k in t for k in ["油", "出油"]):
        return "oily"
    if any(k in t for k in ["干", "紧绷"]):
        return "dry"
    if "混合" in t:
        return "combination"
    if any(k in t for k in ["敏感", "泛红", "易红"]):
        return "sensitive"
    if any(k in t for k in ["不确定", "不知道", "不清楚", "未知"]):
        return "unknown"

    return None


def _normalize_barrier_answer(text: str, lang_code: Literal["EN", "CN"]) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None

    # Options from our UI: No / Mild / Yes / Not sure
    if any(k in t for k in ["no", "none", "not really", "never"]):
        return "healthy"
    if any(k in t for k in ["mild", "sometimes", "a little"]):
        return "unknown"
    if any(k in t for k in ["yes", "stinging", "burn", "burning", "redness", "flaking", "peeling", "irritat"]):
        return "impaired"
    if any(k in t for k in ["not sure", "unsure", "unknown", "idk", "don't know"]):
        return "unknown"

    # Chinese
    if any(k in t for k in ["没有", "不", "不会", "从不"]):
        return "healthy"
    if any(k in t for k in ["轻微", "偶尔", "有点"]):
        return "unknown"
    if any(k in t for k in ["刺痛", "灼", "红", "脱皮", "起皮", "刺激", "过敏"]):
        return "impaired"
    if any(k in t for k in ["不确定", "不知道", "不清楚", "未知"]):
        return "unknown"

    return None


def _looks_like_reco_rationale_request(text: str) -> bool:
    """
    Detect follow-ups like "why these recommendations / what evidence?" so we can
    explain the *current* recommended plan instead of generating a new one.
    """

    t = text.strip().lower()
    if not t:
        return False

    asks_why_or_evidence = any(
        k in t
        for k in [
            "why",
            "reason",
            "rationale",
            "evidence",
            "scientific",
            "science",
            "based on",
            "explain",
        ]
    ) or any(k in text for k in ["为什么", "为啥", "原因", "依据", "科学", "证据", "解释"])

    refers_to_recos = any(
        k in t
        for k in [
            "recommend",
            "recommendation",
            "recommended",
            "these",
            "those",
            "picked",
            "choose",
            "this plan",
            "this regimen",
            "this routine",
            "your picks",
            "your choices",
        ]
    ) or any(k in text for k in ["推荐", "这些", "那这些", "这套", "这组", "你选", "为什么选"])

    return asks_why_or_evidence and refers_to_recos


def _bucket_strength(val: Any) -> str:
    try:
        v = float(val)
    except Exception:
        return "unknown"
    if v >= 0.75:
        return "high"
    if v >= 0.45:
        return "medium"
    if v > 0:
        return "low"
    return "unknown"


def _format_strength(label: str, strength: str, *, language: Literal["EN", "CN"]) -> str:
    if language == "CN":
        mapping = {"high": "高", "medium": "中", "low": "低", "unknown": "未知"}
        return f"{label}：{mapping.get(strength, '未知')}"
    mapping = {"high": "high", "medium": "medium", "low": "low", "unknown": "unknown"}
    return f"{label}: {mapping.get(strength, 'unknown')}"


def _explain_routine_from_aurora_context(
    aurora_context: dict[str, Any],
    *,
    language: Literal["EN", "CN"],
) -> Optional[str]:
    routine = aurora_context.get("routine")
    if not isinstance(routine, dict):
        routine = aurora_context.get("routine_primary") if isinstance(aurora_context.get("routine_primary"), dict) else None
    if not isinstance(routine, dict):
        return None

    am = routine.get("am")
    pm = routine.get("pm")
    steps_am = am if isinstance(am, list) else []
    steps_pm = pm if isinstance(pm, list) else []
    if not steps_am and not steps_pm:
        return None

    detected = aurora_context.get("detected") if isinstance(aurora_context.get("detected"), dict) else {}

    if language == "CN":
        lines: list[str] = []
        lines.append("下面是“为什么推荐这些”的科学解释（按功效机制 + 风险/兼容性来讲）：")
        if detected:
            parts = []
            if detected.get("oily_acne") is True:
                parts.append("偏油/易长痘")
            if detected.get("sensitive_skin") is True:
                parts.append("偏敏感")
            if detected.get("barrier_impaired") is True:
                parts.append("屏障可能受损")
            if parts:
                lines.append(f"你的画像要点：{ '、'.join(parts) }。")
        lines.append("")
    else:
        lines = []
        lines.append("Here’s the scientific rationale for the current recommendations (mechanism + safety + routine-fit):")
        if detected:
            parts = []
            if detected.get("oily_acne") is True:
                parts.append("oily/acne-prone tendency")
            if detected.get("sensitive_skin") is True:
                parts.append("sensitive/reactive tendency")
            if detected.get("barrier_impaired") is True:
                parts.append("possible barrier stress")
            if parts:
                lines.append(f"Your profile signals: {', '.join(parts)}.")
        lines.append("")

    def explain_steps(title: str, steps: list[dict[str, Any]]) -> None:
        if language == "CN":
            lines.append(title)
        else:
            lines.append(title)

        for step in steps:
            if not isinstance(step, dict):
                continue
            sku = step.get("sku") if isinstance(step.get("sku"), dict) else {}
            brand = str(sku.get("brand") or "").strip()
            name = str(sku.get("name") or "").strip()
            display = " ".join([p for p in [brand, name] if p]).strip() or "Unknown product"

            mech = sku.get("mechanism") if isinstance(sku.get("mechanism"), dict) else {}
            strengths: list[str] = []
            if mech:
                # Keep labels minimal and user-friendly.
                mapping = [
                    ("oil_control", "控油" if language == "CN" else "Oil control"),
                    ("acne_comedonal", "痘痘/闭口" if language == "CN" else "Acne/comedones"),
                    ("soothing", "舒缓" if language == "CN" else "Soothing"),
                    ("repair", "修护" if language == "CN" else "Barrier repair"),
                    ("brightening", "提亮" if language == "CN" else "Brightening"),
                ]
                for key, label in mapping:
                    if key in mech:
                        strengths.append(_format_strength(label, _bucket_strength(mech.get(key)), language=language))

            risk_flags = sku.get("risk_flags")
            risks: list[str] = []
            if isinstance(risk_flags, list):
                for rf in risk_flags:
                    if rf == "high_irritation":
                        risks.append("刺激性偏高" if language == "CN" else "Higher irritation risk")
                    elif rf:
                        risks.append(str(rf))

            notes = step.get("notes")
            note_text = ""
            if isinstance(notes, list) and notes:
                note_text = " ".join(str(n) for n in notes if n)

            if language == "CN":
                lines.append(f"- {display}")
                if strengths:
                    lines.append(f"  - 机制匹配：{'; '.join(strengths)}")
                if note_text:
                    lines.append(f"  - 目的：{note_text}")
                if risks:
                    lines.append(f"  - 注意：{'; '.join(risks)}（先从低频开始，耐受后再加）")
            else:
                lines.append(f"- {display}")
                if strengths:
                    lines.append(f"  - Mechanism fit: {'; '.join(strengths)}")
                if note_text:
                    lines.append(f"  - Purpose: {note_text}")
                if risks:
                    lines.append(f"  - Cautions: {'; '.join(risks)} (start low frequency and titrate)")

        lines.append("")

    explain_steps("🌞 AM" if language == "EN" else "🌞 早上（AM）", steps_am)
    explain_steps("🌙 PM" if language == "EN" else "🌙 晚上（PM）", steps_pm)

    if language == "CN":
        lines.append("如果你想要“更严格的科学依据/引用”，请点名其中某一款产品（或发链接/成分表），我可以逐条拆解其成分→作用机制→风险点。")
    else:
        lines.append(
            "If you want stricter evidence with ingredient-by-ingredient justification, name one product (or paste a link/ingredients) and I’ll break down MoA, safety flags, and conflicts."
        )

    return "\n".join(lines).strip()

def _require_brief_id(x_brief_id: Optional[str]) -> str:
    if not x_brief_id:
        raise HTTPException(status_code=400, detail="Missing X-Brief-ID")
    return x_brief_id


def _truncate_event_payload(value: Any, limit: int = 2000) -> Any:
    try:
        text = str(value)
    except Exception:
        return value
    if len(text) <= limit:
        return value
    return text[:limit] + "…"


def _format_posthog_timestamp(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, (int, float)):
        try:
            ts = float(raw)
            if ts > 1e12:  # ms
                ts /= 1000.0
            if ts < 1e9:  # too small / invalid
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            return None
    return None


def _append_events_jsonl_sink(*, dir_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    out_dir = Path(dir_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    date_key = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    file_path = out_dir / f"client-events-{date_key}.jsonl"
    with file_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _forward_events_to_posthog(*, rows: list[dict[str, Any]]) -> None:
    if not POSTHOG_API_KEY or not POSTHOG_HOST:
        return
    if not rows:
        return

    batch: list[dict[str, Any]] = []
    for row in rows:
        distinct_id = row.get("brief_id") or "anonymous"
        user_agent = row.get("user_agent")
        properties: dict[str, Any] = {
            "distinct_id": distinct_id,
            "brief_id": row.get("brief_id"),
            "trace_id": row.get("trace_id"),
            "source": row.get("source"),
            "data": row.get("data"),
            "user_agent": user_agent[:400] if isinstance(user_agent, str) else None,
        }
        if EVENTS_INCLUDE_CLIENT_IP:
            properties["client_ip"] = row.get("client_ip")

        item: dict[str, Any] = {
            "event": row.get("event_name"),
            "properties": {k: v for k, v in properties.items() if v is not None and v != ""},
        }
        ts = _format_posthog_timestamp(row.get("timestamp"))
        if ts:
            item["timestamp"] = ts
        batch.append(item)

    if not batch:
        return

    payload = {"api_key": POSTHOG_API_KEY, "batch": batch}
    url = f"{POSTHOG_HOST}/batch/"
    try:
        async with httpx.AsyncClient(timeout=POSTHOG_TIMEOUT_S) as client:
            res = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code >= 400:
            logger.warning("posthog_forward_failed status=%s body=%s", res.status_code, res.text[:500])
    except Exception as exc:
        logger.warning("posthog_forward_failed err=%s", getattr(exc, "message", str(exc)))


async def _write_events_jsonl_sink(*, rows: list[dict[str, Any]]) -> None:
    if not EVENTS_JSONL_SINK_DIR:
        return
    if not rows:
        return
    try:
        await asyncio.to_thread(_append_events_jsonl_sink, dir_path=EVENTS_JSONL_SINK_DIR, rows=rows)
    except Exception as exc:
        logger.warning("events_jsonl_sink_failed err=%s", getattr(exc, "message", str(exc)))


@router.post("/events")
async def ingest_events(
    request: Request,
    body: Any = Body(...),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    events: list[dict[str, Any]] = []
    if isinstance(body, dict) and isinstance(body.get("events"), list):
        for e in body.get("events") or []:
            if isinstance(e, dict):
                events.append(e)
    elif isinstance(body, list):
        for e in body:
            if isinstance(e, dict):
                events.append(e)
    elif isinstance(body, dict):
        events.append(body)

    if not events:
        raise HTTPException(status_code=400, detail="Missing events")

    # Prevent log/ingest abuse.
    if len(events) > 200:
        raise HTTPException(status_code=400, detail="Too many events")

    client_ip = request.headers.get("x-forwarded-for") or request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    accepted = 0
    rows_to_forward: list[dict[str, Any]] = []
    for raw in events:
        event_name = raw.get("event_name") or raw.get("eventName")
        brief_id = raw.get("brief_id") or raw.get("briefId")
        trace_id = raw.get("trace_id") or raw.get("traceId") or x_trace_id
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw.get("properties")

        if not isinstance(event_name, str) or not event_name.strip():
            continue

        row = {
            "event_name": event_name.strip()[:120],
            "brief_id": str(brief_id or "")[:120],
            "trace_id": str(trace_id or "")[:120],
            "timestamp": raw.get("timestamp"),
            "data": _truncate_event_payload(data, 4000) if data is not None else None,
            "source": raw.get("source") or "client",
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
        logger.info("client_event=%s", row)
        rows_to_forward.append(row)
        accepted += 1

        # Store a small tail for debugging (best-effort).
        if row["brief_id"]:
            try:
                stored = await SESSION_STORE.get(row["brief_id"])
                existing = stored.get("client_events")
                tail = existing if isinstance(existing, list) else []
                tail = [*tail, row][-50:]
                await SESSION_STORE.upsert(row["brief_id"], {"client_events": tail})
            except Exception:
                pass

    if rows_to_forward:
        if POSTHOG_API_KEY and POSTHOG_HOST:
            asyncio.create_task(_forward_events_to_posthog(rows=rows_to_forward))
        elif EVENTS_JSONL_SINK_DIR:
            asyncio.create_task(_write_events_jsonl_sink(rows=rows_to_forward))

    return {"ok": True, "accepted": accepted}


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
        "barrierStatus": body.get("barrierStatus") or body.get("barrier_status") or body.get("barrier") or "unknown",
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
    consent: Optional[bool] = Form(default=None),
    trace_id: Optional[str] = None,
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    stored = await SESSION_STORE.get(brief_id)
    existing_photos = stored.get("photos") if isinstance(stored.get("photos"), dict) else {}

    has_upload = daylight is not None or indoor_white is not None
    if has_upload and consent is not True:
        raise HTTPException(status_code=400, detail="Missing consent for photo upload")

    def _get_retry_count(slot: str) -> int:
        raw = existing_photos.get(slot) if isinstance(existing_photos, dict) else None
        if not isinstance(raw, dict):
            return 0
        v = raw.get("retry_count", raw.get("retryCount", 0))
        try:
            return int(v)
        except Exception:
            return 0

    photos_patch: dict[str, Any] = {}
    merged_photos: dict[str, Any] = dict(existing_photos) if isinstance(existing_photos, dict) else {}

    async def _process(slot: str, upload: UploadFile) -> None:
        old = existing_photos.get(slot) if isinstance(existing_photos, dict) else None
        old_upload_id: Optional[str] = None
        if isinstance(old, dict):
            raw_old_upload_id = old.get("upload_id")
            if isinstance(raw_old_upload_id, str) and raw_old_upload_id.strip():
                old_upload_id = raw_old_upload_id.strip()

        blob = await upload.read()
        content_type = (upload.content_type or "image/jpeg").strip() or "image/jpeg"
        result = await _upload_photo_via_pivota(
            blob=blob,
            content_type=content_type,
            file_name=upload.filename,
            user_id=brief_id,
            consent=bool(consent),
        )
        new_upload_id = result.get("upload_id")
        replaced_upload_deleted: Optional[bool] = None
        if old_upload_id and isinstance(new_upload_id, str) and new_upload_id and old_upload_id != new_upload_id:
            try:
                await _photos_api_json("DELETE", "/photos", params={"upload_id": old_upload_id}, timeout_s=DEFAULT_TIMEOUT_S)
                replaced_upload_deleted = True
                logger.info(
                    "photo_old_upload_deleted=%s",
                    {"brief_id": brief_id, "slot": slot, "old_upload_id": old_upload_id},
                )
            except Exception as exc:
                replaced_upload_deleted = False
                logger.info("Photo delete failed (ignored). upload_id=%s err=%s", old_upload_id, exc)
        retry_count = _get_retry_count(slot)
        if slot in merged_photos:
            retry_count += 1
        slot_patch = {
            "upload_id": result.get("upload_id"),
            "qc_status": result.get("qc_status") or "pending",
            "retry_count": max(0, retry_count),
        }
        if old_upload_id and isinstance(new_upload_id, str) and new_upload_id and old_upload_id != new_upload_id:
            slot_patch["replaced_upload_id"] = old_upload_id
            slot_patch["replaced_upload_deleted"] = replaced_upload_deleted
        advice = result.get("qc_advice")
        if isinstance(advice, dict):
            slot_patch["qc_advice"] = advice
        merged_photos[slot] = slot_patch
        photos_patch[slot] = slot_patch

    tasks = []
    if daylight is not None:
        tasks.append(_process("daylight", daylight))
    if indoor_white is not None:
        tasks.append(_process("indoor_white", indoor_white))
    if tasks:
        await asyncio.gather(*tasks)

    # Determine next state: if any uploaded photo fails QC, keep user in QC retry UI.
    issues = []
    for slot in ("daylight", "indoor_white"):
        raw = merged_photos.get(slot)
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("qc_status") or "").strip()
        if status and status != "passed":
            issues.append(slot)
    has_issues = len(issues) > 0
    next_state = "S3a_PHOTO_QC" if has_issues else "S4_ANALYSIS_LOADING"
    patch = {"photos": photos_patch, "next_state": next_state}
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "photos": merged_photos,
        },
    )
    return {"session": patch, "next_state": patch["next_state"]}


@router.get("/photos/qc")
async def photos_qc(
    upload_id: str,
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    upload_id = (upload_id or "").strip()
    if not upload_id:
        raise HTTPException(status_code=400, detail="Missing upload_id")

    result = await _photos_api_json(
        "GET",
        "/photos/qc",
        params={"upload_id": upload_id},
        timeout_s=DEFAULT_TIMEOUT_S,
    )

    raw_qc_status = result.get("qc_status") or result.get("qcStatus")
    if raw_qc_status is None and isinstance(result.get("qc"), dict):
        raw_qc_status = result["qc"].get("qc_status") or result["qc"].get("qcStatus")

    qc_status: str = "pending"
    if isinstance(raw_qc_status, str) and raw_qc_status.strip():
        normalized = raw_qc_status.strip()
        if normalized in {"passed", "too_dark", "has_filter", "blurry"}:
            qc_status = normalized

    qc_advice: Optional[dict[str, Any]] = None
    if isinstance(result.get("qc"), dict) and isinstance(result["qc"].get("advice"), dict):
        qc_advice = result["qc"]["advice"]

    try:
        stored = await SESSION_STORE.get(brief_id)
        photos = stored.get("photos") if isinstance(stored.get("photos"), dict) else {}
        updated: dict[str, Any] = dict(photos) if isinstance(photos, dict) else {}

        updated_any = False
        for slot in ("daylight", "indoor_white"):
            raw = updated.get(slot)
            if not isinstance(raw, dict):
                continue
            slot_upload_id = raw.get("upload_id")
            if isinstance(slot_upload_id, str) and slot_upload_id.strip() == upload_id:
                raw["qc_status"] = qc_status
                if isinstance(qc_advice, dict):
                    raw["qc_advice"] = qc_advice
                updated[slot] = raw
                updated_any = True

        if updated_any:
            await SESSION_STORE.upsert(
                brief_id,
                {
                    "trace_id": x_trace_id,
                    "photos": updated,
                },
            )
    except Exception:
        # Best-effort session update: ignore.
        pass

    return {"upload_id": upload_id, "qc_status": qc_status, "qc_advice": qc_advice}


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

    market = stored.get("market") or body.get("market") or "US"
    budget_tier = stored.get("budget_tier") or body.get("budget_tier") or "$$"
    budget = _budget_tier_to_aurora_budget(budget_tier)

    language = body.get("language")
    lang_code: Literal["EN", "CN"] = "EN"
    reply_language = "English"
    if isinstance(language, str) and language.strip().upper() in {"CN", "ZH", "ZH-CN", "ZH_HANS"}:
        lang_code = "CN"
        reply_language = "Simplified Chinese"

    reply_instruction = (
        "IMPORTANT: Reply ONLY in English. Do not use Chinese."
        if lang_code == "EN"
        else "请只用简体中文回答，不要使用英文。"
    )

    photos_raw = stored.get("photos") if isinstance(stored.get("photos"), dict) else {}
    photo_qc_parts: list[str] = []
    passed_count = 0
    for slot in ("daylight", "indoor_white"):
        raw = photos_raw.get(slot)
        if not isinstance(raw, dict):
            continue
        qc = raw.get("qc_status") or raw.get("qcStatus") or raw.get("qc")
        qc = str(qc).strip().lower() if qc is not None else ""
        if qc:
            photo_qc_parts.append(f"{slot}:{qc}")
        if qc == "passed":
            passed_count += 1
    photos_provided = passed_count > 0

    aurora_context: Optional[dict[str, Any]] = None
    aurora_answer: Optional[str] = None
    try:
        profile_line = _aurora_profile_line(diagnosis=diagnosis_payload, market=market, budget=budget)
        photo_line = f"photos_provided={'yes' if photos_provided else 'no'}; photo_qc={', '.join(photo_qc_parts) if photo_qc_parts else 'none'}."
        prompt = (
            (f"{GLOW_SYSTEM_PROMPT}\n\n" if GLOW_SYSTEM_PROMPT else "")
            + f"{profile_line}\n"
            + f"{photo_line}\n"
            + "Task: Provide a skin assessment that is honest about uncertainty and feels like a cautious dermatologist.\n\n"
            + "Return ONLY a valid JSON object (no markdown) with this exact shape:\n"
            + '{\n'
            + '  "features": [\n'
            + '    {"observation": "…", "confidence": "pretty_sure" | "somewhat_sure" | "not_sure"}\n'
            + '  ],\n'
            + '  "strategy": "…",\n'
            + '  "needs_risk_check": true | false\n'
            + '}\n\n'
            + "Rules:\n"
            + "- DO NOT output any numeric scores/percentages (no “match score”).\n"
            + "- If photos_provided=no: DO NOT make visual claims. Avoid 'looks', 'I see', or 'in the photo'. Base everything on self-report.\n"
            + "- Observations must be about the user's skin goals/safety (barrier, acne risk, pigmentation, irritation, etc.) and include a short reason.\n"
            + "- Strategy must be actionable and stepwise: (a) what to do for the next 7 days, (b) how to introduce actives safely if relevant, (c) END with ONE direct clarifying question (must include a '?' or '？').\n"
            + "- DO NOT recommend specific products/brands yet.\n"
            + "- Keep it concise: 4–6 features; strategy under 900 characters.\n"
            + f"Language: {reply_language}.\n"
            + f"{reply_instruction}\n"
        )

        aurora_payload = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=prompt,
            timeout_s=DEFAULT_TIMEOUT_S,
        )
        aurora_ctx_raw = aurora_payload.get("context") if isinstance(aurora_payload, dict) else None
        aurora_context = aurora_ctx_raw if isinstance(aurora_ctx_raw, dict) else None
        if isinstance(aurora_payload.get("answer"), str):
            aurora_answer = aurora_payload.get("answer")
    except Exception as exc:
        logger.warning("Aurora analysis call failed; falling back. err=%s", exc)

    analysis_result: dict[str, Any]
    parsed = extract_json_object(aurora_answer or "")
    normalized = _normalize_analysis_from_llm(parsed)
    if normalized:
        if _analysis_violates_guardrails(analysis_obj=normalized, photos_provided=photos_provided):
            logger.warning("LLM analysis violated guardrails; using rule-based fallback.")
            analysis_result = _analysis_from_aurora_context(diagnosis_payload, aurora_context)
        else:
            analysis_result = normalized
    else:
        analysis_result = _analysis_from_aurora_context(diagnosis_payload, aurora_context)

    if lang_code == "EN" and _analysis_contains_cjk(analysis_result):
        analysis_result = await _translate_analysis_to_english(
            analysis_obj=analysis_result,
            llm_provider=body.get("llm_provider") if isinstance(body.get("llm_provider"), str) else None,
            llm_model=body.get("llm_model") if isinstance(body.get("llm_model"), str) else None,
        )
    if lang_code == "CN" and not _analysis_contains_cjk(analysis_result):
        analysis_result = await _translate_analysis_to_chinese(
            analysis_obj=analysis_result,
            llm_provider=body.get("llm_provider") if isinstance(body.get("llm_provider"), str) else None,
            llm_model=body.get("llm_model") if isinstance(body.get("llm_model"), str) else None,
        )

    analysis_result["strategy"] = _ensure_strategy_has_question(
        strategy=str(analysis_result.get("strategy") or ""),
        lang_code=lang_code,
    )

    patch = {"analysis": analysis_result, "next_state": "S5_ANALYSIS_SUMMARY"}
    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "state": patch["next_state"],
            "analysis": analysis_result,
            "aurora_context": aurora_context,
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
    current_products_text = stored.get("current_products_text") if isinstance(stored.get("current_products_text"), str) else None

    goal_map = {
        "routine": "a simple, safe daily skincare regimen",
        "breakouts": "reduce breakouts/acne with minimal irritation",
        "brightening": "brighten and fade dark spots with low irritation",
    }
    goal_note = goal_map.get(str(intent_id).strip().lower(), str(intent_id or "a simple skincare regimen"))

    # Default seed queries (fallback only). We run one product search per category,
    # then pick a "premium" and "dupe" by price from the result set.
    seed_queries: dict[str, str] = {
        "cleanser": "gel cleanser",
        "treatment": "niacinamide serum",
        "moisturizer": "face moisturizer",
        "sunscreen": "sunscreen SPF 50",
    }

    categories_am = ["cleanser", "treatment", "moisturizer", "sunscreen"]
    categories_pm = ["cleanser", "treatment", "moisturizer"]
    conflicts: list[str] = []

    def _step_category(step: dict[str, Any]) -> Optional[str]:
        sku = step.get("sku") if isinstance(step.get("sku"), dict) else {}
        cat = sku.get("category")
        if isinstance(cat, str):
            c = cat.strip().lower()
            if c in {"cleanser", "treatment", "moisturizer", "sunscreen"}:
                return c
        step_name = step.get("step")
        if isinstance(step_name, str):
            s = step_name.lower()
            if "clean" in s:
                return "cleanser"
            if "treat" in s or "serum" in s or "active" in s:
                return "treatment"
            if "moist" in s or "cream" in s:
                return "moisturizer"
            if "spf" in s or "sun" in s:
                return "sunscreen"
        return None

    def _step_query(step: dict[str, Any]) -> Optional[str]:
        sku = step.get("sku") if isinstance(step.get("sku"), dict) else {}
        cat = _step_category(step)
        name = sku.get("name")
        brand = sku.get("brand")
        if not isinstance(name, str) or not name.strip():
            return None
        base = f"{brand.strip()} {name.strip()}" if isinstance(brand, str) and brand.strip() and brand.strip().lower() != "unknown" else name.strip()

        suffix = None
        if cat == "cleanser":
            suffix = "cleanser"
        elif cat == "moisturizer":
            suffix = "moisturizer"
        elif cat == "sunscreen":
            suffix = "sunscreen SPF"
        elif cat == "treatment":
            suffix = "serum"

        q = base
        if suffix and suffix.split()[0] not in q.lower():
            q = f"{q} {suffix}"

        return q[:120]

    async def _aurora_routine_for_budget(budget: str) -> Optional[dict[str, Any]]:
        try:
            preference_note = (
                "Prefer cheaper options"
                if preference in {"pref_cheaper", "cheaper", "cheap"}
                else "Prefer gentler / lower irritation"
                  if preference in {"pref_gentler", "gentler", "gentle"}
                  else "Prefer fastest delivery"
                    if preference in {"pref_fastest_delivery", "fastest_delivery"}
                    else "No special preference"
            )
            products_note = ""
            if current_products_text:
                products_note = (
                    f"Current products (try to reuse what fits; suggest minimal replacements): {current_products_text[:1200]}\n"
                )
            payload = await aurora_chat(
                base_url=AURORA_DECISION_BASE_URL,
                query=(
                    f"{_aurora_profile_sentence(diagnosis=diagnosis_payload, market=market, budget=budget)}\n"
                    f"Goal: {goal_note}.\n"
                    f"{products_note}"
                    f"Preference: {preference_note}.\n"
                    "Please recommend a simple AM/PM skincare routine within my budget. Reply in English."
                ),
                timeout_s=DEFAULT_TIMEOUT_S,
            )
            ctx = payload.get("context") if isinstance(payload, dict) else None
            if not isinstance(ctx, dict):
                return None
            routine = ctx.get("routine")
            return routine if isinstance(routine, dict) else None
        except Exception as exc:
            logger.warning("Aurora routine call failed (budget=%s). err=%s", budget, exc)
            return None

    if not USE_PIVOTA_AGENT_SEARCH:
        premium_routine, dupe_routine = await asyncio.gather(_aurora_routine_for_budget("¥1000+"), _aurora_routine_for_budget("¥200"))

        premium_am = premium_routine.get("am") if isinstance(premium_routine, dict) else None
        premium_pm = premium_routine.get("pm") if isinstance(premium_routine, dict) else None
        dupe_am = dupe_routine.get("am") if isinstance(dupe_routine, dict) else None
        dupe_pm = dupe_routine.get("pm") if isinstance(dupe_routine, dict) else None

        premium_steps_am = premium_am if isinstance(premium_am, list) else []
        premium_steps_pm = premium_pm if isinstance(premium_pm, list) else []
        dupe_steps_am = dupe_am if isinstance(dupe_am, list) else []
        dupe_steps_pm = dupe_pm if isinstance(dupe_pm, list) else []

        if not (premium_steps_am or premium_steps_pm or dupe_steps_am or dupe_steps_pm):
            logger.warning("Aurora returned empty routine; falling back to product search. brief_id=%s", brief_id)
        else:
            categories_am = [c for c in (_step_category(s) for s in premium_steps_am) if c] or categories_am
            categories_pm = [c for c in (_step_category(s) for s in premium_steps_pm) if c] or categories_pm
            all_categories = sorted(set(categories_am + categories_pm))

            def _find_step(steps: list[dict[str, Any]], category: str) -> Optional[dict[str, Any]]:
                for step in steps:
                    if _step_category(step) == category:
                        return step
                return None

            def _build_offer(product_id: str, *, price: float, currency: str, is_dupe: bool, q: str) -> dict[str, Any]:
                url = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
                return {
                    "offer_id": _mk_offer_id("aurora_offer", product_id),
                    "seller": "Aurora",
                    "price": round(price, 2),
                    "currency": currency,
                    "original_price": None,
                    "shipping_days": 5,
                    "returns_policy": "See retailer policy",
                    "reliability_score": 75,
                    "badges": ["best_price"] if is_dupe else ["high_reliability"],
                    "in_stock": True,
                    "purchase_route": "affiliate_outbound",
                    "affiliate_url": url,
                }

            def _normalize_score01(value: Any) -> Optional[float]:
                try:
                    num = float(value)
                except Exception:
                    return None
                if not (num == num):  # NaN guard
                    return None
                if num > 1:
                    num = num / 100.0
                if num < 0:
                    num = 0.0
                if num > 1:
                    num = 1.0
                return num

            def _compute_mechanism_similarity(premium: dict[str, Any], dupe: dict[str, Any]) -> Optional[int]:
                a = premium.get("mechanism")
                b = dupe.get("mechanism")
                if not isinstance(a, dict) or not isinstance(b, dict):
                    return None
                keys = [k for k in a.keys() if k in b]
                if not keys:
                    return None
                diffs: list[float] = []
                for k in keys:
                    av = _normalize_score01(a.get(k))
                    bv = _normalize_score01(b.get(k))
                    if av is None or bv is None:
                        continue
                    diffs.append(abs(av - bv))
                if not diffs:
                    return None
                avg = sum(diffs) / len(diffs)
                return max(0, min(100, int(round((1 - avg) * 100))))

            def _tradeoff_note(premium: dict[str, Any], dupe: dict[str, Any]) -> Optional[str]:
                notes: list[str] = []
                exp_d = dupe.get("experience") if isinstance(dupe.get("experience"), dict) else {}
                texture_d = str(exp_d.get("texture") or "").lower()
                if texture_d in {"sticky", "thick"}:
                    notes.append("Dupe texture may feel heavier/stickier.")
                pilling = _normalize_score01(exp_d.get("pilling_risk"))
                if pilling is not None and pilling > 0.6:
                    notes.append("Higher pilling risk under layering.")

                ss_p = premium.get("social_stats") if isinstance(premium.get("social_stats"), dict) else {}
                ss_d = dupe.get("social_stats") if isinstance(dupe.get("social_stats"), dict) else {}
                burn_p = _normalize_score01(ss_p.get("burn_rate"))
                burn_d = _normalize_score01(ss_d.get("burn_rate"))
                if burn_p is not None and burn_d is not None and burn_d - burn_p >= 0.05:
                    notes.append("Slightly higher irritation mentions on social.")

                return notes[0] if notes else None

            def _extract_kb_actives(step: Optional[dict[str, Any]]) -> list[str]:
                if not isinstance(step, dict):
                    return []
                pack = step.get("evidence_pack")
                if isinstance(pack, dict):
                    actives = pack.get("keyActives") or pack.get("key_actives") or pack.get("key_actives_summary")
                    if isinstance(actives, list):
                        return [str(a) for a in actives if a]
                sku = step.get("sku") if isinstance(step.get("sku"), dict) else {}
                actives = sku.get("actives")
                if isinstance(actives, list):
                    return [str(a) for a in actives if a]
                return []

            def _aurora_product(
                category: str,
                step: Optional[dict[str, Any]],
                *,
                fallback_price: float,
                variant: Literal["premium", "dupe"],
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                sku = step.get("sku") if isinstance(step, dict) and isinstance(step.get("sku"), dict) else {}
                source_sku_id = str(sku.get("sku_id") or sku.get("id") or "")
                base_id = source_sku_id or f"aurora_{uuid.uuid4().hex}"
                # Ensure premium/dupe have distinct IDs even if Aurora returns the same SKU for both.
                product_id = f"{base_id}_{variant}"
                name = str(sku.get("name") or f"{category.title()} Pick")
                brand = str(sku.get("brand") or "Aurora")
                currency = str(sku.get("currency") or "USD")
                try:
                    price_f = float(sku.get("price") or 0)
                except Exception:
                    price_f = 0.0
                if price_f <= 0:
                    price_f = fallback_price

                notes = step.get("notes") if isinstance(step, dict) else None
                note_text = " ".join(str(n) for n in notes) if isinstance(notes, list) else ""

                product: dict[str, Any] = {
                    "sku_id": product_id,
                    "source_sku_id": source_sku_id or None,
                    "name": name,
                    "brand": brand,
                    "category": category,
                    "description": note_text[:2000],
                    "image_url": sku.get("image_url")
                    if isinstance(sku.get("image_url"), str) and sku.get("image_url")
                    else "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
                    "size": str(sku.get("size") or "1 unit"),
                }

                # Pass-through scientific + social signals when available.
                if isinstance(sku.get("mechanism"), dict):
                    product["mechanism"] = sku.get("mechanism")
                if isinstance(sku.get("experience"), dict):
                    product["experience"] = sku.get("experience")
                if isinstance(sku.get("social_stats"), dict):
                    product["social_stats"] = sku.get("social_stats")
                if isinstance(sku.get("risk_flags"), list):
                    product["risk_flags"] = [str(r) for r in sku.get("risk_flags") if r]

                if isinstance(step, dict) and isinstance(step.get("evidence_pack"), dict):
                    product["evidence_pack"] = step.get("evidence_pack")
                if isinstance(step, dict) and isinstance(step.get("ingredients"), dict):
                    product["ingredients"] = step.get("ingredients")

                actives = _extract_kb_actives(step)
                if actives:
                    product["key_actives"] = actives

                offer = _build_offer(
                    product_id,
                    price=price_f,
                    currency=currency,
                    is_dupe=(variant == "dupe"),
                    q=f"{brand} {name}",
                )
                return product, offer

            pairs_by_cat: dict[str, dict[str, Any]] = {}
            for cat in all_categories:
                premium_step = _find_step(premium_steps_am + premium_steps_pm, cat) if premium_steps_am or premium_steps_pm else None
                dupe_step = _find_step(dupe_steps_am + dupe_steps_pm, cat) if dupe_steps_am or dupe_steps_pm else None

                premium_product, premium_offer = _aurora_product(cat, premium_step, fallback_price=55, variant="premium")
                dupe_product, dupe_offer = _aurora_product(cat, dupe_step, fallback_price=18, variant="dupe")

                similarity = _compute_mechanism_similarity(premium_product, dupe_product)
                tradeoff_note = _tradeoff_note(premium_product, dupe_product)

                # Mark dupe offer.
                dupe_offer = {**dupe_offer, "badges": ["best_price"], "reliability_score": 70}

                pair: dict[str, Any] = {
                    "category": cat,
                    "premium": {"product": premium_product, "offers": [premium_offer]},
                    "dupe": {"product": dupe_product, "offers": [dupe_offer]},
                }
                if similarity is not None:
                    pair["similarity"] = similarity
                if tradeoff_note:
                    pair["tradeoff_note"] = tradeoff_note

                pairs_by_cat[cat] = pair

            if pairs_by_cat:
                am_pairs = [pairs_by_cat[c] for c in categories_am if c in pairs_by_cat]
                pm_pairs = [pairs_by_cat[c] for c in categories_pm if c in pairs_by_cat]

                selected_offers: dict[str, str] = {}
                for p in pairs_by_cat.values():
                    selected_offers[p["premium"]["product"]["sku_id"]] = p["premium"]["offers"][0]["offer_id"]
                    selected_offers[p["dupe"]["product"]["sku_id"]] = p["dupe"]["offers"][0]["offer_id"]

                patch = {
                    "productPairs": {"am": am_pairs, "pm": pm_pairs},
                    "selected_offers": selected_offers,
                    "routine_conflicts": [],
                    "next_state": "S7_PRODUCT_RECO",
                }
                await SESSION_STORE.upsert(
                    brief_id,
                    {
                        "trace_id": x_trace_id,
                        "state": patch["next_state"],
                        "productPairs": patch["productPairs"],
                        "selected_offers": patch["selected_offers"],
                        "preference": preference,
                        "market": market,
                        "budget_tier": budget_tier,
                    },
                )
                return {"session": patch, "next_state": patch["next_state"], "productPairs": patch["productPairs"]}

    # Ask Aurora for a routine blueprint; use its products as seed queries.
    try:
        budget = _budget_tier_to_aurora_budget(budget_tier)
        routine = await _aurora_routine_for_budget(budget)

        routine_am = routine.get("am") if isinstance(routine, dict) else None
        routine_pm = routine.get("pm") if isinstance(routine, dict) else None
        steps_am = routine_am if isinstance(routine_am, list) else []
        steps_pm = routine_pm if isinstance(routine_pm, list) else []

        categories_am = [c for c in (_step_category(s) for s in steps_am) if c] or categories_am
        categories_pm = [c for c in (_step_category(s) for s in steps_pm) if c] or categories_pm

        for step in steps_am + steps_pm:
            cat = _step_category(step)
            q = _step_query(step)
            if cat and q:
                seed_queries[cat] = q
    except Exception as exc:
        logger.warning("Aurora routine planning failed; using fallback queries. err=%s", exc)

    async def get_pair(category: str) -> dict[str, Any]:
        seed = seed_queries.get(category) or category
        products = await _find_products(seed, limit=10, timeout_s=DEFAULT_TIMEOUT_S)

        premium_raw: Optional[dict[str, Any]] = None
        dupe_raw: Optional[dict[str, Any]] = None
        if products:
            ranked = _sort_by_price([p for p in products if isinstance(p, dict)])
            if ranked:
                dupe_raw = ranked[0]
                premium_raw = ranked[-1]
                if (
                    premium_raw.get("product_id") == dupe_raw.get("product_id")
                    and len(ranked) > 1
                ):
                    premium_raw = ranked[-2]

        if not premium_raw or not dupe_raw:
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


@router.post("/offers/resolve")
async def offers_resolve(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    """
    Best-effort resolve outbound purchase links (and price/image) for a list of
    affiliate items, using the Pivota Agent shopping gateway.

    This is intentionally called by the frontend only when the user taps
    "checkout/open links", to keep routine generation fast.
    """

    _ = _require_brief_id(x_brief_id)

    raw_items = body.get("items") or body.get("affiliateItems") or body.get("affiliate_items") or []
    if not isinstance(raw_items, list):
        raise HTTPException(status_code=400, detail="`items` must be a list")

    sem = asyncio.Semaphore(4)

    async def _resolve_one(item: Any) -> Optional[dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        product = item.get("product") if isinstance(item.get("product"), dict) else None
        offer = item.get("offer") if isinstance(item.get("offer"), dict) else None
        if not product or not offer:
            return None

        brand = str(product.get("brand") or "").strip()
        name = str(product.get("name") or "").strip()
        desired = f"{brand} {name}".strip() or name
        if not desired:
            return {"product": product, "offer": offer}

        category = str(product.get("category") or _guess_category_from_query(desired)).strip().lower() or "treatment"

        async with sem:
            candidates = await _find_products_with_category(
                desired,
                category=category,
                limit=24,
                timeout_s=OFFERS_RESOLVE_TIMEOUT_S,
            )

        best = _best_product_match(desired, category=category, candidates=candidates, brand_hint=brand)
        if not best:
            return {"product": product, "offer": offer}

        # Patch offer + image URL without changing the product identity (sku_id).
        patched_product = dict(product)
        patched_offer = dict(offer)

        if isinstance(best.get("image_url"), str) and best["image_url"].strip():
            patched_product["image_url"] = best["image_url"].strip()
        else:
            imgs = best.get("image_urls")
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                patched_product["image_url"] = imgs[0]

        affiliate_url = best.get("external_redirect_url") or best.get("external_url") or patched_offer.get("affiliate_url")
        if affiliate_url:
            patched_offer["purchase_route"] = "affiliate_outbound"
            patched_offer["affiliate_url"] = str(affiliate_url)

        # Best-effort numeric price/currency/seller.
        try:
            patched_offer["price"] = round(float(best.get("price") or patched_offer.get("price") or 0), 2)
        except Exception:
            pass

        currency = best.get("currency") or patched_offer.get("currency")
        if currency:
            patched_offer["currency"] = str(currency)

        seller = best.get("merchant_name") or best.get("merchant_id") or patched_offer.get("seller")
        if seller:
            patched_offer["seller"] = str(seller)

        return {"product": patched_product, "offer": patched_offer}

    resolved = await asyncio.gather(*(_resolve_one(i) for i in raw_items))
    items = [r for r in resolved if r is not None]

    # trace_id not used yet; reserved for future audit logs.
    _ = x_trace_id
    return {"items": items}


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


@router.post("/chat")
async def chat(
    body: dict[str, Any],
    x_brief_id: Optional[str] = Header(default=None, alias="X-Brief-ID"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-ID"),
):
    brief_id = _require_brief_id(x_brief_id)
    stored = await SESSION_STORE.get(brief_id)

    message = body.get("message") or body.get("query")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="Missing `message`")

    diagnosis_payload = stored.get("diagnosis") if isinstance(stored.get("diagnosis"), dict) else None
    market = stored.get("market") or body.get("market") or "US"
    budget_tier = stored.get("budget_tier") or body.get("budget_tier") or "$$"
    budget = _budget_tier_to_aurora_budget(budget_tier)
    language = body.get("language")
    anchor_product_id = body.get("anchor_product_id") or body.get("anchorProductId")
    anchor_product_url = body.get("anchor_product_url") or body.get("anchorProductUrl")
    lang_code: Literal["EN", "CN"] = "EN"
    reply_language = "English"
    if isinstance(language, str) and language.strip().upper() in {"CN", "ZH", "ZH-CN", "ZH_HANS"}:
        lang_code = "CN"
        reply_language = "Simplified Chinese"

    pending = stored.get("pending_clarification")
    pending_user_request = stored.get("pending_user_request") if isinstance(stored.get("pending_user_request"), str) else None
    pending_request_kind = stored.get("pending_request_kind") if isinstance(stored.get("pending_request_kind"), str) else None
    pending_profile_field = stored.get("pending_profile_field") if isinstance(stored.get("pending_profile_field"), str) else None
    current_products_text = stored.get("current_products_text") if isinstance(stored.get("current_products_text"), str) else None

    sys_prompt = f"{GLOW_SYSTEM_PROMPT}\n\n" if GLOW_SYSTEM_PROMPT else ""
    reply_instruction = (
        "IMPORTANT: Reply ONLY in English. Do not use Chinese."
        if lang_code == "EN"
        else "请只用简体中文回答，不要使用英文。"
    )

    # Multi-turn: when we ask the user to list their current products, the next
    # message is treated as the product list and we answer the *original* request.
    effective_message = message.strip()
    products_review_mode = False

    def _clarify_skin_type() -> dict[str, Any]:
        question = "Which skin type fits you best?" if lang_code == "EN" else "你的肤质更接近哪一种？"
        options = ["Oily", "Dry", "Combination", "Sensitive", "Not sure"] if lang_code == "EN" else ["油皮", "干皮", "混合皮", "敏感肌", "不确定"]
        answer = "One quick question so I can answer accurately:" if lang_code == "EN" else "为了给你更准确的建议，我需要先确认一项信息："
        return {
            "answer": answer,
            "intent": "clarify",
            "clarification": {"questions": [{"id": "skin_type", "question": question, "options": options}], "missing_fields": ["skinType"]},
            "context": None,
        }

    def _clarify_barrier() -> dict[str, Any]:
        question = (
            "Lately, do you get stinging/redness/flaking (possible barrier stress)?"
            if lang_code == "EN"
            else "最近你会刺痛/泛红/起皮吗？（可能是屏障压力）"
        )
        options = ["No", "Mild", "Yes (stinging/redness)", "Not sure"] if lang_code == "EN" else ["没有", "轻微", "有（刺痛/泛红）", "不确定"]
        answer = "One quick question so I can answer accurately:" if lang_code == "EN" else "为了给你更准确的建议，我需要再确认一项信息："
        return {
            "answer": answer,
            "intent": "clarify",
            "clarification": {"questions": [{"id": "barrier_status", "question": question, "options": options}], "missing_fields": ["barrierStatus"]},
            "context": None,
        }

    # If we captured a routine list first, we may need to collect minimal skin profile
    # before we can safely review/compare products.
    if pending == "skin_profile_for_products":
        field = pending_profile_field or "skinType"
        diag = diagnosis_payload if isinstance(diagnosis_payload, dict) else {}
        diagnosis_payload = {
            "skinType": diag.get("skinType"),
            "concerns": diag.get("concerns") if isinstance(diag.get("concerns"), list) else [],
            "currentRoutine": diag.get("currentRoutine") or "basic",
            "barrierStatus": diag.get("barrierStatus") or "unknown",
        }

        if field == "skinType":
            parsed = _normalize_skin_type_answer(effective_message, lang_code)
            if not parsed:
                return _clarify_skin_type()
            diagnosis_payload["skinType"] = parsed
            await SESSION_STORE.upsert(
                brief_id,
                {
                    "trace_id": x_trace_id,
                    "diagnosis": diagnosis_payload,
                    "pending_clarification": "skin_profile_for_products",
                    "pending_profile_field": "barrierStatus",
                },
            )
            return _clarify_barrier()

        parsed = _normalize_barrier_answer(effective_message, lang_code)
        if not parsed:
            return _clarify_barrier()

        diagnosis_payload["barrierStatus"] = parsed
        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "diagnosis": diagnosis_payload,
                "pending_clarification": None,
                "pending_profile_field": None,
                "pending_user_request": None,
                "pending_request_kind": None,
            },
        )
        products_review_mode = True
        effective_message = (
            "Please review my current skincare products and tell me what to keep/change before recommending anything."
            if lang_code == "EN"
            else "在推荐之前，请先评估我现在用的护肤品：哪些适合保留，哪些需要替换/注意。"
        )

    # Users often paste their routine list directly. Capture it and (if needed) ask for
    # the minimal skin profile to evaluate safely.
    if (
        not current_products_text
        and not pending
        and not anchor_product_id
        and not anchor_product_url
        and _looks_like_product_list_input(effective_message, lang_code)
    ):
        provided = "" if _is_no_products_reply(effective_message) else effective_message
        current_products_text = provided[:4000] if provided else None

        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "current_products_text": current_products_text,
            },
        )

        diag = diagnosis_payload if isinstance(diagnosis_payload, dict) else {}
        has_skin = isinstance(diag.get("skinType"), str) and str(diag.get("skinType") or "").strip()
        has_barrier = isinstance(diag.get("barrierStatus"), str) and str(diag.get("barrierStatus") or "").strip()

        if not has_skin or not has_barrier:
            await SESSION_STORE.upsert(
                brief_id,
                {
                    "trace_id": x_trace_id,
                    "pending_clarification": "skin_profile_for_products",
                    "pending_profile_field": "skinType",
                },
            )
            return _clarify_skin_type()

        products_review_mode = True
        effective_message = (
            "Please review my current skincare products and tell me what to keep/change before recommending anything."
            if lang_code == "EN"
            else "在推荐之前，请先评估我现在用的护肤品：哪些适合保留，哪些需要替换/注意。"
        )

    if pending == "current_products":
        provided = "" if _is_no_products_reply(effective_message) else effective_message
        current_products_text = provided[:4000] if provided else None

        kind = str(pending_request_kind or "").strip().lower()
        if kind == "routine":
            base = pending_user_request or ""
            sanitized = re.sub(r"\broutine\b", "regimen", base, flags=re.IGNORECASE).strip()
            effective_message = sanitized or (
                "Please propose a simple AM/PM skincare regimen based on my profile."
                if lang_code == "EN"
                else "请基于我的情况给一套简单的早晚护肤方案（尽量温和）。"
            )
        else:
            effective_message = (
                pending_user_request
                or (
                    "Please review my current skincare products and tell me what to keep/change before recommending anything."
                    if lang_code == "EN"
                    else "在推荐之前，请先评估我现在用的护肤品：哪些适合保留，哪些需要替换/注意。"
                )
            )
        products_review_mode = True

        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "pending_clarification": None,
                "pending_user_request": None,
                "pending_request_kind": None,
                "current_products_text": current_products_text,
            },
        )

    elif (
        (_looks_like_current_products_review_request(effective_message) or _looks_like_routine_request(effective_message))
        and not current_products_text
        and not anchor_product_id
        and not anchor_product_url
    ):
        # Ask for product history first so we can evaluate what the user already uses
        # instead of jumping straight into blind recommendations.
        kind = "review" if _looks_like_current_products_review_request(effective_message) else "routine"
        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "pending_clarification": "current_products",
                "pending_user_request": effective_message[:2000],
                "pending_request_kind": kind,
            },
        )

        if lang_code == "CN":
            question = (
                "在推荐之前，我想先评估你现在用的护肤品是否适合。\n"
                "请把你正在用的产品按步骤列出来（洁面/精华/保湿/防晒/处方药等），或者回复“无/从零开始”。"
            )
            options = ["无 / 从零开始"]
            answer = "为了给你更准确的建议，我需要先确认一件事："
        else:
            question = (
                "Before I recommend anything, I want to evaluate what you’re already using.\n"
                "Please list your current products by step (cleanser/actives/moisturizer/SPF/any prescriptions), or reply “none / start fresh”."
            )
            options = ["None / start fresh"]
            answer = "One quick question so I can answer accurately:"

        return {
            "answer": answer,
            "intent": "clarify",
            "clarification": {"questions": [{"id": "current_products", "question": question, "options": options}], "missing_fields": ["current_products"]},
            "context": None,
        }

    if isinstance(anchor_product_id, str) and anchor_product_id.strip():
        profile = _aurora_profile_sentence(diagnosis=diagnosis_payload, market=market, budget=budget)
        query = (
            f"{sys_prompt}{profile}\n"
            f"User question: {effective_message}\n"
            "Please give a detailed product-fit assessment (suitability, risks/cautions, how to use, and 1–2 alternatives).\n"
            f"{reply_instruction}"
        )
    else:
        if _looks_like_reco_rationale_request(effective_message) and isinstance(stored.get("aurora_context"), dict):
            explained = _explain_routine_from_aurora_context(stored["aurora_context"], language=lang_code)
            if explained:
                return {
                    "answer": explained,
                    "intent": "explain",
                    "clarification": None,
                    "context": stored.get("aurora_context"),
                }

        if current_products_text and (_looks_like_current_products_review_request(effective_message) or _looks_like_routine_request(effective_message)):
            products_review_mode = True

        if products_review_mode and current_products_text:
            profile = _aurora_profile_sentence(diagnosis=diagnosis_payload, market=market, budget=budget)
            query = (
                f"{sys_prompt}{profile}\n"
                f"current_products={current_products_text}\n"
                f"User request: {effective_message}\n"
                "Task:\n"
                "1) Evaluate each listed product: keep/adjust/replace + a short reason.\n"
                "2) Call out conflicts/irritation risks and how to use safely.\n"
                "3) Then propose a minimal AM/PM skincare regimen that reuses the kept products.\n"
                "4) Only suggest 1–2 new additions if truly necessary; keep it budget-aware.\n"
                "5) Use a natural chat style. Do NOT use a rigid template like “Part 1/Part 2/Part 3/Part 4”.\n"
                f"{reply_instruction}"
            )
        else:
            profile = _aurora_profile_line(diagnosis=diagnosis_payload, market=market, budget=budget)
            products_block = f"\ncurrent_products={current_products_text}\n" if current_products_text else ""
            query = f"{sys_prompt}{profile}{products_block}\nUser message: {effective_message}\n{reply_instruction}"

    try:
        payload = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=query,
            timeout_s=DEFAULT_TIMEOUT_S,
            llm_provider=body.get("llm_provider") if isinstance(body.get("llm_provider"), str) else None,
            llm_model=body.get("llm_model") if isinstance(body.get("llm_model"), str) else None,
            anchor_product_id=str(anchor_product_id).strip() if isinstance(anchor_product_id, str) and anchor_product_id.strip() else None,
            anchor_product_url=str(anchor_product_url).strip() if isinstance(anchor_product_url, str) and anchor_product_url.strip() else None,
        )
    except Exception as exc:
        logger.error("Aurora chat failed. err=%s", exc)
        raise HTTPException(status_code=502, detail={"upstream": "aurora", "error": str(exc)}) from exc

    answer = payload.get("answer") if isinstance(payload, dict) else None
    intent = payload.get("intent") if isinstance(payload, dict) else None
    llm_error = payload.get("llm_error") if isinstance(payload, dict) else None
    clarification = payload.get("clarification") if isinstance(payload, dict) else None
    context = payload.get("context") if isinstance(payload, dict) else None

    # Aurora sometimes replies in Chinese even for EN users (including fallback answers).
    # For EN users, translate into English so the experience remains bilingual.
    if (
        lang_code == "EN"
        and intent != "clarify"
        and isinstance(answer, str)
        and any("\u4e00" <= ch <= "\u9fff" for ch in answer)
    ):
        try:
            translation = await aurora_chat(
                base_url=AURORA_DECISION_BASE_URL,
                query=(
                    "Translate the following text into English. Keep the bullet formatting.\n"
                    "IMPORTANT: Reply ONLY in English. Do not use Chinese.\n\n"
                    f"TEXT:\n{answer}"
                ),
                timeout_s=DEFAULT_TIMEOUT_S,
                llm_provider=body.get("llm_provider") if isinstance(body.get("llm_provider"), str) else None,
                llm_model=body.get("llm_model") if isinstance(body.get("llm_model"), str) else None,
            )
            translated_answer = translation.get("answer") if isinstance(translation, dict) else None
            if isinstance(translated_answer, str) and translated_answer.strip():
                answer = translated_answer
        except Exception as exc:
            logger.warning("Fallback translation failed. err=%s", exc)

    if intent == "clarify":
        clarification = _normalize_clarification(clarification, language=lang_code)
        if lang_code == "CN":
            answer = "为了给你更准确的建议，我需要先确认一个信息："
        else:
            answer = "One quick question so I can answer accurately:"

    await SESSION_STORE.upsert(
        brief_id,
        {
            "trace_id": x_trace_id,
            "last_user_message": effective_message[:2000],
            "last_aurora_intent": intent,
            "last_aurora_answer": answer,
            "aurora_context": context if isinstance(context, dict) else None,
        },
    )

    return {
        "answer": answer,
        "intent": intent,
        "clarification": clarification,
        "context": context,
    }
