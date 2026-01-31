from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import time
import uuid
import urllib.parse
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.services.aurora import aurora_chat
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

    if isinstance(diagnosis, dict):
        skin_type = diagnosis.get("skinType") or diagnosis.get("skin_type")
        concerns_raw = diagnosis.get("concerns")
        if isinstance(concerns_raw, list):
            concerns = [str(c) for c in concerns_raw if c]
        current_routine = diagnosis.get("currentRoutine") or diagnosis.get("current_routine")

    concerns_str = ", ".join(concerns) if concerns else "none"
    skin_str = str(skin_type or "unknown")
    routine_str = str(current_routine or "basic")

    return (
        # NOTE: Avoid the substring "routine" in this line because Aurora's /api/chat
        # intent heuristics treat any mention of "routine" as a routine request.
        f"skin_type={skin_str}; concerns={concerns_str}; region={market}; budget={budget}; current_regimen={routine_str}."
    )

def _aurora_profile_sentence(
    *,
    diagnosis: Optional[dict[str, Any]],
    market: str,
    budget: str,
) -> str:
    skin_type = None
    concerns: list[str] = []

    if isinstance(diagnosis, dict):
        skin_type = diagnosis.get("skinType") or diagnosis.get("skin_type")
        concerns_raw = diagnosis.get("concerns")
        if isinstance(concerns_raw, list):
            concerns = [str(c) for c in concerns_raw if c]

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
    return f"User profile: skin type {skin_str}; concerns: {concerns_str}; region: {market}; budget: {budget}."

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
        return base

    features: list[dict[str, Any]] = []
    if detected.get("oily_acne") is True:
        features.extend(
            [
                {"observation": "Higher oil/shine patterns (acne-prone tendency)", "confidence": "pretty_sure"},
                {"observation": "Pores may clog more easily without gentle balancing", "confidence": "somewhat_sure"},
            ]
        )
    if detected.get("sensitive_skin") is True:
        features.extend(
            [
                {"observation": "Skin looks reactive/sensitive (irritation risk)", "confidence": "somewhat_sure"},
            ]
        )
    if detected.get("barrier_impaired") is True:
        features.extend(
            [
                {"observation": "Barrier may be stressed (prioritize repair + low irritation)", "confidence": "pretty_sure"},
            ]
        )

    merged_features = features + base.get("features", [])
    if merged_features:
        base["features"] = merged_features[:6]

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


def _is_no_products_reply(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False

    if any(k in t for k in ["none", "no products", "no routine", "start fresh", "from scratch"]):
        return True
    return any(k in text for k in ["没有", "无", "从零开始", "不用", "没用"])


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

    market = stored.get("market") or body.get("market") or "US"
    budget_tier = stored.get("budget_tier") or body.get("budget_tier") or "$$"
    budget = _budget_tier_to_aurora_budget(budget_tier)

    aurora_context: Optional[dict[str, Any]] = None
    try:
        aurora_payload = await aurora_chat(
            base_url=AURORA_DECISION_BASE_URL,
            query=(
                f"{_aurora_profile_line(diagnosis=diagnosis_payload, market=market, budget=budget)}\n"
                "Give a brief diagnosis and a safe, minimal routine strategy. Reply in English."
            ),
            timeout_s=DEFAULT_TIMEOUT_S,
        )
        aurora_ctx_raw = aurora_payload.get("context") if isinstance(aurora_payload, dict) else None
        aurora_context = aurora_ctx_raw if isinstance(aurora_ctx_raw, dict) else None
    except Exception as exc:
        logger.warning("Aurora analysis call failed; falling back. err=%s", exc)

    analysis_result = _analysis_from_aurora_context(diagnosis_payload, aurora_context)

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
        premium_routine, dupe_routine = await asyncio.gather(
            _aurora_routine_for_budget("¥1000+"),
            _aurora_routine_for_budget("¥200"),
        )

        premium_am = premium_routine.get("am") if isinstance(premium_routine, dict) else None
        premium_pm = premium_routine.get("pm") if isinstance(premium_routine, dict) else None
        dupe_am = dupe_routine.get("am") if isinstance(dupe_routine, dict) else None
        dupe_pm = dupe_routine.get("pm") if isinstance(dupe_routine, dict) else None

        premium_steps_am = premium_am if isinstance(premium_am, list) else []
        premium_steps_pm = premium_pm if isinstance(premium_pm, list) else []
        dupe_steps_am = dupe_am if isinstance(dupe_am, list) else []
        dupe_steps_pm = dupe_pm if isinstance(dupe_pm, list) else []

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

        def _aurora_product(
            category: str,
            step: Optional[dict[str, Any]],
            *,
            fallback_price: float,
            variant: Literal["premium", "dupe"],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            sku = step.get("sku") if isinstance(step, dict) and isinstance(step.get("sku"), dict) else {}
            base_id = str(sku.get("sku_id") or sku.get("id") or f"aurora_{uuid.uuid4().hex}")
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

            product = {
                "sku_id": product_id,
                "name": name,
                "brand": brand,
                "category": category,
                "description": note_text[:2000],
                "image_url": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
                "size": "1 unit",
            }
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

            # Mark dupe offer.
            dupe_offer = {**dupe_offer, "badges": ["best_price"], "reliability_score": 70}

            pairs_by_cat[cat] = {
                "category": cat,
                "premium": {"product": premium_product, "offers": [premium_offer]},
                "dupe": {"product": dupe_product, "offers": [dupe_offer]},
            }

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
    if pending == "current_products":
        provided = "" if _is_no_products_reply(effective_message) else effective_message
        current_products_text = provided[:4000] if provided else None

        effective_message = (
            pending_user_request
            or (
                "Please review my current skincare products and tell me what to keep/change before recommending anything."
                if lang_code == "EN"
                else "在推荐之前，请先评估我现在用的护肤品：哪些适合保留，哪些需要替换/注意。"
            )
        )

        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "pending_clarification": None,
                "pending_user_request": None,
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
        await SESSION_STORE.upsert(
            brief_id,
            {
                "trace_id": x_trace_id,
                "pending_clarification": "current_products",
                "pending_user_request": effective_message[:2000],
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
