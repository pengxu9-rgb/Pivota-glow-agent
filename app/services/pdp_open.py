from __future__ import annotations

import re
import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote_plus


ResolveOnceFn = Callable[..., Awaitable[dict[str, Any]]]


_UUID_LIKE_RE = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.IGNORECASE)
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        text = _as_str(value)
        if text:
            return text
    return ""


def _get_case_insensitive(d: dict[str, Any], *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        value = lower_map.get(str(key).lower())
        if value is not None:
            return value
    return None


def is_opaque_product_id(value: Any) -> bool:
    text = _as_str(value)
    if not text:
        return False
    if text.lower().startswith("kb:"):
        return True
    compact = re.sub(r"[\s_-]+", "", text)
    if _UUID_LIKE_RE.match(text):
        return True
    if _HEX32_RE.match(compact):
        return True
    return False


def normalize_reason_code(raw: Any) -> str:
    text = _as_str(raw).lower()
    if not text:
        return "fallback_external"
    normalized = text.replace("-", "_").replace(" ", "_")
    mapping = {
        "subject_direct": "subject_direct",
        "canonical_ref_direct": "canonical_ref_direct",
        "mapped_hit": "mapped_hit",
        "resolve_once": "mapped_hit",
        "no_candidates": "no_candidates",
        "db_timeout": "db_timeout",
        "db_error": "db_timeout",
        "upstream_timeout": "upstream_timeout",
        "invalid_id": "invalid_id",
        "fallback_external": "fallback_external",
        "no_match": "no_candidates",
        "not_found": "no_candidates",
        "timeout": "upstream_timeout",
        "resolve_timeout": "upstream_timeout",
    }
    return mapping.get(normalized, "fallback_external")


def _extract_display_title(payload: dict[str, Any]) -> str:
    display = _as_obj(payload.get("display"))
    title = _first_non_empty_str(
        display.get("title"),
        payload.get("title"),
        payload.get("name"),
        payload.get("display_name"),
    )
    if title:
        return title
    brand = _first_non_empty_str(payload.get("brand"))
    name = _first_non_empty_str(payload.get("name"), payload.get("display_name"))
    combined = " ".join([part for part in [brand, name] if part]).strip()
    return combined


def _normalize_product_ref(raw: Any) -> tuple[Optional[dict[str, str]], Optional[str]]:
    ref = _as_obj(raw)
    if not ref:
        return None, None
    product_id = _first_non_empty_str(
        _get_case_insensitive(ref, "product_id", "productId"),
    )
    merchant_id = _first_non_empty_str(
        _get_case_insensitive(ref, "merchant_id", "merchantId"),
    )
    if not product_id:
        return None, "invalid_id"
    if is_opaque_product_id(product_id):
        return None, "invalid_id"
    return {
        "product_id": product_id,
        "merchant_id": merchant_id,
    }, None


def _subject_from_product_group_id(product_group_id: Any) -> Optional[dict[str, Any]]:
    pgid = _as_str(product_group_id)
    if not pgid:
        return None
    return {
        "kind": "product_group",
        "product_group_id": pgid,
    }


def _subject_from_canonical_ref(raw_ref: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    ref, err = _normalize_product_ref(raw_ref)
    if not ref:
        return None, err
    return {
        "kind": "canonical_product",
        "product_ref": {
            "merchant_id": ref.get("merchant_id") or "",
            "product_id": ref.get("product_id") or "",
        },
    }, None


def _extract_subject_direct(payload: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    subject = _as_obj(payload.get("subject"))
    kind = _first_non_empty_str(subject.get("kind")).lower()

    if kind == "product_group":
        direct = _subject_from_product_group_id(
            _first_non_empty_str(
                _get_case_insensitive(subject, "product_group_id", "productGroupId"),
                _get_case_insensitive(payload, "product_group_id", "productGroupId"),
            )
        )
        if direct:
            return direct, None
        return None, "invalid_id"

    if kind == "canonical_product":
        canonical, err = _subject_from_canonical_ref(
            _get_case_insensitive(subject, "product_ref", "productRef")
        )
        if canonical:
            return canonical, None
        return None, err or "invalid_id"

    direct = _subject_from_product_group_id(
        _first_non_empty_str(
            _get_case_insensitive(subject, "product_group_id", "productGroupId"),
            _get_case_insensitive(payload, "product_group_id", "productGroupId"),
        )
    )
    if direct:
        return direct, None
    return None, None


def _extract_canonical_direct(payload: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    candidates = [
        _get_case_insensitive(payload, "canonical_product_ref", "canonicalProductRef"),
        _get_case_insensitive(payload, "product_ref", "productRef"),
    ]
    for raw_ref in candidates:
        canonical, err = _subject_from_canonical_ref(raw_ref)
        if canonical:
            return canonical, None
        if err:
            return None, err
    return None, None


def _extract_resolved_subject(response: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    subject_direct, subject_err = _extract_subject_direct(response)
    if subject_direct:
        return subject_direct, None
    if subject_err:
        return None, subject_err

    canonical_direct, canonical_err = _extract_canonical_direct(response)
    if canonical_direct:
        return canonical_direct, None
    if canonical_err:
        return None, canonical_err
    return None, None


def _extract_reason_code_from_response(response: dict[str, Any]) -> str:
    root = _as_obj(response)
    reason = _first_non_empty_str(
        _get_case_insensitive(root, "reason_code", "reasonCode"),
        _get_case_insensitive(_as_obj(root.get("error")), "reason_code", "reasonCode"),
        _get_case_insensitive(_as_obj(root.get("trace")), "reason_code", "reasonCode"),
    )
    return normalize_reason_code(reason)


def _build_google_fallback_url(query: str, language: str) -> str:
    q = _as_str(query)
    if not q:
        return ""
    lang = "zh-CN" if _as_str(language).upper() == "CN" else "en"
    return f"https://www.google.com/search?q={quote_plus(q)}&hl={lang}"


def _collect_legacy_refs(payload: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for key in ["sku_id", "skuId", "product_id", "productId", "merchant_id", "merchantId", "product_group_id", "productGroupId"]:
        value = _get_case_insensitive(payload, key)
        if value is None:
            continue
        refs[key] = value
    subject = _as_obj(payload.get("subject"))
    if subject:
        refs["subject"] = subject
    canonical = _get_case_insensitive(payload, "canonical_product_ref", "canonicalProductRef")
    if isinstance(canonical, dict):
        refs["canonical_product_ref"] = canonical
    product_ref = _get_case_insensitive(payload, "product_ref", "productRef")
    if isinstance(product_ref, dict):
        refs["product_ref"] = product_ref
    refs["authoritative"] = False
    return refs


def _build_resolve_hints(payload: dict[str, Any], *, canonical_subject: Optional[dict[str, Any]]) -> dict[str, Any]:
    hints: dict[str, Any] = {}

    if canonical_subject and canonical_subject.get("kind") == "canonical_product":
        product_ref = canonical_subject.get("product_ref")
        if isinstance(product_ref, dict):
            pid = _as_str(product_ref.get("product_id"))
            mid = _as_str(product_ref.get("merchant_id"))
            if pid and not is_opaque_product_id(pid):
                hints["product_ref"] = {"product_id": pid, "merchant_id": mid}

    sku_id = _first_non_empty_str(_get_case_insensitive(payload, "sku_id", "skuId"))
    if sku_id and not is_opaque_product_id(sku_id):
        hints["sku_id"] = sku_id

    product_id = _first_non_empty_str(_get_case_insensitive(payload, "product_id", "productId"))
    if product_id and not is_opaque_product_id(product_id):
        hints["product_id"] = product_id

    brand = _first_non_empty_str(payload.get("brand"))
    title = _first_non_empty_str(payload.get("name"), payload.get("display_name"), payload.get("title"))
    if brand:
        hints["brand"] = brand
    if title:
        hints["title"] = title
        hints["aliases"] = [title]
    return hints


def _build_pdp_target(
    *,
    subject: dict[str, Any],
    display_title: str,
    resolution_path: str,
    reason_code: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "schema_version": "pdp_target.v1",
        "subject": subject,
        "display": {"title": display_title},
        "trace": {
            "resolution_path": resolution_path,
            "reason_code": reason_code,
            "confidence": confidence,
        },
    }


def build_direct_pdp_target(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    direct_subject, _ = _extract_subject_direct(payload)
    if direct_subject:
        return _build_pdp_target(
            subject=direct_subject,
            display_title=_extract_display_title(payload),
            resolution_path="subject_direct",
            reason_code="subject_direct",
            confidence=1.0,
        )
    canonical_subject, _ = _extract_canonical_direct(payload)
    if canonical_subject:
        return _build_pdp_target(
            subject=canonical_subject,
            display_title=_extract_display_title(payload),
            resolution_path="canonical_ref_direct",
            reason_code="canonical_ref_direct",
            confidence=1.0,
        )
    return None


async def resolve_pdp_open_contract(
    *,
    payload: dict[str, Any],
    language: str,
    resolve_once: ResolveOnceFn,
    request_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    request_meta = request_meta or {}
    display_title = _extract_display_title(payload)
    legacy_refs = _collect_legacy_refs(payload)

    subject_direct, subject_err = _extract_subject_direct(payload)
    if subject_direct:
        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
        return {
            "request_meta": request_meta,
            "pdp_target": _build_pdp_target(
                subject=subject_direct,
                display_title=display_title,
                resolution_path="subject_direct",
                reason_code="subject_direct",
                confidence=1.0,
            ),
            "pdp_open_path": "subject_direct",
            "reason_code": "subject_direct",
            "fallback_reason_code": None,
            "resolve_attempt_count": 0,
            "resolve_hints": {},
            "legacy_refs": legacy_refs,
            "internal_candidate_found": True,
            "time_to_pdp_ms": elapsed_ms,
            "external_url": None,
            "open_mode": "internal",
        }

    canonical_direct, canonical_err = _extract_canonical_direct(payload)
    if canonical_direct:
        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
        return {
            "request_meta": request_meta,
            "pdp_target": _build_pdp_target(
                subject=canonical_direct,
                display_title=display_title,
                resolution_path="canonical_ref_direct",
                reason_code="canonical_ref_direct",
                confidence=1.0,
            ),
            "pdp_open_path": "canonical_ref_direct",
            "reason_code": "canonical_ref_direct",
            "fallback_reason_code": None,
            "resolve_attempt_count": 0,
            "resolve_hints": {},
            "legacy_refs": legacy_refs,
            "internal_candidate_found": True,
            "time_to_pdp_ms": elapsed_ms,
            "external_url": None,
            "open_mode": "internal",
        }

    resolve_query = _first_non_empty_str(
        _get_case_insensitive(payload, "resolve_query", "resolveQuery"),
        display_title,
    )
    resolve_hints = _build_resolve_hints(payload, canonical_subject=canonical_direct)

    reason_code = normalize_reason_code(subject_err or canonical_err)
    resolved_subject: Optional[dict[str, Any]] = None
    resolve_response: dict[str, Any] = {}

    if resolve_query:
        resolve_response = await resolve_once(query=resolve_query, hints=resolve_hints)
        resolved_subject, resolve_subject_err = _extract_resolved_subject(resolve_response)
        if resolved_subject:
            elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
            return {
                "request_meta": request_meta,
                "pdp_target": _build_pdp_target(
                    subject=resolved_subject,
                    display_title=display_title or resolve_query,
                    resolution_path="resolve_once",
                    reason_code="mapped_hit",
                    confidence=0.9,
                ),
                "pdp_open_path": "resolve_once",
                "reason_code": "mapped_hit",
                "fallback_reason_code": None,
                "resolve_attempt_count": 1,
                "resolve_hints": resolve_hints,
                "legacy_refs": legacy_refs,
                "internal_candidate_found": True,
                "time_to_pdp_ms": elapsed_ms,
                "external_url": None,
                "open_mode": "internal",
            }

        upstream_reason = _extract_reason_code_from_response(resolve_response)
        reason_code = normalize_reason_code(
            resolve_subject_err or upstream_reason or reason_code or "fallback_external"
        )
    else:
        reason_code = normalize_reason_code(reason_code or "invalid_id")

    fallback_url = _build_google_fallback_url(resolve_query or display_title, language)
    elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
    return {
        "request_meta": request_meta,
        "pdp_target": None,
        "pdp_open_path": "fallback_external",
        "reason_code": reason_code or "fallback_external",
        "fallback_reason_code": reason_code or "fallback_external",
        "resolve_attempt_count": 1 if resolve_query else 0,
        "resolve_hints": resolve_hints,
        "legacy_refs": legacy_refs,
        "internal_candidate_found": False,
        "time_to_pdp_ms": elapsed_ms,
        "external_url": fallback_url,
        "open_mode": "external_new_tab",
    }
