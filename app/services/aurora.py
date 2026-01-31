from __future__ import annotations

import json
from typing import Any, Optional

import httpx


async def aurora_chat(
    *,
    base_url: str,
    query: str,
    timeout_s: float,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    anchor_product_id: Optional[str] = None,
    anchor_product_url: Optional[str] = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {"query": query}
    if llm_provider:
        payload["llm_provider"] = llm_provider
    if llm_model:
        payload["llm_model"] = llm_model
    if anchor_product_id:
        payload["anchor_product_id"] = anchor_product_id
    if anchor_product_url:
        payload["anchor_product_url"] = anchor_product_url

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        res = await client.post(url, json=payload)

    try:
        data = res.json()
    except Exception:
        data = {"raw": res.text}

    if res.status_code >= 400:
        raise httpx.HTTPStatusError("Aurora returned error", request=res.request, response=res)

    return data if isinstance(data, dict) else {"data": data}


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    for start in (i for i, ch in enumerate(text) if ch == "{"):
        candidate = _extract_braced(text, start)
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    return None


def _extract_braced(text: str, start: int) -> Optional[str]:
    depth = 0
    in_str = False
    escape = False
    end: Optional[int] = None

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None or depth != 0:
        return None
    return text[start : end + 1]
