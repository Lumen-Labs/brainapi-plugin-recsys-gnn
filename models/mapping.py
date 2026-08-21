"""Interaction → event-hub triple mapping for structured deterministic ingest."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


BEHAVIOR_MAP: dict[str, tuple[str, str]] = {
    "purchase": ("Purchase", "TARGETED"),
    "purchased": ("Purchase", "TARGETED"),
    "buy": ("Purchase", "TARGETED"),
    "bought": ("Purchase", "TARGETED"),
    "view": ("View", "TARGETED"),
    "viewed": ("View", "TARGETED"),
    "click": ("View", "TARGETED"),
    "clicked": ("View", "TARGETED"),
    "cart": ("AddToCart", "TARGETED"),
    "add_to_cart": ("AddToCart", "TARGETED"),
    "added_to_cart": ("AddToCart", "TARGETED"),
    "addtocart": ("AddToCart", "TARGETED"),
}


def user_uuid(user_id: str) -> str:
    return f"user:{user_id}"


def item_uuid(item_id: str) -> str:
    return f"item:{item_id}"


def _normalize_behavior(behavior: str) -> tuple[str, str]:
    key = (behavior or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in BEHAVIOR_MAP:
        return BEHAVIOR_MAP[key]
    label = (behavior or "Interaction").strip() or "Interaction"
    return (label[:1].upper() + label[1:], "TARGETED")


def _format_happened_at(timestamp: Optional[str]) -> Optional[str]:
    if not timestamp:
        return None
    raw = timestamp.strip()
    if not raw:
        return None
    if len(raw) == 10 and raw[2] == "/" and raw[5] == "/":
        return raw
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ):
        try:
            candidate = (
                raw.replace("Z", "+0000")
                if fmt.endswith("%z") and raw.endswith("Z")
                else raw
            )
            parsed = datetime.strptime(candidate, fmt)
            return parsed.strftime("%m/%d/%Y")
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%m/%d/%Y")
    except ValueError:
        return raw


def interaction_to_triple(
    *,
    user_id: str,
    item_id: str,
    behavior: str,
    timestamp: Optional[str] = None,
    seq: int = 1,
) -> dict[str, Any]:
    event_name, edge_name = _normalize_behavior(behavior)
    event_node: dict[str, Any] = {
        "name": event_name,
        "type": "EVENT",
        "uuid": f"evt:{user_id}:{item_id}:{seq}",
    }
    happened_at = _format_happened_at(timestamp)
    if happened_at:
        event_node["happened_at"] = happened_at
    return {
        "subject": {
            "name": user_id,
            "type": "USER",
            "uuid": user_uuid(user_id),
        },
        "subj_event": {
            "name": "MADE",
            "uuid": f"rel:{user_id}:{item_id}:made:{seq}",
        },
        "event": event_node,
        "event_obj": {
            "name": edge_name,
            "uuid": f"rel:{user_id}:{item_id}:tgt:{seq}",
        },
        "object": {
            "name": item_id,
            "type": "PRODUCT",
            "uuid": item_uuid(item_id),
        },
    }


def structured_ingest_payload(
    *,
    user_id: str,
    item_id: str,
    behavior: str,
    timestamp: Optional[str] = None,
    brain_id: str = "demorecsys",
    seq: int = 1,
) -> dict[str, Any]:
    return {
        "mode": "deterministic",
        "brain_id": brain_id,
        "data": [
            interaction_to_triple(
                user_id=user_id,
                item_id=item_id,
                behavior=behavior,
                timestamp=timestamp,
                seq=seq,
            )
        ],
    }
