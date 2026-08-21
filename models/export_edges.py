"""Export bipartite user–item edges from BrainAPI event hubs."""

from __future__ import annotations

from typing import Any, Protocol


FORBIDDEN_BRAIN_PREFIXES = ("beam1m", "locomoconv")
PRODUCT_LABELS = frozenset({"PRODUCT", "ITEM"})
USER_LABELS = frozenset({"USER"})
EVENT_WEIGHTS = {
    "purchase": 3.0,
    "purchased": 3.0,
    "buy": 3.0,
    "bought": 3.0,
    "addtocart": 2.0,
    "add_to_cart": 2.0,
    "added_to_cart": 2.0,
    "cart": 2.0,
    "view": 1.0,
    "viewed": 1.0,
    "click": 1.0,
    "clicked": 1.0,
}


class GraphLike(Protocol):
    def search_entities(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        node_labels: list[str] | None = None,
        query_text: str | None = None,
    ) -> Any: ...

    def get_event_centric_neighbors(
        self,
        nodes: list[Any],
        brain_id: str = "default",
    ) -> list[tuple[Any, Any, Any, Any, Any]]: ...


def assert_recsys_brain(brain_id: str) -> str:
    bid = (brain_id or "").strip()
    if not bid:
        raise ValueError("brain_id is required")
    if bid.startswith(FORBIDDEN_BRAIN_PREFIXES):
        raise ValueError(
            f"Refusing brain_id={bid!r}. Use demorecsys "
            "(or another dedicated *recsys* brain)."
        )
    return bid


def node_stable_id(node: Any) -> str | None:
    if node is None:
        return None
    uuid = getattr(node, "uuid", None)
    if uuid is None and isinstance(node, dict):
        uuid = node.get("uuid")
    if uuid is not None and str(uuid).strip():
        return str(uuid).strip()
    name = getattr(node, "name", None)
    if name is None and isinstance(node, dict):
        name = node.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()
    return None


def node_name(node: Any) -> str | None:
    if node is None:
        return None
    name = getattr(node, "name", None)
    if name is None and isinstance(node, dict):
        name = node.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()
    return node_stable_id(node)


def node_labels(node: Any) -> set[str]:
    raw = getattr(node, "labels", None)
    if raw is None and isinstance(node, dict):
        raw = node.get("labels") or []
    return {str(x).upper() for x in (raw or [])}


def event_weight(event_node: Any) -> float:
    name = (node_name(event_node) or "").strip().lower().replace("-", "_").replace(" ", "_")
    if name in EVENT_WEIGHTS:
        return EVENT_WEIGHTS[name]
    compact = name.replace("_", "")
    if compact in EVENT_WEIGHTS:
        return EVENT_WEIGHTS[compact]
    return 1.0


def _page_users(graph: GraphLike, brain_id: str, *, page_size: int = 200) -> list[Any]:
    users: list[Any] = []
    skip = 0
    total = None
    while True:
        page = graph.search_entities(
            brain_id=brain_id,
            limit=page_size,
            skip=skip,
            node_labels=["USER"],
        )
        results = getattr(page, "results", None)
        if results is None and isinstance(page, dict):
            results = page.get("results") or []
        results = list(results or [])
        if total is None:
            total = getattr(page, "total", None)
            if total is None and isinstance(page, dict):
                total = page.get("total")
            total = int(total or 0)
        users.extend(results)
        skip += len(results)
        if not results or skip >= total:
            break
    return users


def export_user_item_edges(
    graph: GraphLike,
    brain_id: str = "demorecsys",
    *,
    page_size: int = 200,
    neighbor_batch: int = 50,
) -> list[dict[str, Any]]:
    """
    Project USER —EVENT→ PRODUCT hubs into unique bipartite edges.

    Returns dicts: user_id, item_id, user_uuid, item_uuid, user_name, item_name, weight.
    """
    bid = assert_recsys_brain(brain_id)
    users = _page_users(graph, bid, page_size=page_size)
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for i in range(0, len(users), neighbor_batch):
        batch = users[i : i + neighbor_batch]
        hubs = graph.get_event_centric_neighbors(batch, brain_id=bid) or []
        for row in hubs:
            if not row or len(row) < 5:
                continue
            subject, _r1, event, _r2, tip = row[0], row[1], row[2], row[3], row[4]
            subj_labels = node_labels(subject)
            tip_labels = node_labels(tip)
            if not (subj_labels & USER_LABELS):
                continue
            if not (tip_labels & PRODUCT_LABELS):
                continue
            user_uuid = node_stable_id(subject)
            item_uuid = node_stable_id(tip)
            user_name = node_name(subject)
            item_name = node_name(tip)
            if not user_uuid or not item_uuid:
                continue
            weight = event_weight(event)
            key = (user_uuid, item_uuid)
            prev = best.get(key)
            if prev is None or weight > float(prev["weight"]):
                best[key] = {
                    "user_id": user_name or user_uuid,
                    "item_id": item_name or item_uuid,
                    "user_uuid": user_uuid,
                    "item_uuid": item_uuid,
                    "user_name": user_name,
                    "item_name": item_name,
                    "weight": weight,
                }

    return sorted(best.values(), key=lambda e: (e["user_id"], e["item_id"]))
