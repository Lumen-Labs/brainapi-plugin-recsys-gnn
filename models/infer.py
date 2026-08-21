"""Inference helpers for trained LightGCN artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from models.artifacts import LightGCNArtifacts, load_artifacts


def recommend_for_user(
    brain_id: str,
    user_id: str,
    *,
    top_k: int = 20,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    arts = load_artifacts(brain_id)
    if arts is None:
        return {
            "status": "model_missing",
            "message": "No LightGCN artifacts for this brain. POST /recsys/train first.",
            "user_id": user_id,
            "brain_id": brain_id,
            "model": "lightgcn",
            "items": [],
        }
    return rank_items(arts, user_id, top_k=top_k, exclude_seen=exclude_seen)


def rank_items(
    arts: LightGCNArtifacts,
    user_id: str,
    *,
    top_k: int = 20,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    idx = arts.resolve_user_index(user_id)
    if idx is None:
        return {
            "status": "user_unknown",
            "message": f"User {user_id!r} not in trained id map.",
            "user_id": user_id,
            "brain_id": arts.brain_id,
            "model": "lightgcn",
            "items": [],
        }

    user_vec = arts.user_emb[idx]
    scores = arts.item_emb @ user_vec
    canonical_user = arts.user_ids[idx]
    seen_ids = set(arts.seen.get(canonical_user) or [])
    if exclude_seen and seen_ids:
        for j, item_id in enumerate(arts.item_ids):
            if item_id in seen_ids:
                scores[j] = -np.inf

    k = max(1, min(int(top_k), scores.shape[0]))
    # argpartition then sort the top-k
    if k < scores.shape[0]:
        part = np.argpartition(-scores, k - 1)[:k]
    else:
        part = np.arange(scores.shape[0])
    part = part[np.argsort(-scores[part])]
    items = []
    for j in part:
        if not np.isfinite(scores[j]):
            continue
        items.append({"item_id": arts.item_ids[int(j)], "score": float(scores[j])})
    return {
        "status": "ok",
        "user_id": user_id,
        "brain_id": arts.brain_id,
        "model": "lightgcn",
        "top_k": top_k,
        "items": items,
        "meta": {
            "n_users": arts.meta.get("n_users"),
            "n_items": arts.meta.get("n_items"),
            "n_edges": arts.meta.get("n_edges"),
            "artifact_path": str(arts.path),
            "backend": arts.meta.get("backend", "numpy"),
        },
    }
