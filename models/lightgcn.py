"""Minimal CPU LightGCN (He et al.) with BPR — NumPy only (no torch required)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from models.artifacts import artifacts_dir, save_artifacts


def _build_maps(edges: list[dict[str, Any]]) -> tuple[
    list[str],
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    list[tuple[int, int]],
]:
    user_canonical: dict[str, str] = {}
    item_canonical: dict[str, str] = {}
    user_aliases: dict[str, str] = {}
    item_aliases: dict[str, str] = {}
    seen: dict[str, set[str]] = defaultdict(set)

    for e in edges:
        u_can = str(e.get("user_name") or e.get("user_id") or e["user_uuid"])
        i_can = str(e.get("item_name") or e.get("item_id") or e["item_uuid"])
        for key in (e.get("user_uuid"), e.get("user_id"), e.get("user_name"), u_can):
            if key:
                user_aliases[str(key)] = u_can
        for key in (e.get("item_uuid"), e.get("item_id"), e.get("item_name"), i_can):
            if key:
                item_aliases[str(key)] = i_can
        user_canonical[u_can] = u_can
        item_canonical[i_can] = i_can
        seen[u_can].add(i_can)

    user_ids = sorted(user_canonical)
    item_ids = sorted(item_canonical)
    u_index = {u: i for i, u in enumerate(user_ids)}
    i_index = {it: i for i, it in enumerate(item_ids)}
    pairs: list[tuple[int, int]] = []
    for e in edges:
        u_can = user_aliases[
            str(e.get("user_name") or e.get("user_id") or e["user_uuid"])
        ]
        i_can = item_aliases[
            str(e.get("item_name") or e.get("item_id") or e["item_uuid"])
        ]
        pairs.append((u_index[u_can], i_index[i_can]))

    seen_lists = {u: sorted(v) for u, v in seen.items()}
    return user_ids, item_ids, user_aliases, item_aliases, seen_lists, pairs


def _normalized_adj(n_users: int, n_items: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    n = n_users + n_items
    adj = np.zeros((n, n), dtype=np.float64)
    for u, i in pairs:
        ui = n_users + i
        adj[u, ui] = 1.0
        adj[ui, u] = 1.0
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    # D^{-1/2} A D^{-1/2}
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def _propagate(E0: np.ndarray, norm_adj: np.ndarray, n_layers: int) -> np.ndarray:
    embs = [E0]
    x = E0
    for _ in range(n_layers):
        x = norm_adj @ x
        embs.append(x)
    return np.mean(np.stack(embs, axis=0), axis=0)


def train_and_save(
    edges: list[dict[str, Any]],
    *,
    brain_id: str,
    epochs: int = 20,
    embedding_dim: int = 64,
    n_layers: int = 3,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if not edges:
        raise ValueError("No user–item edges to train on")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    user_ids, item_ids, user_aliases, item_aliases, seen, pairs = _build_maps(edges)
    n_users, n_items = len(user_ids), len(item_ids)
    if n_users == 0 or n_items == 0:
        raise ValueError("Need at least one user and one item")

    n = n_users + n_items
    E = rng.normal(0.0, 0.1, size=(n, embedding_dim)).astype(np.float64)
    norm_adj = _normalized_adj(n_users, n_items, pairs)

    user_pos: dict[int, set[int]] = defaultdict(set)
    for u, i in pairs:
        user_pos[u].add(i)

    pair_list = list(pairs)
    last_loss = 0.0
    for _epoch in range(max(1, int(epochs))):
        random.shuffle(pair_list)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(pair_list), batch_size):
            batch = pair_list[start : start + batch_size]
            out = _propagate(E, norm_adj, n_layers)
            user_emb = out[:n_users]
            item_emb = out[n_users:]

            grad = np.zeros_like(E)
            batch_loss = 0.0
            for u, i in batch:
                for _ in range(50):
                    j = int(rng.integers(0, n_items))
                    if j not in user_pos[u]:
                        break
                else:
                    j = int(rng.integers(0, n_items))
                u_e = user_emb[u]
                pos_e = item_emb[i]
                neg_e = item_emb[j]
                pos_score = float(np.dot(u_e, pos_e))
                neg_score = float(np.dot(u_e, neg_e))
                x = pos_score - neg_score
                # -log σ(x)
                sig = 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))
                loss = -math.log(max(sig, 1e-12))
                batch_loss += loss
                # d(-log σ)/dx = σ - 1
                d = sig - 1.0
                # grads w.r.t embeddings at final layer (approx through mean of layers:
                # treat as identity for simplicity on E0 — LightGCN often trains E0 only)
                gu = d * (pos_e - neg_e)
                gi = d * u_e
                gj = -d * u_e
                grad[u] += gu
                grad[n_users + i] += gi
                grad[n_users + j] += gj

            grad /= max(1, len(batch))
            grad += weight_decay * E
            E -= lr * grad
            total_loss += batch_loss / max(1, len(batch))
            n_batches += 1
        last_loss = total_loss / max(1, n_batches)

    out = _propagate(E, norm_adj, n_layers)
    user_emb = out[:n_users].astype(np.float32)
    item_emb = out[n_users:].astype(np.float32)

    dest = out_dir or artifacts_dir(brain_id)
    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_edges": len(pairs),
        "epochs": int(epochs),
        "embedding_dim": embedding_dim,
        "n_layers": n_layers,
        "backend": "numpy",
        "final_bpr_loss": last_loss,
        "artifact_path": str(dest),
    }
    save_artifacts(
        dest,
        brain_id=brain_id,
        user_ids=user_ids,
        item_ids=item_ids,
        user_aliases=user_aliases,
        item_aliases=item_aliases,
        user_emb=user_emb,
        item_emb=item_emb,
        seen=seen,
        meta=meta,
    )
    return meta
