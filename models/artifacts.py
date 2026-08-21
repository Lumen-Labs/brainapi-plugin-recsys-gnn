"""Artifact paths and load/save helpers for LightGCN embeddings (NumPy)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_dir(brain_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in brain_id)
    return plugin_root() / "models" / "artifacts" / safe


@dataclass
class LightGCNArtifacts:
    brain_id: str
    user_ids: list[str]
    item_ids: list[str]
    user_aliases: dict[str, str]
    item_aliases: dict[str, str]
    user_emb: np.ndarray
    item_emb: np.ndarray
    seen: dict[str, list[str]]
    meta: dict[str, Any]
    path: Path

    def resolve_user_index(self, user_key: str) -> int | None:
        key = str(user_key).strip()
        canonical = self.user_aliases.get(key, key)
        try:
            return self.user_ids.index(canonical)
        except ValueError:
            return None


def save_artifacts(
    out_dir: Path,
    *,
    brain_id: str,
    user_ids: list[str],
    item_ids: list[str],
    user_aliases: dict[str, str],
    item_aliases: dict[str, str],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    seen: dict[str, list[str]],
    meta: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "user_emb.npy", np.asarray(user_emb, dtype=np.float32))
    np.save(out_dir / "item_emb.npy", np.asarray(item_emb, dtype=np.float32))
    id_maps = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_aliases": user_aliases,
        "item_aliases": item_aliases,
        "seen": seen,
    }
    (out_dir / "id_maps.json").write_text(
        json.dumps(id_maps, indent=2) + "\n", encoding="utf-8"
    )
    payload = {"brain_id": brain_id, "model": "lightgcn", **meta}
    (out_dir / "meta.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def load_artifacts(brain_id: str, base: Path | None = None) -> LightGCNArtifacts | None:
    out_dir = base or artifacts_dir(brain_id)
    if not (out_dir / "meta.json").exists():
        return None
    user_path = out_dir / "user_emb.npy"
    item_path = out_dir / "item_emb.npy"
    if not user_path.exists() or not item_path.exists():
        return None
    id_maps = json.loads((out_dir / "id_maps.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return LightGCNArtifacts(
        brain_id=brain_id,
        user_ids=list(id_maps.get("user_ids") or []),
        item_ids=list(id_maps.get("item_ids") or []),
        user_aliases=dict(id_maps.get("user_aliases") or {}),
        item_aliases=dict(id_maps.get("item_aliases") or {}),
        user_emb=np.load(user_path),
        item_emb=np.load(item_path),
        seen={str(k): list(v) for k, v in (id_maps.get("seen") or {}).items()},
        meta=meta,
        path=out_dir,
    )
