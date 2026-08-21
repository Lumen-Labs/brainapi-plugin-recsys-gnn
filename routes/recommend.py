from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.plugins.context import PluginContext


class InteractionIngestBody(BaseModel):
    user_id: str
    item_id: str
    behavior: str = Field(
        ...,
        description="Interaction type, e.g. VIEWED, ADDED_TO_CART, PURCHASED",
    )
    timestamp: Optional[str] = None
    brain_id: str = "demorecsys"
    wait: bool = Field(
        default=True,
        description="If true, poll until structured ingest reaches a terminal status",
    )
    timeout_s: float = 120.0


class TrainBody(BaseModel):
    brain_id: str = "demorecsys"
    model: str = Field(
        default="lightgcn",
        description="Training recipe id: lightgcn (pinsage/comirec later)",
    )
    epochs: int = 20
    embedding_dim: int = 64
    n_layers: int = 3
    wait: bool = True
    timeout_s: float = 600.0


TERMINAL = frozenset({"completed", "failed", "partial_failed", "timeout"})


def create_router(context: type[PluginContext] | PluginContext) -> APIRouter:
    router = APIRouter(prefix="/recsys", tags=["recsys-gnn-plugin"])
    cache = context.adapters.cache

    async def _poll_task(
        task_id: str, brain_id: str, timeout_s: float
    ) -> tuple[str, dict[str, Any] | None]:
        status = "queued"
        task_payload: dict[str, Any] | None = None
        deadline = time.monotonic() + max(1.0, timeout_s)
        while True:
            raw = cache.get_task(task_id, brain_id=brain_id)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if isinstance(raw, str):
                    try:
                        task_payload = json.loads(raw)
                    except json.JSONDecodeError:
                        task_payload = {"raw": raw}
                elif isinstance(raw, dict):
                    task_payload = raw
                status = str((task_payload or {}).get("status") or "unknown")
                if status in TERMINAL:
                    break
            if time.monotonic() >= deadline:
                status = "timeout" if status not in TERMINAL else status
                break
            await asyncio.sleep(0.5)
        return status, task_payload

    @router.post("/interactions")
    async def ingest_interactions(body: InteractionIngestBody) -> dict[str, Any]:
        from models.mapping import structured_ingest_payload
        from src.workers.tasks.ingestion import (
            ingest_structured_data as ingest_structured_data_task,
            set_ingestion_task_status,
        )

        brain_id = (body.brain_id or "demorecsys").strip()
        if brain_id.startswith(("beam1m", "locomoconv")):
            return {
                "status": "rejected",
                "message": (
                    "Refusing memory-eval brains. Use demorecsys "
                    "(or another dedicated *recsys* brain)."
                ),
                "brain_id": brain_id,
            }

        payload = structured_ingest_payload(
            user_id=body.user_id,
            item_id=body.item_id,
            behavior=body.behavior,
            timestamp=body.timestamp,
            brain_id=brain_id,
            seq=1,
        )
        task_id = str(uuid4())
        set_ingestion_task_status(task_id, brain_id, "queued", stage="queued")
        ingest_structured_data_task.apply_async(
            args=[payload],
            task_id=task_id,
        )

        status = "queued"
        task_payload: dict[str, Any] | None = None
        if body.wait:
            status, task_payload = await _poll_task(task_id, brain_id, body.timeout_s)

        return {
            "status": status,
            "task_id": task_id,
            "brain_id": brain_id,
            "message": (
                "Forwarded to core POST /ingest/structured mode=deterministic "
                "(event-hub triples). Does not call free-text POST /ingest/."
            ),
            "triple": payload["data"][0],
            "task": task_payload,
        }

    @router.post("/train")
    async def train(body: TrainBody) -> dict[str, Any]:
        from workers.tasks import run_train_recsys, train_recsys

        brain_id = (body.brain_id or "demorecsys").strip()
        if brain_id.startswith(("beam1m", "locomoconv")):
            return {
                "status": "rejected",
                "message": "Refusing memory-eval brains. Use demorecsys.",
                "brain_id": brain_id,
            }
        if (body.model or "lightgcn").lower() != "lightgcn":
            return {
                "status": "rejected",
                "message": f"Model {body.model!r} not implemented; use lightgcn.",
                "brain_id": brain_id,
            }

        task_id = str(uuid4())
        args = {
            "brain_id": brain_id,
            "epochs": body.epochs,
            "embedding_dim": body.embedding_dim,
            "n_layers": body.n_layers,
            "task_id": task_id,
        }

        # wait=true: train in-process so eval works even when Celery was
        # started without the recsys_gnn queue (default brainapi start).
        if body.wait:
            cache.set(
                f"task:{task_id}",
                json.dumps(
                    {
                        "status": "running",
                        "task_id": task_id,
                        "brain_id": brain_id,
                        "stage": "train",
                    }
                ),
                brain_id=brain_id,
            )
            result = await asyncio.to_thread(run_train_recsys, args)
            status = str(result.get("status") or "failed")
            cache.set(
                f"task:{task_id}",
                json.dumps({**result, "task_id": task_id}),
                brain_id=brain_id,
            )
            return {
                "status": status,
                "task_id": task_id,
                "brain_id": brain_id,
                "model": "lightgcn",
                "message": (
                    "LightGCN train finished in-process (wait=true). "
                    "Celery queue recsys_gnn used only when wait=false."
                ),
                "task": result,
            }

        cache.set(
            f"task:{task_id}",
            json.dumps({"status": "queued", "task_id": task_id, "brain_id": brain_id}),
            brain_id=brain_id,
        )
        train_recsys.apply_async(args=[args], task_id=task_id, queue="recsys_gnn")
        return {
            "status": "queued",
            "task_id": task_id,
            "brain_id": brain_id,
            "model": "lightgcn",
            "message": (
                "LightGCN train queued on Celery queue recsys_gnn. "
                "Ensure the worker includes -Q ...,recsys_gnn."
            ),
            "task": {"status": "queued", "task_id": task_id, "brain_id": brain_id},
        }

    @router.get("/recommend")
    async def recommend(
        user_id: str,
        top_k: int = 20,
        brain_id: str = "demorecsys",
        exclude_seen: bool = True,
    ) -> Any:
        from models.infer import recommend_for_user

        bid = (brain_id or "demorecsys").strip()
        result = recommend_for_user(
            bid, user_id, top_k=top_k, exclude_seen=exclude_seen
        )
        if result.get("status") == "model_missing":
            return JSONResponse(status_code=503, content=result)
        if result.get("status") == "user_unknown":
            return JSONResponse(status_code=404, content=result)
        return result

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "plugin": "recsys-gnn",
            "interactions": "forwards_to_/ingest/structured",
            "train": "lightgcn_on_graph_export",
            "recommend": "lightgcn_embeddings",
        }

    return router
