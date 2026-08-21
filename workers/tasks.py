from __future__ import annotations

from typing import Any

from src.config import config
from src.core.plugins.context import PluginContext
from src.core.plugins.prompts import prompt_registry
from src.workers.app import ingestion_app


def _worker_context() -> PluginContext:
    return PluginContext(
        adapters=PluginContext._build_adapters(),
        prompts=prompt_registry,
        config=config,
    )


def _set_status(cache, task_id: str, brain_id: str, status: str, **extra: Any) -> None:
    import json

    payload = {"status": status, "task_id": task_id, "brain_id": brain_id, **extra}
    cache.set(f"task:{task_id}", json.dumps(payload), brain_id=brain_id)


def run_train_recsys(args: dict | None = None, *, cache=None) -> dict:
    """
    Export user–item edges from BrainAPI graph adapters and train LightGCN.

    Callable from the API (wait=true) or Celery. Never mutates LoCoMo/BEAM brains.
    """
    payload = dict(args or {})
    brain_id = str(payload.get("brain_id") or "demorecsys").strip()
    epochs = int(payload.get("epochs") or 20)
    embedding_dim = int(payload.get("embedding_dim") or 64)
    n_layers = int(payload.get("n_layers") or 3)
    task_id = str(payload.get("task_id") or "")

    ctx = _worker_context()
    cache = cache or ctx.adapters.cache

    try:
        from models.export_edges import assert_recsys_brain, export_user_item_edges
        from models.lightgcn import train_and_save

        assert_recsys_brain(brain_id)
        if task_id:
            _set_status(cache, task_id, brain_id, "running", stage="export")

        edges = export_user_item_edges(ctx.adapters.graph, brain_id)
        if task_id:
            _set_status(
                cache,
                task_id,
                brain_id,
                "running",
                stage="train",
                n_edges=len(edges),
            )

        meta = train_and_save(
            edges,
            brain_id=brain_id,
            epochs=epochs,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
        )
        result = {
            "status": "completed",
            "model": "lightgcn",
            "brain_id": brain_id,
            "n_users": meta["n_users"],
            "n_items": meta["n_items"],
            "n_edges": meta["n_edges"],
            "artifact_path": meta["artifact_path"],
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "final_bpr_loss": meta.get("final_bpr_loss"),
        }
        if task_id:
            _set_status(
                cache,
                task_id,
                brain_id,
                "completed",
                stage="completed",
                model=result["model"],
                n_users=result["n_users"],
                n_items=result["n_items"],
                n_edges=result["n_edges"],
                artifact_path=result["artifact_path"],
                epochs=epochs,
                embedding_dim=embedding_dim,
                final_bpr_loss=result.get("final_bpr_loss"),
            )
        return result
    except Exception as exc:  # noqa: BLE001 — surface to cache/status
        err = {"status": "failed", "error": str(exc), "brain_id": brain_id}
        if task_id:
            try:
                _set_status(
                    cache, task_id, brain_id, "failed", stage="failed", error=str(exc)
                )
            except Exception:
                pass
        return err


@ingestion_app.task(bind=True, name="workers.tasks.train_recsys")
def train_recsys(self, args: dict | None = None) -> dict:
    payload = dict(args or {})
    if not payload.get("task_id"):
        payload["task_id"] = str(getattr(self.request, "id", "") or "")
    return run_train_recsys(payload)
