# recsys-gnn

Optional LightGCN recommender on top of BrainAPI. The core product is still the knowledge graph: structured ingest writes user–item event hubs; this plugin **exports** those edges, trains a NumPy LightGCN, and serves `GET /recsys/recommend`.

It does **not** replace `POST /ingest/`, core synergies, or `GET /retrieve/recommend`. Additive only.

| | |
|---|---|
| Registry name | `recsys-gnn` |
| Version | `0.2.0` |
| BrainAPI | `>=2.14.0` |
| Route prefix | `/recsys` |
| Extra pip deps | `numpy` |
| Celery queue | `recsys_gnn` (only when `wait=false` on train) |

## Install

```bash
git clone https://github.com/Lumen-Labs/brainapi-plugin-recsys-gnn.git plugins/recsys-gnn
```

Or:

```bash
./bin/brainapi install recsys-gnn
```

Restart the API. `numpy` is installed from `plugin.yaml` on load.

For async train (`wait=false`), the worker must include queue **`recsys_gnn`**. With `wait=true` (default), training runs **in-process** so eval works even when Celery was started without that queue.

## Quick start

```bash
# 1) Write an interaction into the KB (deterministic structured ingest)
curl -X POST "$BRAINAPI_URL/recsys/interactions" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -d '{
    "user_id": "u1",
    "item_id": "sku-42",
    "behavior": "purchase",
    "brain_id": "demorecsys",
    "wait": true
  }'

# 2) Train LightGCN from the graph
curl -X POST "$BRAINAPI_URL/recsys/train" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -d '{
    "brain_id": "demorecsys",
    "model": "lightgcn",
    "epochs": 20,
    "embedding_dim": 64,
    "n_layers": 3,
    "wait": true
  }'

# 3) Rank
curl "$BRAINAPI_URL/recsys/recommend?user_id=u1&top_k=20&brain_id=demorecsys" \
  -H "BrainPAT: $BRAINPAT_TOKEN"
```

Health: `GET /recsys/health`.

## Safety

Brain ids starting with `beam1m` or `locomoconv` are **rejected** on ingest and train. Use `demorecsys` (or another dedicated recsys brain). Never train on LoCoMo/BEAM eval brains.

## API

### `POST /recsys/interactions`

Forwards to core `POST /ingest/structured` with `mode=deterministic` (event-hub triples). Does not call free-text `POST /ingest/`.

| Field | Default | Description |
|---|---|---|
| `user_id`, `item_id`, `behavior` | required | Interaction |
| `timestamp` | optional | Event `happened_at` |
| `brain_id` | `demorecsys` | |
| `wait` | `true` | Poll until terminal ingest status |
| `timeout_s` | `120` | |

Mapped behaviors: view/click → `View`; cart → `AddToCart`; purchase/buy → `Purchase`. Unknown labels are title-cased.

For attribute preferences and Favorite/Wishlist weights, use [features-rec](https://github.com/Lumen-Labs/brainapi-plugin-features-rec) instead (or as well). This route only writes the interaction triple.

### `POST /recsys/train`

Exports USER→EVENT→PRODUCT bipartite edges from the graph and trains LightGCN (BPR, NumPy, no PyTorch required).

| Field | Default | Description |
|---|---|---|
| `brain_id` | `demorecsys` | |
| `model` | `lightgcn` | Only `lightgcn` is implemented (`pinsage` / `comirec` rejected) |
| `epochs` | `20` | |
| `embedding_dim` | `64` | |
| `n_layers` | `3` | |
| `wait` | `true` | In-process train vs Celery `recsys_gnn` |
| `timeout_s` | `600` | Used when polling a queued train |

Edge weights used at export: purchase `3`, add-to-cart `2`, view `1`.

Artifacts are written under `models/artifacts/<brain_id>/` (`user_emb.npy`, `item_emb.npy`, `id_maps.json`, `meta.json`). That directory is gitignored.

### `GET /recsys/recommend`

Query params: `user_id`, `top_k=20`, `brain_id=demorecsys`, `exclude_seen=true`.

| Status | HTTP | Meaning |
|---|---|---|
| `ok` | 200 | Ranked `{ item_id, score }[]` |
| `model_missing` | 503 | No artifacts — `POST /recsys/train` first |
| `user_unknown` | 404 | User not in the trained id map |

Scores are `item_emb @ user_vec`. Seen items (from training edges) can be masked to `-inf`.

## Layout

```text
recsys-gnn/
  plugin.yaml
  main.py
  routes/recommend.py
  models/mapping.py       # interaction → structured triple
  models/export_edges.py  # graph → bipartite edges
  models/lightgcn.py      # CPU LightGCN + BPR
  models/artifacts.py     # save/load
  models/infer.py         # rank
  models/artifacts/       # gitignored trained weights
  workers/tasks.py
  workers/celery.py
```

## Publishing

Pushes to `main` publish to the BrainAPI registry via GitHub Actions. Trained embeddings are not part of the package.

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).

## Related

- [features-rec](https://github.com/Lumen-Labs/brainapi-plugin-features-rec) — train-free attribute prefs
- [BrainAPI](https://github.com/Lumen-Labs/brainapi2)
- `docs/research/16-recsys-eval-protocol.md` on brainapi2
