# models/

- `mapping.py` — interaction → structured triple (KB write)
- `export_edges.py` — USER→EVENT→PRODUCT → bipartite edges (KB read)
- `lightgcn.py` — CPU LightGCN + BPR train (NumPy; no torch required)
- `artifacts.py` / `infer.py` — save/load embeddings, rank items
- `artifacts/` — per-brain trained weights (gitignored)
