# TomatoGuard AI — Research Viewer API (Phase 5)

Local research viewer for Track B (TOBRFV-LMID) plant-disjoint TEST replay.
Read-only: no file-upload endpoints. Does not bundle the dataset.

## Run
```bash
cd "TomatoGuard_Project (1)"
.venv/bin/pip install -r api/requirements.txt   # or use pinned venv
TOBRFV_ROOT="/path/to/TOBRFV-LMID" .venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000 . The dataset is read from `TOBRFV_ROOT` (read-only); images are served only by resolving paths present in the precomputed plant-split manifests.

## Tests
```bash
.venv/bin/python -m pytest api/tests -q
```
