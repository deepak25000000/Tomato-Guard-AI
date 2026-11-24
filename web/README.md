# Web (PWA) - TomatoGuard AI

Research prototype PWA. Runs leaf-model ONNX in-browser via `onnxruntime-web` (WASM). No upload by default. Local-only history in IndexedDB.

## Build
```bash
cd web
npm ci   # or npm install (package-lock.json pinned after install)
npm run build
npm run preview
```

## Assets
Copy model artifacts into `web/public/assets/` before build:
- `leaf_mobilenet_v3_large_seed42.onnx` (from `build/models/`)
- `model_meta.json` (from `build/model_meta.json`)
- `ui_strings.json` (from `build/ui_strings.json`)

## Parity
```bash
cd web
node scripts/golden_parity.mjs   # writes build/parity/web_node.json
```
Browser Canvas parity is verified via Playwright (see tests).

## Notes
- `onnxruntime-web` pinned in `package.json`.
- Service worker caches app shell; no image upload endpoints.
- Deployment requires explicit approval (spec).
