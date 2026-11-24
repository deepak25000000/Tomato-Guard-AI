from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "results" / "phase3" / "plant_splits"
PHASE4 = ROOT / "results" / "phase4"
BUILD = ROOT / "build"
UI = json.loads((BUILD / "ui_strings.json").read_text())
BANNER = UI["RESEARCH_BANNER"]
MIXED_DAYS = set(range(9, 30))
EARLY_DAYS = set(range(1, 9))
MODES = ("rgb", "spectral", "fusion")
SEEDS = (42, 43, 44)

TOBRFV_ROOT = Path(os.environ.get("TOBRFV_ROOT", str(ROOT / "TOBRFV-LMID")))
ALLOWED_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if o]

app = FastAPI(title="TomatoGuard AI Research Viewer", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET"], allow_headers=["*"])

# --- load read-only artifacts ---
MANIFESTS = {s: pd.read_csv(SPLITS / f"manifest_split_seed{s}.csv") for s in SEEDS}
PREDS = {(s, m): pd.read_csv(PHASE4 / f"seed{s}" / m / "preds_test.csv") for s in SEEDS for m in MODES}
# allowed image paths (absolute resolved) - only these can be served
ALLOWED_PATHS: set[str] = set()
for df in MANIFESTS.values():
    for col in ("rgb", "nir800", "nir1000"):
        for p in df[col].dropna().astype(str):
            ALLOWED_PATHS.add(str(Path(p).resolve()))

# lazy ONNX sessions
_SESS: dict[tuple[int,str], object] = {}

def get_session(seed: int, mode: str):
    key = (seed, mode)
    if key in _SESS: return _SESS[key]
    p = BUILD / "models" / f"trackb_{mode}_seed{seed}.onnx"
    if not p.exists(): return None
    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    _SESS[key] = sess
    return sess

def _preprocess_stream(path: str, gray: bool) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    im = im.convert("L" if gray else "RGB").resize((224, 224))
    a = np.asarray(im, np.float32) / 255.0
    if gray:
        return a
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return (a - mean) / std

def live_predict(seed: int, mode: str, row: pd.Series) -> float | None:
    sess = get_session(seed, mode)
    if sess is None: return None
    rgb = _preprocess_stream(str(row["rgb"]), False).transpose(2,0,1)[None].astype(np.float32)
    g8 = _preprocess_stream(str(row["nir800"]), True)
    g10 = _preprocess_stream(str(row["nir1000"]), True)
    spec = (np.stack([g8, g10, (g8+g10)/2]) - 0.5) / 0.5
    spec = spec[None].astype(np.float32)
    in_names = [i.name for i in sess.get_inputs()]
    feed = {}
    if "rgb" in in_names: feed["rgb"] = rgb
    if "spec" in in_names: feed["spec"] = spec
    logits = sess.run(["logits"], feed)[0][0]
    e = np.exp(logits - logits.max()); p = e / e.sum()
    return float(p[1])

@app.get("/api/health")
def health():
    return {"status": "ok", "banner": BANNER, "tobrfv_root": str(TOBRFV_ROOT), "tobrfv_root_exists": TOBRFV_ROOT.exists()}

@app.get("/api/model-info")
def model_info():
    models = []
    for s in SEEDS:
        for m in MODES:
            p = BUILD / "models" / f"trackb_{m}_seed{s}.onnx"
            models.append({"seed": s, "mode": m, "exists": p.exists(), "path": str(p.relative_to(ROOT)) if p.exists() else None})
    return {"banner": BANNER, "models": models, "ui_strings": UI,
            "attribution": "TOBRFV-LMID (CC BY 4.0). Do not bundle the dataset with this viewer."}

@app.get("/api/plants")
def plants(seed: int = Query(42), split: Literal["test"]="test", mixed_only: bool = Query(True)):
    if seed not in SEEDS: raise HTTPException(400, "seed must be 42,43,44")
    df = MANIFESTS[seed]
    df = df[df["split"] == split]
    if mixed_only: df = df[df["day"].isin(MIXED_DAYS)]
    out = (df.groupby(["plant_key","label"]).size().reset_index(name="n_images")
             .sort_values("plant_key"))
    return {"banner": BANNER, "seed": seed, "mixed_only": mixed_only, "plants": out.to_dict(orient="records")}

@app.get("/api/replay")
def replay(seed: int = Query(42), plant_key: str = Query(..., min_length=1),
           day: int = Query(..., ge=1, le=29), mixed_only: bool = Query(True)):
    if seed not in SEEDS: raise HTTPException(400, "seed must be 42,43,44")
    if mixed_only and day not in MIXED_DAYS:
        raise HTTPException(400, "day outside mixed-label range 9-29; set mixed_only=false to view day 1-8 with warning")
    man = MANIFESTS[seed]
    rows = man[(man["split"]=="test") & (man["plant_key"]==plant_key) & (man["day"]==day)]
    if rows.empty: raise HTTPException(404, "no test rows for that plant/day")
    warning = UI["SINGLE_CLASS_DAY_WARNING"] if day in EARLY_DAYS else None
    results = []
    for _, r in rows.iterrows():
        item = {"plant_key": plant_key, "day": int(day), "angle": int(r["angle"]), "batch": r["batch"],
                "true_label_inoculation_status": int(r["label"]),
                "true_label_name": "infected" if int(r["label"])==1 else "healthy_control",
                "streams": {k: str(r[k]) for k in ("rgb","nir800","nir1000")},
                "warning": warning, "precomputed": {}, "live_onnx": {}, "match_delta": {}}
        for m in MODES:
            pre = PREDS[(seed,m)]
            match = pre[(pre["plant_key"]==plant_key) & (pre["day"]==day) & (pre["angle"]==int(r["angle"]))]
            if match.empty:
                # fallback: match by index in manifest order
                pre_p = None
            else:
                pre_p = float(match.iloc[0]["p_infected"])
            live_p = live_predict(seed, m, r)
            item["precomputed"][m] = pre_p
            item["live_onnx"][m] = live_p
            if pre_p is not None and live_p is not None:
                item["match_delta"][m] = abs(pre_p - live_p)
        results.append(item)
    return {"banner": BANNER, "seed": seed, "plant_key": plant_key, "day": day,
            "day_warning": warning, "results": results}

@app.get("/api/image")
def image(path: str = Query(..., min_length=1)):
    # read-only serving of manifest-listed files only
    rp = str(Path(path).resolve())
    if rp not in ALLOWED_PATHS:
        raise HTTPException(403, "path not in the allowed plant-split manifest")
    p = Path(rp)
    if not p.exists(): raise HTTPException(404, "file not found on this machine")
    return FileResponse(p)

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

INDEX_HTML = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>TomatoGuard Research Viewer</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}} .banner{{background:#fff3cd;padding:10px;border-radius:8px;font-weight:700}} .warn{{color:#b71c1c;font-weight:700}} .muted{{color:#555}} img{{max-width:240px;border:1px solid #ccc;margin:4px}} .card{{border:1px solid #ddd;padding:12px;margin:10px 0;border-radius:8px}} pre{{white-space:pre-wrap}}</style>
</head><body>
<div class='banner'>{BANNER}</div>
<h1>Track B Research Viewer (read-only replay)</h1>
<p class='muted'>Labels = inoculation status (RT-PCR verified), not symptom onset. Days 1-8 are single-class days; default view shows mixed-label days 9-29 only. TOBRFV-LMID, CC BY 4.0 — attribution required; dataset not bundled.</p>
<label>Seed <select id='seed'><option>42</option><option>43</option><option>44</option></select></label>
<label><input type='checkbox' id='mixed' checked> Mixed-label days only (9-29)</label>
<button onclick='loadPlants()'>Load plants</button>
<div id='plants'></div>
<div id='out'></div>
<script>
async function loadPlants(){{const seed=document.getElementById('seed').value; const mixed=document.getElementById('mixed').checked;
  const r=await fetch(`./api/plants?seed=${{seed}}&mixed_only=${{mixed}}`); const d=await r.json();
  const box=document.getElementById('plants'); box.innerHTML='<h3>Test plants</h3>';
  d.plants.forEach(p=>{{const b=document.createElement('button'); b.textContent=`${{p.plant_key}} (y=${{p.label}}, n=${{p.n_images}})`; b.style.margin='4px';
    b.onclick=()=>loadPlant(p.plant_key); box.appendChild(b);}});}}
async function loadPlant(pk){{const seed=document.getElementById('seed').value; const mixed=document.getElementById('mixed').checked;
  const days=mixed?[9,11,13,15,17,19,21,23,25,27,29]:Array.from({{length:29}},(_,i)=>i+1);
  const out=document.getElementById('out'); out.innerHTML='<h3>'+pk+'</h3>';
  for(const day of days){{const r=await fetch(`./api/replay?seed=${{seed}}&plant_key=${{encodeURIComponent(pk)}}&day=${{day}}&mixed_only=${{mixed}}`);
    if(!r.ok) continue; const d=await r.json();
    d.results.forEach(item=>{{const card=document.createElement('div'); card.className='card';
      const warn=item.warning?`<div class='warn'>${{item.warning}}</div>`:'';
      const imgs=['rgb','nir800','nir1000'].map(k=>`<div style='display:inline-block'><div>${{k}}</div><img src='./api/image?path=${{encodeURIComponent(item.streams[k])}}'></div>`).join('');
      card.innerHTML=`<b>day ${{day}} angle ${{item.angle}}</b> true=${{item.true_label_name}} (inoculation status)${{warn}}<div>${{imgs}}</div><pre>${{JSON.stringify({{precomputed:item.precomputed,live_onnx:item.live_onnx,match_delta:item.match_delta}},null,2)}}</pre>`;
      out.appendChild(card);}});}}}}
</script></body></html>"""
