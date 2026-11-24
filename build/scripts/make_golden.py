"""Phase 1.4 - golden vectors: 50 fixed S2-test JPEGs + expected PyTorch logits.

  .venv/bin/python build/scripts/make_golden.py

Deterministic, class-stratified selection from results/splits_S2/test.csv (sorted by
path; evenly spaced indices within each class). For each image: copy the JPEG bytes
into build/golden/images/, record SHA-256, apply the documented preprocessing contract
(docs/PREPROCESSING.md: EXIF orientation, RGB, direct bilinear resize 224x224, /255,
ImageNet mean/std, NCHW) and run the FROZEN model to get expected float32 logits.

Output: build/golden/manifest.json (new files only). Restart-proof via existence check.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from models import build_model  # noqa: E402

CKPT = ROOT / "build" / "checkpoint" / "mobilenet_v3_large_seed42.pt"
S2 = ROOT / "results" / "splits_S2"
GDIR = ROOT / "build" / "golden"
MANIFEST = GDIR / "manifest.json"
CFG = json.loads((ROOT / "build" / "config.json").read_text())
N = int(CFG["golden_vectors"]["n"])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_jpeg_bytes(path: Path) -> tuple[np.ndarray, int]:
    """The documented contract, starting from JPEG bytes. Returns (nchw, exif_orientation)."""
    with Image.open(path) as im:
        orientation = int(im.getexif().get(274, 1))  # 274 = EXIF Orientation tag
        im = ImageOps.exif_transpose(im)              # step 2: apply EXIF orientation
        im = im.convert("RGB")                        # step 3
        im = im.resize((224, 224), Image.BILINEAR)    # step 4: direct resize, no crop
        a = np.asarray(im, dtype=np.float32) / 255.0  # step 5
    a = (a - MEAN) / STD                              # step 6
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None], orientation  # step 7 NCHW


def main() -> int:
    if MANIFEST.exists():
        print(f"[skip] {MANIFEST} exists")
        return 0
    t0 = time.time()
    df = pd.read_csv(S2 / "test.csv")
    classes = json.loads((S2 / "classes.json").read_text())
    # deterministic stratified selection: sort by (class index, path), even spacing
    df = df.copy()
    df["cls"] = df["class"].map({c: i for i, c in enumerate(classes)})
    df = df.sort_values(["cls", "path"], kind="stable").reset_index(drop=True)
    per_class = {c: 0 for c in classes}
    chosen = []
    # round-robin over classes in sorted order until N reached
    groups = [g.index.tolist() for _, g in df.groupby("cls", sort=True)]
    while len(chosen) < N and any(groups):
        for gi in range(len(groups)):
            if not groups[gi]:
                continue
            # even spacing: pick the middle-most remaining index for stability
            pick = groups[gi].pop(len(groups[gi]) // 2)
            chosen.append(pick)
            if len(chosen) >= N:
                break
    chosen = sorted(chosen)
    sub = df.loc[chosen].reset_index(drop=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    (GDIR / "images").mkdir(parents=True, exist_ok=True)
    vectors = []
    n_exif = 0
    for _, row in sub.iterrows():
        src = Path(row["path"])
        dst = GDIR / "images" / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
        x, orient = preprocess_jpeg_bytes(src)
        if orient != 1:
            n_exif += 1
        with torch.no_grad():
            lg = model(torch.from_numpy(x)).numpy()[0]
        e = np.exp(lg - lg.max())
        prob = (e / e.sum()).astype(np.float32)
        vectors.append({
            "file": f"images/{src.name}",
            "source_path": str(src),
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "true_label": row["class"],
            "true_index": int(row["cls"]),
            "exif_orientation": orient,
            "logits": [float(v) for v in lg.astype(np.float32)],
            "probs": [float(v) for v in prob],
            "top1_index": int(prob.argmax()),
            "top1_label": ck["classes"][int(prob.argmax())],
        })
    manifest = {
        "n": len(vectors),
        "model": ck["arch"],
        "seed": ck["seed"],
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "preprocessing": "docs/PREPROCESSING.md (EXIF, RGB, bilinear 224x224, /255, ImageNet mean/std, NCHW)",
        "selection": "S2 test.csv sorted by (class, path), round-robin even spacing, n=50",
        "classes": ck["classes"],
        "n_exif_orientation_ne_1": n_exif,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vectors": vectors,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST} with {len(vectors)} vectors; "
          f"exif!=1: {n_exif}; runtime {round(time.time()-t0,1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
