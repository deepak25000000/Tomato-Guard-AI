"""Cross-dataset check: evaluate a trained run on ANY other folder-per-class tomato leaf dataset
(e.g. a PlantVillage tomato subset, PlantDoc, or your own phone photos). Measures domain shift, which is the
main reason a lab-trained model disappoints in the field.

  python 06_external_eval.py --run runs/mobilenet_v3_small --data-dir /path/to/external \
      --class-map class_map.json --out ext_eval_plantvillage

class_map.json maps EXTERNAL folder names -> MODEL class names; unmapped external classes are skipped, e.g.
  {"Tomato_Bacterial_spot": "Bacterial Spot", "Tomato_healthy": "Healthy", "Tomato_Early_blight": "Early Blight"}
Check dataset licences and that the mapping is semantically right before reporting anything.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from common import IMG_EXT
from models import ImageCSV, build_model, eval_tf, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--out", default="ext_eval")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    classes = meta["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    cmap = json.loads(Path(a.class_map).read_text())
    bad = [v for v in cmap.values() if v not in c2i]
    if bad:
        raise SystemExit(f"class_map targets not in model classes: {bad}\nmodel classes: {classes}")
    rows = [{"path": str(p), "class": cmap[d.name]} for d in Path(a.data_dir).iterdir()
            if d.is_dir() and d.name in cmap for p in d.rglob("*") if p.suffix.lower() in IMG_EXT]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no images matched the class map")
    print(f"{len(df)} external images over {df['class'].nunique()} mapped classes")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["arch"], len(classes), pretrained=False).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    lg, y = predict_logits(model, DataLoader(ImageCSV(df, c2i, eval_tf()), batch_size=32, num_workers=2), dev)
    pred = lg.argmax(1)
    present = sorted(set(y))
    res = {"n": int(len(y)), "accuracy": accuracy_score(y, pred),
           "macro_f1_present_classes": f1_score(y, pred, labels=present, average="macro", zero_division=0),
           "in_domain_reference": json.loads((run / "eval.json").read_text()) if (run / "eval.json").exists() else None}
    (out / "external_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    rep = pd.DataFrame(classification_report(y, pred, labels=present, target_names=[classes[i] for i in present],
                                             output_dict=True, zero_division=0)).T
    rep.to_csv(out / "per_class.csv")
    pd.DataFrame(confusion_matrix(y, pred, labels=present), index=[classes[i] for i in present],
                 columns=[classes[i] for i in present]).to_csv(out / "confusion.csv")
    print(rep.round(3).to_string())
    print(f"\nexternal accuracy {res['accuracy']:.3f} vs in-domain "
          f"{res['in_domain_reference']['accuracy'] if res['in_domain_reference'] else 'n/a'}  "
          "<- the gap is your domain-shift evidence")


if __name__ == "__main__":
    main()
