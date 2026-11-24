"""Step 3 - Evaluate a trained run on the held-out test split (same metrics as your notebook,
plus calibration/ECE, per-class report and bootstrap CIs).

  python 03_evaluate.py --run runs/mobilenet_v3_small
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from models import ImageCSV, build_model, eval_tf, predict_logits


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    hit = pred == y
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    classes, T = meta["classes"], meta["temperature"]
    c2i = {c: i for i, c in enumerate(classes)}
    test = pd.read_csv(Path(a.splits_dir) / "test.csv")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(meta["arch"], len(classes), pretrained=False).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    loader = DataLoader(ImageCSV(test, c2i, eval_tf()), batch_size=32, num_workers=2)
    logits, y = predict_logits(model, loader, dev)
    p_raw, p_cal = softmax(logits), softmax(logits / T)
    pred = p_raw.argmax(1)  # temperature scaling never changes the argmax

    yb = label_binarize(y, classes=np.arange(len(classes)))
    m = {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_roc_auc": roc_auc_score(yb, p_cal, average="macro"),
        "macro_pr_auc": average_precision_score(yb, p_cal, average="macro"),
        "ece_before_temperature": ece(p_raw, y), "ece_after_temperature": ece(p_cal, y),
        "temperature": T, "n_test": int(len(y)),
    }
    rng = np.random.default_rng(42)
    accs, f1s = [], []
    for _ in range(a.boot):
        idx = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum(), replace=True)
                              for k in range(len(classes)) if (y == k).any()])
        accs.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], labels=np.arange(len(classes)), average="macro", zero_division=0))
    m["accuracy_95ci"] = np.percentile(accs, [2.5, 97.5]).tolist()
    m["macro_f1_95ci"] = np.percentile(f1s, [2.5, 97.5]).tolist()

    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    spec = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp; tn = cm.sum() - tp - fp - fn
        spec[c] = float(tn / (tn + fp)) if (tn + fp) else None
    m["specificity"] = spec
    rep = pd.DataFrame(classification_report(y, pred, labels=np.arange(len(classes)), target_names=classes,
                                             output_dict=True, zero_division=0)).T
    rep.to_csv(run / "classification_report.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.xticks(rotation=40, ha="right"); plt.tight_layout()
    plt.savefig(run / "confusion_matrix.png", dpi=300)
    (run / "eval.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame({"path": test["path"], "y_true": y, "y_pred": pred, "confidence": p_cal.max(1)}
                 ).to_csv(run / "test_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in m.items() if k != "specificity"}, indent=2))
    print(rep.round(3).to_string())


if __name__ == "__main__":
    main()
