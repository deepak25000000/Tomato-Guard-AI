"""Step 5 - compare all trained runs in one table (for the paper's model-comparison table).
   python 05_compare.py
"""
import json
from pathlib import Path

import pandas as pd

rows = []
for d in sorted(Path("runs").glob("*")):
    e, m = d / "eval.json", d / "meta.json"
    if e.exists() and m.exists():
        ev, me = json.loads(e.read_text()), json.loads(m.read_text())
        rows.append({"run": d.name, "accuracy": ev["accuracy"], "acc_CI_low": ev["accuracy_95ci"][0],
                     "acc_CI_high": ev["accuracy_95ci"][1], "macro_F1": ev["macro_f1"],
                     "macro_recall": ev["macro_recall"], "ROC_AUC": ev["macro_roc_auc"],
                     "ECE_cal": ev["ece_after_temperature"], "ckpt_MB": round((d / "best.pt").stat().st_size / 2**20, 2)})
df = pd.DataFrame(rows).round(4)
print(df.to_string(index=False))
df.to_csv("runs/comparison.csv", index=False)
