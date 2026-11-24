"""Step 7 - Merge several tomato datasets into ONE label space with JOINT duplicate removal/grouping.

Unified labels: Bacterial Spot, Early Blight, Late Blight, Healthy, Yellow Leaf Curl Virus,
                Other Problem, Not Tomato.
Mapping is rule-based on the image's parent-folder name (datasets/default_rules.json, first match wins;
label null = skip). Anything unmatched is SKIPPED and listed - nothing is silently relabelled.

  cp datasets/datasets_local.example.json datasets/datasets_local.json   # edit the paths
  python 07_harmonize.py --config datasets/datasets_local.json --out unified
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import IMG_EXT, dhash_variants, near_duplicate_groups, sha256_file

UNIFIED = ["Bacterial Spot", "Early Blight", "Late Blight", "Healthy",
           "Yellow Leaf Curl Virus", "Other Problem", "Not Tomato"]
SKIP = object()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def map_folder(folder: str, rules: list, not_tomato_unless):
    n = norm(folder)
    if not_tomato_unless and norm(not_tomato_unless) not in n:
        return "Not Tomato"
    for r in rules:
        if not any(k in r for k in ("exact", "all", "any")):
            continue
        ok = True
        if "exact" in r:
            ok = n in [norm(x) for x in r["exact"]]
        if ok and r.get("all"):
            ok = all(k in n for k in r["all"])
        if ok and r.get("any"):
            ok = any(k in n for k in r["any"])
        if ok:
            return SKIP if r["label"] is None else r["label"]
    return None  # unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="unified")
    ap.add_argument("--near-dup-threshold", type=int, default=5)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    default_rules = json.loads((Path(__file__).parent / "datasets" / "default_rules.json").read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for i, d in enumerate(cfg["datasets"]):
        root = Path(d["root"])
        if not root.exists():
            print(f"WARNING: {d['name']}: {root} not found - skipped")
            continue
        rules = json.loads(Path(d["rules"]).read_text()) if d.get("rules") else default_rules
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
        for p in tqdm(files, desc=d["name"]):
            lab = map_folder(p.parent.name, rules, d.get("not_tomato_unless"))
            if lab is None or lab is SKIP:
                skipped.append({"source": d["name"], "folder": p.parent.name,
                                "reason": "unmatched" if lab is None else "explicit_skip"})
                continue
            rows.append({"path": str(p), "source": d["name"], "folder": p.parent.name, "label": lab,
                         "priority": d.get("priority", i)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No images mapped - check paths and folder names")
    sk = pd.DataFrame(skipped)
    if len(sk):
        sk = sk.groupby(["source", "folder", "reason"]).size().reset_index(name="n_images")
        sk.to_csv(out / "skipped.csv", index=False)
        print("\nSKIPPED folders (review - add rules if a folder should be mapped):\n", sk.to_string(index=False))

    ok = []
    for p in tqdm(df["path"], desc="verify"):
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(True)
        except Exception:  # noqa: BLE001
            ok.append(False)
    df = df[ok].sort_values(["priority", "path"]).reset_index(drop=True)

    df["sha256"] = [sha256_file(p) for p in tqdm(df["path"], desc="sha256")]
    nl = df.groupby("sha256")["label"].nunique()
    conflict = set(nl[nl > 1].index)
    n_conflict = int(df["sha256"].isin(conflict).sum())
    df = df[~df["sha256"].isin(conflict)]
    first = df.drop_duplicates("sha256", keep="first").set_index("sha256")["source"]
    dup = df.duplicated("sha256", keep="first")
    cross_exact = int((df.loc[dup, "source"].values != first.loc[df.loc[dup, "sha256"]].values).sum())
    df = df[~dup].reset_index(drop=True)
    print(f"\nexact duplicates removed: {int(dup.sum())} (of which across different datasets: {cross_exact}); "
          f"dropped for conflicting labels: {n_conflict}")

    hashes = np.array([dhash_variants(p) for p in tqdm(df["path"], desc="dhash")], dtype=np.uint64)
    df["group"] = near_duplicate_groups(hashes, a.near_dup_threshold, block=128)
    g = df.groupby("group")
    cross_groups = int((g["source"].nunique() > 1).sum())
    print(f"near-duplicate groups: {g.ngroups}; groups spanning >1 dataset: {cross_groups}; "
          f"largest group: {int(g.size().max())}")

    df.drop(columns="priority").to_csv(out / "all.csv", index=False)
    table = df.pivot_table(index="source", columns="label", values="path", aggfunc="count", fill_value=0)
    table = table.reindex(columns=[c for c in UNIFIED if c in table.columns])
    table.to_csv(out / "counts_source_by_label.csv")
    print("\nimages per dataset x unified label:\n", table.to_string())
    (out / "report.json").write_text(json.dumps({
        "images": int(len(df)), "exact_dups_removed": int(dup.sum()), "exact_dups_cross_dataset": cross_exact,
        "conflicting_label_hash_images_dropped": n_conflict, "near_dup_groups": int(g.ngroups),
        "groups_spanning_multiple_datasets": cross_groups, "labels": UNIFIED}, indent=2))
    (out / "classes.json").write_text(json.dumps([c for c in UNIFIED if c in set(df["label"])], indent=2))
    print("\nwrote", out / "all.csv")


if __name__ == "__main__":
    main()
