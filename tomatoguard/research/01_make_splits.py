"""Step 1 - Build leakage-aware splits for the 7-class RGB leaf dataset.

The Mendeley RGB dataset has NO plant IDs, so true plant-disjoint splitting is impossible.
Best available substitute: exact-duplicate removal + near-duplicate grouping (perceptual hash,
including flipped copies) + a group-disjoint, class-stratified split. Report it as
"near-duplicate-group-disjoint", NOT "plant-disjoint".

Usage:
  python 01_make_splits.py --data-dir "/path/Tomato Leaf Disease Classification Dataset/Tomato Dataset"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from common import IMG_EXT, SEED, assert_disjoint, dhash_variants, near_duplicate_groups, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--near-dup-threshold", type=int, default=5, help="max Hamming distance (of 64 bits)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    root, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    print("Classes:", classes)

    rows, corrupted = [], []
    for c in classes:
        for p in sorted((root / c).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    rows.append({"path": str(p), "class": c})
                except Exception as e:  # noqa: BLE001
                    corrupted.append({"path": str(p), "error": str(e)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid images found - check --data-dir")
    print(f"Valid images: {len(df)}  corrupted: {len(corrupted)}")

    # exact duplicates
    df["sha256"] = [sha256_file(p) for p in tqdm(df["path"], desc="sha256")]
    conflict = df.groupby("sha256")["class"].nunique()
    conflict_hashes = set(conflict[conflict > 1].index)
    n_conf = int(df["sha256"].isin(conflict_hashes).sum())
    df = df[~df["sha256"].isin(conflict_hashes)]
    n_exact = int(df.duplicated("sha256").sum())
    df = df.drop_duplicates("sha256").reset_index(drop=True)
    print(f"Dropped {n_conf} images with identical bytes but conflicting labels; "
          f"removed {n_exact} exact duplicates -> {len(df)} images")

    # near duplicates
    hashes = np.array([dhash_variants(p) for p in tqdm(df["path"], desc="dhash")], dtype=np.uint64)
    df["group"] = near_duplicate_groups(hashes, a.near_dup_threshold)
    sizes = df.groupby("group").size()
    multi_class = int((df.groupby("group")["class"].nunique() > 1).sum())
    print(f"Near-duplicate groups: {len(sizes)}  largest: {sizes.max()}  "
          f"groups spanning >1 class: {multi_class}")
    if sizes.max() > 0.05 * len(df):
        print("WARNING: a very large group exists - threshold may be too high. "
              "Inspect splits/groups.csv or lower --near-dup-threshold.")

    # group-disjoint, class-stratified 3-way split
    y = df["class"].values
    n_test = max(2, round(1 / a.test_frac))
    sg = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=a.seed)
    rest_idx, test_idx = next(sg.split(df, y, df["group"]))
    rest = df.iloc[rest_idx].reset_index(drop=True)
    val_share = a.val_frac / (1 - a.test_frac)
    n_val = max(2, round(1 / val_share))
    sg2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=a.seed)
    tr_idx, va_idx = next(sg2.split(rest, rest["class"].values, rest["group"]))
    parts = {"train": rest.iloc[tr_idx], "validation": rest.iloc[va_idx], "test": df.iloc[test_idx]}

    assert_disjoint(parts, "group")      # leakage check by near-dup group
    assert_disjoint(parts, "sha256")     # leakage check by exact hash
    for name, part in parts.items():
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"\n{name}: {len(part)} images\n{part['class'].value_counts().reindex(classes).to_string()}")

    df.sort_values("group").to_csv(out / "groups.csv", index=False)
    pd.DataFrame(corrupted).to_csv(out / "corrupted.csv", index=False)
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    (out / "split_report.json").write_text(json.dumps({
        "images_after_exact_dedup": len(df), "near_dup_threshold": a.near_dup_threshold,
        "groups": int(len(sizes)), "largest_group": int(sizes.max()),
        "groups_spanning_multiple_classes": multi_class, "seed": a.seed,
        "sizes": {k: len(v) for k, v in parts.items()},
        "note": "Group-disjoint by near-duplicate cluster; NOT plant-disjoint (no plant IDs in this dataset).",
    }, indent=2))
    print("\nLeakage checks passed. Splits written to", out)


if __name__ == "__main__":
    main()
