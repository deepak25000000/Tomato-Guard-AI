"""CAEP step 1 - plant-disjoint, label+batch-stratified splits (all angles/days of a plant stay together).

  python split_by_plant.py --manifest manifest.csv --seeds 42,43,44,45,46 --out-dir plant_splits
Writes one manifest_split_seed<k>.csv per seed (adds a 'split' column) - use several seeds because
167 plants is a small sample and a single split can be lucky or unlucky.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(plants: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> dict:
    strat = plants["label"].astype(str) + "_" + plants["batch"].astype(str)
    if strat.value_counts().min() < 3:       # too few plants in a cell -> stratify by label only
        strat = plants["label"].astype(str)
    rest, test = train_test_split(plants, test_size=test_frac, stratify=strat, random_state=seed)
    s2 = rest["label"].astype(str) + "_" + rest["batch"].astype(str)
    if s2.value_counts().min() < 2:
        s2 = rest["label"].astype(str)
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac), stratify=s2, random_state=seed)
    out = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        out.update({k: name for k in part["plant_key"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="plant_splits")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    plants = df.drop_duplicates("plant_key")[["plant_key", "label", "batch"]].reset_index(drop=True)
    print(f"{len(plants)} plants; healthy/infected = {plants['label'].value_counts().sort_index().to_dict()}")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        mapping = make_split(plants, a.test_frac, a.val_frac, s)
        d = df.copy(); d["split"] = d["plant_key"].map(mapping)
        d.to_csv(Path(a.out_dir) / f"manifest_split_seed{s}.csv", index=False)
        print(f"seed {s}: plants per split =",
              d.drop_duplicates("plant_key")["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
