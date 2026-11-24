"""CAEP step 2 - hard leakage check + confound report. Exit code 1 on leakage.

  python check_leakage.py --manifest plant_splits/manifest_split_seed42.csv --report-cols angle,batch,day
"""
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from common import assert_disjoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--group-col", default="plant_key")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--report-cols", default="angle,batch")
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    frames = {k: v for k, v in df.groupby(a.split_col)}
    try:
        assert_disjoint(frames, a.group_col)
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
    print(f"PASS: no '{a.group_col}' appears in more than one split "
          f"({', '.join(f'{k}={v[a.group_col].nunique()} plants' for k, v in frames.items())})")
    print("\nConfound report - infected prevalence (%) and n images per split:")
    for col in a.report_cols.split(","):
        t = df.groupby([col, a.split_col])[a.label_col].agg(["mean", "size"])
        t["mean"] = (t["mean"] * 100).round(1)
        print(f"\nby {col}:\n{t.unstack(a.split_col).to_string()}")
    print("\nIf prevalence differs strongly across angle/batch/day, a model can win by learning the "
          "acquisition setting. Report metrics stratified by these factors.")


if __name__ == "__main__":
    main()
