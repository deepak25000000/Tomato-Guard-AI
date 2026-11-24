"""Re-run the paper's TOBRFV-LMID audit on YOUR copy of the data (Section V-A) and check it against the
published numbers. Input = manifest.csv from build_manifest.py (one row per plant/day/angle with one
path column per stream).

  python audit_tobrfv.py --manifest manifest.csv --streams rgb,vnir,nir800,nir1000 --out audit_out

Computes, per stream: image-level and plant-level Cohen's d (healthy minus infected, plant-level with
bootstrap 95% CI over plants), infected prevalence by angle / batch / day, d within strata (angle, batch,
mixed-label days), and the uncalibrated NDVI-like proxy drift (Spearman vs day).
Nothing here is a classifier - it is the confound audit.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


def load_stats(path):
    """Mean normalised intensity (all bands) and mean of band 0 (red when the image is RGB)."""
    with Image.open(path) as im:
        if im.mode in ("P", "RGBA", "LA", "CMYK"):
            im = im.convert("RGB")
        arr = np.asarray(im)
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0 if arr.dtype == np.uint8 else (float(arr.max()) or 1.0)
    a = arr.astype(np.float64) / scale
    return float(a.mean()), float(a[..., 0].mean() if a.ndim == 3 else a.mean())


def cohen_d(healthy, infected):
    h, i = np.asarray(healthy, float), np.asarray(infected, float)
    if len(h) < 2 or len(i) < 2:
        return float("nan")
    sp = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(i) - 1) * i.var(ddof=1)) / (len(h) + len(i) - 2))
    return float((h.mean() - i.mean()) / sp) if sp > 0 else float("nan")


def d_of(frame, col):
    return cohen_d(frame.loc[frame["label"] == 0, col], frame.loc[frame["label"] == 1, col])


def plant_level(frame, col):
    return frame.groupby("plant_key").agg(label=("label", "first"), v=(col, "mean"))


def plant_d_boot(frame, col, n=2000, seed=42):
    pl = plant_level(frame, col)
    h, i = pl.loc[pl.label == 0, "v"].values, pl.loc[pl.label == 1, "v"].values
    rng = np.random.default_rng(seed)
    bs = [cohen_d(rng.choice(h, len(h)), rng.choice(i, len(i))) for _ in range(n)]
    return cohen_d(h, i), np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(h), len(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--streams", default="rgb,nir800,nir1000")
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.manifest)
    streams = a.streams.split(",")
    cache = out / "intensities.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print("loaded cached intensities from", cache)
    else:
        for s in streams:
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(load_stats, df[s].tolist()))
            df[f"{s}_mean"] = [r[0] for r in res]
            df[f"{s}_red"] = [r[1] for r in res]
            print("done stream", s)
        df.to_csv(cache, index=False)

    R = {"n_rows_plant_day_angle": int(len(df)), "n_plants": int(df["plant_key"].nunique()),
         "plants_healthy_infected": df.drop_duplicates("plant_key")["label"].value_counts().sort_index().to_dict(),
         "images_per_stream": int(len(df)), "images_total": int(len(df) * len(streams)),
         "images_healthy_infected_per_stream": df["label"].value_counts().sort_index().to_dict()}

    prev = {}
    for col in ("angle", "batch"):
        g = df.groupby(col)["label"].agg(["sum", "size"])
        prev[col] = {str(k): {"infected": int(r["sum"]), "n": int(r["size"]), "pct": round(100 * r["sum"] / r["size"], 1)}
                     for k, r in g.iterrows()}
    both = df.groupby("day")["label"].nunique()
    mixed_days = sorted(both[both == 2].index.tolist())
    prev["mixed_label_days"] = mixed_days
    prev["day_prevalence_pct"] = (df.groupby("day")["label"].mean() * 100).round(1).to_dict()
    R["prevalence"] = prev

    R["cohens_d_healthy_minus_infected"] = {}
    for s in streams:
        c = f"{s}_mean"
        pd_, ci, nh, ni = plant_d_boot(df, c, a.boot)
        sd = {"image_level": d_of(df, c), "plant_level": pd_, "plant_level_95ci": ci,
              "n_plants_healthy": nh, "n_plants_infected": ni}
        mdf = df[df["day"].isin(mixed_days)]
        sd["mixed_days_image_level"] = d_of(mdf, c)
        sd["mixed_days_plant_level"] = plant_d_boot(mdf, c, 200)[0]
        sd["by_angle_image_level"] = {str(k): d_of(g, c) for k, g in df.groupby("angle")}
        sd["by_batch_image_level"] = {str(k): d_of(g, c) for k, g in df.groupby("batch")}
        R["cohens_d_healthy_minus_infected"][s] = sd

    if {"rgb", "nir800"} <= set(streams):
        px = df.assign(ndvi=lambda x: (x["nir800_mean"] - x["rgb_red"]) / (x["nir800_mean"] + x["rgb_red"] + 1e-9))
        px = px[px["day"].isin(mixed_days)].groupby(["plant_key", "label", "day"], as_index=False)["ndvi"].mean()
        nd = {}
        for lab, name in ((0, "healthy"), (1, "infected")):
            sub = px[px["label"] == lab]
            per = [spearmanr(g["day"], g["ndvi"])[0] for _, g in sub.groupby("plant_key") if len(g) >= 4]
            pooled = spearmanr(sub["day"], sub["ndvi"])[0] if len(sub) > 3 else float("nan")
            nd[name] = {"median_per_plant_spearman": float(np.nanmedian(per)) if per else None,
                        "pooled_spearman": float(pooled), "n_plants": len(per)}
        nd["caveat"] = "uncalibrated red(RGB) vs 800 nm from different cameras: a drift indicator only"
        R["ndvi_like_proxy"] = nd

    (out / "audit_summary.json").write_text(json.dumps(R, indent=2, default=float))
    rows = [{"stream": s, "level": lv, "d": R["cohens_d_healthy_minus_infected"][s][k]}
            for s in streams for lv, k in (("image", "image_level"), ("plant", "plant_level"))]
    t = pd.DataFrame(rows).pivot(index="stream", columns="level", values="d").loc[streams]
    t.to_csv(out / "cohens_d_table.csv")
    ax = t.plot.bar(figsize=(7, 4)); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("Cohen's d (healthy - infected)"); plt.tight_layout(); plt.savefig(out / "cohens_d.png", dpi=200)
    print(json.dumps(R, indent=2, default=float))
    print("\nCOMPARE WITH THE PAPER:\n"
          "  images/stream 4,808 | plants 167 (77 healthy / 90 infected) | infected prevalence 68.0% @0deg, 52.1% @5deg, 52.2% @15deg\n"
          "  64.6% batch I vs 60.0% batch II | 11 mixed-label days (9,11,...,29)\n"
          "  plant-level d: RGB +0.543, VNIR +0.647, 800nm -0.014, 1000nm -0.897 | image-level +0.296, +0.281, -0.012, -0.304\n"
          "Differences mean either a parsing issue (check the regex / plant key) or the paper needs correcting - find out which.")


if __name__ == "__main__":
    main()
