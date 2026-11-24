"""PHASE 1 - Track A (RGB leaf images) data audit.

One file per dataset, as requested: this is the dataset-1 audit.

  .venv/bin/python analysis/dataset1_phase1_audit.py

Produces (all numbers machine-generated, nothing hand-typed):
  results/phase1/class_counts.csv            (a) counts vs the user's earlier run
  results/phase1/file_properties.csv         (b) readability, mode, format, dimensions, size
  results/phase1/file_summary.json           (b) distributions
  results/phase1/file_size_hist.png          (b)
  results/phase1/exact_duplicates.json       (c) SHA-256 duplicates + label conflicts
  results/phase1/near_dup_thresholds.csv     (d) groups / largest / multiclass per threshold
  results/phase1/near_dup_group_table.csv    (d) every group at the chosen threshold
  results/phase1/montage_matched_pairs.png   (d) 20 random matched pairs
  results/phase1/montage_unmatched_nearest_pairs.png  (d) 20 random unmatched nearest pairs
  results/phase1/ncc_agreement.csv           (d) 2nd independent method vs dhash
  results/phase1/split_recreation.json       (e) notebook split recreated + membership check
  results/phase1/split_membership.csv        (e) remapped original split per image
  results/phase1/test_nearest_train.csv      (e) nearest train image for every test/val image
  results/phase1/leakage_by_threshold.csv    (e) % test (and val) with a near-dup in train
  results/phase1/cross_label_near_duplicates.csv  (f) visually-identical pairs with DIFFERENT labels
  results/phase1/cross_label_split_overlap.csv    (f) ...and where they land in the original split
  results/phase1/label_conflict_summary.json      (f)
  results/phase1/phase1_summary.json         summary + warnings
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase1"
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

import common  # noqa: E402
from common import IMG_EXT, dhash_variants, near_duplicate_groups, sha256_file  # noqa: E402

DATA = ROOT / "Tomato Leaf Disease Classification Dataset" / "Tomato Dataset"
ORIG = Path.home() / "Downloads" / "Tomato_Research_Results"   # user's original split CSVs (outside workspace)
CLAIMS = json.loads((ROOT / "analysis" / "reference_claims.json").read_text())
THRESHOLDS_USER = [0, 2, 4, 6, 8, 10]      # exactly the thresholds the user asked to be reported
THRESHOLDS_SWEEP = list(range(0, 11))       # superset used to SELECT the grouping threshold
THRESHOLDS = THRESHOLDS_SWEEP               # internal: everything is computed over the sweep
SEED = 42

# HARD RULE 2: the folder typo is NEVER touched on disk - display labels only, in outputs.
DISPLAY = {"Bacteiral Spot": "Bacterial Spot"}
FILE_CACHE: dict = {"path": [], "class": []}


def disp(folder: str) -> str:
    return DISPLAY.get(folder, folder)


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------- (a) class counts
def a_class_counts(files: pd.DataFrame) -> dict:
    claim = CLAIMS["phase1_trackA_class_counts"]["by_display_label"]
    rows = []
    for folder, n in sorted(Counter(files["class"]).items()):
        d = disp(folder)
        rows.append({"folder_name": folder, "display_label": d, "n_images": int(n),
                     "earlier_run_count": int(claim[d]) if d in claim else None,
                     "difference": int(n) - int(claim[d]) if d in claim else None})
    extra = sorted(set(claim) - {disp(c) for c in files["class"].unique()})
    pd.DataFrame(rows).to_csv(RES / "class_counts.csv", index=False)
    summary = {"rows": rows, "total_images": int(len(files)),
               "earlier_total": CLAIMS["phase1_trackA_class_counts"]["total"],
               "total_difference": int(len(files)) - CLAIMS["phase1_trackA_class_counts"]["total"],
               "display_labels_in_claim_but_not_on_disk": extra,
               "note": "'Bacteiral Spot' folder mapped to display label 'Bacterial Spot' in outputs only "
                       "(raw data untouched, HARD RULE 2)."}
    (RES / "class_counts.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------- (b) file properties
def b_file_properties(files: pd.DataFrame) -> dict:
    recs, bad = [], []
    for p, c in zip(files["path"], files["class"]):
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                mode, fmt = im.mode, im.format
            recs.append({"path": p, "class": c, "width": int(w), "height": int(h), "mode": mode,
                         "format": fmt, "size_bytes": Path(p).stat().st_size})
        except Exception as e:  # noqa: BLE001
            bad.append({"path": p, "class": c, "error": repr(e)})
    fp = pd.DataFrame(recs)
    fp.to_csv(RES / "file_properties.csv", index=False)
    pd.DataFrame(bad).to_csv(RES / "unreadable.csv", index=False)
    size_kb = (fp["size_bytes"] / 1024).to_numpy()
    dims = fp.groupby(["width", "height"]).size().sort_values(ascending=False).head(10)
    summary = {
        "n_files_attempted": int(len(files)), "n_readable": int(len(fp)), "n_unreadable": int(len(bad)),
        "unreadable": bad[:20],
        "modes": {str(k): int(v) for k, v in fp["mode"].value_counts().items()},
        "formats": {str(k): int(v) for k, v in fp["format"].value_counts().items()},
        "n_distinct_dimensions": int(fp[["width", "height"]].drop_duplicates().shape[0]),
        "dimensions_top": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in dims.items()},
        "width_min_max": [int(fp["width"].min()), int(fp["width"].max())],
        "height_min_max": [int(fp["height"].min()), int(fp["height"].max())],
        "size_bytes_stats": {k: float(v) for k, v in
                             {"min": fp["size_bytes"].min(), "max": fp["size_bytes"].max(),
                              "mean": fp["size_bytes"].mean(), "median": fp["size_bytes"].median(),
                              "std": fp["size_bytes"].std(ddof=1)}.items()},
        "size_kb_p01_p25_p75_p99": [float(x) for x in np.percentile(size_kb, [1, 25, 75, 99])],
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(size_kb, bins=60)
    ax.set_xlabel("file size (KB)"); ax.set_ylabel("images"); ax.set_title("Track A file size distribution")
    fig.tight_layout(); fig.savefig(RES / "file_size_hist.png", dpi=150); plt.close(fig)
    (RES / "file_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------- (c) exact SHA-256 duplicates
def c_exact_duplicates(files: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    files = files.copy()
    files["sha256"] = [sha256_file(p) for p in files["path"]]
    grp = files.groupby("sha256")
    dup_hashes = {h for h, g in grp if len(g) > 1}
    conflicts = {h for h, g in grp if g["class"].nunique() > 1}
    dup_rows = files[files["sha256"].isin(dup_hashes)].sort_values(["sha256", "path"])
    dup_rows.to_csv(RES / "exact_duplicate_rows.csv", index=False)
    summary = {
        "n_images": int(len(files)), "n_unique_sha256": int(files["sha256"].nunique()),
        "n_duplicate_groups": int(len(dup_hashes)),
        "n_images_in_duplicate_groups": int(len(dup_rows)),
        "n_exact_duplicates_removed_if_dedup": int(len(dup_rows) - len(dup_hashes)),
        "n_label_conflict_hashes": int(len(conflicts)),
        "n_images_with_label_conflict": int(files["sha256"].isin(conflicts).sum()),
        "conflict_examples": dup_rows[dup_rows["sha256"].isin(conflicts)].head(20).to_dict("records"),
        "user_claim_earlier_run": "zero exact duplicates",
    }
    (RES / "exact_duplicates.json").write_text(json.dumps(summary, indent=2))
    return files, summary


# ----------------------------------------------------------- (d) near-duplicates
def blocked_min_distance(hashes: np.ndarray) -> np.ndarray:
    """(n,n) uint8: min Hamming distance between image i's ORIGINAL hash and every flip
    variant of image j - exactly the criterion common.near_duplicate_groups uses."""
    n = len(hashes)
    D = np.zeros((n, n), dtype=np.uint8)
    for start in range(0, n, 256):
        rows = hashes[start:start + 256, 0]
        best = None
        for v in range(hashes.shape[1]):
            d = common._hamming_block(rows, hashes[:, v])
            best = d if best is None else np.minimum(best, d)
        D[start:start + 256] = best
    np.fill_diagonal(D, 255)
    return D


def d_near_duplicates(files: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    hashes = np.array([dhash_variants(p) for p in files["path"]], dtype=np.uint64)
    np.save(RES / "_dhash_variants.npy", hashes)
    D = blocked_min_distance(hashes)
    np.save(RES / "_dhash_distance_matrix.npy", D)

    rows = []
    for t in THRESHOLDS:
        gid = near_duplicate_groups(hashes, t)
        sizes = pd.Series(gid).value_counts()
        per_class = files.assign(g=gid).groupby("g")["class"].nunique()
        multi = per_class[per_class > 1]
        members = files.assign(g=gid)
        n_multi_img = int(members[members["g"].isin(multi.index)].shape[0])
        rows.append({"threshold": t, "user_requested": t in THRESHOLDS_USER,
                     "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
                     "largest_group_pct": round(100 * int(sizes.max()) / len(files), 4),
                     "n_images_in_groups_gt1": int(sizes[sizes > 1].sum()),
                     "n_singleton_groups": int((sizes == 1).sum()),
                     "groups_spanning_gt1_class": int(len(multi)),
                     "n_images_in_multiclass_groups": n_multi_img,
                     "pct_images_in_multiclass_groups": round(100 * n_multi_img / len(files), 4)})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(RES / "near_dup_thresholds.csv", index=False)

    # Threshold choice uses research/01_make_splits.py's own sanity bound (largest group <=5% of
    # the dataset); take the LARGEST tested threshold that still satisfies it, so the split is as
    # leakage-safe as the data allows without the grouping degenerating into one giant cluster.
    ok = thr_df.loc[thr_df["largest_group"] <= 0.05 * len(files), "threshold"].tolist()
    chosen = int(max(ok)) if ok else int(min(THRESHOLDS))
    bound = 0.05 * len(files)
    rationale = (f"largest threshold in the sweep {THRESHOLDS_SWEEP} whose largest group stays <= 5% of "
                 f"{len(files)} images = {bound:.1f} (the same bound research/01_make_splits.py warns "
                 f"about); largest group at t={chosen} is "
                 f"{int(thr_df.loc[thr_df.threshold == chosen, 'largest_group'].iloc[0])}. "
                 f"The user-requested reporting thresholds {THRESHOLDS_USER} are a subset of this sweep.")

    gid = near_duplicate_groups(hashes, chosen)
    members = files.assign(group=gid, display_label=[disp(c) for c in files["class"]])
    members.to_csv(RES / "near_dup_all_assignments.csv", index=False)
    sizes = members.groupby("group").size().sort_values(ascending=False)
    members[members["group"].isin(sizes[sizes > 1].index)].sort_values(["group", "path"]).to_csv(
        RES / "near_dup_group_table.csv", index=False)

    info = {"thresholds_tested": THRESHOLDS, "per_threshold": thr_df.to_dict("records"),
            "chosen_threshold": chosen, "chosen_threshold_rationale": rationale,
            "chosen_n_groups": int(len(sizes)), "chosen_largest_group": int(sizes.max()),
            "chosen_n_groups_gt1": int((sizes > 1).sum()),
            "chosen_n_images_in_groups_gt1": int(sizes[sizes > 1].sum())}
    return D, members, info


def _thumb(path: str, px: int = 140) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB").resize((px, px), Image.BILINEAR)


def save_montage(pairs, title: str, out: Path, dists=None) -> int:
    if not pairs:
        (RES / f"{out.stem}_EMPTY.txt").write_text("no pairs available to render\n")
        return 0
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.15 * n), squeeze=False)
    for k, (i, j) in enumerate(pairs):
        if i == j:
            raise RuntimeError("montage pair degenerated to an identical index")
        for col, idx in ((0, i), (1, j)):
            axes[k][col].imshow(_thumb(FILE_CACHE["path"][idx]))
            axes[k][col].set_xticks([]); axes[k][col].set_yticks([])
        d = "" if dists is None else f"  d={dists[k]}"
        axes[k][0].set_ylabel(f"{disp(FILE_CACHE['class'][i])}\nvs {disp(FILE_CACHE['class'][j])}{d}",
                              fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.14, 0, 1, 0.98))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return n


def montages_and_ncc(files: pd.DataFrame, D: np.ndarray, info: dict) -> dict:
    t = info["chosen_threshold"]
    rng = np.random.default_rng(SEED)
    n = len(files)

    tri_i, tri_j = np.triu_indices(n, 1)
    dvec = D[tri_i, tri_j]

    # matched: directly within threshold AND in the same connected group (i<j)
    matched_mask = dvec <= t
    matched = list(zip(tri_i[matched_mask].tolist(), tri_j[matched_mask].tolist()))
    # unmatched nearest: the closest pairs that are NOT within threshold
    order = np.argsort(dvec, kind="stable")
    nearest_unmatched = [(int(tri_i[k]), int(tri_j[k])) for k in order if dvec[k] > t][:100]

    rng_m = rng.choice(len(matched), min(20, len(matched)), replace=False) if matched else []
    rng_u = rng.choice(len(nearest_unmatched), min(20, len(nearest_unmatched)), replace=False) \
        if nearest_unmatched else []
    pick_m = [matched[int(k)] for k in rng_m]
    pick_u = [nearest_unmatched[int(k)] for k in rng_u]
    n_m = save_montage(pick_m, f"20 random MATCHED pairs (dhash threshold t={t})",
                       RES / "montage_matched_pairs.png", [int(D[i, j]) for i, j in pick_m])
    n_u = save_montage(pick_u, "20 random NEAREST-but-UNMATCHED pairs",
                       RES / "montage_unmatched_nearest_pairs.png", [int(D[i, j]) for i, j in pick_u])
    # export the exact pairs rendered, so the montages are auditable and reproducible
    for tag, picks in (("matched", pick_m), ("unmatched_nearest", pick_u)):
        pd.DataFrame([{"pair_id": k, "i": i, "j": j, "hamming": int(D[i, j]),
                       "class_i": disp(files["class"].iloc[i]), "class_j": disp(files["class"].iloc[j]),
                       "same_class": str(files["class"].iloc[i]) == str(files["class"].iloc[j]),
                       "path_i": files["path"].iloc[i], "path_j": files["path"].iloc[j]}
                      for k, (i, j) in enumerate(picks)]).to_csv(
            RES / f"montage_{tag}_pairs.csv", index=False)

    # ---- second, independent method: NCC of 32x32 grayscale incl. flips -------------
    base = np.empty((n, 32, 32), dtype=np.float32)
    for k, p in enumerate(files["path"]):
        with Image.open(p) as im:
            g = ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.LANCZOS)
        base[k] = np.asarray(g, dtype=np.float32)
    base = base.reshape(n, -1)
    base -= base.mean(axis=1, keepdims=True)
    nv = np.linalg.norm(base, axis=1, keepdims=True); nv[nv == 0] = 1.0
    base = base / nv

    grid = base.reshape(n, 32, 32)
    variants = [base, grid[:, :, ::-1].reshape(n, -1), grid[:, ::-1, :].reshape(n, -1),
                grid[:, ::-1, ::-1].reshape(n, -1)]
    R = None
    for v in variants:
        vv = v.reshape(n, -1)
        m = np.linalg.norm(vv, axis=1, keepdims=True); m[m == 0] = 1.0
        c = (vv / m) @ (vv / m).T
        R = c if R is None else np.maximum(R, c)
    np.fill_diagonal(R, 1.0)
    np.save(RES / "_ncc_matrix.npy", R.astype(np.float16))

    rvec = R[tri_i, tri_j]
    rows = []
    for th in THRESHOLDS:
        A = dvec <= th
        for r in (0.90, 0.95, 0.98):
            B = rvec >= r
            inter = int((A & B).sum()); union = int((A | B).sum())
            rows.append({"threshold": th, "user_requested": th in THRESHOLDS_USER, "ncc_r": r,
                         "n_dhash_pairs": int(A.sum()), "n_ncc_pairs": int(B.sum()),
                         "n_intersect": inter, "n_union": union,
                         "jaccard": round(inter / union, 6) if union else None,
                         "pct_dhash_pairs_confirmed_by_ncc": round(100 * inter / A.sum(), 4) if A.sum() else None,
                         "pct_ncc_pairs_confirmed_by_dhash": round(100 * inter / B.sum(), 4) if B.sum() else None})
    pd.DataFrame(rows).to_csv(RES / "ncc_agreement.csv", index=False)

    # ---- cross-LABEL near-duplicate candidates (possible label noise), from BOTH methods ----
    cls = files["class"].to_numpy()
    recs = []
    ii, jj = np.where(np.triu(D <= t, 1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        if cls[a] != cls[b]:
            recs.append({"method": f"dhash<=chosen_t{t}", "i": a, "j": b, "hamming": int(D[a, b]),
                         "ncc_max_over_flips": round(float(R[a, b]), 6),
                         "class_i": disp(cls[a]), "class_j": disp(cls[b]),
                         "path_i": files["path"].iloc[a], "path_j": files["path"].iloc[b]})
    # independent method: NCC alone says near-identical, labels disagree (regardless of dhash)
    ni, nj = np.where(np.triu(R >= 0.998, 1))
    for a, b in zip(ni.tolist(), nj.tolist()):
        if cls[a] != cls[b]:
            recs.append({"method": "ncc>=0.998", "i": a, "j": b, "hamming": int(D[a, b]),
                         "ncc_max_over_flips": round(float(R[a, b]), 6),
                         "class_i": disp(cls[a]), "class_j": disp(cls[b]),
                         "path_i": files["path"].iloc[a], "path_j": files["path"].iloc[b]})
    pd.DataFrame(recs).to_csv(RES / "cross_label_near_duplicates.csv", index=False)

    # per-pair agreement for exactly the pairs shown in the montages (auditable)
    for tag, picks in (("matched", pick_m), ("unmatched_nearest", pick_u)):
        if not picks:
            continue
        p = RES / f"montage_{tag}_pairs.csv"
        df = pd.read_csv(p)
        df["ncc_max_over_flips"] = [round(float(R[i, j]), 6) for i, j in picks]
        df["ncc_ge_0.98"] = df["ncc_max_over_flips"] >= 0.98
        df.to_csv(p, index=False)

    info.update({"n_direct_matched_pairs_at_chosen_t": len(matched),
                 "n_nearest_unmatched_pool": len(nearest_unmatched),
                 "montage_matched_pairs_rendered": n_m, "montage_unmatched_pairs_rendered": n_u,
                 "montage_pair_sampling": "seed 42; unmatched pool = 100 smallest Hamming distances > threshold",
                 "ncc_method": "max NCC over original/h-flip/v-flip/180deg of zero-mean unit-norm 32x32 grayscale",
                 "ncc_agreement_file": "results/phase1/ncc_agreement.csv",
                 "n_cross_label_near_dup_rows": int(len(recs)),
                 "n_cross_label_by_dhash": int(sum(1 for r in recs if r["method"].startswith("dhash"))),
                 "n_cross_label_by_ncc_alone": int(sum(1 for r in recs if r["method"] == "ncc>=0.998")),
                 "cross_label_file": "results/phase1/cross_label_near_duplicates.csv"})
    return info


# ------------------------------------------------------- (e) original-split leakage
def _order_variants(allp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Candidate row orders for re-running the notebook's train_test_split - membership depends
    on row order, and the notebook's rglob() order is filesystem-dependent."""
    out = {}
    out["sorted_class_then_path"] = allp.sort_values(["class", "path"]).reset_index(drop=True)

    def natkey(s: str):
        # tag each token so int/str never collide during comparison
        return [(0, int(x)) if x.isdigit() else (1, x)
                for x in re.split(r"(\d+)", s) if x != ""]
    out["sorted_class_then_natural_filename"] = allp.assign(
        _k=allp["path"].map(lambda p: (Path(p).parent.name, natkey(Path(p).name)))).sort_values(
        "_k").drop(columns="_k").reset_index(drop=True)

    rows = []
    for c in sorted({Path(p).parent.name for p in allp["path"]}):
        d = DATA / c
        if d.is_dir():
            for p in sorted(d.iterdir(), key=lambda x: x.name):   # scandir-like, name order here
                if p.is_file():
                    rows.append(str(p))
    m = {p: i for i, p in enumerate(rows)}
    out["filesystem_dir_order"] = allp[allp["path"].isin(m)].assign(
        _k=allp["path"].map(m)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return out


def e_original_split(files: pd.DataFrame) -> dict:
    out: dict = {}

    # ---- 1. remap the stale absolute paths (uniqueness asserted) -------------------
    idx = defaultdict(list)
    for p, c in zip(files["path"], files["class"]):
        idx[(c, Path(p).name)].append(p)
    parts, rows = {}, []
    for name in ("train", "validation", "test"):
        df = pd.read_csv(ORIG / f"{name}.csv")
        resolved, missing, ambig = [], 0, 0
        for _, r in df.iterrows():
            cands = idx.get((r["class"], Path(r["path"]).name), [])
            if len(cands) == 1:
                resolved.append(cands[0])
            elif len(cands) == 0:
                missing += 1; resolved.append(None)
            else:
                ambig += 1; resolved.append(None)
        keep = df.drop(columns=["path"]).rename(columns={"path": "original_stale_path"})
        keep.insert(0, "path", resolved)
        keep["split"] = name
        parts[name] = keep
        rows.append({"split": name, "n_rows": int(len(keep)),
                     "resolved_unique": int(len(keep) - missing - ambig),
                     "missing": int(missing), "ambiguous": int(ambig)})
    allp = pd.concat(parts.values(), ignore_index=True)
    out["path_remap"] = {"per_split": rows, "n_rows_total": int(len(allp)),
                         "n_paths_null": int(allp["path"].isna().sum())}
    if allp["path"].isna().any():
        raise SystemExit("FATAL: original split rows could not be remapped - stopping (HARD RULE 6)")

    # integrity: the sha256 the notebook recorded must still match the bytes on disk
    fresh = dict(zip(files["path"], files["sha256"]))
    allp["sha256_recomputed"] = allp["path"].map(fresh)
    match = int((allp["hash"] == allp["sha256_recomputed"]).sum())
    out["path_remap"]["sha256_in_csv_equals_recomputed"] = match
    out["path_remap"]["sha256_mismatch"] = int(len(allp) - match)
    allp.to_csv(RES / "split_membership.csv", index=False)
    # propagate the recomputed hash back onto each split frame (allp is a concat, i.e. a copy)
    sha = dict(zip(allp["path"], allp["sha256_recomputed"]))
    for k in parts:
        parts[k]["sha256_recomputed"] = parts[k]["path"].map(sha)

    # ---- 2. independently recreate the notebook's split --------------------------
    claim = CLAIMS["phase1_original_split_sizes"]
    recreate = {}
    for oname, frame in _order_variants(allp).items():
        tr, temp = train_test_split(frame, test_size=0.30, stratify=frame["class"], random_state=42)
        va, te = train_test_split(temp, test_size=0.50, stratify=temp["class"], random_state=42)
        got = {"train": set(tr["path"]), "validation": set(va["path"]), "test": set(te["path"])}
        recreate[oname] = {
            "sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
            "sizes_match_claim": {"train": len(tr) == claim["train"],
                                  "validation": len(va) == claim["validation"],
                                  "test": len(te) == claim["test"]},
            "membership_identical_to_original": {k: got[k] == set(parts[k]["path"])
                                                 for k in ("train", "validation", "test")},
            "n_membership_symmetric_diff": {k: len(got[k] ^ set(parts[k]["path"]))
                                            for k in ("train", "validation", "test")},
        }
    recreate["_claim"] = claim
    recreate["_note"] = ("stratified split SIZES are order-independent; MEMBERSHIP depends on the row order "
                         "of the notebook's rglob() walk, which is filesystem-dependent")
    out["split_recreation"] = recreate
    (RES / "split_recreation.json").write_text(json.dumps(recreate, indent=2, default=float))

    # ---- 3. exact-hash leakage across the ORIGINAL split -------------------------
    tr_h = set(parts["train"]["sha256_recomputed"])
    exact = {}
    for k in ("validation", "test"):
        s = parts[k]["sha256_recomputed"]
        exact[k] = {"n": int(len(s)), "n_with_exact_dup_in_train": int(s.isin(tr_h).sum()),
                    "pct": round(100 * float(s.isin(tr_h).mean()), 4)}
    out["exact_hash_leakage_original_split"] = exact

    # ---- 4. nearest TRAIN image for every test / validation image ----------------
    path2i = {p: i for i, p in enumerate(files["path"])}
    D = np.load(RES / "_dhash_distance_matrix.npy")
    tr_paths = list(parts["train"]["path"])
    tr_idx = np.array([path2i[p] for p in tr_paths])
    tr_cls = list(parts["train"]["class"])
    recs = []
    for k in ("test", "validation"):
        q_paths = list(parts[k]["path"])
        q_idx = np.array([path2i[p] for p in q_paths])
        sub = D[np.ix_(q_idx, tr_idx)]
        arg = sub.argmin(axis=1)
        dist = sub[np.arange(len(q_idx)), arg]
        for qi in range(len(q_idx)):
            recs.append({"split": k, "query_path": q_paths[qi],
                         "nearest_train_path": tr_paths[int(arg[qi])],
                         "hamming_min_over_flips": int(dist[qi]),
                         "same_class": str(files["class"].iloc[q_idx[qi]]) == str(tr_cls[int(arg[qi])])})
        out.setdefault("nearest_train", {})[k] = {
            "n": int(len(q_idx)), "min": int(dist.min()), "median": float(np.median(dist)),
            "mean": round(float(dist.mean()), 4), "max": int(dist.max())}
    pd.DataFrame(recs).to_csv(RES / "test_nearest_train.csv", index=False)

    # ---- 5. % with a near-duplicate in train, per threshold ----------------------
    leak_rows = []
    by_split = {"test": [], "validation": []}
    for r in recs:
        by_split[r["split"]].append(r)
    for t in THRESHOLDS:
        row = {"threshold": t}
        for k in ("test", "validation"):
            hit = sum(1 for r in by_split[k] if r["hamming_min_over_flips"] <= t)
            row[f"n_{k}"] = len(by_split[k])
            row[f"n_{k}_near_dup_in_train"] = hit
            row[f"pct_{k}_near_dup_in_train"] = round(100 * hit / len(by_split[k]), 4)
        row["pct_diff_test_minus_validation"] = round(
            row["pct_test_near_dup_in_train"] - row["pct_validation_near_dup_in_train"], 4)
        leak_rows.append(row)
    pd.DataFrame(leak_rows).to_csv(RES / "leakage_by_threshold.csv", index=False)
    out["leakage_by_threshold"] = leak_rows
    (RES / "leakage.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def f_label_conflict_split_overlap() -> dict:
    """Where do the cross-label near-duplicate candidates land in the ORIGINAL split?
    A visually identical pair split across train/test is simultaneously leakage and unlearnable."""
    cross = pd.read_csv(RES / "cross_label_near_duplicates.csv")
    mem = pd.read_csv(RES / "split_membership.csv")
    s2split = dict(zip(mem["path"], mem["split"]))
    rows = []
    for _, r in cross.iterrows():
        si, sj = s2split.get(r["path_i"]), s2split.get(r["path_j"])
        rows.append({"method": r["method"], "class_i": r["class_i"], "class_j": r["class_j"],
                     "hamming": r["hamming"], "ncc_max_over_flips": r["ncc_max_over_flips"],
                     "split_i": si, "split_j": sj,
                     "same_split": si == sj and si is not None,
                     "crosses_split_boundary": si is not None and sj is not None and si != sj,
                     "involves_test": "test" in (si, sj), "involves_validation": "validation" in (si, sj),
                     "path_i": r["path_i"], "path_j": r["path_j"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cross_label_split_overlap.csv", index=False)
    hi = df[df["method"] == "ncc>=0.998"]
    out = {
        "n_rows_total": int(len(df)),
        "n_rows_dhash": int((df["method"].str.startswith("dhash")).sum()),
        "n_rows_ncc_high_conf": int(len(hi)),
        "n_distinct_images_involved_all": int(len(set(df["path_i"]) | set(df["path_j"]))),
        "n_distinct_images_involved_ncc_high_conf": int(len(set(hi["path_i"]) | set(hi["path_j"]))),
        "n_cross_label_pairs_crossing_a_split_boundary": int(df["crosses_split_boundary"].sum()),
        "n_cross_label_pairs_ncc_high_conf_crossing_boundary": int(hi["crosses_split_boundary"].sum()),
        "n_cross_label_pairs_ncc_high_conf_within_test": int(hi["involves_test"].sum()),
        "n_cross_label_pairs_ncc_high_conf_within_validation": int(hi["involves_validation"].sum()),
        "interpretation": "visually near-identical images (NCC>=0.998 on 32x32 grayscale, incl. flips) "
                           "carrying DIFFERENT class labels. SHA-256 reported zero duplicates because the "
                           "bytes differ (re-encoding); the label conflict is only visible after perceptual "
                           "comparison. These are label-noise candidates, not yet proven annotations errors.",
    }
    (RES / "label_conflict_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    global FILE_CACHE
    t0 = time.time()
    files = []
    for c in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        for p in sorted((DATA / c).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                files.append({"path": str(p), "class": c})
    files = pd.DataFrame(files)
    if files.empty:
        raise SystemExit("no images found under " + str(DATA))
    log(f"[a] {len(files)} image files found")
    FILE_CACHE = {"path": files["path"].tolist(), "class": files["class"].tolist()}

    summary: dict = {"phase": 1, "data_dir": str(DATA), "thresholds_tested": THRESHOLDS,
                     "seed": SEED, "artifacts_dir": str(RES)}

    summary["a_class_counts"] = a_class_counts(files)
    log("[b] file properties...")
    summary["b_file_properties"] = b_file_properties(files)
    log("[c] sha256 exact duplicates...")
    files, summary["c_exact_duplicates"] = c_exact_duplicates(files)
    log("[d] dhash near-duplicates + montages + NCC (this is the slowest step)...")
    D, _members, info = d_near_duplicates(files)
    summary["d_near_duplicates"] = montages_and_ncc(files, D, info)
    log("[e] original-split leakage...")
    summary["e_original_split"] = e_original_split(files)
    log("[f] cross-label near-duplicate vs original split...")
    summary["f_label_conflicts"] = f_label_conflict_split_overlap()

    summary["runtime_s"] = round(time.time() - t0, 1)
    (RES / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {summary['runtime_s']}s -> {RES}")

    # ---- run log (HARD RULE 5): command, runtime, artifacts, decided threshold ----
    import datetime
    runlog = ROOT / "results" / "run_log.md"
    old = runlog.read_text() if runlog.exists() else "# Run log\n"
    marker = "## Phase 1 - Track A data audit"
    if marker in old:          # block-scoped: remove ONLY our previous entry (CHANGELOG)
        i = old.index(marker)
        j = old.find("\n## ", i + 1)
        head = old[:i].rstrip("\n")
        tail = "" if j == -1 else old[j + 1:]
        old = head + ("\n\n" + tail if tail else "\n")
    entry = [
        f"\n{marker}\n",
        f"- time (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "- command: `.venv/bin/python analysis/dataset1_phase1_audit.py`",
        f"- runtime: {summary['runtime_s']} s",
        f"- images scanned: {summary['a_class_counts']['total_images']}",
        f"- near-duplicate thresholds tested: {THRESHOLDS}",
        f"- **chosen near-dup threshold: {info['chosen_threshold']}** - {info['chosen_threshold_rationale']}",
        f"- artifacts: results/phase1/ (summary = phase1_summary.json)",
        "",
    ]
    runlog.write_text(old + "\n".join(entry))

    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
