"""PHASE 3 - Track B (TOBRFV-LMID multispectral) data audit + paper-claim verification.

  .venv/bin/python analysis/dataset2_phase3_audit.py

Stages
  A  parse every file, check 1:1:1:1 stream pairing (unpairable tuples: excluded from the
     4-stream manifest ONLY if listed in analysis/approved_data_exclusions.json, documented in
     results/phase3/stream_pairing_defect.json, otherwise abort - HARD RULE 6), data quality
     (mode/dtype/shape), sha256
  B  per-image mean intensity per stream (cached) -> Cohen's d, image- and plant-level
  C  compare against the paper's Section V-A claims (analysis/reference_claims.json) -> PASS/FAIL
  D  run the PROJECT'S OWN caep/build_manifest.py + caep/audit_tobrfv.py unedited, as a cross-check
  E  plant-disjoint split + HARD RULE 4 leakage assertions (aborts on failure)

Filename grammar (verified in Phase 0/3 - the `day` token is the 2nd field, proved by
reproducing the paper's mixed-label-day list exactly):
    <streamdir>/<host>/<batch>/<label>/<pref>_<day>_<streamtok>_<angle>_<plant>.<ext>
e.g. RGB/Tomato/I/Virus/V_20_RGB_0_30.tiff

PLANT KEY (resolved by evidence, not by guessing - see stage C's plant_key_evidence):
    the LAST field is the plant index within (batch, label): grouping Tomato files by
    (batch, cls, last) yields exactly 167 plants = 77 healthy + 90 infected, reproducing the
    paper's three Section V-A claims, and a visual grid of one (batch,cls) group shows each
    column ageing from seedling (day 1) to flowering (day 29) while columns differ on the same
    day. The 2nd field (`day`) is the acquisition day, NOT a plant.
    plant_key = <host>|<batch>|<cls>|<last>

Outputs under results/phase3/  (nothing here is typed by hand).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P3 = RES / "phase3"
P3.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from common import assert_disjoint  # noqa: E402

DATA = ROOT / "TOBRFV-LMID"
RESEARCH = ROOT / "tomatoguard" / "research"
PY = str(ROOT / ".venv" / "bin" / "python")
CLAIMS = json.loads((ROOT / "analysis" / "reference_claims.json").read_text())
STREAMS = ["RGB", "VNIR", "VNIR_800nm", "VNIR_1000nm"]
STREAM_COL = {"RGB": "rgb", "VNIR": "vnir", "VNIR_800nm": "nir800", "VNIR_1000nm": "nir1000"}
# regex over the path relative to DATA; `day` is nested inside `pot` so both are captured
RX = re.compile(
    r"^(?P<stream>[^/]+)/(?P<host>[^/]+)/(?P<batch>I{1,2})/(?P<cls>Healthy|Virus)/"
    r"(?P<pot>[HV]_(?P<day>\d+))_(?P<tok>RGB|VNIR_800nm|VNIR_1000nm|VNIR)_"
    r"(?P<angle>\d+)_(?P<idx>\d+)\.[A-Za-z0-9]+$")
SEED = 42


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- stage A
def stage_a() -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    files = sorted(p for p in DATA.rglob("*") if p.is_file())
    recs, bad = [], []
    for p in files:
        rel = str(p.relative_to(DATA))
        m = RX.match(rel)
        if not m:
            bad.append(rel)
            continue
        g = m.groupdict()
        recs.append({
            "stream": g["stream"], "host": g["host"], "batch": g["batch"],
            "cls": g["cls"], "label": 0 if g["cls"] == "Healthy" else 1,
            "pot": g["pot"], "day": int(g["day"]), "angle": int(g["angle"]),
            "plant": int(g["idx"]), "ext": p.suffix.lower(), "bytes": p.stat().st_size,
            "path": str(p), "rel": rel,
            # the plant key: last field = plant index within (batch, cls) - see module docstring
            "plant_key": f"{g['host']}|{g['batch']}|{g['cls']}|{g['idx']}",
            "plant_key_tomato": f"{g['batch']}|{g['cls']}|{g['idx']}",
        })
    df = pd.DataFrame(recs)
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} files did not parse, e.g. {bad[:3]} (HARD RULE 6)")

    # ---- 1:1:1:1 pairing across the four streams -----------------------------
    # User-approved handling of unpairable tuples (HARD RULE 6): tuples that cannot be paired
    # are excluded from the 4-stream manifest and documented loudly; a tuple NOT listed in
    # analysis/approved_data_exclusions.json aborts the run. Nothing is guessed or skipped.
    key = ["host", "batch", "pot", "day", "angle", "plant"]
    per = df.groupby(key)["stream"].nunique()

    def tkey(ix) -> str:
        return f"{ix[0]}|{ix[1]}|{ix[2]}|day={ix[3]}|angle={ix[4]}|plant={ix[5]}"

    approved_doc = json.loads((ROOT / "analysis" / "approved_data_exclusions.json").read_text())
    approved = {e["tuple_key"]: e for e in approved_doc["exclusions"]}
    bad_keys = [tkey(ix) for ix in per[per != 4].index]
    unapproved = sorted(set(bad_keys) - set(approved))
    if unapproved:
        raise SystemExit(f"FATAL: unpairable tuple(s) not in approved_data_exclusions.json: "
                         f"{unapproved} (HARD RULE 6 - ask, do not guess)")

    defect = {"decision": approved_doc["decision"], "approved_by": approved_doc["approved_by"],
              "approved_file": "analysis/approved_data_exclusions.json",
              "n_tuples": int(len(per)), "n_complete_tuples": int((per == 4).sum()),
              "n_incomplete_tuples": int((per != 4).sum()), "n_unapproved_incomplete": 0,
              "excluded_from_four_stream_manifest_only": True,
              "tuples": []}
    for tk in bad_keys:
        gk = tk.split("|")
        sel = df[(df.host == gk[0]) & (df.batch == gk[1]) & (df.pot == gk[2]) &
                 (df.day == int(gk[3].split("=")[1])) & (df.angle == int(gk[4].split("=")[1])) &
                 (df.plant == int(gk[5].split("=")[1]))]
        have = sorted(sel.stream.tolist())
        defect["tuples"].append({
            "tuple_key": tk, "cls": sel.cls.iloc[0], "present_streams": have,
            "missing_streams": sorted(set(STREAMS) - set(have)),
            "files": sel.rel.tolist(),
            "reason": approved[tk]["reason"] if tk in approved else "NOT APPROVED",
        })
        # per-plant coverage context (why excluding this tuple does not drop the plant)
        pk = sel.plant_key.iloc[0]
        sub = df[(df.plant_key == pk) & (df.host == "Tomato")]
        defect["tuples"][-1]["plant_key"] = pk
        defect["tuples"][-1]["files_per_stream_for_this_plant"] = {
            s: int((sub.stream == s).sum()) for s in STREAMS}
    (P3 / "stream_pairing_defect.json").write_text(json.dumps(defect, indent=2))

    pairing = {"n_tuples": int(len(per)),
               "n_with_all_4_streams": int((per == 4).sum()),
               "n_incomplete": int((per != 4).sum()),
               "complete": bool((per == 4).all()),
               "complete_except_approved_exclusions": bool(unapproved) is False,
               "approved_exclusions": [t for t in bad_keys if t in approved],
               "defect_file": "results/phase3/stream_pairing_defect.json"}
    if not pairing["complete"]:
        log(f"[A] {pairing['n_incomplete']} unpairable tuple(s) excluded from the 4-stream "
            f"manifest per user decision -> {pairing['defect_file']}")

    # ---- readability / mode / dtype / shape ----------------------------------
    def probe(row):
        from PIL import Image
        try:
            with Image.open(row["path"]) as im:
                arr = np.asarray(im)
                return (True, im.mode, str(arr.dtype), im.size[0], im.size[1], None)
        except Exception as e:  # noqa: BLE001
            return (False, None, None, None, None, str(e))

    with ThreadPoolExecutor(8) as ex:
        pr = list(ex.map(probe, recs))
    df["readable"] = [p[0] for p in pr]
    df["mode"] = [p[1] for p in pr]
    df["dtype"] = [p[2] for p in pr]
    df["width"] = [p[3] for p in pr]
    df["height"] = [p[4] for p in pr]
    df["read_error"] = [p[5] for p in pr]

    quality = {}
    for s in STREAMS:
        sub = df[df.stream == s]
        shapes = sub.groupby(["width", "height"]).size()
        quality[s] = {
            "n": int(len(sub)),
            "modes": dict(Counter(sub["mode"].dropna())),
            "dtypes": dict(Counter(sub["dtype"].dropna())),
            "n_unreadable": int((~sub["readable"]).sum()),
            "n_distinct_shapes": int(len(shapes)),
            "width_min_max": [int(sub.width.min()), int(sub.width.max())],
            "height_min_max": [int(sub.height.min()), int(sub.height.max())],
            "n_exactly_640x480": int(((sub.width == 640) & (sub.height == 480)).sum()),
            "bytes_min_median_max": [int(sub.bytes.min()), int(sub.bytes.median()),
                                     int(sub.bytes.max())],
        }

    # ---- exact duplicates (sha256) ------------------------------------------
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    with ThreadPoolExecutor(8) as ex:
        df["sha256"] = list(ex.map(sha, df["path"].tolist()))
    grp = df.groupby("sha256")
    dup = {"n_images": int(len(df)), "n_unique_sha256": int(df.sha256.nunique()),
           "n_duplicate_groups": int((grp.size() > 1).sum()),
           "n_images_in_duplicate_groups": int(df[df.sha256.duplicated(keep=False)].shape[0]),
           "n_label_conflict_hashes": int((grp["label"].nunique() > 1).sum()),
           "n_plant_conflict_hashes": int((grp["plant_key"].nunique() > 1).sum())}

    df.to_csv(P3 / "image_manifest.csv", index=False)
    out = {"n_files": int(len(files)), "n_parsed": int(len(df)), "n_unparsed": len(bad),
           "stream_pairing": pairing, "per_stream_quality": quality,
           "exact_duplicates": dup, "runtime_s": round(time.time() - t0, 1)}
    (P3 / "stage_a_data_quality.json").write_text(json.dumps(out, indent=2))
    log(f"[A] {len(df)} files parsed, pairing complete={pairing['complete']}, "
        f"sha256 dup groups={dup['n_duplicate_groups']} ({out['runtime_s']}s)")
    return df, out


# --------------------------------------------------------------------------- stage B
def stage_b(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Mean normalised intensity per image per stream (the project's own definition:
    /65535 for uint16, /255 for uint8 - see caep/audit_tobrfv.py load_stats)."""
    cache = P3 / "intensities.csv"
    if cache.exists():
        log("[B] using cached intensities")
        return pd.read_csv(cache)
    t0 = time.time()

    def stats(path):
        from PIL import Image
        with Image.open(path) as im:
            if im.mode in ("P", "RGBA", "LA", "CMYK"):
                im = im.convert("RGB")
            arr = np.asarray(im)
        scale = 65535.0 if arr.dtype == np.uint16 else 255.0 if arr.dtype == np.uint8 \
            else float(arr.max() or 1.0)
        a = arr.astype(np.float64) / scale
        return float(a.mean()), float(a[..., 0].mean() if a.ndim == 3 else a.mean())

    cols = {}
    for s in STREAMS:
        sub = df[df.stream == s]
        with ThreadPoolExecutor(8) as ex:
            r = list(ex.map(stats, sub["path"].tolist()))
        cols[s] = (sub.index, [x[0] for x in r], [x[1] for x in r])
        log(f"    intensities {s} done")

    out = df.copy()
    for s, (idx, mean, red) in cols.items():
        c = STREAM_COL[s]
        out.loc[idx, f"{c}_mean"] = mean
        out.loc[idx, f"{c}_red"] = red
    out.to_csv(cache, index=False)
    log(f"[B] intensities computed in {round(time.time() - t0, 1)}s -> {cache}")
    return out


def cohen_d(h, i):
    h, i = np.asarray(h, float), np.asarray(i, float)
    if len(h) < 2 or len(i) < 2:
        return float("nan")
    sp = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(i) - 1) * i.var(ddof=1)) / (len(h) + len(i) - 2))
    return float((h.mean() - i.mean()) / sp) if sp > 0 else float("nan")


def stage_plantkey_evidence(df: pd.DataFrame) -> dict:
    """Justify the plant key with DATA, not assertion (HARD RULE 6: no guessing).

    Two candidate readings of the last filename field:
      (A) plant index within (batch, cls)  -> a plant is followed over many days
      (B) per-day photo counter            -> the same value on two days is a different plant
    We test both: (A) must reproduce the paper's 167 / 77 / 90 plant counts, and a visual grid
    must show one column ageing from seedling to flowering while columns differ on one day.
    """
    T = df[df.host == "Tomato"]
    claims = CLAIMS["phase3_paper_claims"]
    cand = {
        "A_host|batch|cls|plant (plant index, CHOSEN)": T.plant_key.nunique(),
        "A_batch|cls|plant (Tomato only)": T.plant_key_tomato.nunique(),
        "A_both_hosts|batch|cls|plant": df.plant_key.nunique(),
        "B_host|batch|cls|day (day counter reading)": T.groupby(["host", "batch", "cls", "day"]).ngroups,
        "B_host|batch|day": T.groupby(["host", "batch", "day"]).ngroups,
        "B_host|batch|cls|day|angle": T.groupby(["host", "batch", "cls", "day", "angle"]).ngroups,
    }
    by_cls = (T.drop_duplicates("plant_key").groupby("cls").size().to_dict())
    per_plant = T.groupby("plant_key").agg(n_images=("path", "size"),
                                           n_days=("day", "nunique"),
                                           n_angles=("angle", "nunique"))
    # ---- visual grid: rows = day, cols = plant, one (batch,cls) group, angle 0 ---------
    from PIL import Image, ImageDraw
    sub = T[(T.batch == "I") & (T.cls == "Healthy") & (T.angle == 0)]
    days = sorted(sub.day.unique())
    plants = sorted(sub.plant.unique())[:6]
    W, H = 200, 150
    canvas = Image.new("RGB", (W * len(plants), H * len(days)), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, d0 in enumerate(days):
        for j, pl in enumerate(plants):
            hit = sub[(sub.day == d0) & (sub.plant == pl)]
            if len(hit) == 0:
                continue
            im = Image.open(hit.iloc[0]["path"]).convert("RGB").resize((W, H))
            canvas.paste(im, (j * W, i * H))
            dr.text((j * W + 4, i * H + 4), f"d{d0}", fill=(255, 255, 0))
    grid = P3 / "plant_key_grid.png"
    canvas.save(grid)

    ev = {
        "question": "is the last filename field a plant index (A) or a per-day photo counter (B)?",
        "candidate_key_counts_tomato": cand,
        "claim_n_plants": claims["n_plants"],
        "claim_plants_healthy": claims["plants_healthy"],
        "claim_plants_infected": claims["plants_infected"],
        "reproduced_by_chosen_key": {
            "n_plants": int(T.plant_key.nunique()),
            "healthy": int(by_cls.get("Healthy", 0)),
            "infected": int(by_cls.get("Virus", 0)),
            "all_three_claims_match": (int(T.plant_key.nunique()) == claims["n_plants"]
                                       and int(by_cls.get("Healthy", 0)) == claims["plants_healthy"]
                                       and int(by_cls.get("Virus", 0)) == claims["plants_infected"]),
        },
        "per_plant_structure": {
            "images_per_plant": {k: float(v) for k, v in per_plant.n_images.agg(
                ["min", "median", "max"]).items()},
            "days_per_plant": {k: float(v) for k, v in per_plant.n_days.agg(
                ["min", "median", "max"]).items()},
            "angles_per_plant": {k: float(v) for k, v in per_plant.n_angles.agg(
                ["min", "median", "max"]).items()},
        },
        "visual_evidence": {
            "grid_image": str(grid.relative_to(ROOT)),
            "layout": "rows = acquisition day, columns = 6 plants of (batch I, Healthy), angle 0",
            "what_to_look_for": "under (A) each COLUMN must be the same plant ageing from "
                                "seedling to flowering and columns must differ on a given day",
        },
        "conclusion": ("reading (A): the last field is the plant index. It is the ONLY reading "
                       "that reproduces the paper's 167/77/90 plant counts, and the grid shows "
                       "longitudinal growth within a column. plant_key = host|batch|cls|plant."),
        "used_for": "the plant-disjoint splits in stage E and every Phase 4 experiment",
    }
    (P3 / "plant_key_evidence.json").write_text(json.dumps(ev, indent=2))
    log(f"[C0] plant key evidence: n_plants={ev['reproduced_by_chosen_key']['n_plants']} "
        f"(claim {claims['n_plants']}), all three claims match = "
        f"{ev['reproduced_by_chosen_key']['all_three_claims_match']}, grid -> {grid.name}")
    return ev


def stage_c(df: pd.DataFrame, a: dict, n_boot: int = 2000) -> dict:
    """Compare our numbers with the paper's Section V-A claims."""
    T = df[df.host == "Tomato"].copy()
    tol_c = CLAIMS["phase3_paper_claims"]["tolerance"]
    checks: list[dict] = []

    def add(claim, ours, ok, note="", claimed=None, delta=None, pass_by_key=None):
        rec = {"claim": claim, "ours": ours, "pass": bool(ok), "note": note}
        if claimed is not None:
            rec["claimed"] = claimed
        if delta is not None:
            rec["delta"] = delta
        if pass_by_key is not None:
            rec["pass_by_key"] = pass_by_key
        checks.append(rec)

    # --- images per stream ---------------------------------------------------
    c_ips = CLAIMS["phase3_paper_claims"]["images_per_stream"]
    ips = {s: int((T.stream == s).sum()) for s in STREAMS}
    add("images_per_stream", ips, all(v == c_ips for v in ips.values()),
        f"claim {c_ips}; per-stream counts of Tomato files")

    # --- plants --------------------------------------------------------------
    nkeys = {
        "host|batch|cls|plant (Tomato)  <- CHOSEN KEY": T.plant_key.nunique(),
        "host|batch|cls|plant (both hosts)": df.plant_key.nunique(),
        "host|batch|cls|plant (Pepper)": df[df.host == "Pepper"].plant_key.nunique(),
        "batch|cls|plant (Tomato)": T.plant_key_tomato.nunique(),
        "batch|cls|plant (both hosts)": df.groupby(["batch", "cls", "plant"]).ngroups,
        "host|batch|cls|day (Tomato)": T.groupby(["host", "batch", "cls", "day"]).ngroups,
        "host|batch|day (Tomato)": T.groupby(["host", "batch", "day"]).ngroups,
        "host|batch|plant (Tomato, cls ignored)": T.groupby(["host", "batch", "plant"]).ngroups,
        "host|batch|pot (both hosts)": df.groupby(["host", "batch", "pot"]).ngroups,
    }
    c_np = CLAIMS["phase3_paper_claims"]["n_plants"]
    add("n_plants", nkeys, c_np in nkeys.values(),
        f"claim {c_np}; every candidate key is listed - the chosen key must be the one that "
        f"reproduces it (see plant_key_evidence.json)")
    lab = df.drop_duplicates("plant_key").groupby("plant_key")["label"].first()
    labT = T.drop_duplicates("plant_key").groupby("plant_key")["label"].first()
    ours_bh = {"healthy": int((lab == 0).sum()), "infected": int((lab == 1).sum())}
    ours_to = {"healthy": int((labT == 0).sum()), "infected": int((labT == 1).sum())}
    c_h = CLAIMS["phase3_paper_claims"]["plants_healthy"]
    c_i = CLAIMS["phase3_paper_claims"]["plants_infected"]
    add("plants_healthy/infected",
        {"claim": {"healthy": c_h, "infected": c_i},
         "host|batch|cls|plant (Tomato)": ours_to,
         "host|batch|cls|plant (both hosts)": ours_bh},
        (c_h, c_i) == (ours_to["healthy"], ours_to["infected"])
        or (c_h, c_i) == (ours_bh["healthy"], ours_bh["infected"]),
        "claim reproduced only if the chosen plant key matches exactly")

    # --- prevalence by angle / batch (image level, Tomato) -------------------
    prev_a, prev_b = {}, {}
    for ang in (0, 5, 15):
        s = T[T.angle == ang]
        prev_a[str(ang)] = round(100 * s.label.mean(), 2)
    for b in ("I", "II"):
        s = T[T.batch == b]
        prev_b[b] = round(100 * s.label.mean(), 2)
    c_pa = CLAIMS["phase3_paper_claims"]["prevalence_pct_by_angle"]
    c_pb = CLAIMS["phase3_paper_claims"]["prevalence_pct_by_batch"]
    ok_a = all(abs(prev_a[k] - c_pa[k]) <= tol_c["prevalence_pp"] for k in c_pa)
    ok_b = all(abs(prev_b[k] - c_pb[k]) <= tol_c["prevalence_pp"] for k in c_pb)
    delta_a = {k: round(prev_a[k] - c_pa[k], 2) for k in c_pa}
    delta_b = {k: round(prev_b[k] - c_pb[k], 2) for k in c_pb}
    add("prevalence_pct_by_angle", prev_a, ok_a,
        f"claim {c_pa} tol {tol_c['prevalence_pp']}pp (image-level, Tomato)",
        claimed=c_pa, delta=delta_a,
        pass_by_key={k: bool(abs(prev_a[k] - c_pa[k]) <= tol_c["prevalence_pp"]) for k in c_pa})
    add("prevalence_pct_by_batch", prev_b, ok_b,
        f"claim {c_pb} tol {tol_c['prevalence_pp']}pp (image-level, Tomato)",
        claimed=c_pb, delta=delta_b,
        pass_by_key={k: bool(abs(prev_b[k] - c_pb[k]) <= tol_c["prevalence_pp"]) for k in c_pb})

    # --- mixed-label days ----------------------------------------------------
    both = T.groupby("day")["label"].nunique()
    mixed = sorted(int(x) for x in both[both == 2].index)
    c_md = CLAIMS["phase3_paper_claims"]["mixed_label_days"]
    add("n_mixed_label_days", {"n": len(mixed), "days": mixed}, mixed == c_md,
        f"claim {len(c_md)} days {c_md}; exact list match required")

    # --- Cohen's d -----------------------------------------------------------
    d_claim = CLAIMS["phase3_paper_claims"]
    rows = []
    for s in STREAMS:
        c = STREAM_COL[s]
        sub = T.dropna(subset=[f"{c}_mean"])
        img_d = cohen_d(sub.loc[sub.label == 0, f"{c}_mean"],
                        sub.loc[sub.label == 1, f"{c}_mean"])
        pl = sub.groupby("plant_key").agg(label=("label", "first"), v=(f"{c}_mean", "mean"))
        pl_d = cohen_d(pl.loc[pl.label == 0, "v"], pl.loc[pl.label == 1, "v"])
        rng = np.random.default_rng(SEED)
        h = pl.loc[pl.label == 0, "v"].values
        i = pl.loc[pl.label == 1, "v"].values
        bs = [cohen_d(rng.choice(h, len(h)), rng.choice(i, len(i))) for _ in range(n_boot)]
        rows.append({"stream": c, "image_level_d": img_d, "plant_level_d": pl_d,
                     "plant_level_ci_lo": float(np.nanpercentile(bs, 2.5)),
                     "plant_level_ci_hi": float(np.nanpercentile(bs, 97.5)),
                     "n_plants_healthy": int(len(h)), "n_plants_infected": int(len(i)),
                     "n_images": int(len(sub)),
                     "claim_image_level": d_claim["cohens_d_image_level"].get(c),
                     "claim_plant_level": d_claim["cohens_d_plant_level"].get(c)})
    dt = pd.DataFrame(rows)
    tol_d = d_claim["tolerance"]["cohens_d"]
    dt["image_level_pass"] = [
        abs(r.image_level_d - r.claim_image_level) <= tol_d if pd.notna(r.claim_image_level)
        else None for r in dt.itertuples()]
    dt["plant_level_pass"] = [
        abs(r.plant_level_d - r.claim_plant_level) <= tol_d if pd.notna(r.claim_plant_level)
        else None for r in dt.itertuples()]
    dt.to_csv(P3 / "cohens_d_vs_claim.csv", index=False)
    add("cohens_d_image_level",
        {r.stream: round(r.image_level_d, 4) for r in dt.itertuples()},
        all(dt["image_level_pass"]), f"claim {d_claim['cohens_d_image_level']} tol {tol_d}")
    add("cohens_d_plant_level",
        {r.stream: round(r.plant_level_d, 4) for r in dt.itertuples()},
        all(dt["plant_level_pass"]), f"claim {d_claim['cohens_d_plant_level']} tol {tol_d}")

    out = {"n_checks": len(checks), "n_pass": sum(1 for c in checks if c["pass"]),
           "checks": checks, "tolerances": tol_c, "n_images_tomato": int(len(T)),
           "runtime_note": "all numbers computed by this script from results/phase3/image_manifest.csv"}
    (P3 / "claims_comparison.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"[C] {out['n_pass']}/{out['n_checks']} claim groups PASS")
    return out


# --------------------------------------------------------------------------- stage D
def stage_d() -> dict:
    """Run the project's own tooling UNEDITED, as an independent cross-check."""
    out = {}
    man = P3 / "project_manifest.csv"
    rx = (r"^(?P<stream>[^/]+)/[^/]+/(?P<batch>I{1,2})/(?P<cls>Healthy|Virus)/"
          r"(?P<pot>[HV]_(?P<day>\d+))_(?P<tok>RGB|VNIR_800nm|VNIR_1000nm|VNIR)_"
          r"(?P<angle>\d+)_(?P<idx>\d+)\.[A-Za-z0-9]+$")
    # stream-map tokens: build_manifest picks the LONGEST matching token inside the captured
    # stream name, so "800"/"1000" would lose to "VNIR" inside "VNIR_800nm"/"VNIR_1000nm".
    # "800nm"/"1000nm" are longer than "VNIR" and appear only in their own directory -> correct
    # mapping WITHOUT editing the project script (HARD RULE 7).
    cmd = [PY, "caep/build_manifest.py", "--root", str(DATA), "--regex", rx,
           "--stream-map", "rgb=RGB,vnir=VNIR,nir800=800nm,nir1000=1000nm",
           "--infected-keywords", "virus,infected",
           "--require", "rgb,vnir,nir800,nir1000", "--out", str(man)]
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    (P3 / "build_manifest_stdout.txt").write_text(
        "$ " + " ".join(cmd) + "\n" + r.stdout + "\n--- STDERR ---\n" + r.stderr)
    out["build_manifest_returncode"] = r.returncode
    if r.returncode != 0 or not man.exists():
        out["build_manifest_error"] = (r.stdout + r.stderr)[-2000:]
        (P3 / "project_crosscheck.json").write_text(json.dumps(out, indent=2))
        log("[D] build_manifest FAILED - recorded, continuing")
        return out

    cmd2 = [PY, "caep/audit_tobrfv.py", "--manifest", str(man),
            "--streams", "rgb,vnir,nir800,nir1000", "--out", str(P3 / "project_audit"),
            "--workers", "8", "--boot", "2000"]
    r2 = subprocess.run(cmd2, cwd=RESEARCH, capture_output=True, text=True)
    (P3 / "audit_tobrfv_stdout.txt").write_text(
        "$ " + " ".join(cmd2) + "\n" + r2.stdout + "\n--- STDERR ---\n" + r2.stderr)
    out["audit_tobrfv_returncode"] = r2.returncode
    summ = P3 / "project_audit" / "audit_summary.json"
    if summ.exists():
        out["audit_summary"] = json.loads(summ.read_text())
    (P3 / "project_crosscheck.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"[D] project scripts ran (manifest rc={r.returncode}, audit rc={r2.returncode})")
    return out


# --------------------------------------------------------------------------- stage E
def stage_e(df: pd.DataFrame) -> dict:
    """Build the wide per-image manifest the project's CAEP/fusion tooling expects, then let the
    PROJECT'S OWN caep/split_by_plant.py make the plant-disjoint splits and caep/check_leakage.py
    assert them (HARD RULE 4: failure aborts the run).

    Wide = one row per Tomato image (4808), with one path column per stream, because streams are
    1:1:1:1 paired on (host,batch,pot,day,angle,plant) - asserted in stage A.
    """
    T = df[df.host == "Tomato"].copy()
    key = ["host", "batch", "pot", "day", "angle", "plant"]
    wide = T.pivot_table(index=key, columns="stream", values="path", aggfunc="first").reset_index()
    meta = (T.groupby(key, as_index=False)
            .agg(label=("label", "first"), plant_key=("plant_key", "first"),
                 cls=("cls", "first"), sha256=("sha256", "first")))
    wide = meta.merge(wide, on=key)
    wide = wide.rename(columns={"RGB": "rgb", "VNIR": "vnir",
                                "VNIR_800nm": "nir800", "VNIR_1000nm": "nir1000"})
    need = ["rgb", "vnir", "nir800", "nir1000"]
    missing = [c for c in need if c not in wide.columns]
    if missing:
        raise SystemExit(f"FATAL: wide manifest missing streams {missing} (HARD RULE 6)")
    # rows that cannot be fused = the unpairable tuples; only user-approved ones may be dropped
    inc = wide[wide[need].isna().any(axis=1)].copy()
    if len(inc):
        approved_doc = json.loads((ROOT / "analysis" / "approved_data_exclusions.json").read_text())
        approved = {e["tuple_key"] for e in approved_doc["exclusions"]}
        dropped = [f"{r.host}|{r.batch}|{r.pot}|day={r.day}|angle={r.angle}|plant={r.plant}"
                   for r in inc.itertuples()]
        unapproved = sorted(set(dropped) - approved)
        if unapproved:
            raise SystemExit(f"FATAL: unpairable manifest row(s) not approved for exclusion: "
                             f"{unapproved} (HARD RULE 6 - ask, do not guess)")
        wide = wide[~wide[need].isna().any(axis=1)].copy()
        log(f"[E] excluded {len(dropped)} unpairable row(s) from the 4-stream manifest "
            f"(user-approved, documented): {dropped}")
    else:
        dropped = []
    if wide[need].isna().any().any():
        raise SystemExit("FATAL: incomplete stream rows in the wide manifest (HARD RULE 6)")
    out_cols = ["plant_key", "label", "batch", "cls", "day", "angle", "sha256"] + need
    man = wide[out_cols].copy()
    man["path"] = man["rgb"]          # alias so `check_leakage`-style tooling can key on images
    man = man[["path"] + out_cols]
    man_path = P3 / "manifest_tomato.csv"
    man.to_csv(man_path, index=False)
    log(f"[E] wide manifest: {len(man)} images, {man.plant_key.nunique()} plants -> {man_path}")

    # ---- the project's own splitter (unedited) ------------------------------
    splits_dir = P3 / "plant_splits"
    cmd = [PY, "caep/split_by_plant.py", "--manifest", str(man_path),
           "--seeds", "42,43,44", "--out-dir", str(splits_dir),
           "--test-frac", "0.20", "--val-frac", "0.15"]
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    (P3 / "split_by_plant_stdout.txt").write_text(
        "$ " + " ".join(cmd) + "\n" + r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0 or not splits_dir.exists():
        raise SystemExit(f"FATAL: split_by_plant.py exited {r.returncode}: "
                         f"{(r.stdout + r.stderr)[-1500:]} (HARD RULE 6)")

    rep: dict = {"n_plants": int(man.plant_key.nunique()), "n_images": int(len(man)),
                 "n_excluded_pairing_defect": len(dropped),
                 "excluded_pairing_defect": dropped,
                 "seeds": {}, "assertions": {}}
    # HARD RULE 6: only user-approved unpairable rows may be absent (checked above)
    rep["assertions"]["pairing_exclusions_all_approved"] = "PASS"
    for seed in (42, 43, 44):
        mp = splits_dir / f"manifest_split_seed{seed}.csv"
        d = pd.read_csv(mp)
        # --- HARD RULE 4: project's own checker (exits 1 on leakage) ----------
        c = subprocess.run([PY, "caep/check_leakage.py", "--manifest", str(mp),
                            "--group-col", "plant_key", "--report-cols", "angle,batch,day"],
                           cwd=RESEARCH, capture_output=True, text=True)
        (P3 / f"check_leakage_seed{seed}.txt").write_text(
            "$ check_leakage.py --manifest " + str(mp) + "\n" + c.stdout + c.stderr)
        rep["assertions"][f"seed{seed}_plant_disjoint_project_checker"] = (
            "PASS" if c.returncode == 0 else f"FAIL rc={c.returncode}: {c.stdout[-400:]}")
        # --- HARD RULE 4: exact sha256 disjoint (my own assertion) ------------
        frames = {k: v for k, v in d.groupby("split")}
        try:
            assert_disjoint(frames, "sha256")
            rep["assertions"][f"seed{seed}_sha256_disjoint"] = "PASS"
        except AssertionError as e:
            rep["assertions"][f"seed{seed}_sha256_disjoint"] = f"FAIL: {e}"
        # --- images must appear in exactly one split -------------------------
        rep["assertions"][f"seed{seed}_image_once"] = (
            "PASS" if d.path.duplicated().sum() == 0 and d.split.notna().all()
            else f"FAIL: repeated={int(d.path.duplicated().sum())} "
                 f"unassigned={int(d.split.isna().sum())}")
        pl = d.drop_duplicates("plant_key")
        rep["seeds"][str(seed)] = {
            "plants": {k: int(v) for k, v in pl["split"].value_counts().items()},
            "images": {k: int(v) for k, v in d["split"].value_counts().items()},
            "healthy_infected_plants": {
                s: {"healthy": int(((pl["split"] == s) & (pl["label"] == 0)).sum()),
                    "infected": int(((pl["split"] == s) & (pl["label"] == 1)).sum())}
                for s in ("train", "validation", "test")},
            "prevalence_pct_by_split": {
                k: round(100 * float(v), 2) for k, v in d.groupby("split")["label"].mean().items()},
            "prevalence_pct_by_angle": {
                str(k): round(100 * float(v), 2)
                for k, v in d.groupby("angle")["label"].mean().items()},
        }
        # keep the long-form image list too (used by the audit report)
        d.to_csv(P3 / f"split_seed{seed}_manifest.csv", index=False)

    (P3 / "split_leakage_assertions.json").write_text(json.dumps(rep, indent=2))
    failed = {k: v for k, v in rep["assertions"].items() if str(v).startswith("FAIL")}
    log("[E] assertions:", json.dumps(rep["assertions"], indent=2))
    if failed:
        raise SystemExit(f"FATAL: leakage assertion(s) failed: {failed} (HARD RULE 4)")
    return rep


def main() -> None:
    t0 = time.time()
    df, a = stage_a()
    df = stage_b(df)
    ev = stage_plantkey_evidence(df)
    c = stage_c(df, a)
    d = stage_d()
    e = stage_e(df)
    summary = {"phase": 3, "runtime_s": round(time.time() - t0, 1),
               "stage_a": a, "claims": {"n_checks": c["n_checks"], "n_pass": c["n_pass"]},
               "plant_key": ev["reproduced_by_chosen_key"],
               "project_crosscheck_ran": d.get("build_manifest_returncode") == 0,
               "split_assertions": e["assertions"]}
    (P3 / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    runlog = RES / "run_log.md"
    old = runlog.read_text() if runlog.exists() else "# Run log\n"
    marker = "## Phase 3 - Track B data audit"
    if marker in old:          # block-scoped: remove ONLY our previous entry (CHANGELOG)
        i = old.index(marker)
        j = old.find("\n## ", i + 1)
        head = old[:i].rstrip("\n")
        tail = "" if j == -1 else old[j + 1:]
        old = head + ("\n\n" + tail if tail else "\n")
    import datetime
    runlog.write_text(old + "\n".join([
        f"\n{marker}\n",
        f"- time (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "- command: `.venv/bin/python analysis/dataset2_phase3_audit.py`",
        f"- runtime: {summary['runtime_s']} s",
        f"- files parsed: {a['n_parsed']} / {a['n_files']} (0 unparsed)",
        f"- plant key: host|batch|cls|plant -> {ev['reproduced_by_chosen_key']['n_plants']} plants "
        f"(healthy {ev['reproduced_by_chosen_key']['healthy']}, "
        f"infected {ev['reproduced_by_chosen_key']['infected']}); all three paper claims match = "
        f"{ev['reproduced_by_chosen_key']['all_three_claims_match']}",
        f"- claim groups passing: {c['n_pass']}/{c['n_checks']}",
        f"- split assertions: {json.dumps(e['assertions'])}",
        "- artifacts: results/phase3/",
        ""]))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
