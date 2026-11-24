"""GROUP B / item B1 - root-cause the one failed claim: prevalence at 15 deg
(ours 52.96 vs claimed 52.2, tolerance 0.1 pp).

  .venv/bin/python analysis/dataset2_phase3_claim_rootcause.py

Everything is computed from results/phase3/image_manifest.csv (per-stream audit of all
40,716 files, orphan tuples INCLUDED) and analysis/reference_claims.json /
analysis/approved_data_exclusions.json (inputs). For every candidate definition we
record the raw NUMERATOR and DENOMINATOR (HARD RULE 1) at all three angles, the delta
vs the claim, and whether the definition reproduces ALL THREE claimed angles within
0.1 pp (an "explanation" that fixes 15 deg but breaks 0/5 deg is not an explanation).

Definitions tested (pre-declared):
  a. base: image-level, Tomato, all stream rows           <- the audited definition
  b. minus the 2 approved orphan tuples
  c. both hosts (Tomato + Pepper)
  d. per stream separately (RGB / VNIR / 800nm / 1000nm)
  e. tuple-level (each capture tuple counted once)
  f. tuple-level, complete tuples only (minus orphans)
  g. plant-level: infected plants / plants photographed at that angle
  h. per-plant macro average of image-level prevalence
  i. macro average over recording days
  j. exclude recording day 20 (the orphans' day)
  k. batch I only / l. batch II only
  m. label taken from the PLANT key (not the image row)
Plus the "required numerator" bookkeeping: what numerator the claim implies for the
audited denominator.

Verdict field: which definition(s), if any, reproduce the claim -> otherwise
"unexplained by the N tested definitions".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results" / "phase3"
OUT = P3 / "claim_rootcause"
TOL = 0.1           # pp, from claims_comparison tolerances
CLAIM = json.loads((ROOT / "analysis" / "reference_claims.json").read_text())[
    "phase3_paper_claims"]["prevalence_pct_by_angle"]
EXCL = json.loads((ROOT / "analysis" / "approved_data_exclusions.json").read_text())
ANGLES = [0, 5, 15]
STREAMS = ["RGB", "VNIR", "VNIR_800nm", "VNIR_1000nm"]


def parse_tuple_key(k: str) -> tuple:
    # "Tomato|I|V_20|day=20|angle=15|plant=14"  (raw cls spelling V_20 == Virus);
    # returned in the manifest's column order: host,batch,cls,plant,day,angle
    host, batch, cls, day, angle, plant = k.split("|")
    return (host, batch, "Virus" if cls == "V_20" else cls,
            int(plant.split("=")[1]), int(day.split("=")[1]), int(angle.split("=")[1]))


def pct(num: float, den: float) -> float | None:
    return None if den == 0 else round(100.0 * num / den, 4)


def main() -> None:
    t0 = time.time()
    man = pd.read_csv(P3 / "image_manifest.csv")
    T = man[man.host == "Tomato"].copy()

    # ---------- integrity: exactly the 2 approved tuples are incomplete -------------
    key_cols = ["host", "batch", "cls", "plant", "day", "angle"]
    n_streams = man.groupby(key_cols)["stream"].transform("count")
    incomplete = man[n_streams != 4]
    inc_keys = sorted(set(map(tuple, incomplete[key_cols].itertuples(index=False, name=None))))
    approved = sorted(parse_tuple_key(e["tuple_key"]) for e in EXCL["exclusions"])
    if inc_keys != approved:
        raise SystemExit(f"FATAL: incomplete tuples {inc_keys} != approved {approved} "
                         f"(HARD RULE 6)")

    orph_idx = set(incomplete.index)
    T["is_orphan"] = T.index.isin(orph_idx)

    # label consistency image-row vs plant-key
    plant_lab = T.drop_duplicates("plant_key").set_index("plant_key")["label"]
    T["label_plant"] = T["plant_key"].map(plant_lab)

    variants: dict[str, dict] = {}

    def add(name: str, note: str, frame: pd.DataFrame,
            how: str = "image") -> None:
        """how: image | tuple | plant | macro_plant | macro_day."""
        res = {}
        for ang in ANGLES:
            s = frame[frame.angle == ang]
            if how == "image":
                num, den = int(s.label.sum()), int(len(s))
            elif how == "tuple":
                g = s.drop_duplicates(key_cols)
                num, den = int(g.label.sum()), int(len(g))
            elif how == "plant":
                g = s.drop_duplicates("plant_key")
                num, den = int(g.label.sum()), int(len(g))
            elif how == "macro_plant":
                per = s.groupby("plant_key")["label"].mean()
                num, den = float(per.sum()), int(len(per))     # mean of means
            elif how == "macro_day":
                per = s.groupby("day")["label"].mean()
                num, den = float(per.sum()), int(len(per))
            else:
                raise ValueError(how)
            res[str(ang)] = {"numerator": num, "denominator": den,
                             "pct": pct(num, den)}
        deltas = {a: (None if res[a]["pct"] is None
                      else round(res[a]["pct"] - CLAIM[a], 4)) for a in map(str, ANGLES)}
        ok = {a: (deltas[a] is not None and abs(deltas[a]) <= TOL) for a in map(str, ANGLES)}
        variants[name] = {
            "note": note, "per_angle": res, "delta_pp": deltas,
            "pass_within_0.1pp_by_angle": ok,
            "reproduces_all_three_claims": bool(all(ok.values()))}

    # a. base --------------------------------------------------------------------
    add("base_image_level_Tomato_all_streams",
        "image-level, Tomato, every stream row (the audited definition)", T)
    # b. minus approved orphans ---------------------------------------------------
    add("minus_approved_orphan_tuples",
        "drop the 2 approved unpairable capture tuples entirely", T[~T.is_orphan])
    # c. both hosts ---------------------------------------------------------------
    both = man.copy()
    both["is_orphan"] = both.index.isin(orph_idx)
    add("both_hosts_included", "Tomato + Pepper pooled", both)
    # d. per stream ---------------------------------------------------------------
    for s in STREAMS:
        add(f"stream_{s}", f"only the {s} stream rows", T[T.stream == s])
    # e/f. tuple-level ------------------------------------------------------------
    add("tuple_level_all_tuples", "each capture tuple counted once (any stream)",
        T, how="tuple")
    add("tuple_level_complete_tuples_only", "tuple counted once, complete tuples only",
        T[~T.is_orphan], how="tuple")
    # g/h. plant-level ------------------------------------------------------------
    add("plant_level", "infected plants / plants photographed at that angle", T,
        how="plant")
    add("plant_macro_average", "mean over plants of within-plant image prevalence", T,
        how="macro_plant")
    # i. macro over days ----------------------------------------------------------
    add("day_macro_average", "mean over recording days of daily prevalence", T,
        how="macro_day")
    # j. exclude day 20 -----------------------------------------------------------
    add("exclude_day20", "drop every image recorded on day 20", T[T.day != 20])
    # k/l. per batch --------------------------------------------------------------
    for b in ("I", "II"):
        add(f"batch_{b}_only", f"only batch {b}", T[T.batch == b])
    # m. plant-key label ----------------------------------------------------------
    Tp = T.copy()
    Tp["label"] = Tp["label_plant"]
    add("label_from_plant_key", "label taken from the plant key, not the image row", Tp)

    # ---------- bookkeeping: numerator the claim implies --------------------------
    implied = {}
    for ang in ANGLES:
        base = variants["base_image_level_Tomato_all_streams"]["per_angle"][str(ang)]
        den = base["denominator"]
        want = CLAIM[str(ang)] / 100.0 * den
        implied[str(ang)] = {
            "audited_numerator": base["numerator"], "audited_denominator": den,
            "claim_pct": CLAIM[str(ang)],
            "numerator_implied_by_claim": round(want, 2),
            "nearest_integer_numerator": round(want),
            "gap_images": round(want - base["numerator"], 2)}

    matches = [k for k, v in variants.items() if v["reproduces_all_three_claims"]]
    verdict = (f"EXPLAINED by: {', '.join(matches)}" if matches else
               f"UNEXPLAINED - none of the {len(variants)} tested definitions "
               f"reproduces all three claimed angles within {TOL} pp")

    out = {"phase": 3, "analysis": "root-cause of the prevalence_pct_by_angle claim FAIL",
           "claim": CLAIM, "tolerance_pp": TOL,
           "audited_definition": "image-level prevalence over Tomato stream rows, "
                                 "results/phase3/image_manifest.csv (orphans included)",
           "integrity": {"incomplete_tuples_in_raw_data": [list(k) for k in inc_keys],
                         "match_approved_exclusions": True,
                         "n_orphan_rows_at_angle15": int(incomplete.angle.eq(15).sum())},
           "implied_numerator_bookkeeping": implied,
           "variants": variants,
           "definitions_reproducing_all_three_claims": matches,
           "verdict": verdict,
           "sources": ["results/phase3/image_manifest.csv",
                       "analysis/reference_claims.json",
                       "analysis/approved_data_exclusions.json"],
           "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "claim_rootcause.json").write_text(json.dumps(out, indent=2))
    rows = []
    for name, v in variants.items():
        r = {"definition": name, "note": v["note"],
             "reproduces_all_three": v["reproduces_all_three_claims"]}
        for a in map(str, ANGLES):
            r[f"num_{a}"] = v["per_angle"][a]["numerator"]
            r[f"den_{a}"] = v["per_angle"][a]["denominator"]
            r[f"pct_{a}"] = v["per_angle"][a]["pct"]
            r[f"delta_pp_{a}"] = v["delta_pp"][a]
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "claim_rootcause.csv", index=False)

    print(json.dumps({"verdict": verdict, "implied": implied}, indent=2))
    for name, v in variants.items():
        p15 = v["per_angle"]["15"]["pct"]
        print(f"  {name:42s} 15deg={p15!s:>9} delta={v['delta_pp']['15']!s:>9} "
              f"all3={v['reproduces_all_three_claims']}")
    print(f"wrote {OUT}/claim_rootcause.json (+csv) in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
