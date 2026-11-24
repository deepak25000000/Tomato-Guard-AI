"""Build a per-image manifest for TOBRFV-LMID, with RGB / 800nm / 1000nm streams paired per
(plant, day, angle).

I could not see your copy of the dataset, so the file-naming scheme is NOT hard-coded. Provide a
regex with NAMED groups applied to each file's path relative to --root:
   (?P<batch>..) (?P<cls>..) (?P<pot>..) (?P<day>..) (?P<angle>..) (?P<stream>..)

Example (adapt to your real names):
  python build_manifest.py --root /data/TOBRFV_LMID/tomato \
     --regex '(?P<batch>B\\d+)/(?P<cls>healthy|infected)/(?P<pot>P\\d+)_D(?P<day>\\d+)_A(?P<angle>\\d+)_(?P<stream>\\w+)\\.\\w+' \
     --stream-map rgb=RGB,nir800=800,nir1000=1000

Run once with a deliberately wrong --regex to see sample paths printed.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--regex", required=True)
    ap.add_argument("--stream-map", default="rgb=RGB,nir800=800,nir1000=1000",
                    help="column=token pairs; token is matched (case-insens.) inside the 'stream' group")
    ap.add_argument("--infected-keywords", default="infected,inoculated,tobrfv")
    ap.add_argument("--healthy-keywords", default="healthy,control")
    ap.add_argument("--require", default="rgb,nir800,nir1000")
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()

    rx = re.compile(a.regex)
    smap = dict(p.split("=") for p in a.stream_map.split(","))
    inf = [k.strip().lower() for k in a.infected_keywords.split(",")]
    hea = [k.strip().lower() for k in a.healthy_keywords.split(",")]
    root = Path(a.root)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    print(f"{len(files)} image files under {root}")
    print("sample relative paths:", *[str(p.relative_to(root)) for p in files[:5]], sep="\n  ")

    recs, unmatched, nostream = [], 0, 0
    for p in files:
        m = rx.search(str(p.relative_to(root)).replace("\\", "/"))
        if not m:
            unmatched += 1
            continue
        g = m.groupdict()
        col = max((c for c, tok in smap.items() if tok.lower() in g["stream"].lower()),
                  key=lambda c: len(smap[c]), default=None)   # longest token wins: VNIR_800nm > VNIR
        if col is None:
            nostream += 1
            continue
        cls = g["cls"].lower()
        if any(k in cls for k in inf):
            label = 1
        elif any(k in cls for k in hea):
            label = 0
        else:
            raise SystemExit(f"Cannot map class '{g['cls']}' to healthy/infected - edit the keyword args")
        recs.append({"plant_key": f"{g['batch']}|{g['cls']}|{g['pot']}", "batch": g["batch"], "cls": g["cls"],
                     "pot": g["pot"], "day": int(g["day"]), "angle": g["angle"], "label": label,
                     "stream": col, "path": str(p)})
    print(f"matched {len(recs)}  unmatched-regex {unmatched}  unknown-stream {nostream}")
    if not recs:
        raise SystemExit("Nothing matched. Fix --regex (see sample paths above).")
    df = pd.DataFrame(recs)
    key = ["plant_key", "batch", "cls", "pot", "day", "angle", "label"]
    wide = df.pivot_table(index=key, columns="stream", values="path", aggfunc="first").reset_index()
    need = [c for c in a.require.split(",") if c]
    missing = [c for c in need if c not in wide.columns]
    if missing:
        raise SystemExit(f"No files found for stream(s) {missing}; check --stream-map")
    before = len(wide)
    wide = wide.dropna(subset=need)
    print(f"{before} (plant,day,angle) rows -> {len(wide)} with all required streams")
    print(f"plants: {wide['plant_key'].nunique()}  healthy/infected plants: "
          f"{wide.drop_duplicates('plant_key')['label'].value_counts().to_dict()}")
    wide.to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
