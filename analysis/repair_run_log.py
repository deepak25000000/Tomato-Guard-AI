#!/usr/bin/env python3
"""One-time repair of results/run_log.md (HARD RULE 5 completeness; logged in CHANGELOG.md).

What happened: log rewriters used `old = old[: old.index(marker)]` to stay idempotent.
That is only safe when the marker block is the LAST thing in the file. Several Phase 5
smoke tests ran while later phases kept appending, so each rewrite cut at the (no longer
final) Phase 5 block and destroyed every entry appended after it: the architecture sweep,
S1 seeds 43/44, the label-shuffle negative control, the unseen-eval diagnostic, the first
Phase 3 failure, and the first Phase 4 chain stop. The Phase 3 retry-failure entry was
destroyed the same way by a substring-matching marker.

This script re-inserts those entries. EVERY number is read from the surviving source
files named inside each entry (HARD RULE 1 - nothing is typed by hand); commands are
reconstructed from the json fields (mode/tag/seed) or from the shell script that ran
them. Idempotent: blocks already present are skipped.

The rewriters themselves now remove only their own block (block-scoped cut) - see
CHANGELOG.md.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LOG = RES / "run_log.md"


def j(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def day_of(rel: str) -> str:
    """YYYY-MM-DD of a log file (chain logs store only HH:MM:SS, all same day)."""
    return datetime.fromtimestamp((RES / rel).stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")


def stamped(rel: str, needle: str) -> tuple[str, str]:
    """First line of `rel` containing `needle` -> (ISO time from its [HH:MM:SS], line)."""
    for line in (RES / rel).read_text().splitlines():
        if needle in line:
            m = re.match(r"\[(\d\d:\d\d:\d\d)\]", line)
            if m:
                return f"{day_of(rel)}T{m.group(1)}Z", line
    raise SystemExit(f"repair_run_log: {needle!r} not found in results/{rel}")


def faithful_cmd(d: dict) -> str:
    cmd = (f".venv/bin/python analysis/dataset1_phase2_faithful.py"
           f" --mode {d['mode']} --tag {d['tag']} --seed {d['seed']}")
    return cmd + (" --shuffle-labels" if d.get("shuffle_labels") else "")


def faithful_entry(rel: str, heading: str) -> list[str]:
    d = j(f"phase2/{rel}")
    t = datetime.fromtimestamp((RES / f"phase2/{rel}").stat().st_mtime,
                               timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"### {heading}",
             f"- time (UTC): {t} (mtime of the result file)",
             f"- command: `{faithful_cmd(d)}`",
             f"- runtime: {d['runtime_s']} s | test accuracy: {d['accuracy']:.4f} "
             f"(results/phase2/{rel})"]
    if "abs_diff_vs_published" in d:
        lines.append(f"- abs diff vs published accuracy: {d['abs_diff_vs_published']:.4f} "
                     f"(results/phase2/{rel})")
    if "chance_accuracy_1_over_n_classes" in d:
        lines.append(f"- chance level: {d['chance_accuracy_1_over_n_classes']:.4f} "
                     f"(results/phase2/{rel})")
    if "n_unseen" in d:
        lines.append(f"- n unseen test images: {d['n_unseen']} | accuracy CI: "
                     f"[{d['accuracy_95_ci'][0]:.4f}, {d['accuracy_95_ci'][1]:.4f}] "
                     f"(results/phase2/{rel})")
    return lines


def block_a() -> str:
    """Faithful runs: launched directly, never logged (they predate every chain log)."""
    lines = [
        "## Phase 2 - faithful reproduction runs (added retroactively from result files)",
        "",
        "- NOTE: added by `analysis/repair_run_log.py` - these runs were launched directly",
        "  and never received a run-log entry; every value below is read from the file",
        "  named at the end of its line (HARD RULE 1). See CHANGELOG.md.",
    ]
    for rel, heading in (("faithful_original.json", "original checkpoint evaluation (mode=original)"),
                         ("faithful_run1.json", "retraining run 1 (mode=reproduce)"),
                         ("faithful_run2.json", "retraining run 2 (mode=reproduce)")):
        lines += faithful_entry(rel, heading)
    return "\n".join(lines) + "\n"


def block_b() -> str:
    """Entries destroyed by the truncation bug; times/runtimes from the chain logs."""
    pipe_t, pipe_l = stamped("phase2/pipeline.log", "arch sweep done")
    m = re.search(r"rc=(\d+), (\d+)s", pipe_l)
    sel = j("phase2/arch_selection.json")["selected_architecture"]
    lines = [
        "## Phase 2-3 - chained runs (reconstructed from chain logs)",
        "",
        "- NOTE: added by `analysis/repair_run_log.py` - the original entries were",
        "  destroyed by the run-log truncation bug (see CHANGELOG.md); each value is",
        "  re-read from the file named in its line (HARD RULE 1).",
        "### architecture sweep (S2, validation-only selection)",
        f"- time (UTC): {pipe_t} (results/phase2/pipeline.log)",
        "- command: `.venv/bin/python analysis/dataset1_phase2_arch.py`",
        f"- exit code: {m.group(1)} | runtime: {m.group(2)} s (results/phase2/pipeline.log)",
        f"- selected architecture: {sel} on S2 validation macro-F1 "
        f"(results/phase2/arch_selection.json)",
    ]
    chain_cmds = {
        "Phase 2 - S1 seed 43": (".venv/bin/python analysis/dataset1_phase2_faithful.py"
                                 " --mode reproduce --tag seed43 --seed 43",
                                 "faithful_seed43.json", "S1 seed 43"),
        "Phase 2 - S1 seed 44": (".venv/bin/python analysis/dataset1_phase2_faithful.py"
                                 " --mode reproduce --tag seed44 --seed 44",
                                 "faithful_seed44.json", "S1 seed 44"),
        "Phase 2 - NEGATIVE CONTROL": (".venv/bin/python analysis/dataset1_phase2_faithful.py"
                                       " --mode reproduce --tag shuffle_control --seed 42"
                                       " --shuffle-labels",
                                       "faithful_shuffle_control.json",
                                       "NEGATIVE CONTROL (shuffled labels)"),
        "Phase 2 - original checkpoint on UNSEEN": (
            ".venv/bin/python analysis/dataset1_phase2_unseen_eval.py",
            "faithful_origckpt_on_unseen.json",
            "original checkpoint on UNSEEN S2 test images"),
    }
    for needle, (cmd, jrel, heading) in chain_cmds.items():
        t, line = stamped("phase2/chain.log", needle)
        m = re.search(r"rc=(\d+) \((\d+)s\)", line)
        d = j(f"phase2/{jrel}")
        lines += [f"### {heading}",
                  f"- time (UTC): {t} (results/phase2/chain.log)",
                  f"- command: `{cmd}`",
                  f"- exit code: {m.group(1)} | runtime: {m.group(2)} s "
                  f"(results/phase2/chain.log)",
                  f"- test accuracy: {d['accuracy']:.4f} (results/phase2/{jrel})"]
        if "n_unseen" in d:
            lines[-1] = (f"- n unseen: {d['n_unseen']} | accuracy: {d['accuracy']:.4f} | "
                         f"CI [{d['accuracy_95_ci'][0]:.4f}, {d['accuracy_95_ci'][1]:.4f}] "
                         f"(results/phase2/{jrel})")
    # --- the two failed chain attempts ----------------------------------------
    t3, _ = stamped("phase2/chain.log", "Phase 3 - Track B data audit -> rc=")
    err = next(ln for ln in (RES / "phase2/chain.log").read_text().splitlines()
               if "No such file" in ln)
    lines += [
        "### Phase 3 first attempt - FAILED before Python started",
        f"- time (UTC): {t3} (results/phase2/chain.log)",
        "- exit code: 1 | runtime: 0 s (results/phase2/chain.log)",
        f"- error: `{err}`",
        "- impact: none - no analysis code ran, no data touched; fixed with "
        "`mkdir -p` (CHANGELOG.md)",
    ]
    t4, line4 = stamped("phase4_chain.log", "FATAL")
    lines += [
        "### Phase 4 chain first attempt - stopped on missing Phase 3 outputs",
        f"- time (UTC): {t4} (results/phase4_chain.log)",
        f"- {line4.split('] ', 1)[1]}",
    ]
    return "\n".join(lines) + "\n"


def block_c() -> str:
    """The Phase 3 retry that crashed on NameError (its stdout was overwritten)."""
    t, line = stamped("phase3_chain.log", "rc=1")
    m = re.search(r"rc=(\d+) \((\d+)s\)", line)
    return "\n".join([
        "## Phase 3 - Track B data audit (retry attempt 1, reconstructed)",
        "",
        "- NOTE: reconstructed by `analysis/repair_run_log.py` from "
        "results/phase3_chain.log;",
        "  the original entry was destroyed by the run-log truncation bug (CHANGELOG.md).",
        f"- time (UTC): {t} (results/phase3_chain.log)",
        f"- exit code: {m.group(1)} | runtime: {m.group(2)} s (results/phase3_chain.log)",
        "- failure: `NameError: name 'n_boot' is not defined` in `stage_c` - fixed as "
        "logged in",
        "  CHANGELOG.md; that attempt's stdout was overwritten by the successful retry,",
        "  so results/phase3_chain.log is the surviving record.",
    ]) + "\n"


def insert_before(old: str, anchor: str, block: str) -> str:
    n = old.count(anchor)
    if n != 1:
        raise SystemExit(f"repair_run_log: anchor {anchor!r} occurs {n} times (expected 1)")
    i = old.find(anchor)
    head = old[:i].rstrip("\n")
    return head + "\n\n" + block.rstrip("\n") + "\n\n" + old[i:]


def main() -> None:
    old = LOG.read_text() if LOG.exists() else "# Run log\n"
    orig = old
    # (idempotency key, anchor header line, builder) - insertion order matters:
    # each anchor must still be unique after earlier insertions (they are: the blocks
    # contain no anchor text).
    plan = [
        ("faithful reproduction runs (added retroactively",
         "## Phase 2 - Track A experiments on split S2\n", block_a),
        ("chained runs (reconstructed from chain logs)",
         "## Phase 5 - RESULTS.md generation\n", block_b),
        ("retry attempt 1, reconstructed",
         "## Phase 3 - Track B data audit\n", block_c),
    ]
    done = []
    for key, anchor, build in plan:
        if key in old:
            continue
        old = insert_before(old, anchor, build())
        done.append(key)
    if old == orig:
        print("run_log.md already repaired (idempotent re-run) - nothing to do")
        return
    LOG.write_text(old)
    print(f"run_log.md: inserted {len(done)} block(s): {done}")
    print("header order now:")
    for i, ln in enumerate(old.splitlines(), 1):
        if ln.startswith("## "):
            print(f"  {i}: {ln}")


if __name__ == "__main__":
    main()
