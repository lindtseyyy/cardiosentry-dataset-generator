#!/usr/bin/env python3
"""
Copy the five backbone baseline runs in from cardiosentry_baselines_v2.

    python import_baselines.py --from "$DRIVE/runs/baselines_v2" --to "$RUNS"

WHY THIS IS A SCRIPT AND NOT `cp -r`

Because `cp` cannot tell you that the run you just imported was trained under a
different protocol. Every hybrid in this package is reported as a DELTA against
its bare backbone, and that subtraction is only meaningful if the baseline was
trained on the same data, at the same geometry, with the same optimiser, schedule
and threshold rule. `config.fingerprint()` is the hash of exactly those settings,
and this script refuses to import a run whose fingerprint does not match the one
the hybrids are being trained under.

If it refuses, nothing is broken - it means the baseline predates a config change
and the honest fix is to re-train that backbone here, not to compare across it.

WHAT IT DOES NOT COPY: weights/ and predictions/. compare.py reads result.json;
the checkpoints are 28-350 MB apiece and are already on Drive under
runs/baselines_v2/ where they belong. Importing them would duplicate ~1.5 GB to
build a table that never opens them.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import config as C
import models

# result.json is the only file compare.py needs; the rest are copied because they
# are small and make the imported directory readable on its own terms.
COPY = ("result.json", "report.txt", "metrics.csv", "history.csv",
        "config_snapshot.json", "STATUS.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="the baselines_v2 runs folder")
    ap.add_argument("--to", dest="dst", required=True, type=Path,
                    help="this package's runs folder")
    ap.add_argument("--backbone", default=None,
                    help="import only this backbone's row (or a comma-separated "
                         "list). The two notebooks run in parallel into separate "
                         "folders, and each one only needs its own group's "
                         "reference rows - e.g. --backbone vit_b16,swinv2_base.")
    ap.add_argument("--force", action="store_true",
                    help="import even if the fingerprint differs. Read the "
                         "docstring before using this: it makes every delta in "
                         "compare.py a comparison across two protocols.")
    a = ap.parse_args()

    a.dst.mkdir(parents=True, exist_ok=True)
    want = C.fingerprint()

    wanted = models.BASELINE_RUNS
    if a.backbone:
        keys = [k.strip() for k in a.backbone.split(",") if k.strip()]
        unknown = [k for k in keys if k not in models.BASELINE_RUNS]
        if unknown:
            print(f"unknown backbone(s): {unknown}. known: "
                  f"{sorted(models.BASELINE_RUNS)}")
            return 1
        wanted = {k: models.BASELINE_RUNS[k] for k in keys}

    print(f"looking for {len(wanted)} baseline run(s) under {a.src}")
    print(f"required fingerprint: {want}   (geometry {C.DIM_NAME})\n")

    done, skipped = [], []
    for backbone, run in wanted.items():
        src = a.src / run
        rj = src / "result.json"
        if not rj.exists():
            print(f"  {run:34} NOT FOUND at {src}")
            skipped.append(run)
            continue
        try:
            got = json.loads(rj.read_text()).get("fingerprint")
        except Exception as exc:
            print(f"  {run:34} unreadable result.json: {exc}")
            skipped.append(run)
            continue
        if got != want and not a.force:
            print(f"  {run:34} REFUSED - fingerprint {got} != {want}")
            print(f"  {'':34} that baseline was trained under different "
                  f"controlled settings,")
            print(f"  {'':34} so a delta against it would measure the protocol "
                  f"as well as")
            print(f"  {'':34} the component. Re-train {backbone} here, or pass "
                  f"--force if you")
            print(f"  {'':34} already know why they differ.")
            skipped.append(run)
            continue

        out = a.dst / run
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in COPY:
            if (src / f).exists():
                shutil.copy2(src / f, out / f)
                n += 1
        print(f"  {run:34} imported ({n} files)"
              + ("   FORCED, fingerprints differ" if got != want else ""))
        done.append(run)

    print()
    if done:
        print(f"imported {len(done)}: {', '.join(done)}")
        print("compare.py will now emit the DELTA AGAINST THE BACKBONE BASELINE "
              "section.")
    if skipped:
        print(f"missing or refused ({len(skipped)}): {', '.join(skipped)}")
        print("Those backbones' hybrids will still be tabulated - they just have "
              "no reference row.")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
