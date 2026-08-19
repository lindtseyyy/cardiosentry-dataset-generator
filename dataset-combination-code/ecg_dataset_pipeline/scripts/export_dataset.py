#!/usr/bin/env python3
"""
STEP 13 -- Export a VERSIONED, SELF-CONTAINED dataset release.

Turns the metadata CSVs (which only *reference* signal files scattered across
three source distributions) into one directory you can archive, copy to another
machine, or hand to a collaborator, and that a future you can point at and say
exactly what it contained.

Why this exists instead of `--copy-signals`
-------------------------------------------
`create_balanced_dataset.py --copy-signals` renames each `.hea` on disk to the
`global_record_id` but leaves the header's *contents* untouched. A WFDB header
names its own record on line 1 and names its signal file on every signal line:

    JS00001 12 500 5000
    JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I

Renaming the files to `CHAPMAN_000001.*` without rewriting those tokens gives a
directory where `wfdb.rdrecord("CHAPMAN_000001")` raises FileNotFoundError,
looking for a `JS00001.mat` that is not there. Verified on real records.

This script rewrites those two tokens and nothing else. Signal files are copied
byte for byte -- no decode, no re-encode, no resampling, no rounding -- so the
exported samples are bit-identical to the originals, which `--verify` proves by
reading both back and comparing.

Layout
------
Each cohort is a COMPLETE STANDALONE DATASET in its own directory. Copy out
just `balanced/` and it works: its CSV, manifest, checksums and every signal
file it references live inside it, and nothing points back at the source
distributions.

    data/releases/<version>/
        VERSION.json                 what this release is, and from what
        README.md                    generated, human-readable
        harmonized/
            harmonized_ecg_metadata.csv
            MANIFEST.csv             per record: paths, sha256, cohort flags
            checksums.sha256         standard `sha256sum -c` format
            signals/<SOURCE>/<global_record_id>.hea + .dat|.mat
        balanced/
            balanced_ecg_metadata.csv
            MANIFEST.csv
            checksums.sha256
            signals/<SOURCE>/<global_record_id>.hea + .dat|.mat
    data/releases/CHANGELOG.md       appended on every export

The balanced cohort is a subset of the harmonized one, so a both-cohort export
writes its ~9.3k shared records twice: ~3.5 GB instead of ~2.4 GB. `--hardlink`
gets the disk saving back while keeping both trees complete -- the shared
records become one set of bytes with two names. Read-only archives only.

Usage
-----
    python scripts/export_dataset.py                        # next free version
    python scripts/export_dataset.py --version v1
    python scripts/export_dataset.py --version v2 --note "accepted SNOMED 55827005"
    python scripts/export_dataset.py --cohort balanced      # balanced records only
    python scripts/export_dataset.py --hardlink             # dedupe the overlap
    python scripts/export_dataset.py --verify all           # read back every record
    python scripts/export_dataset.py --npy                  # + one stacked array
    python scripts/export_dataset.py --check v1             # re-verify an old release
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (PIPELINE_ROOT, TARGET_CLASSES, banner, info, load_paths,
                    warn, write_csv)

TOOL_VERSION = "1.0"
SAMPLE_VERIFY_N = 200

# Columns in the exported CSVs that must be repointed at the release.
PATH_COLUMNS = ["signal_path", "header_path", "data_path"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def next_version(releases: Path) -> str:
    """Lowest unused vN, so two exports never silently share a name."""
    used = set()
    if releases.is_dir():
        for p in releases.iterdir():
            m = re.fullmatch(r"v(\d+)", p.name)
            if m and p.is_dir():
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"v{n}"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# the one non-trivial operation: rewriting a WFDB header
# --------------------------------------------------------------------------
def rewrite_header(text: str, old_name: str, new_name: str) -> str:
    """Rename the record inside a WFDB header, touching nothing else.

    A header is:
        <record> <nsig> <fs> <nsamp> [...]        <- line 1, record name first
        <file> <fmt> <gain> ...                   <- one per signal
        # free-text comments                      <- optional, preserved

    Only the leading record token and the leading filename token of each signal
    line are replaced. Gains, baselines, checksums, the `16+24` offset format
    Chapman uses, trailing whitespace and every comment survive untouched --
    which matters, because those fields are what make the copied signal bytes
    decode to the same millivolts as the original.
    """
    out = []
    for i, line in enumerate(text.splitlines(keepends=True)):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            out.append(line)                       # comment or blank: verbatim
            continue

        indent = line[: len(line) - len(stripped)]
        parts = stripped.split(" ", 1)
        token, rest = parts[0], (parts[1] if len(parts) > 1 else "")

        if i == 0:
            # Line 1: the record name, with no extension.
            if token != old_name:
                raise ValueError(
                    f"header line 1 names {token!r}, expected {old_name!r}")
            token = new_name
        else:
            # Signal line: <name>.<ext>, or `~` for a null signal.
            if token.startswith(old_name + "."):
                token = new_name + token[len(old_name):]
            elif token != "~":
                raise ValueError(
                    f"signal line references {token!r}, expected {old_name}.*")
        out.append(indent + token + (" " + rest if rest else ""))
    return "".join(out)


def export_record(row: dict, dest_dir: Path) -> dict:
    """Copy one record into the release. Returns its manifest entry."""
    rec_id = row["global_record_id"]
    header_src = Path(row["header_path"])
    data_src = Path(row["data_path"])
    if not header_src.exists():
        raise FileNotFoundError(f"missing header: {header_src}")
    if not data_src.exists():
        raise FileNotFoundError(f"missing signal: {data_src}")

    old_name = header_src.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_dst = dest_dir / f"{rec_id}{data_src.suffix}"
    header_dst = dest_dir / f"{rec_id}.hea"

    # Signal: byte-for-byte. Never decoded, so it cannot be altered.
    shutil.copyfile(data_src, data_dst)
    header_dst.write_text(
        rewrite_header(header_src.read_text(encoding="utf-8"), old_name, rec_id),
        encoding="utf-8")

    return {
        "global_record_id": rec_id,
        "source_dataset": row["source_dataset"],
        "original_record_id": row["original_record_id"],
        "header_file": header_dst,
        "data_file": data_dst,
    }


def link_record(prev: dict, dest_dir: Path) -> dict:
    """Hard-link an already-exported record into a second cohort directory.

    The bytes are identical -- the same record, already renamed and rewritten --
    so the second cohort can share the inode instead of paying for the copy
    again. The directory tree still *looks* like a complete standalone dataset,
    which is the point. Falls back to a real copy across filesystems.

    Caveat worth knowing: hard links mean editing one path edits both. That is
    fine for archived read-only data and wrong for a working copy, which is why
    this is opt-in rather than the default.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in ("header_file", "data_file"):
        src: Path = prev[key]
        dst = dest_dir / src.name
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src)
        except OSError:                            # different filesystem, etc.
            shutil.copyfile(src, dst)
        out[key] = dst
    return {**prev, **out}


def merge_verification(cohort_info: dict) -> dict:
    """Roll the per-cohort verification results into one summary."""
    modes = {c["verification"]["mode"] for c in cohort_info.values()}
    checked = sum(c["verification"]["checked"] for c in cohort_info.values())
    failed = sum(c["verification"]["failed"] for c in cohort_info.values())
    if modes == {"none"}:
        return {"mode": "none", "checked": 0, "failed": 0,
                "summary": "skipped (--verify none)", "failures": []}
    parts = [f"{n}: {c['verification']['summary']}"
             for n, c in cohort_info.items()]
    return {
        "mode": "/".join(sorted(modes)),
        "checked": checked,
        "failed": failed,
        "summary": " | ".join(parts),
        "failures": [f for c in cohort_info.values()
                     for f in c["verification"]["failures"]][:200],
    }


def verify_record(release: Path, rel_base: str, original_base: str) -> str | None:
    """Read the exported record and the original back; compare sample by sample.

    Returns None if identical, or a short reason. Compares the raw ADC array
    (`d_signal`), not the scaled one: that catches a mangled gain or baseline
    field as a shape/scale difference rather than hiding it behind the same
    conversion applied twice.
    """
    import numpy as np
    import wfdb

    try:
        got = wfdb.rdrecord(str(release / rel_base))
    except Exception as exc:                      # noqa: BLE001
        return f"unreadable in release: {type(exc).__name__}: {exc}"
    try:
        want = wfdb.rdrecord(original_base)
    except Exception as exc:                      # noqa: BLE001
        return f"original unreadable: {type(exc).__name__}: {exc}"

    a = np.asarray(got.p_signal, dtype=np.float64)
    b = np.asarray(want.p_signal, dtype=np.float64)
    if a.shape != b.shape:
        return f"shape {a.shape} != original {b.shape}"
    if not np.array_equal(np.nan_to_num(a), np.nan_to_num(b)):
        return f"samples differ (max |d| = {np.nanmax(np.abs(a - b)):.6g} mV)"
    if [s.upper() for s in got.sig_name] != [s.upper() for s in want.sig_name]:
        return "lead order differs"
    if got.fs != want.fs:
        return f"fs {got.fs} != {want.fs}"
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--version", default=None,
                    help="release name (default: next unused vN)")
    ap.add_argument("--cohort", choices=["both", "harmonized", "balanced"],
                    default="both",
                    help="which records to materialise (default: %(default)s)")
    ap.add_argument("--note", default="",
                    help="free text recorded in VERSION.json and the changelog")
    ap.add_argument("--verify", choices=["sample", "all", "none"], default="sample",
                    help=f"read exported records back and compare against the "
                         f"originals (default: sample = {SAMPLE_VERIFY_N} records)")
    ap.add_argument("--layout", choices=["split"], default="split",
                    help="one self-contained directory per cohort "
                         "(harmonized/ and balanced/), each with its own "
                         "signals/, metadata CSV, MANIFEST and checksums")
    ap.add_argument("--hardlink", action="store_true",
                    help="hard-link the balanced cohort's files to the "
                         "harmonized ones instead of copying them: two complete "
                         "directory trees, but the ~1.1 GB overlap is stored "
                         "once. Read-only archives only -- editing one path "
                         "edits both.")
    ap.add_argument("--npy", action="store_true",
                    help="also write signals.npy, one stacked (N, 5000, 12) "
                         "float32 array in mV, plus its row index")
    ap.add_argument("--path-style", choices=["relative", "absolute"],
                    default="relative",
                    help="how the exported CSVs point at signals "
                         "(default: %(default)s, so the release stays portable)")
    ap.add_argument("--check", metavar="VERSION", default=None,
                    help="verify an existing release's checksums and exit")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="smoke test: export at most N records per source")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the version directory if it already exists")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    releases = PIPELINE_ROOT / "data" / "releases"

    if args.check:
        return check_release(releases / args.check)

    hpath = Path(paths["output"]["harmonized_dir"]) / "harmonized_ecg_metadata.csv"
    bpath = Path(paths["output"]["balanced_dir"]) / "balanced_ecg_metadata.csv"

    if not hpath.exists():
        sys.exit(f"harmonized CSV not found: {hpath}\nRun run_pipeline.py first.")
    harmonized = read_csv_rows(hpath)
    balanced, balanced_ids = [], set()
    if bpath.exists():
        balanced = read_csv_rows(bpath)
        balanced_ids = {r["global_record_id"] for r in balanced}
    elif args.cohort in ("both", "balanced"):
        warn(f"balanced CSV not found ({bpath}); exporting harmonized only")

    # Which cohorts get their own self-contained directory.
    cohorts: list[tuple[str, list[dict]]] = []
    if args.cohort in ("both", "harmonized"):
        cohorts.append(("harmonized", harmonized))
    if args.cohort in ("both", "balanced") and balanced:
        cohorts.append(("balanced", balanced))
    if not cohorts:
        sys.exit(f"--cohort {args.cohort} requested but that CSV is missing")

    if args.limit:
        trimmed = []
        for name, rows in cohorts:
            per_source, kept = collections.Counter(), []
            for r in rows:
                if per_source[r["source_dataset"]] < args.limit:
                    per_source[r["source_dataset"]] += 1
                    kept.append(r)
            trimmed.append((name, kept))
            warn(f"--limit {args.limit}: {name} trimmed to {len(kept)} records "
                 f"({dict(per_source)})")
        cohorts = trimmed
        warn("this is a SMOKE TEST, not a real release")

    version = args.version or next_version(releases)
    root = releases / version
    if root.exists():
        if not args.force:
            sys.exit(f"{root} already exists. Pick another --version, or pass "
                     f"--force to overwrite it.")
        warn(f"overwriting existing release {version}")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    print(banner(f"STEP 13 -- EXPORTING RELEASE {version}"))
    info(f"cohorts     : {', '.join(f'{n} ({len(r):,})' for n, r in cohorts)}")
    info(f"destination : {root}")
    info(f"layout      : {args.layout}"
         + ("  (balanced records hard-linked to harmonized)"
            if args.hardlink else ""))

    # ---- one self-contained directory per cohort -----------------------
    # Each gets its own signals/, metadata CSV, MANIFEST and checksums, so a
    # single cohort directory can be copied out on its own and still work.
    cohort_info: dict[str, dict] = {}
    all_failures: list[tuple[str, str]] = []
    total_bytes = 0
    first_manifest: list[dict] = []
    linked = 0
    # id -> already-exported entry, for --hardlink to point the second cohort at
    already: dict[str, dict] = {}

    for name, rows in cohorts:
        cdir = root / name
        cdir.mkdir(parents=True, exist_ok=True)
        info(f"[{name}] copying {len(rows):,} records ...")

        manifest, failures, exported = [], [], {}
        for i, row in enumerate(rows, 1):
            try:
                dest = cdir / "signals" / row["source_dataset"]
                prev = already.get(row["global_record_id"]) if args.hardlink else None
                if prev is not None:
                    entry = link_record(prev, dest)
                    linked += 1
                else:
                    entry = export_record(row, dest)
            except Exception as exc:              # noqa: BLE001 - report, continue
                failures.append((row["global_record_id"],
                                 f"{type(exc).__name__}: {exc}"))
                continue
            exported[entry["global_record_id"]] = entry
            already.setdefault(entry["global_record_id"], entry)
            manifest.append(entry)
            if i % 2000 == 0:
                info(f"  {i:,}/{len(rows):,}")

        if not manifest:
            sys.exit(f"[{name}] nothing exported -- check the paths in the CSV")
        for rec, why in failures[:10]:
            warn(f"  [{name}] {rec}: {why}")
        all_failures += [(f"{name}:{r}", w) for r, w in failures]

        # ---- checksums + manifest, both relative to THIS cohort dir ----
        info(f"[{name}] hashing {2*len(manifest):,} files ...")
        rows_manifest, checksum_lines, cohort_bytes = [], [], 0
        for entry in manifest:
            hea, dat = entry["header_file"], entry["data_file"]
            hea_rel = hea.relative_to(cdir).as_posix()
            dat_rel = dat.relative_to(cdir).as_posix()
            hea_sha, dat_sha = sha256_file(hea), sha256_file(dat)
            cohort_bytes += hea.stat().st_size + dat.stat().st_size
            checksum_lines += [f"{hea_sha}  {hea_rel}", f"{dat_sha}  {dat_rel}"]
            rows_manifest.append({
                "global_record_id": entry["global_record_id"],
                "source_dataset": entry["source_dataset"],
                "original_record_id": entry["original_record_id"],
                "record_base": dat_rel.rsplit(".", 1)[0],
                "header_file": hea_rel,
                "data_file": dat_rel,
                "header_sha256": hea_sha,
                "data_sha256": dat_sha,
                "in_balanced": 1 if entry["global_record_id"] in balanced_ids else 0,
            })
        (cdir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n",
                                               encoding="utf-8")
        write_csv(cdir / "MANIFEST.csv", rows_manifest,
                  list(rows_manifest[0].keys()))

        # ---- the metadata CSV, repointed inside this cohort dir --------
        out, dropped = [], 0
        for r in rows:
            entry = exported.get(r["global_record_id"])
            if entry is None:
                dropped += 1
                continue
            r = dict(r)
            base = entry["data_file"]
            stem = (base.relative_to(cdir).as_posix()
                    if args.path_style == "relative" else base.as_posix())
            r["signal_path"] = stem.rsplit(".", 1)[0]
            r["header_path"] = r["signal_path"] + ".hea"
            r["data_path"] = stem
            out.append(r)
        csv_name = f"{name}_ecg_metadata.csv"
        write_csv(cdir / csv_name, out, list(out[0].keys()))

        npy_info = write_npy(cdir, rows_manifest) if args.npy else None
        verify = run_verification(args.verify, cdir, rows_manifest, rows)

        cohort_info[name] = {
            "directory": name,
            "metadata_csv": f"{name}/{csv_name}",
            "records": len(manifest),
            "rows": len(out),
            "files": 2 * len(manifest),
            "bytes": cohort_bytes,
            "gigabytes": round(cohort_bytes / 1e9, 3),
            "failures": len(failures),
            "verification": verify,
            "npy": npy_info,
        }
        total_bytes += cohort_bytes
        if not first_manifest:
            first_manifest = rows_manifest
        info(f"[{name}] done: {len(manifest):,} records, "
             f"{cohort_bytes/1e9:.2f} GB, {verify['summary']}")

    exported_all = already
    failures = all_failures
    n_h = cohort_info.get("harmonized", {}).get("rows", 0)
    n_b = cohort_info.get("balanced", {}).get("rows", 0)
    n_records = sum(c["records"] for c in cohort_info.values())
    n_files = sum(c["files"] for c in cohort_info.values())
    verify_report = merge_verification(cohort_info)
    exported = exported_all
    rows_manifest = first_manifest
    npy_info = {n: c["npy"] for n, c in cohort_info.items() if c["npy"]} or None

    # ---- provenance ----------------------------------------------------
    counts = tally(harmonized, balanced, exported)
    meta = {
        "version": version,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": f"scripts/export_dataset.py {TOOL_VERSION}",
        "note": args.note,
        "cohort_exported": args.cohort,
        "layout": args.layout,
        "hardlinked": bool(args.hardlink),
        "path_style": args.path_style,
        "cohorts": cohort_info,
        "records": {
            "materialised": n_records,
            "distinct": len(exported),
            "harmonized_rows": n_h,
            "balanced_rows": n_b,
            "export_failures": len(failures),
        },
        "files": {"count": n_files, "bytes": total_bytes,
                  "gigabytes": round(total_bytes / 1e9, 3),
                  "hardlinked_files": 2 * linked},
        "counts": counts,
        "inputs": {
            "harmonized_csv": {"path": str(hpath), "sha256": sha256_file(hpath),
                               "rows": len(harmonized)},
            "balanced_csv": ({"path": str(bpath), "sha256": sha256_file(bpath),
                              "rows": len(balanced)} if balanced else None),
        },
        # A release is only reproducible if you know which label decisions
        # produced it, so the configs are fingerprinted too.
        "config": {
            name: sha256_file(PIPELINE_ROOT / "config" / name)
            for name in ("label_mapping.yaml", "paths.yaml")
            if (PIPELINE_ROOT / "config" / name).exists()
        },
        "verification": verify_report,
        "npy": npy_info,
        "failures": [{"record": r, "reason": w} for r, w in failures[:200]],
    }
    (root / "VERSION.json").write_text(json.dumps(meta, indent=2) + "\n",
                                       encoding="utf-8")
    (root / "README.md").write_text(render_readme(meta), encoding="utf-8")
    append_changelog(releases, meta)

    # ---- print ---------------------------------------------------------
    print(banner(f"RELEASE {version} WRITTEN"))
    print(f"  location        : {root}")
    print()
    for name, c in cohort_info.items():
        print(f"  {name+'/':14s} {c['records']:>7,} records  "
              f"{c['files']:>7,} files  {c['gigabytes']:>6.2f} GB  "
              f"-> {Path(c['metadata_csv']).name}")
        print(f"  {'':14s} {c['verification']['summary']}")
    print()
    print(f"  on-disk total   : {total_bytes/1e9:.2f} GB"
          + (f"  ({linked:,} records hard-linked, not duplicated)"
             if linked else ""))
    print(f"  distinct records: {len(exported):,}")
    print()
    print(f"  {'Class':8s} {'harmonized':>12s} {'balanced':>10s}")
    for cls in TARGET_CLASSES:
        print(f"  {cls:8s} {counts['harmonized'].get(cls, 0):12,} "
              f"{counts['balanced'].get(cls, 0):10,}")
    print()
    print(f"  provenance      : {root / 'VERSION.json'}")
    print(f"  changelog       : {releases / 'CHANGELOG.md'}")
    if failures:
        print(f"\n  WARNING: {len(failures)} records failed to export "
              f"(listed in VERSION.json)")
    return 1 if (failures or verify_report["failed"]) else 0


def tally(harmonized, balanced, exported) -> dict:
    out = {}
    for name, rows in (("harmonized", harmonized), ("balanced", balanced)):
        c = collections.Counter()
        for r in rows:
            if r["global_record_id"] not in exported:
                continue
            for cls in TARGET_CLASSES:
                if str(r.get(cls, "0")).strip() in ("1", "1.0"):
                    c[cls] += 1
        out[name] = dict(c)
    out["by_source"] = dict(collections.Counter(
        e["source_dataset"] for e in exported.values()))
    return out


def write_npy(root: Path, rows_manifest: list[dict]) -> dict:
    """One stacked (N, 5000, 12) float32 array in mV, for 1D model training.

    Written incrementally through a memmap: 19,743 records is ~4.7 GB, which
    should not have to fit in RAM to be produced.
    """
    import numpy as np
    import wfdb

    from common import TARGET_N_SAMPLES

    n = len(rows_manifest)
    path = root / "signals.npy"
    info(f"writing {path.name}  ({n:,} x {TARGET_N_SAMPLES} x 12 float32, "
         f"~{n*TARGET_N_SAMPLES*12*4/1e9:.1f} GB) ...")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n, TARGET_N_SAMPLES, 12))
    index, bad = [], 0
    for i, row in enumerate(rows_manifest):
        try:
            rec = wfdb.rdrecord(str(root / row["record_base"]))
            sig = np.asarray(rec.p_signal, dtype=np.float32)
            if sig.shape != (TARGET_N_SAMPLES, 12):
                raise ValueError(f"shape {sig.shape}")
            arr[i] = sig
            leads = [s.upper() for s in rec.sig_name]
        except Exception as exc:                  # noqa: BLE001
            warn(f"  npy: {row['global_record_id']}: {type(exc).__name__}: {exc}")
            arr[i] = 0.0
            leads, bad = [], bad + 1
        index.append({"row": i, "global_record_id": row["global_record_id"],
                      "source_dataset": row["source_dataset"],
                      "lead_order": "|".join(leads)})
        if (i + 1) % 2000 == 0:
            info(f"  {i+1:,}/{n:,}")
    arr.flush()
    write_csv(root / "signals_index.csv", index, list(index[0].keys()))
    return {"file": "signals.npy", "shape": [n, TARGET_N_SAMPLES, 12],
            "dtype": "float32", "units": "mV", "failed_rows": bad,
            "index": "signals_index.csv"}


def run_verification(mode: str, root: Path, rows_manifest: list[dict],
                     source_rows: list[dict]) -> dict:
    if mode == "none":
        return {"mode": "none", "checked": 0, "failed": 0,
                "summary": "skipped (--verify none)", "failures": []}

    originals = {r["global_record_id"]: r for r in source_rows}
    targets = rows_manifest
    if mode == "sample":
        import random
        rng = random.Random(42)
        targets = rng.sample(rows_manifest, min(SAMPLE_VERIFY_N, len(rows_manifest)))

    info(f"verifying {len(targets):,} exported records against their originals ...")
    failures = []
    for i, row in enumerate(targets, 1):
        src = originals.get(row["global_record_id"])
        if src is None:
            continue
        why = verify_record(root, row["record_base"], src["signal_path"])
        if why:
            failures.append({"record": row["global_record_id"], "reason": why})
        if i % 2000 == 0:
            info(f"  {i:,}/{len(targets):,}")

    ok = len(targets) - len(failures)
    summary = (f"{ok:,}/{len(targets):,} records bit-identical to source"
               + (f"  -- {len(failures)} FAILED" if failures else ""))
    for f in failures[:10]:
        warn(f"    {f['record']}: {f['reason']}")
    return {"mode": mode, "checked": len(targets), "failed": len(failures),
            "summary": summary, "failures": failures[:200]}


def check_release(root: Path) -> int:
    """Re-verify an existing release from its own checksums.sha256."""
    if not root.is_dir():
        sys.exit(f"no such release: {root}")
    # One checksums.sha256 per cohort directory; older single-directory
    # releases keep theirs at the top level.
    sum_files = sorted(root.glob("*/checksums.sha256")) or \
        ([root / "checksums.sha256"] if (root / "checksums.sha256").exists() else [])
    if not sum_files:
        sys.exit(f"no checksums.sha256 under {root} -- "
                 f"was this release written by this tool?")

    print(banner(f"CHECKING RELEASE {root.name}"))
    ok = True
    for sums in sum_files:
        base = sums.parent
        label = base.name if base != root else "(root)"
        bad, missing, n = [], [], 0
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            want, rel = line.split("  ", 1)
            p = base / rel
            n += 1
            if not p.exists():
                missing.append(rel)
            elif sha256_file(p) != want:
                bad.append(rel)
            if n % 10000 == 0:
                info(f"  [{label}] {n:,} files")
        print(f"  {label:14s} {n:>7,} files   missing {len(missing)}   "
              f"corrupted {len(bad)}")
        for rel in (missing + bad)[:10]:
            warn(f"    {label}/{rel}")
        ok &= not (bad or missing)

    print("\n" + ("RELEASE INTACT" if ok else "RELEASE DAMAGED"))
    return 0 if ok else 1


def render_readme(meta: dict) -> str:
    c = meta["counts"]
    rows = "\n".join(
        f"| {cls} | {c['harmonized'].get(cls, 0):,} | {c['balanced'].get(cls, 0):,} |"
        for cls in TARGET_CLASSES)
    src = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(c["by_source"].items()))
    note = f"\n**Note:** {meta['note']}\n" if meta["note"] else ""
    cohorts = meta.get("cohorts", {})
    tree, blocks = [], []
    for name, c in cohorts.items():
        tree.append(
            f"{name}/\n"
            f"├── {Path(c['metadata_csv']).name}   {c['rows']:,} rows\n"
            f"├── MANIFEST.csv          {c['records']:,} records, with SHA-256\n"
            f"├── checksums.sha256      {c['files']:,} files\n"
            f"└── signals/<SOURCE>/<global_record_id>.hea + .dat|.mat"
            + (f"\n└── signals.npy          {tuple(c['npy']['shape'])} "
               f"{c['npy']['dtype']}" if c.get("npy") else ""))
        blocks.append(f"| `{name}/` | {c['records']:,} | {c['files']:,} | "
                      f"{c['gigabytes']} | {c['verification']['summary']} |")
    tree_s = "\n\n".join(tree)
    blocks_s = "\n".join(blocks)
    link_note = ""
    if meta.get("hardlinked") and meta["files"].get("hardlinked_files"):
        link_note = (
            f"\n> The two directories overlap by "
            f"{meta['files']['hardlinked_files']//2:,} records. Those are stored "
            f"once and **hard-linked** into both, so the tree costs "
            f"{meta['files']['gigabytes']} GB rather than the sum of its parts. "
            f"Each directory is still complete and independently readable. Do not "
            f"edit files in place: the two paths are the same bytes on disk.\n")
    return f"""# CardioSentry ECG dataset — release `{meta['version']}`

Created {meta['created_utc']} by `{meta['tool']}`.
{note}
**Each cohort directory is a complete, standalone dataset.** Copy out just
`balanced/` and it works on its own — its CSV, its manifest, its checksums and
every signal file it references are inside it. Nothing points back at the
original source distributions.

Signal files were copied byte for byte; only the WFDB header's record name and
signal-file reference were rewritten to match the new filename.

## Cohorts

| Directory | records | files | GB | verification |
| --- | ---: | ---: | ---: | --- |
{blocks_s}
{link_note}
```
{tree_s}
```

## Contents

| Class | harmonized | balanced |
| --- | ---: | ---: |
{rows}

| Source | records |
| --- | ---: |
{src}

`VERSION.json` carries full provenance, including SHA-256 of the input CSVs and
of `config/label_mapping.yaml`, so you can tell whether two releases were built
from the same clinical decisions.

## Verification

{meta['verification']['summary']}

Re-check integrity at any time:

```bash
python scripts/export_dataset.py --check {meta['version']}
# or, per cohort:
cd {meta['version']}/balanced && sha256sum -c checksums.sha256
```

## Reading it

Paths in each CSV are **{meta['path_style']}** to that cohort's own directory.

```python
import pandas as pd, wfdb
df = pd.read_csv("balanced/balanced_ecg_metadata.csv")
rec = wfdb.rdrecord("balanced/" + df.signal_path[0])
print(rec.p_signal.shape, rec.sig_name, rec.fs)
```

## Licensing

The Chongqing ACS/STEMI portion is CC BY-NC-**ND** 4.0. This release is a
compilation of licence-gated source databases (PhysioNet credentialed / DUA)
and **must not be redistributed**. It is for internal use by people who already
hold access to all three sources.
"""


def append_changelog(releases: Path, meta: dict) -> None:
    path = releases / "CHANGELOG.md"
    if not path.exists():
        path.write_text("# Dataset releases\n\n"
                        "Newest first. Written by `scripts/export_dataset.py`.\n\n",
                        encoding="utf-8")
    head = path.read_text(encoding="utf-8")
    marker = "Written by `scripts/export_dataset.py`.\n\n"
    c = meta["counts"]["harmonized"]
    entry = (f"## `{meta['version']}` — {meta['created_utc'][:10]}\n\n"
             f"{meta['records']['materialised']:,} records, "
             f"{meta['files']['gigabytes']} GB. "
             f"STEMI {c.get('STEMI', 0):,} · AF {c.get('AF', 0):,} · "
             f"LVH {c.get('LVH', 0):,} · NORMAL {c.get('NORMAL', 0):,}. "
             f"{meta['verification']['summary']}.\n"
             + (f"\n{meta['note']}\n" if meta["note"] else "")
             + f"\nlabel_mapping.yaml `"
               f"{meta['config'].get('label_mapping.yaml', '?')[:12]}`\n\n")
    path.write_text(head.replace(marker, marker + entry, 1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
