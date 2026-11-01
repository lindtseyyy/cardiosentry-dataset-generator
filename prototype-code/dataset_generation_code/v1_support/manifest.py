"""Accepted-manifest loading + waveform gates (PLAN 3).

Fails loudly with offender lists; nothing is ever skipped silently.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .config import ACCEPT_ACTIONS, EXPECTED_ACCEPTED, TARGET_LABELS
from .labels import parse_target_labels, to_multihot
from .util import V1Error, log, sha256_file
from . import wfdbio


class ManifestError(V1Error):
    pass


_SUBSET_TOP = "V1_10S_STANDARDIZED_SUBSET"


def rebase_subset_path(p: str, subset_dir: Path, rid: str) -> str:
    """Anchor a CSV path onto the runtime subset dir (PLAN v1.10 §3.0).

    `subset_output_path` is stored ABSOLUTE — valid only on the machine
    that cut the subset (using it verbatim fails every record on EC2).
    Everything after the `V1_10S_STANDARDIZED_SUBSET/` part is portable,
    so re-anchor that tail under the subset_dir actually in use (local
    copy or S3-extracted). Already-relative values anchor directly;
    absolute values without the marker fail loudly.
    """
    parts = Path(p).parts
    if _SUBSET_TOP in parts:
        tail = parts[parts.index(_SUBSET_TOP) + 1:]
        if not tail:
            raise ManifestError(f"{rid}: path has no tail: {p!r}")
        return str(subset_dir / Path(*tail))
    if not Path(p).is_absolute():
        return str(subset_dir / p)
    raise ManifestError(
        f"{rid}: cannot rebase absolute path outside the subset: {p!r}")


def load_accepted(subset_dir: str | Path) -> list[dict]:
    """Load subset_records_extended.csv; assert 31,109 rows; parse labels."""
    subset_dir = Path(subset_dir)
    path = subset_dir / "metadata" / "subset_records_extended.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_ACCEPTED:
        raise ManifestError(f"{path}: {len(rows)} rows, need {EXPECTED_ACCEPTED}")
    seen = set()
    for r in rows:
        if r["record_id"] in seen:
            raise ManifestError(f"duplicate record_id {r['record_id']}")
        seen.add(r["record_id"])
        r["subset_output_path"] = rebase_subset_path(
            r["subset_output_path"], subset_dir, r["record_id"])
        labs = parse_target_labels(r["target_labels"])
        if not labs:
            raise ManifestError(f"{r['record_id']}: empty target_labels")
        r["labels_tuple"] = labs
        r["multihot"] = to_multihot(labs)
        if r["action_taken"] not in ACCEPT_ACTIONS:
            raise ManifestError(f"{r['record_id']}: action {r['action_taken']!r}"
                                " not in accepted set")
        if (r.get("exclusion_reason") or "").strip():
            raise ManifestError(f"{r['record_id']}: non-empty exclusion_reason")
    log(f"accepted manifest: {len(rows)} rows, all actions accepted")
    return rows


def cross_check_manifests(subset_dir: str | Path, accepted: list[dict]) -> dict:
    """Verify accepted IDs == standardization_manifest non-excluded rows.

    The 61 unresolved + 19 excluded IDs must be absent; counts must match.
    """
    path = Path(subset_dir) / "manifests" / "standardization_manifest.csv"
    with open(path, newline="") as fh:
        full = list(csv.DictReader(fh))
    acc_ids = {r["record_id"] for r in accepted}
    keep = {r["record_id"] for r in full if r["action_taken"] in ACCEPT_ACTIONS}
    out = {r["record_id"] for r in full if r["action_taken"] not in ACCEPT_ACTIONS}
    if acc_ids != keep:
        missing = sorted(keep - acc_ids)[:10]
        extra = sorted(acc_ids - keep)[:10]
        raise ManifestError(
            f"accepted/manifest mismatch: missing={missing} extra={extra}")
    acts = Counter(r["action_taken"] for r in full)
    log(f"cross-check OK: {len(full)} manifest rows, "
        f"{len(out)} unresolved/excluded absent "
        f"({acts.get('unresolved', 0)} unresolved, {acts.get('excluded', 0)} excluded)")
    return {"manifest_rows": len(full), "omitted": sorted(out),
            "actions": dict(acts)}


def waveform_gate(records: list[dict]) -> dict:
    """Per-record gate: file exists, 12 canonical leads, exactly 10.0 s."""
    offenders: list[str] = []
    formats = Counter()
    for r in records:
        hea = Path(r["subset_output_path"])
        try:
            if not hea.exists():
                raise ManifestError("missing file")
            meta = wfdbio.parse_header(hea)
            wfdbio._assert_convention(meta)  # noqa: SLF001 (same package)
            wfdbio.lead_order_index(meta)
            dur = meta["nsamp"] / meta["fs"]
            if abs(dur - 10.0) > 1e-9:
                raise ManifestError(f"duration {dur}")
            r["fs"] = meta["fs"]
            r["nsamp"] = meta["nsamp"]
            r["header_age"] = meta["comments"].get("Age", "")
            r["header_sex"] = meta["comments"].get("Sex", "")
            wave = hea.with_suffix(".mat") if r["subset_waveform_format"] == "mat" \
                else hea.with_suffix(".dat")
            if not wave.exists():
                raise ManifestError(f"missing waveform {wave.name}")
            formats[r["subset_waveform_format"]] += 1
        except (ManifestError, V1Error, OSError, ValueError) as exc:
            offenders.append(f"{r['record_id']}: {exc}")
    if offenders:
        raise ManifestError(f"waveform gate failed for {len(offenders)}: "
                            f"{offenders[:10]}")
    per_label = Counter()
    for r in records:
        for lab in r["labels_tuple"]:
            per_label[lab] += 1
    log(f"waveform gate OK: {len(records)} records, formats {dict(formats)}")
    return {"formats": dict(formats),
            "per_label": {lab: per_label[lab] for lab in TARGET_LABELS}}


def verify_file_hashes(records: list[dict], expect_csv_hashes: bool = True) -> int:
    """Recompute SHA-256 of every .hea + waveform; compare to CSV columns."""
    bad: list[str] = []
    for r in records:
        hea = Path(r["subset_output_path"])
        wave = hea.with_suffix(".mat") if r["subset_waveform_format"] == "mat" \
            else hea.with_suffix(".dat")
        if expect_csv_hashes:
            if sha256_file(hea) != r["subset_header_sha256"]:
                bad.append(f"{r['record_id']}: header hash")
            if sha256_file(wave) != r["subset_waveform_sha256"]:
                bad.append(f"{r['record_id']}: waveform hash")
    if bad:
        raise ManifestError(f"hash mismatch for {len(bad)}: {bad[:10]}")
    log(f"hash verify OK: {2 * len(records)} files match manifest SHAs")
    return 2 * len(records)
