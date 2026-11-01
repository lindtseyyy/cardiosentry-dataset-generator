"""Patient map loading + verification (PLAN 0.2, 4.1).

The map is precomputed where the upstream copy lives (build_patient_map.py)
and shipped in v1_support/data. Every row is re-verified at notebook time:

  * filename rows: join recomputed from the bundled database extract
  * content rows: subset signal re-hashed and compared to stored signal_sha256

No upstream copy is needed on EC2.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .util import V1Error, log
from .wfdbio import signal_int16_hash_mv, read_record_mv

DATA_DIR = Path(__file__).resolve().parent / "data"
MAP_FILE = DATA_DIR / "ptbxl_patient_map.csv"
DB_FILE = DATA_DIR / "ptbxl_database_extract.csv"


def load_database_extract() -> dict[str, dict]:
    out = {}
    with open(DB_FILE, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename_hr"]] = row
    return out


def load_map() -> dict[str, dict]:
    out = {}
    with open(MAP_FILE, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["record_id"]] = row
    return out


def verify_map(records: list[dict]) -> dict[str, dict]:
    """Verify every map row; return record_id -> patient info for ALL records.

    Every accepted record gets an entry: recovered ones carry patient_id +
    method; the rest carry method no_upstream_identity /
    unrecoverable_incomplete_upstream with empty patient_id.
    """
    db = load_database_extract()
    pmap = load_map()
    by_id = {r["record_id"]: r for r in records}
    unknown = [rid for rid in pmap if rid not in by_id]
    if unknown:
        raise V1Error(f"patient map has {len(unknown)} unknown record_ids: "
                      f"{unknown[:5]}")
    out: dict[str, dict] = {}
    checked = {"filename": 0, "content": 0}
    for r in records:
        rid = r["record_id"]
        row = pmap.get(rid)
        if row is None:
            if r["source_dataset"] == "ptb-xl":
                method = "unrecoverable_incomplete_upstream"
            else:
                method = "no_upstream_identity"
            out[rid] = {"patient_id": "", "method": method,
                        "upstream_filename_hr": "", "ecg_id": "",
                        "strat_fold": ""}
            continue
        if row["method"] == "filename":
            key = row["upstream_filename_hr"]
            v1p = r["original_v1_path"]
            expect = v1p[v1p.find("records500/"):].rsplit(".", 1)[0]
            if key != expect or key not in db:
                raise V1Error(f"{rid}: filename join mismatch {key} vs {expect}")
            if db[key]["patient_id"] != row["patient_id"]:
                raise V1Error(f"{rid}: patient mismatch vs database extract")
            checked["filename"] += 1
        elif row["method"] == "content":
            sig, _ = read_record_mv(r["subset_output_path"])
            if signal_int16_hash_mv(sig) != row["signal_sha256"]:
                raise V1Error(f"{rid}: subset signal hash != join-time hash; "
                              "file changed or map corrupt")
            if row["upstream_filename_hr"] not in db:
                raise V1Error(f"{rid}: upstream file not in database extract")
            checked["content"] += 1
        else:
            raise V1Error(f"{rid}: unknown method {row['method']}")
        out[rid] = {"patient_id": row["patient_id"], "method": row["method"],
                    "upstream_filename_hr": row["upstream_filename_hr"],
                    "ecg_id": row["ecg_id"], "strat_fold": row["strat_fold"]}
    log(f"patient map verified: filename={checked['filename']} "
        f"content={checked['content']}")
    return out
