"""Build v1_support/data patient artifacts (run HERE, where upstream lives).

Joins:
  * ptb-xl-clean-explicit-normal -> ptbxl_database by upstream filename
  * ptb-xl (HR)                  -> upstream signals by byte-exact content hash

Outputs (shipped in the code bundle so EC2 needs no upstream copy):
  v1_support/data/ptbxl_patient_map.csv      record_id -> patient + provenance
  v1_support/data/ptbxl_database_extract.csv filename_hr,ecg_id,patient_id,...
  v1_support/data/PROVENANCE.txt

EC2 re-verifies every row (patients.verify_map): filename rows are recomputed
from the extract; content rows by re-hashing the subset signal.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SUBSET_DEFAULT = ("/home/lindtseyxvii/Codes/cardiosentry/dataset-files"
                  "/MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET")
V1_DEFAULT = "/home/lindtseyxvii/Codes/cardiosentry/dataset-files/MAIN_DATASET/V1"
UP_DEFAULT = "/home/lindtseyxvii/Codes/cardiosentry/dataset-files/ptb-xl-dataset"


def norm_pid(raw: str) -> str:
    raw = str(raw).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return f"ptbxl:{raw}"


def leadmajor_int16_bytes_subset(row: dict) -> bytes:
    hea = Path(row["subset_output_path"])
    if row["subset_waveform_format"] == "mat":
        arr = np.asarray(loadmat(str(hea.with_suffix(".mat")))["val"])
        if arr.shape[0] != 12:
            arr = arr.T
        return np.ascontiguousarray(arr.astype("<i2")).tobytes()
    with open(hea) as fh:
        fl = fh.readline().split()
    n = int(fl[1]) * int(fl[3])
    raw = np.fromfile(str(hea.with_suffix(".dat")), dtype="<i2", count=n)
    return np.ascontiguousarray(raw.reshape(int(fl[3]), 12).T).tobytes()


def main(subset_dir: str = SUBSET_DEFAULT, v1_dir: str = V1_DEFAULT,
         upstream_dir: str = UP_DEFAULT) -> None:
    t0 = time.time()
    out_dir = HERE / "v1_support" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(v1_dir) / "metadata" / "ptbxl_reference" / "ptbxl_database.csv"
    up: dict[str, dict] = {}
    with open(db_path, newline="") as fh:
        for row in csv.DictReader(fh):
            up[row["filename_hr"]] = row
    print(f"database rows: {len(up)}")

    with open(out_dir / "ptbxl_database_extract.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename_hr", "ecg_id", "patient_id", "strat_fold", "age", "sex"])
        for key in sorted(up):
            r = up[key]
            w.writerow([key, r["ecg_id"], norm_pid(r["patient_id"]),
                        r["strat_fold"], r["age"], r["sex"]])

    # upstream content map (streaming; keep hashes only)
    h2r: dict[str, str] = {}
    n_up = 0
    for rec in (Path(upstream_dir) / "RECORDS").read_text().split():
        if not rec.startswith("records500/"):
            continue
        base = Path(upstream_dir) / rec
        try:
            with open(base.with_suffix(".hea")) as fh:
                fl = fh.readline().split()
            if not (int(fl[1]) == 12 and int(fl[2]) == 500 and int(fl[3]) == 5000):
                continue
            raw = np.fromfile(str(base) + ".dat", dtype="<i2")
            if raw.size != 60000:
                continue
            h2r.setdefault(hashlib.sha256(
                np.ascontiguousarray(raw.reshape(5000, 12).T).tobytes()
            ).hexdigest(), rec)
            n_up += 1
            del raw
        except Exception:
            continue
    print(f"upstream 500 Hz signals hashed: {n_up}")

    with open(Path(subset_dir) / "metadata" / "subset_records_extended.csv",
              newline="") as fh:
        sub = list(csv.DictReader(fh))

    rows_out = []
    methods = Counter()
    for x in sub:
        rid = x["record_id"]
        src = x["source_dataset"]
        if src == "ptb-xl-clean-explicit-normal":
            v1p = x["original_v1_path"]
            key = v1p[v1p.find("records500/"):].rsplit(".", 1)[0]
            db = up.get(key)
            if db is None:
                raise SystemExit(f"filename join miss: {rid} -> {key}")
            sig = leadmajor_int16_bytes_subset(x)
            rows_out.append([rid, src, norm_pid(db["patient_id"]), "filename",
                             key, db["ecg_id"], db["strat_fold"],
                             hashlib.sha256(sig).hexdigest(),
                             db["age"], db["sex"], "", ""])
            methods["filename"] += 1
        elif src == "ptb-xl":
            sig = leadmajor_int16_bytes_subset(x)
            m = h2r.get(hashlib.sha256(sig).hexdigest())
            if m is None or m not in up:
                methods["unrecoverable_incomplete_upstream"] += 1
                continue
            db = up[m]
            with open(x["subset_output_path"]) as fh:
                hea = fh.read()
            import re
            ma = re.search(r"# Age: (\d+)", hea)
            ms = re.search(r"# Sex: (\w+)", hea)
            age_agree = ""
            if ma and db["age"] not in ("", None):
                age_agree = str(abs(int(ma.group(1)) - float(db["age"])) <= 1)
            sex_agree = ""
            if ms:
                ds = {"1": "Male", "0": "Female"}.get(str(db["sex"]).strip(),
                                                      str(db["sex"]))
                sex_agree = str(ms.group(1) == ds)
            rows_out.append([rid, src, norm_pid(db["patient_id"]), "content",
                             m, db["ecg_id"], db["strat_fold"],
                             hashlib.sha256(sig).hexdigest(),
                             db["age"], db["sex"], age_agree, sex_agree])
            methods["content"] += 1
        else:
            methods["no_upstream_identity"] += 1

    with open(out_dir / "ptbxl_patient_map.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "source_dataset", "patient_id", "method",
                    "upstream_filename_hr", "ecg_id", "strat_fold",
                    "signal_sha256", "db_age", "db_sex", "age_agree", "sex_agree"])
        w.writerows(sorted(rows_out))

    pats = {r[2] for r in rows_out}
    prov = (f"ptbxl patient map provenance\n"
            f"built_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"subset: {subset_dir}\n"
            f"database: {db_path} ({len(up)} rows)\n"
            f"upstream_signals: {upstream_dir} ({n_up} records500 hashed)\n"
            f"methods: {dict(methods)}\n"
            f"distinct_patients: {len(pats)}\n"
            f"note: 567 HR misses have no counterpart even at 8uV coarse match;\n"
            f"  local upstream copy holds 19526/21799 records500 files.\n")
    (out_dir / "PROVENANCE.txt").write_text(prov)
    print(dict(methods), f"patients={len(pats)}",
          f"elapsed={(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:])
