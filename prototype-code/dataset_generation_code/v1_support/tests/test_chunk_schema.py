"""Regression test for the missing hash column false-LEAKAGE (PLAN v1.15).

Chunk rows never carried v1_waveform_sha256, so gate_leakage read ''
for every row and cried LEAKAGE on a phantom key. Locks the schema
contract (gate keys ⊆ CHUNK_FIELDS), the no-false-positive behavior on
real-schema rows, the manifest backfill (atomic, idempotent, strict),
and the gate's all-empty-column pre-check.
Run:  python3 -m unittest v1_support.tests.test_chunk_schema -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

from v1_support import chunk as CH
from v1_support import validate as VV
from v1_support.util import V1Error

GATE_KEYS = ("record_id", "dup_group_id", "v1_waveform_sha256", "patient_id")


def _row(rid, split, h):
    r = {k: "" for k in CH.CHUNK_FIELDS}
    r.update({"record_id": rid, "dup_group_id": f"g-{rid}",
              "v1_waveform_sha256": h, "patient_id": f"p-{rid}",
              "split": split})
    return r


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class TestChunkSchema(unittest.TestCase):
    def test_gate_keys_in_chunk_fields(self):
        self.assertLessEqual(set(GATE_KEYS), set(CH.CHUNK_FIELDS))

    def test_real_schema_rows_pass_leakage(self):
        rows = [_row("a", "train", "h1"), _row("b", "val", "h2"),
                _row("c", "test", "h3")]
        VV.gate_leakage(rows)  # must not raise

    def test_gate_rejects_all_empty_hash_column(self):
        rows = [_row("a", "train", ""), _row("b", "val", "")]
        with self.assertRaises(VV.ValidationError) as cm:
            VV.gate_leakage(rows)
        self.assertIn("v1_waveform_sha256", str(cm.exception))
        self.assertNotIn("LEAKAGE", str(cm.exception))

    def test_backfill_adds_column_atomically(self):
        old = [k for k in CH.CHUNK_FIELDS if k != "v1_waveform_sha256"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chunk_00000.csv"
            _write_csv(p, [{k: ("a" if k == "record_id" else "x")
                            for k in old}], old)
            rep = CH.backfill_hash_column(td, {"a": "HASH-A"})
            self.assertEqual(rep, {"files": 1, "added": 1, "verified": 0})
            with open(p, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["v1_waveform_sha256"], "HASH-A")
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_backfill_idempotent_second_run(self):
        old = [k for k in CH.CHUNK_FIELDS if k != "v1_waveform_sha256"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chunk_00000.csv"
            _write_csv(p, [{k: ("a" if k == "record_id" else "x")
                            for k in old}], old)
            CH.backfill_hash_column(td, {"a": "HASH-A"})
            rep = CH.backfill_hash_column(td, {"a": "HASH-A"})
            self.assertEqual(rep, {"files": 1, "added": 0, "verified": 1})

    def test_backfill_refuses_unknown_record(self):
        old = [k for k in CH.CHUNK_FIELDS if k != "v1_waveform_sha256"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chunk_00000.csv"
            _write_csv(p, [{k: ("ghost" if k == "record_id" else "x")
                            for k in old}], old)
            with self.assertRaises(V1Error):
                CH.backfill_hash_column(td, {"a": "HASH-A"})

    def test_backfill_refuses_conflict_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chunk_00000.csv"
            _write_csv(p, [_row("a", "train", "WRONG")], CH.CHUNK_FIELDS)
            with self.assertRaises(V1Error):
                CH.backfill_hash_column(td, {"a": "RIGHT"})
            with self.assertRaises(V1Error):
                CH.backfill_hash_column(td, {"a": ""})


if __name__ == "__main__":
    unittest.main()
