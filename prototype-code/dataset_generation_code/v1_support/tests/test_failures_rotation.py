"""Regression test for stale failures.csv on §13 resume (PLAN v1.14).

A chunk failure appends a row and kills the §13 cell; the resumed run
that overcomes it must not trip the end-of-§13 "failures.csv empty"
assert on the stale row. rotate_failures_log moves data-bearing logs
aside (audit preserved) and leaves clean/missing logs alone.
Run:  python3 -m unittest v1_support.tests.test_failures_rotation -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

from v1_support import chunk as CH


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chunk_id", "stage", "error"])
        w.writerows(rows)


class TestRotateFailuresLog(unittest.TestCase):
    def test_stale_rows_rotate_with_audit_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "failures.csv"
            _write(p, [[9, "chunk", "V1Error: staging failed for x"]])
            name = CH.rotate_failures_log(p)
            self.assertTrue(name.startswith("failures_"))
            self.assertFalse(p.exists())
            with open(Path(td) / name, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["chunk_id"], "9")

    def test_header_only_log_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "failures.csv"
            _write(p, [])
            self.assertIsNone(CH.rotate_failures_log(p))
            self.assertTrue(p.exists())
            self.assertEqual(list(Path(td).glob("failures_*.csv")), [])

    def test_missing_log_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "failures.csv"
            self.assertIsNone(CH.rotate_failures_log(p))

    def test_resume_story_end_to_end(self):
        # run 1 fails -> row; run 2 rotates, stays clean, end assert passes
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "failures.csv"
            CH._append_failures(p, 9, "chunk", RuntimeError("boom"))
            CH.rotate_failures_log(p)
            if not p.exists():
                _write(p, [])
            with open(p, newline="") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 0)


if __name__ == "__main__":
    unittest.main()
