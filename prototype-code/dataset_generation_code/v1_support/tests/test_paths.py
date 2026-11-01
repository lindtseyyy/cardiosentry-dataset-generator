"""Regression tests for subset path rebasing (PLAN v1.10 §3.0).

subset_output_path is stored ABSOLUTE (valid only on the cutting
machine); load_accepted must re-anchor the portable tail under the
runtime subset_dir or every record fails on EC2 with "missing file".
Run:  python3 -m unittest v1_support.tests.test_paths -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v1_support.manifest import (ManifestError,  # noqa: E402
                                 rebase_subset_path)

LOCAL = ("/home/lindtseyxvii/Codes/cardiosentry/dataset-files/MAIN_DATASET/"
         "V1_10S_STANDARDIZED_SUBSET/records/chapman_shaoxing/g1/JS00001.hea")
EC2DIR = Path("/home/ubuntu/CardioSentry/outputs/input/"
              "V1_10S_STANDARDIZED_SUBSET")
LOCALDIR = Path("/home/lindtseyxvii/Codes/cardiosentry/dataset-files/"
                "MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET")


class TestRebase(unittest.TestCase):
    def test_absolute_rebases_onto_runtime_dir(self):
        got = rebase_subset_path(LOCAL, EC2DIR, "r1")
        self.assertEqual(got, str(
            EC2DIR / "records/chapman_shaoxing/g1/JS00001.hea"))

    def test_local_dir_reproduces_original(self):
        # On the cutting machine the rebase must be a no-op.
        self.assertEqual(rebase_subset_path(LOCAL, LOCALDIR, "r1"), LOCAL)

    def test_idempotent_when_dir_has_marker(self):
        once = rebase_subset_path(LOCAL, EC2DIR, "r1")
        self.assertEqual(rebase_subset_path(once, EC2DIR, "r1"), once)

    def test_relative_anchors_directly(self):
        got = rebase_subset_path("records/a/b.hea", EC2DIR, "r1")
        self.assertEqual(got, str(EC2DIR / "records/a/b.hea"))

    def test_absolute_without_marker_raises(self):
        with self.assertRaises(ManifestError):
            rebase_subset_path("/elsewhere/x.hea", EC2DIR, "r1")

    def test_marker_without_tail_raises(self):
        with self.assertRaises(ManifestError):
            rebase_subset_path("/x/V1_10S_STANDARDIZED_SUBSET", EC2DIR, "r1")


if __name__ == "__main__":
    unittest.main()
