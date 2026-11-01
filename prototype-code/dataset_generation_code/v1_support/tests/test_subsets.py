"""Regression tests for eval-subset separation (PLAN v1.2: §11, §14, §15.7).

Covers: subset_for() derivation for all (split, variant) combos, rejection
of invalid combos, and gate_subset_separation() pair completeness —
every eval record owns exactly one <split>_clean + one <split>_stress row
(same record, same split), train rows all carry subset "train".
Run:  python3 -m unittest v1_support.tests.test_subsets -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v1_support import validate as VV  # noqa: E402
from v1_support.config import (EVAL_SUBSETS, SUBSETS, TRAIN_VARIANTS,  # noqa: E402
                               subset_for)


def _row(rid, split, subset, variant):
    return {"record_id": rid, "split": split, "subset": subset,
            "variant_id": variant}


def _eval_pair(rid, split):
    return [_row(rid, split, f"{split}_clean", "clean"),
            _row(rid, split, f"{split}_stress", "stress")]


def _train_rows(rid):
    return [_row(rid, "train", "train", v) for v in TRAIN_VARIANTS]


class TestSubsetFor(unittest.TestCase):
    def test_eval_subsets(self):
        self.assertEqual(EVAL_SUBSETS,
                         ("val_clean", "val_stress",
                          "test_clean", "test_stress"))
        self.assertEqual(SUBSETS, ("train",) + EVAL_SUBSETS)

    def test_train_variants_map_to_train(self):
        for v in TRAIN_VARIANTS:
            self.assertEqual(subset_for("train", v), "train")

    def test_eval_mapping(self):
        self.assertEqual(subset_for("val", "clean"), "val_clean")
        self.assertEqual(subset_for("val", "stress"), "val_stress")
        self.assertEqual(subset_for("test", "clean"), "test_clean")
        self.assertEqual(subset_for("test", "stress"), "test_stress")

    def test_rejects_augmented_in_eval(self):
        with self.assertRaises(ValueError):
            subset_for("val", "v01")
        with self.assertRaises(ValueError):
            subset_for("test", "v03")

    def test_rejects_stress_in_train(self):
        with self.assertRaises(ValueError):
            subset_for("train", "stress")

    def test_rejects_unknown_split(self):
        with self.assertRaises(ValueError):
            subset_for("dev", "clean")


class TestGateSubsetSeparation(unittest.TestCase):
    def test_happy_path(self):
        rows = (_eval_pair("v1", "val") + _eval_pair("v2", "val")
                + _eval_pair("t1", "test") + _train_rows("r1"))
        VV.gate_subset_separation(rows)  # must not raise

    def test_missing_stress_fails(self):
        rows = [_row("v1", "val", "val_clean", "clean")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_cross_split_pair_fails(self):
        rows = [_row("v1", "val", "val_clean", "clean"),
                _row("v1", "test", "test_stress", "stress")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_subset_variant_mismatch_fails(self):
        rows = [_row("v1", "val", "val_stress", "clean"),
                _row("v1", "val", "val_stress", "stress")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_subset_split_mismatch_fails(self):
        rows = [_row("v1", "val", "test_clean", "clean"),
                _row("v1", "val", "val_stress", "stress")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_duplicate_subset_row_fails(self):
        rows = (_eval_pair("v1", "val")
                + [_row("v1", "val", "val_clean", "clean")])
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_train_subset_must_be_train(self):
        rows = [_row("r1", "train", "val_clean", "clean")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_train_variant_count_enforced(self):
        rows = [_row("r1", "train", "train", "clean")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)

    def test_unknown_subset_value_fails(self):
        rows = [_row("v1", "val", "val", "clean"),
                _row("v1", "val", "val_stress", "stress")]
        with self.assertRaises(VV.ValidationError):
            VV.gate_subset_separation(rows)


if __name__ == "__main__":
    unittest.main()
