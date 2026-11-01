"""Regression test for the INT16_MIN staging guard (PLAN v1.13).

write_dat rejected any signal with abs() > 32767 — but -32768 (INT16_MIN)
is a legal int16 value (ADC negative rail; 25 accepted records contain
it, e.g. ningbo__JS23786). The guard must accept the true int16 range
[-32768, 32767] and reject only genuine overflow. Synthetic signals only:
runs anywhere, no dataset needed.
Run:  python3 -m unittest v1_support.tests.test_wfdbio_int16 -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

from v1_support import wfdbio
from v1_support.util import V1Error


def _sig(values):
    sig = np.zeros((12, 64), dtype=np.float64)
    sig[:, :] = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return sig


class TestWriteDatInt16Range(unittest.TestCase):
    def test_int16_min_rail_round_trips_exactly(self):
        sig = _sig(0.0)
        sig[0, :26] = -32.768   # INT16_MIN rail, as in ningbo__JS23786
        sig[1, 0] = 32.767      # INT16_MAX
        with tempfile.TemporaryDirectory() as td:
            hea, _ = wfdbio.write_dat(Path(td) / "rail", sig, 500.0)
            back, _ = wfdbio.read_record_mv(hea)
        a = np.round(sig * 1000).astype(np.int64)
        b = np.round(back * 1000).astype(np.int64)
        self.assertTrue(np.array_equal(a, b))

    def test_true_overflow_above_still_rejected(self):
        sig = _sig(0.0)
        sig[3, 10] = 32.768     # round() -> 32768, unrepresentable
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(V1Error):
                wfdbio.write_dat(Path(td) / "hi", sig, 500.0)

    def test_true_overflow_below_still_rejected(self):
        sig = _sig(0.0)
        sig[5, 10] = -32.769    # round() -> -32769, unrepresentable
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(V1Error):
                wfdbio.write_dat(Path(td) / "lo", sig, 500.0)


if __name__ == "__main__":
    unittest.main()
