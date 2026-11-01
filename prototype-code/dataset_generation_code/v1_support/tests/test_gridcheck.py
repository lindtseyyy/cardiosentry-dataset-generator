"""Regression tests for gridcheck (spectral minor-grid validator).

Covers: exact synthetic periods (incl. non-integer-bin, exercising sub-bin
refinement), tilted grids, flat-image failure, and gate_report logic.
Run:  python3 -m unittest v1_support.tests.test_gridcheck -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v1_support import gridcheck as GC  # noqa: E402


def synth_grid(n=200, period=16.0, angle_deg=0.0, minor_period=None,
               noise=0.0, seed=0):
    """Grayscale synthetic grid: thin dark lines on paper white."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    img = np.full((n, n), 250.0)
    img[(xx % period) < 1.0] = 110.0
    img[(yy % period) < 1.0] = 110.0
    if minor_period:
        img[(xx % minor_period) < 0.6] = np.minimum(
            img[(xx % minor_period) < 0.6], 190.0)
        img[(yy % minor_period) < 0.6] = np.minimum(
            img[(yy % minor_period) < 0.6], 190.0)
    if noise:
        img = img + rng.normal(0, noise, size=img.shape)
    if angle_deg:
        from PIL import Image
        img = np.asarray(Image.fromarray(img.astype(np.uint8)).rotate(
            angle_deg, resample=Image.BICUBIC, expand=False)).astype(float)
    return img


class TestMeasureWindow(unittest.TestCase):
    def test_exact_period(self):
        arr = synth_grid(period=16.0)
        m = GC.measure_window(arr, (0, 0, 200, 200), None, 16.0)
        self.assertTrue(m["ok"], m)
        self.assertAlmostEqual(m["px"], 16.0, delta=0.16)  # 1%
        self.assertAlmostEqual(m["py"], 16.0, delta=0.16)

    def test_non_integer_bin(self):
        # 200/17.79 = 11.24: peak between bins -> needs sub-bin refinement
        arr = synth_grid(period=17.79)
        m = GC.measure_window(arr, (0, 0, 200, 200), None, 17.79)
        self.assertTrue(m["ok"], m)
        self.assertAlmostEqual(m["px"], 17.79, delta=0.36)  # 2%

    def test_minor_grid_ignored(self):
        # major 17.79 + strong minor 3.558: estimator must lock major
        arr = synth_grid(period=17.79, minor_period=17.79 / 5)
        m = GC.measure_window(arr, (0, 0, 200, 200), None, 17.79)
        self.assertTrue(m["ok"], m)
        self.assertAlmostEqual(m["px"], 17.79, delta=0.36)

    def test_tilted(self):
        arr = synth_grid(period=17.79, angle_deg=2.5)
        m = GC.measure_window(arr, (0, 0, 200, 200), None, 17.79)
        self.assertTrue(m["ok"], m)
        self.assertAlmostEqual(m["px"], 17.79, delta=0.54)  # 3%

    def test_noisy(self):
        arr = synth_grid(period=16.0, noise=6.0)
        m = GC.measure_window(arr, (0, 0, 200, 200), None, 16.0)
        self.assertTrue(m["ok"], m)
        self.assertAlmostEqual(m["px"], 16.0, delta=0.32)

    def test_flat_fails(self):
        m = GC.measure_window(np.full((200, 200), 128.0), (0, 0, 200, 200),
                              None, 16.0)
        self.assertFalse(m["ok"])


class TestGates(unittest.TestCase):
    def _m(self, path, dev, ok=True):
        return {"path": path, "nominal": 16.0,
                "windows": [{"ok": ok, "dev_x": dev, "dev_y": -dev,
                             "px": 16 * (1 + dev), "py": 16 * (1 - dev)}
                            for _ in range(9)],
                "n_ok": 9 if ok else 0, "n_failed": 0 if ok else 9,
                "max_dev": abs(dev) if ok else None}

    def test_pass(self):
        m = self._m("a.png", 0.01)
        rep = GC.gate_report([m], [(m, self._m("b.png", 0.012))])
        self.assertTrue(rep["pass"], rep)

    def test_nominal_fail(self):
        m = self._m("a.png", 0.06)
        rep = GC.gate_report([m], [])
        self.assertFalse(rep["pass"])

    def test_failed_windows(self):
        m = self._m("a.png", 0.0, ok=False)
        rep = GC.gate_report([m], [])
        self.assertFalse(rep["pass"])


if __name__ == "__main__":
    unittest.main()
