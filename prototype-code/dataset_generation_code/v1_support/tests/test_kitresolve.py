"""Regression test for the §1 re-run kit_dir wipe (PLAN v1.12).

§1 rebuilds cfg from scratch (kit_dir=""), wiping the path §2 found — and
the §12 approval instruction tells the user to do exactly that. Every kit
consumer must self-heal via render.resolve_kit_dir instead of trusting
cfg.kit_dir; run_batch must refuse an empty/missing kit with an actionable
error instead of leaking a bare `can't open file` from the subprocess.
Run:  python3 -m unittest v1_support.tests.test_kitresolve -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

from v1_support import render as R
from v1_support.util import V1Error


def _fake_kit(root: Path) -> Path:
    kit = root / "kit" / "ecg_image_v1_code" / "ecg-image-generator"
    kit.mkdir(parents=True)
    (kit / R.KIT_SCRIPT).write_text("# stub\n")
    return kit


class TestResolveKitDir(unittest.TestCase):
    def test_valid_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            kit = _fake_kit(Path(td))
            self.assertEqual(R.resolve_kit_dir(td, str(kit)), str(kit))

    def test_empty_recovers_from_work_dir(self):
        # simulates: user re-ran §1 (cfg rebuilt, kit_dir="") then §12
        with tempfile.TemporaryDirectory() as td:
            kit = _fake_kit(Path(td))
            self.assertEqual(R.resolve_kit_dir(td, ""), str(kit))

    def test_stale_path_recovers(self):
        with tempfile.TemporaryDirectory() as td:
            kit = _fake_kit(Path(td))
            got = R.resolve_kit_dir(td, "/nonexistent/kit")
            self.assertEqual(got, str(kit))

    def test_missing_kit_points_at_section_2(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(V1Error) as cm:
                R.resolve_kit_dir(td, "")
            self.assertIn("§2", str(cm.exception))

    def test_run_batch_refuses_empty_kit_dir(self):
        with self.assertRaises(V1Error) as cm:
            R.run_batch("", sys.executable, "/tmp", "/tmp", "Arial.ttf")
        self.assertIn("§1 re-run", str(cm.exception))

    def test_run_batch_refuses_missing_script(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(V1Error) as cm:
                R.run_batch(td, sys.executable, "/tmp", "/tmp", "Arial.ttf")
            self.assertIn("§2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
