"""Equivalence test: gate re-derive == chunk.py derivation (PLAN v1.16).

gate_pixels byte-compares re-derived clean/stress finals against shipped
PNGs. The re-derive once saved with default PNG params while chunk.py
uses optimize=True — identical pixels, different bytes, phantom
"nondeterministic re-derive" on EC2. This test mirrors chunk.py's steps
literally (mode pipeline + save params) and pins rederive_final() to
byte-equality for clean and stress, so future drift fails here.
Run:  python3 -m unittest v1_support.tests.test_rederive -v
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

from v1_support import resize as RS
from v1_support import validate as VV
from v1_support import variants as V


def _master(path: Path) -> None:
    arr = np.full((850, 1100, 3), 250, dtype=np.uint8)
    arr[::25, :] = (255, 180, 180)     # fake grid lines
    arr[:, ::25] = (255, 180, 180)
    arr[400:450, 100:1000] = 20        # fake trace block
    Image.fromarray(arr, "RGB").save(path, "PNG")


def _textures(d: Path) -> None:
    rng = np.random.default_rng(7)
    for i in range(2):
        t = (rng.normal(200, 25, (48, 64, 3))).clip(0, 255).astype(np.uint8)
        Image.fromarray(t, "RGB").save(d / f"tex{i}.jpg", "JPEG")


def _chunk_way(master_path: Path, vid: str, tex_dir: Path) -> bytes:
    """Literal mirror of _process_chunk's clean/stress derivation + save."""
    arial = Image.open(master_path).convert("RGB")
    if vid == "clean":
        view, _ = V.make_clean(arial.convert("L"))
    else:
        view, _ = V.make_stress(arial.convert("L"), tex_dir)
    final, _ = RS.to_final(view)
    buf = io.BytesIO()
    final.save(buf, "PNG", optimize=True)
    return buf.getvalue()


class TestRederiveEquivalence(unittest.TestCase):
    def test_clean_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "m.png"
            _master(mp)
            self.assertEqual(VV.rederive_final(mp, "clean", td),
                             _chunk_way(mp, "clean", Path(td)))

    def test_stress_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "m.png"
            _master(mp)
            _textures(Path(td))
            self.assertEqual(VV.rederive_final(mp, "stress", td),
                             _chunk_way(mp, "stress", Path(td)))

    def test_default_save_params_would_differ(self):
        # guards the guard: proves optimize=True is load-bearing, i.e. this
        # test can actually catch the v1.16 bug class.
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "m.png"
            _master(mp)
            arial = Image.open(mp).convert("RGB")
            view, _ = V.make_clean(arial.convert("L"))
            final, _ = RS.to_final(view)
            plain, opt = io.BytesIO(), io.BytesIO()
            final.save(plain, "PNG")
            final.save(opt, "PNG", optimize=True)
            self.assertNotEqual(plain.getvalue(), opt.getvalue())
            self.assertEqual(np.asarray(Image.open(io.BytesIO(plain.getvalue()))).tobytes(),
                             np.asarray(Image.open(io.BytesIO(opt.getvalue()))).tobytes())


if __name__ == "__main__":
    unittest.main()
