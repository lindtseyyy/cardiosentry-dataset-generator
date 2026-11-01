"""Regression test for the §12 display loop (PLAN v1.11).

56 full-size inline PNGs (~100 MB over IOPub) stalled the kernel
mid-cell on EC2. §12 must display thumbnails (full files stay in
preview_dir). Executes the REAL loop slice from the built notebook
against synthetic PNGs — no dataset needed.
Run:  python3 -m unittest v1_support.tests.test_thumbnails -v
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))


def display_loop_source() -> str:
    nb = json.loads((NB_DIR / "v1_image_generation.ipynb").read_text())
    cells = nb["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "## 12." in c["source"]:
            src = cells[i + 1]["source"]
            assert cells[i + 1]["cell_type"] == "code"
            break
    else:
        raise AssertionError("§12 cell moved; update this test")
    start = src.index("from pathlib import Path as _P")
    end = src.index('print("(zoomed 320×200 crop above)")') + len(
        'print("(zoomed 320×200 crop above)")')
    loop = src[start:end]
    assert "for rid in PREVIEW_IDS:" in loop, "loop moved; update test"
    return loop


class TestThumbnails(unittest.TestCase):
    def test_inline_images_are_thumbnails(self):
        from PIL import Image
        shown = []

        def fake_display(im, *a, **k):
            shown.append(im.size)

        ipython = types.ModuleType("IPython")
        ipython_display = types.ModuleType("IPython.display")
        ipython_display.display = fake_display
        ipython.display = ipython_display
        with tempfile.TemporaryDirectory() as tmp:
            rids = ["rec_a", "rec_b"]
            for rid in rids:
                for suf in ("__clean.png", "__stress.png", "__v01.png",
                            "__v02.png"):
                    Image.new("RGB", (1024, 768), (250, 250, 250)).save(
                        Path(tmp) / f"{rid}{suf}")
            pre_ipython, pre_disp = (sys.modules.get("IPython"),
                                     sys.modules.get("IPython.display"))
            sys.modules["IPython"] = ipython
            sys.modules["IPython.display"] = ipython_display
            try:
                ns = {"__name__": "__main__",
                      "cfg": SimpleNamespace(preview_dir=tmp),
                      "PREVIEW_IDS": rids}
                exec(compile(display_loop_source(), "<nb-s12-loop>",
                             "exec"), ns)
            finally:
                for name, mod in (("IPython", pre_ipython),
                                  ("IPython.display", pre_disp)):
                    if mod is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = mod
        # 2 records x (4 thumbnails + 1 crop); order per record:
        # 4 thumbs then 1 crop
        self.assertEqual(len(shown), 10)
        got_thumbs = [s for i, s in enumerate(shown) if i % 5 < 4]
        got_crops = [s for i, s in enumerate(shown) if i % 5 == 4]
        self.assertEqual(len(got_thumbs), 8)
        self.assertEqual(len(got_crops), 2)
        for w, h in got_thumbs:
            self.assertLessEqual(w, 512)
            self.assertLessEqual(h, 384)
        for c in got_crops:
            self.assertEqual(c, (320, 200))


if __name__ == "__main__":
    unittest.main()
