"""Regression tests for TAR packaging (PLAN §16).

Covers: build_tar determinism (same bytes twice) and verify_tar_listing
accepting the five per-subset metadata.csv files while still rejecting
stray members. No rendering — fixture bytes only.
Run:  python3 -m unittest v1_support.tests.test_package -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v1_support import package as PKG  # noqa: E402
from v1_support.config import SUBSETS  # noqa: E402
from v1_support.util import V1Error  # noqa: E402


def _mini_dataset(root: Path) -> Path:
    ds = root / "dataset"
    full_rows = []
    for s in SUBSETS:
        (ds / s / "images").mkdir(parents=True)
        png = ds / s / "images" / f"r1__{s}.png"
        png.write_bytes(b"\x89PNG-fake-" + s.encode())
        rel = f"{s}/images/{png.name}"
        (ds / s / "metadata.csv").write_text(
            "subset,image_path\n" f"{s},{rel}\n")
        full_rows.append({"subset": s, "image_path": rel})
    with open(ds / "full_metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subset", "image_path"])
        w.writeheader()
        w.writerows(full_rows)
    return ds


class TestPackage(unittest.TestCase):
    def test_tar_accepts_subset_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _mini_dataset(Path(tmp))
            top = list(SUBSETS) + ["full_metadata.csv"]
            tar = PKG.build_tar(ds, Path(tmp) / "a.tar", top)
            PKG.verify_tar_listing(tar["path"], ds / "full_metadata.csv",
                                   set(top))  # must not raise

    def test_tar_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _mini_dataset(Path(tmp))
            top = list(SUBSETS) + ["full_metadata.csv"]
            a = PKG.build_tar(ds, Path(tmp) / "a.tar", top)
            b = PKG.build_tar(ds, Path(tmp) / "b.tar", top)
            self.assertEqual(a["sha256"], b["sha256"])

    def test_tar_members_byte_sorted(self):
        # PLAN §16: LC_ALL=C order. Guards against Path-tuple sorting,
        # which orders 'a/b' vs 'a-b' differently from byte order.
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            ds = _mini_dataset(Path(tmp))
            (ds / "train" / "images" / "a-b.png").write_bytes(b"x")
            (ds / "train" / "images" / "a").mkdir()
            (ds / "train" / "images" / "a" / "b.png").write_bytes(b"x")
            top = list(SUBSETS) + ["full_metadata.csv"]
            tar = PKG.build_tar(ds, Path(tmp) / "a.tar", top)
            with tarfile.open(tar["path"]) as tf:
                names = tf.getnames()
            self.assertEqual(names, sorted(names))

    def test_tar_rejects_stray_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _mini_dataset(Path(tmp))
            (ds / "train" / "stray.txt").write_text("junk")
            top = list(SUBSETS) + ["full_metadata.csv"]
            tar = PKG.build_tar(ds, Path(tmp) / "a.tar", top)
            with self.assertRaises(V1Error):
                PKG.verify_tar_listing(tar["path"], ds / "full_metadata.csv",
                                       set(top))

    def test_tar_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = _mini_dataset(Path(tmp))
            (ds / "train" / "images" / "r1__train.png").unlink()
            top = list(SUBSETS) + ["full_metadata.csv"]
            tar = PKG.build_tar(ds, Path(tmp) / "a.tar", top)
            with self.assertRaises(V1Error):
                PKG.verify_tar_listing(tar["path"], ds / "full_metadata.csv",
                                       set(top))


if __name__ == "__main__":
    unittest.main()
