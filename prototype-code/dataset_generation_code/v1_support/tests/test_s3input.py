"""Regression tests for S3 input fetch (PLAN v1.3 §3).

Covers: URI parsing, sidecar parsing, SHA-gated download reuse, the
traversal-safe extractor (incl. attack rejection), and fetch_input_subset
end to end against a fake in-memory S3 client. No network.
Run:  python3 -m unittest v1_support.tests.test_s3input -v
"""
import hashlib
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v1_support import s3util as S3U  # noqa: E402
from v1_support.util import V1Error  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeS3:
    """In-memory stand-in for the boto3 client surface we use."""

    def __init__(self, objects: dict):
        self.objects = dict(objects)
        self.downloads: list = []

    def head_object(self, Bucket, Key):  # noqa: N803 (boto shape)
        if Key not in self.objects:
            raise Exception("404 Not Found")
        return {"ContentLength": len(self.objects[Key])}

    def download_file(self, Bucket, Key, dest, Config=None, Callback=None):  # noqa: N803
        self.downloads.append(Key)
        body = self.objects[Key]
        with open(dest, "wb") as fh:
            fh.write(body)
        if Callback is not None:
            Callback(len(body))


def _make_targz(root: Path, top: str, extra: dict | None = None) -> Path:
    src = root / top
    (src / "metadata").mkdir(parents=True)
    (src / "metadata" / "subset_records_extended.csv").write_text("id\n")
    for name, body in (extra or {}).items():
        p = src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    out = root / f"{top}.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname=top)
    return out


class TestParse(unittest.TestCase):
    def test_uri_ok(self):
        b, k = S3U.parse_s3_uri(
            "s3://my-bucket/a/b/V1_10S_STANDARDIZED_SUBSET.tar.gz")
        self.assertEqual((b, k), ("my-bucket",
                                 "a/b/V1_10S_STANDARDIZED_SUBSET.tar.gz"))

    def test_uri_bad(self):
        for u in ("https://x/y", "s3://nobucket", "s3://b/",
                  "s3:///k", "s3://b", ""):
            with self.assertRaises(V1Error, msg=u):
                S3U.parse_s3_uri(u)

    def test_sidecar_ok(self):
        d = "ab" * 32
        self.assertEqual(S3U.parse_sidecar(f"{d}  f.tar.gz\n", "f.tar.gz"), d)

    def test_sidecar_bad(self):
        with self.assertRaises(V1Error):
            S3U.parse_sidecar("nothex  f.tar.gz\n", "f.tar.gz")
        with self.assertRaises(V1Error):
            S3U.parse_sidecar("onlyonefield\n")
        with self.assertRaises(V1Error):
            S3U.parse_sidecar(f"{'ab' * 32}  other.tar.gz\n", "f.tar.gz")


class TestDownloadVerify(unittest.TestCase):
    def test_reuse_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            p.write_bytes(b"data")
            s3 = FakeS3({"k": b"data"})
            rec = S3U.download_and_verify(s3, "b", "k", p, _sha(b"data"))
            self.assertTrue(rec["reused"])
            self.assertEqual(s3.downloads, [])

    def test_redownload_when_hash_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            p.write_bytes(b"stale")
            s3 = FakeS3({"k": b"fresh"})
            rec = S3U.download_and_verify(s3, "b", "k", p, _sha(b"fresh"))
            self.assertFalse(rec["reused"])
            self.assertEqual(p.read_bytes(), b"fresh")

    def test_mismatch_after_download_raises_and_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            s3 = FakeS3({"k": b"evil"})
            with self.assertRaises(V1Error):
                S3U.download_and_verify(s3, "b", "k", p, _sha(b"good"))
            self.assertFalse(p.exists())

    def test_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            s3 = FakeS3({})
            with self.assertRaises(V1Error):
                S3U.download_file(s3, "b", "nope", Path(tmp) / "f")

    def test_no_credentials_raises_actionable_error(self):
        # Duck-typed on the botocore name — no boto install needed.
        NoCredentialsError = type("NoCredentialsError", (Exception,), {})

        class NoCred(FakeS3):
            def head_object(self, Bucket, Key):
                raise NoCredentialsError("Unable to locate credentials")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(V1Error) as ctx:
                S3U.download_file(NoCred({}), "b", "k", Path(tmp) / "f")
        msg = str(ctx.exception)
        self.assertIn("IAM instance role", msg)
        self.assertIn("re-run this section", msg)

    def test_access_denied_raises_actionable_error(self):
        class Denied(FakeS3):
            def head_object(self, Bucket, Key):
                raise Exception("An error occurred (403) when calling the "
                                "HeadObject operation: Forbidden")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(V1Error) as ctx:
                S3U.download_file(Denied({}), "b", "k", Path(tmp) / "f")
        self.assertIn("refused access", str(ctx.exception))

    def test_unknown_errors_propagate_raw(self):
        class Weird(FakeS3):
            def head_object(self, Bucket, Key):
                raise RuntimeError("something exotic")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                S3U.download_file(Weird({}), "b", "k", Path(tmp) / "f")


class TestExtract(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = _make_targz(Path(tmp), "TOP", {"a/b.bin": b"\x00\x01"})
            out = S3U.safe_extract_tarball(t, Path(tmp) / "x")
            self.assertTrue((out / "TOP" / "metadata"
                             / "subset_records_extended.csv").exists())
            self.assertEqual((out / "TOP" / "a" / "b.bin").read_bytes(),
                             b"\x00\x01")

    def test_dotdot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            evil = Path(tmp) / "evil.tar"
            with tarfile.open(evil, "w") as tf:
                ti = tarfile.TarInfo("TOP/../../pwn")
                body = b"x"
                ti.size = len(body)
                import io
                tf.addfile(ti, io.BytesIO(body))
            with self.assertRaises(V1Error):
                S3U.safe_extract_tarball(evil, Path(tmp) / "x")
            self.assertFalse((Path(tmp) / "pwn").exists())

    def test_absolute_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            evil = Path(tmp) / "evil.tar"
            with tarfile.open(evil, "w") as tf:
                ti = tarfile.TarInfo("/abs/pwn")
                body = b"x"
                ti.size = len(body)
                import io
                tf.addfile(ti, io.BytesIO(body))
            with self.assertRaises(V1Error):
                S3U.safe_extract_tarball(evil, Path(tmp) / "x")


class TestFetch(unittest.TestCase):
    def test_fetch_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = _make_targz(Path(tmp), "V1_10S_STANDARDIZED_SUBSET")
            blob = t.read_bytes()
            key = "datasets/raw_signals_data/V1/V1_10S_STANDARDIZED_SUBSET.tar.gz"
            s3 = FakeS3({key: blob,
                         key + ".sha256":
                         f"{_sha(blob)}  V1_10S_STANDARDIZED_SUBSET.tar.gz\n"
                         .encode()})
            got = S3U.fetch_input_subset(
                s3, f"s3://b/{key}", Path(tmp) / "input")
            self.assertTrue(Path(got).name == "V1_10S_STANDARDIZED_SUBSET")
            self.assertTrue((Path(got) / "metadata"
                             / "subset_records_extended.csv").exists())
            # second run reuses both tarball and extraction
            n = len(s3.downloads)
            got2 = S3U.fetch_input_subset(
                s3, f"s3://b/{key}", Path(tmp) / "input")
            self.assertEqual(got2, got)
            self.assertEqual(s3.downloads[n:], [key + ".sha256"])

    def test_fetch_rejects_non_tarball(self):
        s3 = FakeS3({})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(V1Error):
                S3U.fetch_input_subset(s3, "s3://b/data.csv",
                                       Path(tmp) / "input")


if __name__ == "__main__":
    unittest.main()
