"""Regression tests for the no-checksum S3 fallback (PLAN v1.17).

S3 may accept a multipart upload yet store NO CRC64NVME checksum
(observed on EC2: size-correct object, empty ChecksumCRC64NVME). That
must route to SHA-256 re-download verification (pre-approved path b),
not fail — and re-running §17 must reuse the size-matching object
instead of re-uploading 146 GB. Fake in-memory S3 client, no network,
no boto3/awscrt needed.
Run:  python3 -m unittest v1_support.tests.test_s3_fallback -v
"""
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Stub the one boto3 import upload_file needs (TransferConfig ctor only;
# the fake client ignores it). setdefault: never clobber a real boto3.
if "boto3.s3.transfer" not in sys.modules:
    _b3 = types.ModuleType("boto3")
    _s3 = types.ModuleType("boto3.s3")
    _tr = types.ModuleType("boto3.s3.transfer")

    class TransferConfig:  # noqa: D101
        def __init__(self, *a, **k):
            pass

    _tr.TransferConfig = TransferConfig
    sys.modules.setdefault("boto3", _b3)
    sys.modules.setdefault("boto3.s3", _s3)
    sys.modules.setdefault("boto3.s3.transfer", _tr)

from v1_support import s3util as S3U  # noqa: E402
from v1_support.util import V1Error  # noqa: E402


class FakeS3Upload:
    """In-memory stand-in for the upload/HEAD/GET surface we use."""

    def __init__(self, store_checksum=True):
        self.objects: dict = {}
        self.store_checksum = store_checksum
        self.uploads: list = []

    def upload_file(self, Filename, Bucket, Key,  # noqa: N803 (boto shape)
                    ExtraArgs=None, Config=None, Callback=None):
        self.uploads.append(Key)
        data = Path(Filename).read_bytes()
        assert (ExtraArgs or {}).get("ChecksumAlgorithm") == "CRC64NVME"
        self.objects[Key] = {"data": data, "crc": "ZmFrZS1jcmM=" if
                             self.store_checksum else None}
        if Callback is not None:
            Callback(len(data))

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise Exception("404 Not Found")
        o = self.objects[Key]
        h = {"ContentLength": len(o["data"])}
        if o["crc"]:
            h["ChecksumCRC64NVME"] = o["crc"]
        return h

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key]["data"])}


def _payload(root: Path, name="t.tar", n=4096) -> Path:
    p = root / name
    p.write_bytes(bytes(i % 251 for i in range(n)))
    return p


class TestNoChecksumFallback(unittest.TestCase):
    def test_fresh_upload_without_checksum_redownload_verifies(self):
        s3 = FakeS3Upload(store_checksum=False)
        with tempfile.TemporaryDirectory() as td:
            p = _payload(Path(td))
            rec = S3U.upload_and_verify(s3, p, "b", "k")
        self.assertIsNone(rec["crc64nvme"])
        self.assertFalse(rec["reused"])
        self.assertEqual(rec["verify_method"], "redownload-sha256")
        self.assertEqual(s3.uploads, ["k"])

    def test_reuse_size_match_without_checksum_skips_upload(self):
        s3 = FakeS3Upload(store_checksum=False)
        with tempfile.TemporaryDirectory() as td:
            p = _payload(Path(td))
            s3.objects["k"] = {"data": p.read_bytes(), "crc": None}
            rec = S3U.upload_and_verify(s3, p, "b", "k")
        self.assertTrue(rec["reused"])
        self.assertEqual(s3.uploads, [])
        self.assertEqual(rec["verify_method"], "redownload-sha256")

    def test_size_mismatch_reuploads(self):
        s3 = FakeS3Upload(store_checksum=False)
        with tempfile.TemporaryDirectory() as td:
            p = _payload(Path(td))
            s3.objects["k"] = {"data": b"stale", "crc": None}
            rec = S3U.upload_and_verify(s3, p, "b", "k")
        self.assertFalse(rec["reused"])
        self.assertEqual(s3.uploads, ["k"])
        self.assertEqual(rec["verify_method"], "redownload-sha256")

    def test_corruption_still_caught_by_redownload(self):
        s3 = FakeS3Upload(store_checksum=False)
        with tempfile.TemporaryDirectory() as td:
            p = _payload(Path(td))
            good = bytearray(p.read_bytes())
            good[100] ^= 0xFF
            s3.objects["k"] = {"data": bytes(good), "crc": None}
            # same size, wrong bytes -> reuse attempt must fail verification
            with self.assertRaises(V1Error):
                S3U.upload_and_verify(s3, p, "b", "k")

    def test_checksum_present_without_awscrt_redownloads(self):
        s3 = FakeS3Upload(store_checksum=True)
        real = S3U.ensure_awscrt
        S3U.ensure_awscrt = lambda: False  # no network in tests
        self.addCleanup(setattr, S3U, "ensure_awscrt", real)
        with tempfile.TemporaryDirectory() as td:
            p = _payload(Path(td))
            rec = S3U.upload_and_verify(s3, p, "b", "k")
        self.assertEqual(rec["crc64nvme"], "ZmFrZS1jcmM=")
        self.assertEqual(rec["verify_method"], "redownload-sha256")


if __name__ == "__main__":
    unittest.main()
