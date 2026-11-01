"""S3 upload with S3-native checksums (PLAN 17, v1.1 amendment 10) +
S3 input fetch (PLAN v1.3 §3: the EC2 build pulls the 2 GB input tarball
from S3 when no local copy exists).

Integrity model, stated honestly:
  * Transfer leg: ChecksumAlgorithm=CRC64NVME on a multipart upload. S3
    validates every part on receipt and stores the composed full-object
    checksum; corruption in flight is REJECTED, not merely detected.
  * Independent verification (one required): (a) recompute the composed
    full-object CRC64NVME locally with identical part framing and compare
    to HEAD's ChecksumCRC64NVME (needs awscrt; self-tested against the
    published CRC-64/NVME check vector); else (b) streaming get_object
    re-hash against the local SHA-256.
  * v1.17 fallback: S3 sometimes accepts the upload yet stores NO
    checksum (observed: size-correct object, empty ChecksumCRC64NVME —
    an SDK/S3-side storage gap, not an integrity signal). CRC64NVME
    stays preferred (requested on every upload, used whenever
    stored), but a missing checksum now routes to path (b) instead
    of failing: SHA-256 byte-equality is the stronger proof anyway.
  * HEAD size + user-supplied Metadata.sha256 alone are NOT presented as
    independent verification; the local .sha256 sidecar is provenance.
  * Input fetch: the sidecar OBJECT (<key>.sha256) is the authority for
    the expected SHA-256; the tarball is SHA-verified after download
    (reused across re-runs when the hash matches) and extracted with a
    traversal-safe extractor. boto3 validates response checksums on GET
    when the object carries them.

Auth: boto3 default credential chain (EC2 IAM instance role). No keys here.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import subprocess
import sys
from pathlib import Path

from .config import (S3_MULTIPART_THRESHOLD, S3_MULTIPART_CHUNKSIZE,
                     S3_CHECKSUM_ALGORITHM)
from .util import V1Error, log, sha256_file

# Published CRC-64/NVME check value for ASCII "123456789" (CRC catalogue:
# poly 0x9A6C9329AC4BC9B5, init/xorout all-F, refin/refout true).
CRC64NVME_CHECK_INPUT = b"123456789"
CRC64NVME_CHECK_VALUE = 0xAE8B14860A799888


def ensure_awscrt() -> bool:
    """Import awscrt, pip-installing it if necessary. Returns availability."""
    try:
        import awscrt  # noqa: F401
        return True
    except ImportError:
        pass
    log("awscrt missing; attempting pip install (needs linux wheels)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "awscrt"], check=True, timeout=600)
        import awscrt  # noqa: F401
        return True
    except Exception as exc:
        log(f"awscrt unavailable ({exc}); path (b) re-download required")
        return False


def _crc64nvme_fn():
    from awscrt import checksums as crt
    fn = getattr(crt, "crc64nvme", None)
    if fn is None:
        raise V1Error("awscrt has no crc64nvme API")
    if fn(CRC64NVME_CHECK_INPUT) != CRC64NVME_CHECK_VALUE:
        raise V1Error("awscrt crc64nvme fails the published check vector")
    return fn


def composed_crc64nvme(path: str | Path,
                       chunksize: int = S3_MULTIPART_CHUNKSIZE) -> str:
    """Locally recompute S3's composed full-object CRC64NVME.

    Composition (per AWS 'Checking object integrity'): binary big-endian
    part digests concatenated in part order, CRC64NVME over that, base64,
    suffixed '-N' for multipart. Single part: plain base64 digest.
    """
    fn = _crc64nvme_fn()
    parts = []
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunksize)
            if not block:
                break
            parts.append(struct.pack(">Q", fn(block)))
    if len(parts) == 1:
        return base64.b64encode(parts[0]).decode()
    full = struct.pack(">Q", fn(b"".join(parts)))
    return base64.b64encode(full).decode() + f"-{len(parts)}"


def check_crc64nvme_support() -> None:
    """Preflight: botocore's S3 model must offer CRC64NVME (no silent downgrade)."""
    import botocore.session
    model = botocore.session.get_session().get_service_model("s3")
    try:
        shape = model.shape_for("ChecksumAlgorithm")
    except Exception as exc:
        raise V1Error(f"S3 model has no ChecksumAlgorithm: {exc}") from exc
    if S3_CHECKSUM_ALGORITHM not in (shape.enum or []):
        raise V1Error(f"botocore S3 model lacks {S3_CHECKSUM_ALGORITHM}; "
                      "upgrade boto3/botocore instead of downgrading integrity")
    log(f"S3 ChecksumAlgorithm {S3_CHECKSUM_ALGORITHM} supported")


class Progress:
    def __init__(self, name: str, total: int):
        self.name, self.total, self.seen = name, total, 0

    def __call__(self, n: int):
        self.seen += n
        if self.seen >= getattr(self, "_next", 0):
            pct = 100 * self.seen / max(1, self.total)
            print(f"  {self.name}: {self.seen / 2**30:.2f}/"
                  f"{self.total / 2**30:.2f} GiB ({pct:.0f}%)", flush=True)
            self._next = self.seen + (1 << 30)


def _head_or_none(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if "404" in str(exc) or "Not Found" in str(exc):
            return None
        raise


def upload_file(s3, local: str | Path, bucket: str, key: str,
                sha_hex: str) -> dict:
    """Multipart upload requesting CRC64NVME; returns HEAD record.

    crc64nvme is None when S3 stored no checksum (fresh upload or
    reused size-matching object) — upload_and_verify then proves
    integrity via SHA-256 re-download instead of failing (v1.17).
    """
    from boto3.s3.transfer import TransferConfig
    local = Path(local)
    cfg = TransferConfig(multipart_threshold=S3_MULTIPART_THRESHOLD,
                         multipart_chunksize=S3_MULTIPART_CHUNKSIZE,
                         use_threads=True)
    head = _head_or_none(s3, bucket, key)
    if head is not None:
        if int(head["ContentLength"]) != local.stat().st_size:
            log(f"size differs for {key} "
                f"(s3={head['ContentLength']} local={local.stat().st_size}); "
                f"re-uploading")
        elif head.get("ChecksumCRC64NVME"):
            log(f"reusing matching S3 object (size + CRC64NVME present): {key}")
            return {"key": key, "reused": True,
                    "crc64nvme": head.get("ChecksumCRC64NVME")}
        else:
            log(f"reusing size-matching object without stored checksum: {key} "
                f"(will redownload-verify)")
            return {"key": key, "reused": True, "crc64nvme": None}
    log(f"uploading {local.name} -> s3://{bucket}/{key} "
        f"(ChecksumAlgorithm={S3_CHECKSUM_ALGORITHM})")
    s3.upload_file(
        str(local), bucket, key,
        ExtraArgs={"ChecksumAlgorithm": S3_CHECKSUM_ALGORITHM,
                   "Metadata": {"sha256": sha_hex},
                   "ContentType": "application/x-tar"
                   if local.suffix == ".tar" else "application/octet-stream"},
        Config=cfg, Callback=Progress(local.name, local.stat().st_size))
    head = s3.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != local.stat().st_size:
        raise V1Error(f"S3 size mismatch for {key}")
    crc = head.get("ChecksumCRC64NVME")
    if not crc:
        log(f"S3 stored no {S3_CHECKSUM_ALGORITHM} checksum for {key} "
            f"(size OK); falling back to redownload SHA-256 verification")
        return {"key": key, "reused": False, "crc64nvme": None}
    log(f"HEAD OK: {key} bytes={head['ContentLength']} crc64nvme={crc}")
    return {"key": key, "reused": False, "crc64nvme": crc}


def verify_composed(s3, local: str | Path, bucket: str, key: str,
                    head_crc: str) -> str:
    """Path (a): independent local recomputation of the composed checksum."""
    expect = composed_crc64nvme(local)
    if expect != head_crc:
        raise V1Error(f"composed CRC64NVME mismatch for {key}: "
                      f"local={expect} s3={head_crc} — investigate; "
                      "fall back to path (b) re-download as arbiter")
    log(f"composed CRC64NVME independently verified: {key}")
    return "composed-local"


def verify_redownload(s3, local: str | Path, bucket: str, key: str,
                      sha_hex: str) -> str:
    """Path (b): streaming re-download + SHA-256. Slow but fully independent."""
    size = Path(local).stat().st_size
    log(f"re-download verifying {key} ({size / 2**30:.1f} GiB)...")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    h, seen, nxt = hashlib.sha256(), 0, 1 << 30
    try:
        for block in iter(lambda: body.read(1 << 24), b""):
            h.update(block)
            seen += len(block)
            if seen >= nxt:
                log(f"  verified {seen / 2**30:.0f}/{size / 2**30:.1f} GiB")
                nxt += 1 << 30
    finally:
        body.close()
    if seen != size:
        raise V1Error(f"S3 byte-count mismatch for {key}")
    if h.hexdigest() != sha_hex:
        raise V1Error(f"S3 SHA-256 mismatch for {key}")
    log(f"re-download SHA-256 verified: {key}")
    return "redownload-sha256"


def upload_and_verify(s3, local: str | Path, bucket: str, key: str,
                      force_redownload: bool = False) -> dict:
    """Full upload + required verification. Returns the verification record."""
    local = Path(local)
    digest = sha256_file(local)
    rec = upload_file(s3, local, bucket, key, digest)
    # None-check first: the no-checksum fallback must not require awscrt.
    if rec["crc64nvme"] is None or force_redownload or not ensure_awscrt():
        method = verify_redownload(s3, local, bucket, key, digest)
    else:
        try:
            method = verify_composed(s3, local, bucket, key, rec["crc64nvme"])
        except V1Error:
            log("composed-checksum mismatch; arbitrating via re-download...")
            method = verify_redownload(s3, local, bucket, key, digest)
    return {"s3_uri": f"s3://{bucket}/{key}", "bytes": local.stat().st_size,
            "sha256": digest, "crc64nvme": rec["crc64nvme"],
            "reused": rec["reused"], "verify_method": method}


# ---------------------------------------------------------- input fetch

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/key into (bucket, key). Loud on malformed input."""
    if not uri.startswith("s3://"):
        raise V1Error(f"not an s3:// URI: {uri!r}")
    rest = uri[len("s3://"):]
    if "/" not in rest or rest.startswith("/") or rest.endswith("/"):
        raise V1Error(f"malformed S3 URI (need s3://bucket/key): {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise V1Error(f"malformed S3 URI (need s3://bucket/key): {uri!r}")
    return bucket, key


def parse_sidecar(text: str, expect_name: str | None = None) -> str:
    """Parse a `sha256sum`-style sidecar ("<hex>  <filename>") -> hex digest."""
    parts = text.strip().split()
    if len(parts) != 2:
        raise V1Error(f"malformed .sha256 sidecar: {text.strip()!r}")
    digest, name = parts
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise V1Error(f"sidecar has no 64-hex digest: {text.strip()!r}")
    if expect_name is not None and name != expect_name:
        raise V1Error(f"sidecar names {name!r}, need {expect_name!r}")
    return digest


def _as_friendly(exc: Exception, bucket: str, key: str) -> V1Error | None:
    """Translate known credential/permission shapes into actionable errors.

    Duck-typed on exception name/message (botocore must stay an optional
    import). Returns None for anything unrecognized — those propagate raw.
    """
    name, text = type(exc).__name__, str(exc)
    if name in ("NoCredentialsError", "PartialCredentialsError",
                "CredentialRetrievalError") \
            or "Unable to locate credentials" in text:
        return V1Error(
            "No AWS credentials found (boto3 default chain). This build "
            "authenticates with an EC2 IAM instance role — attach one to "
            "this instance with s3:GetObject on "
            "datasets/raw_signals_data/V1/* (input) plus s3:GetObject + "
            "s3:PutObject on datasets/V1/* (output + HEAD-resume checks), "
            "then re-run this section (no kernel restart needed). See "
            "EC2_NOTES.md §Transfer/IAM for the exact policy.")
    if "403" in text or "AccessDenied" in text or "Forbidden" in text \
            or "Unauthorized" in text:
        return V1Error(
            f"S3 refused access to s3://{bucket}/{key}: the instance role "
            "lacks permission. Grant s3:GetObject on "
            "datasets/raw_signals_data/V1/* (input) plus s3:GetObject + "
            "s3:PutObject on datasets/V1/* (output + HEAD-resume checks), "
            "then re-run this section.")
    return None


def download_file(s3, bucket: str, key: str, dest: str | Path) -> Path:
    """Download one object via tmp+rename (a crash never leaves a half file).

    Returns the destination path. Credential/permission failures raise
    actionable errors; anything else propagates loudly.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception as exc:
        if "404" in str(exc) or "Not Found" in str(exc):
            raise V1Error(f"s3://{bucket}/{key} does not exist") from exc
        friendly = _as_friendly(exc, bucket, key)
        if friendly is not None:
            raise friendly from exc
        raise
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    log(f"downloading s3://{bucket}/{key} ({size / 2**20:.0f} MiB)")
    try:  # tuned multipart framing; None = boto defaults (still correct)
        from boto3.s3.transfer import TransferConfig
        cfg = TransferConfig(multipart_threshold=S3_MULTIPART_THRESHOLD,
                             multipart_chunksize=S3_MULTIPART_CHUNKSIZE,
                             use_threads=True)
    except ImportError:
        cfg = None
    s3.download_file(bucket, key, str(tmp), Config=cfg,
                     Callback=Progress(key.rsplit("/", 1)[-1], size))
    if tmp.stat().st_size != size:
        raise V1Error(f"short download for {key}: "
                      f"{tmp.stat().st_size} != HEAD {size}")
    tmp.replace(dest)
    return dest


def download_and_verify(s3, bucket: str, key: str, dest: str | Path,
                        expect_sha256: str) -> dict:
    """Download unless the destination already hashes correctly; then verify.

    Returns {"path", "bytes", "reused"}. A SHA mismatch after a fresh
    download deletes the file and raises — never a silent bad input.
    """
    dest = Path(dest)
    if dest.exists() and sha256_file(dest) == expect_sha256:
        log(f"reusing verified local copy: {dest.name}")
        return {"path": str(dest), "bytes": dest.stat().st_size,
                "reused": True}
    if dest.exists():
        log(f"{dest.name} present but hash differs; re-downloading")
        dest.unlink()
    download_file(s3, bucket, key, dest)
    if sha256_file(dest) != expect_sha256:
        dest.unlink()
        raise V1Error(f"SHA-256 mismatch after downloading s3://{bucket}/{key}")
    log(f"SHA-256 verified: {dest.name}")
    return {"path": str(dest), "bytes": dest.stat().st_size, "reused": False}


def safe_extract_tarball(tarball: str | Path, dest_dir: str | Path) -> Path:
    """Extract a .tar/.tar.gz, rejecting absolute paths and `..` members."""
    import tarfile
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r") as tf:
        members = tf.getmembers()
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise V1Error(f"tarball member escapes destination: {m.name!r}")
        tf.extractall(dest_dir, members=members)
    log(f"extracted {len(members)} members -> {dest_dir}")
    return dest_dir


def fetch_input_subset(s3, uri: str, input_dir: str | Path) -> str:
    """Fetch the input subset tarball from S3 unless already extracted.

    1. Sidecar object (<key>.sha256) is the SHA-256 authority (always
       re-fetched: 100 bytes).
    2. Tarball downloaded (or reused when the hash matches) + SHA-verified.
    3. Extracted unless the sentinel manifest already exists; the §5 gates
       (31,109 rows + 62,218 file hashes) are the backstop for a partial
       extraction.
    Returns the extracted subset directory path.
    """
    bucket, key = parse_s3_uri(uri)
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    base = key.rsplit("/", 1)[-1]
    if not base.endswith((".tar.gz", ".tgz", ".tar")):
        raise V1Error(f"input URI must point at a tarball, got: {uri!r}")

    sidecar_local = input_dir / (base + ".sha256")
    download_file(s3, bucket, key + ".sha256", sidecar_local)
    expect = parse_sidecar(sidecar_local.read_text(), base)
    log(f"expected input SHA-256: {expect[:16]}… (from sidecar object)")

    tarball = input_dir / base
    dl = download_and_verify(s3, bucket, key, tarball, expect)

    top = base[:-7] if base.endswith(".tar.gz") else base.rsplit(".", 1)[0]
    extracted = input_dir / top
    sentinel = extracted / "metadata" / "subset_records_extended.csv"
    if sentinel.exists():
        log(f"input already extracted: {extracted} "
            f"(tarball {'reused' if dl['reused'] else 'downloaded'})")
    else:
        safe_extract_tarball(tarball, input_dir)
        if not sentinel.exists():
            raise V1Error(f"extraction produced no manifest at {sentinel}")
    return str(extracted)
