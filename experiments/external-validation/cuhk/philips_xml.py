#!/usr/bin/env python3
"""
Reader for the CUHK Philips TraceMasterVue XML corpus (Zenodo 5700069).

    from philips_xml import read
    rec = read("STEMI XML AB/A016XXXX-..._2006-10-12_15-57-44.xml")
    rec.statements     # per-ECG Philips criteria codes -> the label source
    rec.median_beats   # (12, 600) float32 mV, or None
    rec.rhythm         # (12, 5500) - NOT YET AVAILABLE, see XLI note below

WHAT THIS FORMAT IS
-------------------
Philips PageWriter XL carts, exported through TraceMasterVue. UTF-16 XML,
schema PhilipsECG.xsd v1.04. Every file carries:

  * per-ECG interpretation statements from the Philips criteria program,
  * per-lead measurements (R/S/Q amplitudes, ST at onset/mid/80ms, in uV),
  * a 12-lead representative beat, uncompressed base64 int16,
  * the full 11 s 12-lead rhythm at 500 Hz, XLI-compressed.

Signal specs match the CardioSentry pipeline exactly - 500 Hz native, 12 leads,
11 s (the pipeline wants 10) - so nothing needs resampling. Lead labels come in
two dialects across the corpus, "C1..C6" and "V1..V6"; canonical_leads() folds
them together.

THE XLI PROBLEM, STATED HONESTLY
--------------------------------
The 11 s rhythm is compressed with Philips XLI, which is proprietary. This
module gets PART of the way and stops at a known point:

  SOLVED - the container.  The payload is a chain of 16 chunks, each
    `int32 size, int16, int16` followed by `size` bytes. Verified to consume
    the payload exactly, with zero remainder, on every file tested.

  SOLVED - the entropy layer.  Each chunk is LZW: 10-bit FIXED-width codes,
    MSB-first bit packing, dictionary seeded with 256 single-byte entries and
    CAPPED AT 1024 entries then frozen (it does not grow the code width, and it
    does not reset). Every chunk decodes cleanly to ~11000 bytes = 5500 int16,
    which is exactly 11 s at 500 Hz. See `lzw_decompress`.

  NOT SOLVED - the residual layer.  The 5500 int16 per chunk are not samples;
    they are residuals under a predictor this module has not identified.
    Plane-split and interleaved byte orders, crossed with first- and
    second-order integration, were all checked against the XML's own per-lead
    R/S amplitudes: best correlation was +0.43, and chunks 6-15 decode to
    IDENTICAL content, which cannot be true of V1/V2/V3. Something in the
    predictor or the chunk-to-lead mapping is still wrong.

  Also unexplained: 16 chunks for 12 leads.

  Finishing this needs the reference algorithm rather than more guessing. The
  public reverse-engineering of Philips XLI (Watford's XliDecompressor) covers
  the residual step; porting it is a bounded task, and `lzw_decompress` below
  means only the last stage is missing.

WHAT WORKS TODAY
----------------
`median_beats` needs none of that. 10,019 of the 16,755 files carry the 12-lead
representative beat as plain base64 int16, and it decodes correctly: peak-to-
peak per lead correlates +0.986 with the per-lead R/S/Q amplitudes the cart
itself recorded in the same file, across all 12 leads.

It is 600 samples per lead - 1.2 s. That is a representative beat, not a rhythm
strip, and it CANNOT drive the existing image pipeline, which renders 10 s
(5000 samples) per sheet. Tiling a beat to fill 10 s would fabricate rhythm the
patient never had. See FINDINGS.md.
"""
from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The cart records at 5 uV per least-significant bit; every amplitude in the
# file, measurements included, is in microvolts.
RESOLUTION_UV = 5.0
CANONICAL = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# int16 max doubles as the "no sample here" marker inside a <waveform> payload.
PAD = 32767

_WAVEFORM = re.compile(r"<waveform ([^>]*)>(.*?)</waveform>", re.S)
_STATEMENT = re.compile(
    r'code="([A-Z0-9_]+)"[^>]*>\s*<lhsstatement>(.*?)</lhsstatement>'
    r'\s*(?:<rhsstatement>(.*?)</rhsstatement>)?', re.S)
_SEVERITY = re.compile(r'<severity code="([^"]*)"[^>]*>([^<]*)</severity>')
_LEADLABELS = re.compile(r'leadlabels="([^"]*)"')
_PARSEDWF = re.compile(r"<parsedwaveforms([^>]*)>(.*?)</parsedwaveforms>", re.S)
_PATIENT = re.compile(r"(.+?)(\d{4})-(\d{1,2})-(\d{1,2})_")


@dataclass
class Statement:
    code: str
    text: str
    criteria: str = ""


@dataclass
class Record:
    path: Path
    patient: str                       # anonymised prefix; groups a subject's ECGs
    date: str
    severity: str                      # AB / BO / NO / ON / DE
    severity_text: str
    leads: list[str]
    statements: list[Statement] = field(default_factory=list)
    median_beats: np.ndarray | None = None      # (12, n) mV
    measurements: dict = field(default_factory=dict)

    @property
    def codes(self) -> set[str]:
        return {s.code for s in self.statements}

    def __repr__(self) -> str:
        mb = "none" if self.median_beats is None else f"{self.median_beats.shape}"
        return (f"Record({self.path.name[:28]!r} patient={self.patient!r} "
                f"sev={self.severity} codes={len(self.statements)} beats={mb})")


def canonical_leads(labels: list[str]) -> list[str]:
    """Fold the corpus's two lead dialects into one.

    Some carts write the precordials as C1..C6 and some as V1..V6; they are the
    same electrodes. Anything already canonical passes through untouched.
    """
    out = []
    for l in labels:
        l = l.strip()
        if re.fullmatch(r"[Cc][1-6]", l):
            l = "V" + l[1]
        out.append(l.replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF"))
    return out


# ====================================================================== XLI
def lzw_decompress(data: bytes, width: int = 10, cap: int = 1024) -> bytes:
    """The entropy layer of Philips XLI.

    Determined empirically and then checked against the whole corpus: fixed
    10-bit codes (the width never grows), MSB-first packing, and a dictionary
    that stops accepting new entries at `cap` rather than resetting. Getting any
    one of those three wrong makes the stream desynchronise within a few hundred
    codes, which is how they were pinned down - every wrong variant breaks, and
    this one runs to the end of every chunk of every file tried.

    Returns ~11000 bytes per chunk: 5500 int16 at 500 Hz = the declared 11 s.
    Those int16 are RESIDUALS, not samples. See the module docstring.
    """
    out = bytearray()
    dic: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    nxt = 256
    bits = 0
    acc = 0
    prev: bytes | None = None
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= width:
            code = (acc >> (bits - width)) & ((1 << width) - 1)
            bits -= width
            acc &= (1 << bits) - 1
            if code in dic:
                entry = dic[code]
            elif prev is not None and code == nxt:
                entry = prev + prev[:1]
            else:
                # Trailing partial code at the end of a chunk. Everything up to
                # here is valid, so return it rather than raising.
                return bytes(out)
            out += entry
            if prev is not None and nxt < cap:
                dic[nxt] = prev + entry[:1]
                nxt += 1
            prev = entry
    return bytes(out)


def xli_chunks(payload: bytes) -> list[bytes]:
    """Split an XLI payload into its LZW-compressed chunks.

    Header is `int32 size` then two int16 whose meaning is not established.
    A correct split consumes the payload exactly; callers should check.
    """
    import struct
    out, off = [], 0
    while off + 8 <= len(payload):
        size, _a, _b = struct.unpack("<ihh", payload[off:off + 8])
        if size <= 0 or off + 8 + size > len(payload):
            break
        out.append(payload[off + 8:off + 8 + size])
        off += 8 + size
    return out


def read_xli_payload(text: str) -> bytes | None:
    m = _PARSEDWF.search(text)
    if not m:
        return None
    return base64.b64decode(re.sub(r"\s", "", m.group(2)))


# ============================================================== median beats
def read_median_beats(text: str, n_leads: int = 12) -> np.ndarray | None:
    """The 12-lead representative beat, in mV. None when the file has none.

    Each <waveform> element is ONE lead, plain base64 int16 little-endian, with
    int16-max used as padding. Validated against the cart's own per-lead
    measurements: peak-to-peak correlates +0.986 with R-minus-S across the 12
    leads of a record, which is not something a wrong decode produces.
    """
    blocks = _WAVEFORM.findall(text)
    if len(blocks) != n_leads:
        return None
    leads = []
    for _attrs, b64 in blocks:
        try:
            raw = base64.b64decode(re.sub(r"\s", "", b64))
        except Exception:                       # noqa: BLE001
            return None
        a = np.frombuffer(raw, dtype="<i2")
        a = a[a != PAD]
        if a.size == 0:
            return None
        leads.append(a.astype(np.float32))
    n = min(l.size for l in leads)
    if n < 100:
        return None
    return np.stack([l[:n] for l in leads]) * (RESOLUTION_UV / 1000.0)


# =================================================================== reading
def read(path: str | Path, with_beats: bool = True) -> Record:
    path = Path(path)
    text = path.read_text(encoding="utf-16")

    sev = _SEVERITY.search(text)
    lab = _LEADLABELS.search(text)
    m = _PATIENT.match(path.name)

    statements = []
    for code, lhs, rhs in _STATEMENT.findall(text):
        statements.append(Statement(
            code=code,
            # The source is XML, so "ST &gt; .10mV" arrives escaped; these
            # strings get read by humans and matched by eye against the code sets.
            text=html.unescape(re.sub(r"\s+", " ", lhs).strip()),
            criteria=html.unescape(re.sub(r"\s+", " ", rhs or "").strip()),
        ))

    meas = {}
    for tag in ("ramp", "samp", "qamp", "ston", "stmid", "st80", "qrsdur"):
        vals = re.findall(rf"<{tag}[^>]*>([^<]*)</{tag}>", text)
        try:
            meas[tag] = [int(v) for v in vals[:12]]
        except ValueError:
            pass

    return Record(
        path=path,
        patient=(m.group(1) if m else path.stem),
        date=(f"{m.group(2)}-{int(m.group(3)):02d}-{int(m.group(4)):02d}" if m else ""),
        severity=(sev.group(1) if sev else "NONE"),
        severity_text=(sev.group(2).strip() if sev else ""),
        leads=canonical_leads(lab.group(1).split()) if lab else list(CANONICAL),
        statements=statements,
        median_beats=read_median_beats(text) if with_beats else None,
        measurements=meas,
    )


# ============================================== Philips criteria -> label sets
#
# Grouped by WHAT THE STATEMENT ASSERTS, following the project's standing rule
# that only an explicit assertion counts and nothing is inferred. The Philips
# vocabulary is unusually good for this: it names the benign ST-elevation
# mimics with their own codes instead of lumping them in with injury, so the
# mimics can be held out as negatives rather than quietly poisoning the class.
#
# CAVEAT THAT TRAVELS WITH THESE SETS: every statement in this corpus carries
# source="Analysis program". They are the cart algorithm's reading, with no
# physician over-read anywhere in the 16,755 files. That is weaker evidence
# than LUDB (a cardiologist wrote the word) or the curated PTB-XL selection
# (a human inspected each trace).

ACUTE_INFARCT = {          # asserts "INFARCT, ACUTE" / "injury (ACUTE INFARCT)"
    "IMIA", "AMIA", "EAMIA", "AMIEA", "IMIEA", "IPMIA", "AMI61A",
    "IMI67", "ALI67", "IMI68", "ALI68", "IMI69", "ALI69",
}
ST_ELEVATION = {           # asserts ST elevation or injury, stops short of acute MI
    "BSTEI", "BSTEA", "BSTEL", "MSTEI", "MSTEA", "MSTEL",
    "CINJA", "CINJI", "CINJL", "PINJI", "PINJA", "PINJL",
    "STE11", "STE12", "STE20", "STE21", "STE22",
    "STE31", "STE31B", "STE32", "STE33", "STE34", "STE34B",
}
BENIGN_STE_MIMIC = {       # real ST elevation, not infarction - keep as NEGATIVE
    "EREPOL", "STE95", "STE96", "STE97", "STELVH",
}
AF = {"AFIB", "AFIB0", "AFIB1", "AFIB2"}
ATRIAL_FLUTTER = {"AFLT", "AFLUT", "FLUT"}     # excluded from AF, per label_mapping
LVH = {"LVH", "LVH26", "LVHREP", "LVH1", "LVH3"}


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        r = read(arg)
        print(r)
        print("  leads     :", " ".join(r.leads))
        print("  severity  :", r.severity, r.severity_text)
        acute = r.codes & ACUTE_INFARCT
        ste = r.codes & ST_ELEVATION
        mimic = r.codes & BENIGN_STE_MIMIC
        print("  acute MI  :", sorted(acute) or "-")
        print("  ST elev   :", sorted(ste) or "-")
        print("  mimic     :", sorted(mimic) or "-")
        for s in r.statements[:8]:
            print(f"    {s.code:<9} {s.text[:56]:<56} | {s.criteria[:36]}")
        if r.median_beats is not None:
            mb = r.median_beats
            print(f"  beats     : {mb.shape} mV, ptp per lead "
                  f"{[round(float(x), 2) for x in mb.ptp(axis=1)]}")
