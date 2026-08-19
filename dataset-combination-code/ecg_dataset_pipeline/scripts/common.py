"""
CardioSentry dataset pipeline -- shared utilities.

Nothing in this module makes a clinical labelling decision. All label
semantics live in config/label_mapping.yaml.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).resolve().parent.parent

TARGET_CLASSES = ["STEMI", "AF", "LVH", "NORMAL"]

# The common representation CardioSentry targets.
TARGET_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500
TARGET_DURATION_S = 10.0
TARGET_N_SAMPLES = 5000


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_paths(config_path: str | Path | None = None) -> dict:
    cfg = load_yaml(config_path or PIPELINE_ROOT / "config" / "paths.yaml")

    def _resolve(p):
        if p is None:
            return None
        p = Path(p)
        return str(p if p.is_absolute() else (PIPELINE_ROOT / p).resolve())

    for src in ("ptbxl", "chapman", "stemi"):
        if src in cfg and cfg[src].get("root"):
            cfg[src]["root"] = _resolve(cfg[src]["root"])
    for k, v in cfg.get("output", {}).items():
        cfg["output"][k] = _resolve(v)
    return cfg


def load_mapping(config_path: str | Path | None = None) -> dict:
    return load_yaml(config_path or PIPELINE_ROOT / "config" / "label_mapping.yaml")


def reports_dir(paths: dict) -> Path:
    d = Path(paths["output"]["reports_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def harmonized_dir(paths: dict) -> Path:
    d = Path(paths["output"]["harmonized_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Label resolution -- the single place the YAML is interpreted
# --------------------------------------------------------------------------

@dataclass
class LabelDecision:
    """The outcome of resolving one record's raw source labels."""
    positives: dict = field(default_factory=dict)      # class -> [source labels]
    neutral: list = field(default_factory=list)
    disqualifiers: list = field(default_factory=list)  # labels blocking NORMAL
    unknown: list = field(default_factory=list)        # labels absent from YAML
    excluded: list = field(default_factory=list)       # (label, status, reason)
    record_excluders: list = field(default_factory=list)  # labels that void the record
    normal_asserted: bool = False


class LabelResolver:
    """Resolves raw source labels into CardioSentry target labels.

    One instance per source dataset. Pure function of the YAML config --
    change the YAML, change the cohort, no code edit required.
    """

    def __init__(self, mapping: dict, source_key: str):
        self.mapping = mapping
        self.source_key = source_key
        self.src = mapping["sources"][source_key]
        self.labels = self.src.get("labels") or {}
        self.rules = mapping.get("rules", {})
        self.normal_assertion = self.src.get("normal_assertion")
        self.unknown_policy = self.src.get("unknown_code_policy") or {
            "status": "EXCLUDE_UNKNOWN_CODE",
            "disqualifies_normal": True,
        }

    def entry(self, label: str) -> dict | None:
        e = self.labels.get(label)
        if e is None and not isinstance(label, str):
            e = self.labels.get(str(label))
        return e

    def describe(self, label: str) -> dict:
        """Full audit record for a single source label."""
        e = self.entry(label)
        if e is None:
            return {
                "source_label": label,
                "acronym": "",
                "source_label_name": "UNDEFINED IN DISTRIBUTION",
                "maps_to": "",
                "status": self.unknown_policy["status"],
                "disqualifies_normal": bool(self.unknown_policy.get("disqualifies_normal", True)),
                "reason": (
                    "Label present in the data but not declared in "
                    "config/label_mapping.yaml. Never guessed: it cannot create a "
                    "target label, and (per unknown_code_policy) it is treated as a "
                    "possible pathology so it blocks the NORMAL label."
                ),
            }
        status = e.get("status", "EXCLUDE_NOT_TARGET")
        disq = e.get("disqualifies_normal")
        if disq is None:
            # `also: NORMAL_DISQUALIFIER` is the alternative spelling
            disq = e.get("also") == "NORMAL_DISQUALIFIER"
            if status in ("EXCLUDE_UNCERTAIN", "EXCLUDE_NOT_TARGET") and not disq:
                disq = False
            if status in ("NORMAL_DISQUALIFIER", "EXCLUDE_RECORD"):
                disq = True
            if status == "ACCEPT":
                disq = True
        return {
            "source_label": label,
            "acronym": e.get("acronym") or "",
            "source_label_name": e.get("name") or "",
            "maps_to": e.get("maps_to") or "",
            "status": status,
            "disqualifies_normal": bool(disq),
            "reason": " ".join(str(e.get("reason", "")).split()),
        }

    def resolve(self, raw_labels) -> LabelDecision:
        """raw_labels: iterable of source label strings present on the record."""
        d = LabelDecision()
        for lab in raw_labels:
            info = self.describe(lab)
            status, maps_to = info["status"], info["maps_to"]

            if status == "ACCEPT" and maps_to:
                d.positives.setdefault(maps_to, []).append(str(lab))
            elif status == "NORMAL_NEUTRAL":
                d.neutral.append(str(lab))
            elif status == "EXCLUDE_RECORD":
                d.record_excluders.append(str(lab))
                d.excluded.append((str(lab), status, info["reason"]))
            elif status == "EXCLUDE_UNKNOWN_CODE":
                d.unknown.append(str(lab))
            else:
                d.excluded.append((str(lab), status, info["reason"]))

            if info["disqualifies_normal"] and maps_to != "NORMAL":
                d.disqualifiers.append(str(lab))

        if self.normal_assertion is not None:
            d.normal_asserted = str(self.normal_assertion) in {str(x) for x in raw_labels}
        return d

    def finalize(self, d: LabelDecision) -> tuple[dict, str, str]:
        """Apply rules.normal + rules.ambiguity.

        Returns (binary_labels, ambiguity_flag, ambiguity_note).
        """
        nr = self.rules.get("normal", {})
        amb = self.rules.get("ambiguity", {})

        # A label marked EXCLUDE_RECORD removes the entire recording from the
        # cohort -- not just its own contribution. Used when a label's meaning
        # is unresolved and keeping the record under any label would risk a
        # wrong negative on one of the target heads.
        if d.record_excluders:
            return ({c: 0 for c in TARGET_CLASSES}, "RECORD_EXCLUDED_BY_LABEL",
                    "record dropped because it carries: "
                    + ", ".join(sorted(set(d.record_excluders))))

        binary = {c: 0 for c in TARGET_CLASSES}
        for cls in d.positives:
            if cls in binary:
                binary[cls] = 1

        pathological = [c for c in ("STEMI", "AF", "LVH") if binary[c] == 1]

        normal_ok = True
        why = []
        if nr.get("require_positive_normal_assertion", True):
            asserted = d.normal_asserted or "NORMAL" in d.positives
            if not asserted:
                normal_ok = False
        if normal_ok and nr.get("forbid_disqualifiers", True) and d.disqualifiers:
            normal_ok = False
            why.append("disqualifying labels: " + "|".join(sorted(set(d.disqualifiers))))
        if normal_ok and self.unknown_policy.get("disqualifies_normal", True) and d.unknown:
            normal_ok = False
            why.append("undefined labels: " + "|".join(sorted(set(d.unknown))))

        flag, note = "", ""
        if pathological and (d.normal_asserted or "NORMAL" in d.positives):
            # Contradiction: source asserts normal AND a target pathology.
            flag = "NORMAL_VS_PATHOLOGY"
            note = (
                f"source asserts normal but also {'+'.join(pathological)} "
                f"({'; '.join(f'{k}<-{v}' for k, v in d.positives.items() if k != 'NORMAL')})"
            )
            if amb.get("on_conflict") == "exclude":
                return {c: 0 for c in TARGET_CLASSES}, flag, note + " -> record excluded"
            normal_ok = False
            note += " -> NORMAL dropped, pathology kept"
        elif normal_ok and nr.get("forbid_target_positives", True) and pathological:
            normal_ok = False

        binary["NORMAL"] = 1 if (normal_ok and (d.normal_asserted or "NORMAL" in d.positives)) else 0

        if not flag and why and (d.normal_asserted or "NORMAL" in d.positives):
            flag = "NORMAL_BLOCKED"
            note = "; ".join(why)
        return binary, flag, note


# --------------------------------------------------------------------------
# WFDB header parsing (no signal read -- fast, works on 100k files)
# --------------------------------------------------------------------------

@dataclass
class HeaderInfo:
    record_name: str = ""
    n_sig: int | None = None
    fs: float | None = None
    n_samples: int | None = None
    duration_s: float | None = None
    lead_names: list = field(default_factory=list)
    signal_files: list = field(default_factory=list)
    adc_gain: list = field(default_factory=list)
    adc_units: list = field(default_factory=list)
    fmt: list = field(default_factory=list)
    comments: dict = field(default_factory=dict)
    parse_error: str = ""


_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def parse_header(hea_path: str | Path) -> HeaderInfo:
    """Parse a WFDB .hea file defensively.

    Returns HeaderInfo with parse_error set rather than raising, so a single
    malformed header cannot abort a 45k-file scan.
    """
    info = HeaderInfo()
    try:
        with open(hea_path, "r", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except OSError as exc:
        info.parse_error = f"unreadable: {exc}"
        return info

    body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not body:
        info.parse_error = "empty header"
        return info

    rec = body[0].split()
    info.record_name = rec[0] if rec else ""
    try:
        info.n_sig = int(rec[1])
    except (IndexError, ValueError):
        info.parse_error = "unparsable n_sig in record line"
    try:
        info.fs = float(rec[2])
    except (IndexError, ValueError):
        info.parse_error = (info.parse_error + "; " if info.parse_error else "") + \
            "unparsable sampling frequency in record line"
    try:
        info.n_samples = int(rec[3])
    except (IndexError, ValueError):
        info.parse_error = (info.parse_error + "; " if info.parse_error else "") + \
            f"unparsable n_samples in record line (got {rec[3]!r})" if len(rec) > 3 \
            else (info.parse_error + "; " if info.parse_error else "") + "record line truncated"

    if info.fs and info.n_samples:
        info.duration_s = info.n_samples / info.fs

    for ln in body[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        info.signal_files.append(parts[0])
        info.fmt.append(parts[1])
        gain_field = parts[2] if len(parts) > 2 else ""
        m = re.match(r"^([-\d.eE+]+)", gain_field)
        info.adc_gain.append(float(m.group(1)) if m else float("nan"))
        info.adc_units.append(gain_field.split("/")[-1] if "/" in gain_field else "")
        info.lead_names.append(parts[8] if len(parts) > 8 else "")

    for ln in lines:
        if ln.startswith("#") and ":" in ln:
            k, v = ln[1:].split(":", 1)
            info.comments[k.strip()] = v.strip()
    return info


def normalize_lead(name: str) -> str:
    """PTB-XL writes AVR/AVL/AVF, Chapman and the ACS set write aVR/aVL/aVF."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


NORM_TARGET_LEADS = [normalize_lead(l) for l in TARGET_LEADS]


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def write_csv(path: str | Path, rows: list, fieldnames: list | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(fieldnames or ["(no rows)"])
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def banner(title: str, ch: str = "=", width: int = 74) -> str:
    return f"\n{ch * width}\n{title}\n{ch * width}"


def info(msg: str) -> None:
    print(f"[cardiosentry] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[cardiosentry][WARN] {msg}", flush=True)


def assumption(msg: str) -> None:
    """Flag an assumption loudly, as required by the project brief."""
    print(f"[cardiosentry][ASSUMPTION] {msg}", flush=True)
