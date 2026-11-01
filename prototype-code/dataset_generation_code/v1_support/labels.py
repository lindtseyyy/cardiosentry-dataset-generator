"""Target-label parsing (PLAN 3.5, 14.2). The 12 multi-hot columns are the
only label representation; combination strings are never treated as classes.
"""
from __future__ import annotations

from .config import TARGET_LABELS

LABEL_INDEX = {lab: i for i, lab in enumerate(TARGET_LABELS)}


def parse_target_labels(s: str) -> tuple[str, ...]:
    labs = tuple(lab for lab in (s or "").split("|") if lab)
    unknown = [lab for lab in labs if lab not in LABEL_INDEX]
    if unknown:
        raise ValueError(f"unknown target labels {unknown} in {s!r}")
    if len(set(labs)) != len(labs):
        raise ValueError(f"duplicate labels in {s!r}")
    return labs


def to_multihot(labs: tuple[str, ...] | list[str]) -> dict[str, int]:
    out = {lab: 0 for lab in TARGET_LABELS}
    for lab in labs:
        out[lab] = 1
    return out
