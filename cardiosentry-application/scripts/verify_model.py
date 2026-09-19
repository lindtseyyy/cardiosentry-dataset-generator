#!/usr/bin/env python3
"""Load the active model and assert, on a checked-in fixture image, that the
serving path reproduces the prototype's stored score for that image (§10.6).

    python scripts/verify_model.py [--fixture tests/fixtures/corpus_sheet.jpg]
                                   [--expected tests/fixtures/expected_scores.json]
                                   [--model resnet50_hires] [--tolerance 2e-3]

Checks:
  1. weights load with `strict=True`;
  2. output shape is [1, number of checkpoint labels];
  3. scores reproduce the expected vector within tolerance.

Each descriptor names its own expected vector and provenance. Most legacy
vectors came from the prototype's evaluation of ECG000002 on GPU with AMP;
the V1 ConvNeXt fixture is explicitly a CPU-fp32 serving-path vector because
that run did not evaluate the older corpus sheet. This is what catches a
changed timm tag, architecture argument, preprocessing path, or weight file.

Also importable: `verify_model(ld, fixture, expected, tolerance)` is called by
backend.main at startup (warn-only) when the fixture is configured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np                    # noqa: E402
from PIL import Image                 # noqa: E402

from backend.services import preprocess as prep     # noqa: E402
from backend.services import runner                 # noqa: E402
from backend.services.registry import LoadedDescriptor, discover, active  # noqa: E402


def _fixture_rgb(fixture: Path) -> np.ndarray:
    with Image.open(fixture) as im:
        im.load()
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[1] != 1686 or arr.shape[0] != 1311:
        print(f"  warn: fixture is {arr.shape[1]}x{arr.shape[0]}, expected a "
              "1686x1311 stage-4 sheet; scores will not match the stored ones.",
              file=sys.stderr)
    return arr


def verify_model(ld: LoadedDescriptor, fixture: Path, expected: dict | None,
                 tolerance: float) -> dict:
    """Returns a report dict; raises on hard failures."""
    desc = ld.descriptor
    print(f"verify_model: {desc.id}  ({ld.weights_path.name})")
    lm = runner.load_model(ld)
    print(f"  loaded: views={lm.views} epoch={lm.epoch} "
          f"fp={lm.protocol_fingerprint} thresholds={[round(t, 4) for t in lm.thresholds]}")

    rgb = _fixture_rgb(fixture)
    pre = desc.preprocess
    res = prep.preprocess_corrected(
        rgb, input_size=pre.input_hw,
        match_training_jpeg=pre.match_training_jpeg,
        jpeg_quality=pre.jpeg_quality,
        mean=pre.normalize.mean, std=pre.normalize.std)

    # Exercise the exact serving path: the canonical chain writes
    # model_input.png; derive_views reads it back and builds the per-view
    # tensors (post-q95 pixels, BICUBIC for additional views).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_input = Path(td) / "model_input.png"
        tmp_input.write_bytes(res.model_input_png)
        views, prep_ms = runner.derive_views(lm, tmp_input,
                                             pre.normalize.mean, pre.normalize.std)

    logits = runner._forward_views(lm, views)
    expected_shape = [1, len(desc.labels)]
    if list(logits.shape) != expected_shape:
        raise RuntimeError(
            f"expected output shape {expected_shape}, got {list(logits.shape)}")
    scores = 1.0 / (1.0 + np.exp(-logits.float().numpy()))[0]
    report = {"model": desc.id,
              "scores": {l: float(scores[i]) for i, l in enumerate(desc.labels)},
              "shape": list(logits.shape)}

    if expected:
        if expected.get("model") and expected["model"] != desc.id:
            print(f"  WARN: expected scores were produced by "
                  f"{expected['model']!r}, not {desc.id!r} — the comparison is "
                  "not meaningful", file=sys.stderr)
        exp_scores = expected["scores"]
        deltas = {l: abs(report["scores"][l] - exp_scores[l]) for l in desc.labels}
        report["max_delta"] = max(deltas.values())
        report["expected"] = exp_scores
        if report["max_delta"] > tolerance:
            raise RuntimeError(
                f"score drift {report['max_delta']:.2e} exceeds tolerance "
                f"{tolerance:.2e} — timm/weights drift suspected. deltas={deltas}")
        print(f"  scores reproduce expected vector: max |Δ| = {report['max_delta']:.2e}"
              f"  (tolerance {tolerance:.2e})")
    else:
        print("  no expected scores provided — shape check only")
    print("  OK")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path,
                    default=APP_ROOT / "tests" / "fixtures" / "corpus_sheet.jpg")
    ap.add_argument("--expected", type=Path, default=None,
                    help="default: the descriptor's verify.expected_scores, "
                         "else tests/fixtures/expected_scores.json")
    ap.add_argument("--model", default=None, help="default: the active descriptor")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="default: the descriptor's verify.score_tolerance, else 2e-3")
    a = ap.parse_args()

    loaded = discover()
    ld = loaded[a.model] if a.model else active(loaded)

    # An explicit flag wins; otherwise the descriptor's own verify block, which
    # is what makes `--model X` compare X against X's stored vector rather than
    # against whichever file the default happens to point at.
    vspec = ld.descriptor.verify
    expected_path = a.expected or (
        APP_ROOT / vspec.expected_scores if vspec.expected_scores
        else APP_ROOT / "tests" / "fixtures" / "expected_scores.json")
    tolerance = a.tolerance if a.tolerance is not None else (
        vspec.score_tolerance or 2e-3)

    expected = None
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    else:
        print(f"expected scores file {expected_path} not found — shape check only")
    try:
        report = verify_model(ld, a.fixture, expected, tolerance)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
