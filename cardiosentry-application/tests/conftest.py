"""Shared test configuration.

Environment overrides MUST be set before any `backend.*` import (settings is a
module-level singleton), which is why they live here at conftest import time.
Tests never touch the real captures/ directory.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("CARDIOSENTRY_DATA_DIR", str(
    Path(tempfile.gettempdir()) / "cardiosentry_test_captures"))
# The suite pins ONE model on purpose: several assertions are specific to it
# (Grad-CAM's `attn` at 2048 channels on a 24x32 grid is a ResNet-50 fact, not
# an EfficientNetV2-S one). It is therefore independent of whatever
# models/active.yaml points at — but the startup self-check is not, so the
# reference scores must be pinned to the SAME model or every test run logs a
# spurious "self-check FAILED" against another run's numbers.
os.environ.setdefault("CARDIOSENTRY_ACTIVE_MODEL",
                      "resnet50_cbam_nat768")
os.environ.setdefault("CARDIOSENTRY_VERIFY_EXPECTED_SCORES_JSON", str(
    Path(__file__).resolve().parent / "fixtures"
    / "expected_scores_resnet50_cbam_nat768.json"))
# Model inference on CPU is slow; one thread keeps the smoke test fast enough
# without touching the production default.
os.environ.setdefault("CARDIOSENTRY_NUM_THREADS", "2")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def client():
    """TestClient over the real app (lifespan loads the real model once)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c
