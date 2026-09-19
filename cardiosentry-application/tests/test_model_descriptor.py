from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from backend.schemas.model_descriptor import ModelDescriptor
from backend.services import runner


def _descriptor(**output) -> ModelDescriptor:
    return ModelDescriptor.model_validate({
        "id": "twelve-head",
        "display_name": "Twelve head",
        "weights": "weights/test.pt",
        "architecture": {"module": "arch.test", "registry_key": "test"},
        "labels": ["NORM", "AF", "TInv", "LQT", "PRWP"],
        "display_names": {"NORM": "Normal ECG", "AF": "Atrial Fibrillation"},
        "output": output,
    })


def test_suppressed_labels_leave_checkpoint_order_intact():
    descriptor = _descriptor(suppressed_labels=["TInv", "LQT", "PRWP"])

    assert descriptor.labels == ["NORM", "AF", "TInv", "LQT", "PRWP"]
    assert descriptor.served_labels == ["NORM", "AF"]
    assert descriptor.served_display_names == {
        "NORM": "Normal ECG",
        "AF": "Atrial Fibrillation",
    }


def test_suppressed_labels_must_be_real_and_cannot_hide_every_head():
    with pytest.raises(ValidationError, match="unknown"):
        _descriptor(suppressed_labels=["QT"])
    with pytest.raises(ValidationError, match="cannot suppress every label"):
        _descriptor(suppressed_labels=["NORM", "AF", "TInv", "LQT", "PRWP"])


def test_prediction_filters_suppressed_heads_but_reenable_needs_no_new_weights(
        tmp_path, monkeypatch):
    model_input = tmp_path / "model_input.png"
    model_input.write_bytes(b"same model input")
    monkeypatch.setattr(runner, "derive_views", lambda *args: ({}, 0.25))
    monkeypatch.setattr(
        runner, "_forward_views",
        lambda *args: torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]))

    descriptor = _descriptor(suppressed_labels=["TInv", "LQT", "PRWP"])
    loaded = SimpleNamespace(
        descriptor=descriptor,
        thresholds=[0.5] * len(descriptor.labels),
        views=[(1, 1)],
        lock=threading.Lock(),
    )
    suppressed = runner.predict(loaded, model_input)
    assert list(suppressed.scores) == ["NORM", "AF"]
    assert list(suppressed.thresholds) == ["NORM", "AF"]

    # The five logits never changed. Removing only the output policy returns
    # every head on the next run, which is the reversibility guarantee.
    loaded.descriptor = _descriptor(suppressed_labels=[])
    restored = runner.predict(loaded, model_input)
    assert list(restored.scores) == ["NORM", "AF", "TInv", "LQT", "PRWP"]
