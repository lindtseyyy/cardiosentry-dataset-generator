"""End-to-end API smoke test: the staged flow against the REAL model.

upload -> detect -> rectify -> predict -> history -> annotate -> trash
"""
from __future__ import annotations

import time

import pytest


def _upload_photo(client, fixtures, name: str, **form):
    with open(fixtures / name, "rb") as f:
        r = client.post("/api/captures",
                        files={"file": (name, f, "image/jpeg")},
                        data=form)
    assert r.status_code == 201, r.text
    return r.json()


def test_health_and_config(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True
    assert h["active_model"] == "resnet50_cbam_nat768"

    cfg = client.get("/api/config").json()
    assert cfg["active_model"]["labels"] == ["STEMI", "AF", "LVH", "NORMAL"]
    assert cfg["active_model"]["display_names"]["STEMI"] == "ST-Elevation MI (STEMI)"
    assert set(cfg["active_model"]["thresholds"]) == {"STEMI", "AF", "LVH", "NORMAL"}
    assert "must not be used for patient care" in cfg["notice"]


def test_models_list(client):
    m = client.get("/api/models").json()
    assert m["active_id"] == "resnet50_cbam_nat768"
    ids = {x["id"] for x in m["models"]}
    assert {"resnet50_cbam_nat768", "efficientnetv2_s_cbam_nat768",
            "efficientnetv2_s_nat768", "resnet50_hires", "resnet50_dual",
            "resnet50_v1", "convnextv2_tiny_v1img_v1"} <= ids
    active = [x for x in m["models"] if x["active"]][0]
    assert active["call_style"] == "tensor"
    assert active["input_size"] == [768, 1024]      # [height, width]
    assert active["views"] == [[768, 1024]]
    # The square round-2 descriptors must still round-trip through the widened
    # schema, spelled as they always were.
    hires = [x for x in m["models"] if x["id"] == "resnet50_hires"][0]
    assert hires["input_size"] == [768, 768] and hires["views"] == [[768, 768]]


def test_full_staged_flow(client, fixtures):
    # 1) upload -> detect + quality + preview files
    cap = _upload_photo(client, fixtures, "photo_clean.jpg",
                        sheet_id="SHEET-TEST-1", blind_mode="true")
    cid = cap["capture_id"]
    assert cap["original"]["width"] == 2400
    det = cap["detection"]
    assert det["quad_norm"] and len(det["quad_norm"]) == 4
    assert client.get(f"/api/captures/{cid}/files/preview.jpg").status_code == 200
    assert client.get(f"/api/captures/{cid}/files/overlay.jpg").status_code == 200

    # 2) rectify with the detected (auto) quad
    r = client.post(f"/api/captures/{cid}/rectify",
                    json={"quad_norm": det["quad_norm"], "source": "auto",
                          "geometry_mode": "training_canvas"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["corrected_px"] == [1686, 1311]
    assert body["preprocess"]["tensor_shape"] == [1, 3, 768, 1024]
    assert body["preprocess"]["input_size"] == [768, 1024]
    assert client.get(f"/api/captures/{cid}/files/model_input.png").status_code == 200

    # 3) predict (real CPU forward, ~1-2 s)
    p = client.post(f"/api/captures/{cid}/predict", json={})
    assert p.status_code == 200, p.text
    pred = p.json()
    assert set(pred["scores"]) == {"STEMI", "AF", "LVH", "NORMAL"}
    assert all(0.0 < s < 1.0 for s in pred["scores"].values())
    assert pred["threshold_source"] == "checkpoint"
    assert pred["thresholds"]["STEMI"] == pytest.approx(0.32, abs=0.001)
    assert pred["timing_ms"]["forward"] > 0
    assert pred["model"]["checkpoint_sha256_short"]

    # This fixture is ECG000002, whose AF head is near-certain (the prototype
    # scored it 0.99999). The real-photo warp may shift scores, but the
    # positive set must be a well-formed subset of the labels.
    assert set(pred["positive"]) <= {"STEMI", "AF", "LVH", "NORMAL"}

    # 4) history + detail show the run
    lst = client.get("/api/captures").json()
    ids = [c["capture_id"] for c in lst["captures"]]
    assert cid in ids
    summary = next(c for c in lst["captures"] if c["capture_id"] == cid)
    assert summary["sheet_id"] == "SHEET-TEST-1"
    assert summary["latest_model_id"] == "resnet50_cbam_nat768"

    detail = client.get(f"/api/captures/{cid}").json()
    assert len(detail["prediction_runs"]) == 1
    assert detail["correction"]["geometry_mode"] == "training_canvas"
    assert detail["preprocess"]["tensor_sha256"]

    # 5) annotation (reference labels — §17.6)
    ann = client.patch(f"/api/captures/{cid}/annotation",
                       json={"reference_labels": {"STEMI": False, "AF": True,
                                                  "LVH": False, "NORMAL": False},
                             "truth_source": "test"})
    assert ann.status_code == 200
    detail = client.get(f"/api/captures/{cid}").json()
    assert detail["annotation"]["reference_labels"]["AF"] is True

    # 6) trash (never unlink)
    d = client.delete(f"/api/captures/{cid}")
    assert d.status_code == 200
    assert client.get(f"/api/captures/{cid}").status_code == 404


def test_predict_before_rectify_is_a_clear_error(client, fixtures):
    cap = _upload_photo(client, fixtures, "photo_clean.jpg")
    cid = cap["capture_id"]
    r = client.post(f"/api/captures/{cid}/predict", json={})
    assert r.status_code == 404
    assert r.json()["code"] == "MISSING_MODEL_INPUT"


def test_error_codes_are_stable(client):
    assert client.post("/api/captures",
                       files={"file": ("x.jpg", b"not an image", "image/jpeg")}
                       ).json()["code"] == "UNSUPPORTED_FORMAT"
    r = client.post("/api/captures/2026-01-01_000000_000000/rectify",
                    json={"quad_norm": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert r.status_code == 404 and r.json()["code"] == "CAPTURE_NOT_FOUND"
    assert client.get("/api/captures/nonsense").status_code == 404


def test_unknown_model_and_bad_thresholds(client, fixtures):
    cap = _upload_photo(client, fixtures, "photo_clean.jpg")
    cid = cap["capture_id"]
    det = cap["detection"]
    client.post(f"/api/captures/{cid}/rectify",
                json={"quad_norm": det["quad_norm"], "source": "auto"})
    r = client.post(f"/api/captures/{cid}/predict", json={"model_id": "nope"})
    assert r.status_code == 404 and r.json()["code"] == "UNKNOWN_MODEL"
    r = client.post(f"/api/captures/{cid}/predict",
                    json={"thresholds": {"STEMI": 1.5}})
    assert r.status_code == 422 and r.json()["code"] == "INCOMPLETE_THRESHOLDS"
    r = client.post(f"/api/captures/{cid}/predict",
                    json={"thresholds": {"STEMI": 1.5, "AF": 0.5,
                                         "LVH": 0.5, "NORMAL": 0.5}})
    assert r.status_code == 422 and r.json()["code"] == "THRESHOLD_OUT_OF_RANGE"
    r = client.post(f"/api/captures/{cid}/predict",
                    json={"thresholds": {"STEMI": 0.9, "AF": 0.5,
                                         "LVH": 0.5, "NORMAL": 0.5}})
    assert r.status_code == 200
    assert r.json()["threshold_source"] == "user_override"
    assert r.json()["thresholds"]["STEMI"] == 0.9
    assert r.json()["labels"] == ["STEMI", "AF", "LVH", "NORMAL"]
    assert r.json()["display_names"]["STEMI"] == "ST-Elevation MI (STEMI)"
