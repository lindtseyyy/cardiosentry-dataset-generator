# CardioSentry Web Application

An **instrument, not a product**: a single-user web app that puts a real
smartphone photograph of a real printed 12-lead ECG sheet into the CardioSentry
image-space multi-label classifier through a preprocessing chain that
**byte-for-byte mirrors the training pipeline**, and records everything about
how that happened — original, detected quad, corrected sheet, exact model
input, all four sigmoid scores, thresholds, timing, versions.

It answers one question: *what does the model do when the input is a real
photograph instead of a synthetic render?* Everything else follows from that.
Full design rationale: [`CARDIOSENTRY_WEB_APP_PLAN.md`](../CARDIOSENTRY_WEB_APP_PLAN.md).

## Run it (10 lines)

```bash
cd cardiosentry-application

# 1. backend deps (the repo's .venv already has torch/timm; install the rest)
../.venv/bin/pip install -e .[dev]

# 2. weights (gitignored training artifacts — copy from the research tree)
cp ../prototype-results/hybrid_v1/resnet50_hires/weights/best.pt models/weights/resnet50_hires.pt
cp ../prototype-results/hybrid_v1/resnet50_dual/weights/best.pt  models/weights/resnet50_dual.pt
cp ../prototype-results/v1/resnet50/weights/best.pt              models/weights/resnet50_v1.pt

# 3. self-checks: preprocess parity vs the training pipeline + model parity
../.venv/bin/python -m pytest tests/
../.venv/bin/python scripts/verify_model.py

# 4. frontend build (needs Node ≥ 18)
(cd frontend && npm install && npm run build)

# 5. serve API + frontend on the LAN (single process, same origin)
../.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The terminal prints every LAN URL and a QR code. On the phone (same Wi-Fi),
open `http://<laptop-ip>:8000` → *Take photo*. The camera hand-off uses
`<input capture>` which works over plain HTTP; live `getUserMedia` preview and
PWA install arrive with HTTPS (`scripts/make_cert.sh`, plan §15.3).

## What it does, stage by stage

```
upload ──▶ ingest ──▶ detect ──▶ corner edit ──▶ rectify ──▶ preprocess ──▶ predict
            (bytes,    (OpenCV     (4 draggable    (homography   (THE training   (sigmoid,
             EXIF,     quad +      handles,        to 1650×1275,  chain: PIL      per-class
             sha256)  fallbacks)   loupe)          +18px margin)  BICUBIC to the  thresholds)
                                                                  model geometry,
                                                                  q95 JPEG)
```

Every arrow is a recorded transformation, persisted per capture:

```
captures/2026-08-23/2026-08-23_014215_a3f9c1/
├── original.jpg        # uploaded bytes (GPS EXIF stripped by default, recorded)
├── preview.jpg         # ~1280px, phone UI only — never for inference
├── overlay.jpg         # preview + detected/final quad — debugging aid
├── corrected.png       # 1686×1311 rectified sheet, LOSSLESS
├── model_input.png     # the active model's geometry (768×1024 by default),
│                       #   LOSSLESS, exactly the tensorized pixels
├── capture.json        # everything about the image and its transformations
├── predictions/        # run_<stamp>_<model>.json — one per model run
└── explain/            # xai_<stamp>_<method>_<model>/ — one per attribution run
                        #   record.json + <method>_<LABEL>_{heatmap,overlay}.png
```

### Explainability — where the model looked

A fourth, opt-in stage runs on the **same stored `model_input.png`** the scores
came from, so a map and a score always describe the same pixels:

| method | what it answers | resolution | cost (CPU, 768×1024) |
|---|---|---|---|
| **Grad-CAM** | which *region* of the sheet | the 24×32 feature grid (~32 px cells) — a lead, not a segment | ~3 s, **+61 MB**, all four labels |
| **Integrated Gradients** | which *pixels*, and in which direction | per-pixel, signed (red raised the score, blue lowered it) | ~13 s × `steps`, **per label** |
| **Jacobian** | how the whole *head* behaves | per-pixel per label, plus a label-coupling matrix | ~8 s **+2.5 GB**, or ~15 s +0.5 GB checkpointed |

Grad-CAM and the Jacobian default to the whole head; Integrated Gradients is
`steps` full forward+backward passes per label, so it defaults to the labels the
model actually flagged. `GET /api/config` quotes the pass count before the wait.

### Memory — the thing that will bite you

Enabling autograd on this model at 768×1024 costs **~2.3 GB of stored
activations**, against 84 MB for the same forward under `no_grad`. On a laptop
serving the app with a browser open (~2.6 GB free, no swap) that is not "slow":
it is the OOM killer taking the whole server down mid-request. Two mitigations,
both on by default:

- **Grad-CAM records no graph before its target layer.** Its gradient is
  `dlogit/dA` for `A` the target layer's output, so nothing upstream appears in
  it. The forward runs under `no_grad`, and a hook on the target detaches `A`
  into a leaf and switches recording on for the remainder. Same derivative,
  **bit-identical CAMs**, and `+2307 MB` becomes `+44 MB`.
- **Integrated Gradients and the Jacobian genuinely need the input gradient**,
  so the full graph is unavoidable — but gradient checkpointing recomputes
  activations instead of storing them: `+2478 MB / 7.6 s` becomes
  `+506 MB / 14.5 s`, equal to 1e-7. `CARDIOSENTRY_EXPLAIN_MEMORY_POLICY=auto`
  (the default) turns it on only when the machine could not otherwise fit.

If even the checkpointed run will not fit, the request is **refused with a 507
naming both numbers** rather than gambling on the OOM killer — which would lose
the loaded model and a 30 s restart with it. Grad-CAM stays available on
machines where the other two cannot run, and the error says so.

Two things every run reports, and they are the point of the feature:

- **`header_frac` / `lift`** — how much attribution mass landed on the printed
  ID/Age/Sex block, over that block's share of the sheet *area*. The record ID
  printed there separates STEMI **exactly** in the synthetic corpus, and the
  `nat768` checkpoints trained with it painted out while the app deliberately
  does not redact an upload. `lift ≈ 1` is a fair share; `lift ≥ 2` raises a
  loud badge in the UI and a caveat in the record.
- **`completeness_delta`** (Integrated Gradients only) — how far the
  attributions are from summing to `F(x) − F(baseline)`, which is the axiom the
  method is built on. Over ~0.2 means the step count was too low and the map is
  under-resolved; no colour map would have told you.

```bash
curl -X POST localhost:8000/api/captures/$CID/explain \
     -H 'Content-Type: application/json' -d '{"method":"gradcam"}'
curl -X POST localhost:8000/api/captures/$CID/explain \
     -H 'Content-Type: application/json' \
     -d '{"method":"integrated_gradients","labels":["STEMI"],"steps":64}'
curl localhost:8000/api/models/active/layers      # Grad-CAM target candidates
```

Attribution runs are **derived data** — every one is reproducible from
`model_input.png` plus the parameters in its `record.json` — so deleting one
removes it outright rather than moving it to `_trash/`. Re-rectifying a capture
marks its existing runs `stale`: they explain pixels the capture no longer has.

`predict` operates on the persisted `model_input.png`, so the **same real
photograph** can be re-scored by a different `.pt` with a bit-identical input:

```bash
../.venv/bin/python scripts/reprocess.py --all --model resnet50_dual
../.venv/bin/python scripts/export_dataset.py --out captures_export.csv
```

## The non-negotiables (why this code looks the way it does)

- **The original is sacred.** Uploaded bytes are written unmodified before
  anything else touches them; all geometry applies to the full-resolution
  oriented original.
- **Preprocessing lives in exactly one module** — `backend/services/preprocess.py`,
  importable by API and scripts. It reproduces the training chain exactly
  (PIL BICUBIC — *not* OpenCV — for the final resize, in-memory q95 JPEG
  round-trip, ToTensor, ImageNet normalize), and `tests/test_preprocess_parity.py`
  proves it against the actual training code to `1e-6` before any real capture.
- **The model is data, not code.** One `.pt` + one YAML descriptor
  (`models/*.yaml`); architectures are vendored verbatim under `backend/arch/`
  with provenance. `verify_model.py` catches timm/weights drift.
- **Per-class thresholds come from the checkpoint** (`[0.54, 0.21, 0.69, 0.86]`
  for the active `efficientnetv2_s_nat768`); a silent 0.5 default is refused at
  startup.
- **Four independent sigmoid heads**, displayed in fixed order, never argmax,
  never sorted by score; `NORMAL` + pathology contradictions are displayed,
  not resolved. The UI says "Model score", never a probability of disease.
- **Warn, never block** on image quality: a rejected photo is a missing data
  point.
- **Blind mode + reference labels** (§17.6): the app records what the ECG
  actually shows, and can hide scores until labels are entered, so the dataset
  is defensible.

## Models shipped

| id | checkpoint | input (h×w) | call style | purpose |
|---|---|---|---|---|
| `efficientnetv2_s_nat768` *(active)* | baselines_v2 | 768×1024 | `tensor` | round-3 **plain baseline** — the unmodified backbone at the sheet's native aspect. Val macro-AUPRC 0.9501; the default serving model |
| `efficientnetv2_s_cbam_nat768` | hybrids_v2 | 768×1024 | `tensor` | same backbone and geometry, plus CBAM (0.9571) |
| `resnet50_cbam_nat768` | hybrids_v2 | 768×1024 | `tensor` | same protocol, geometry and attention block, other backbone (0.9570) |
| `resnet50_hires` | hybrid_v1 | 768×768 | `view_dict` | round-2 control |
| `resnet50_dual` | hybrid_v1 | 384×384 + 768×768 | `view_dict` | dual-resolution hybrid |
| `resnet50_v1` | runs_v1 | 384×384 | `tensor` | round-1 single stream (exercises the F3 adapter) |

The active model is a **rectangle at the sheet's own aspect**, not a square: the
768×768 geometry squeezed the time axis by 22% and spent ~27% of the vertical
budget on blank paper.

The three round-3 descriptors share a protocol fingerprint
(`36988c5e0ba5d72d`) and a geometry, so they are directly comparable. The
active one is the **plain baseline**: no attention block, no fusion, just
`tf_efficientnetv2_s.in1k` with a 4-logit head. It is also the weakest of the
three by macro-AUPRC — 0.9501, against 0.9571 for the same backbone with CBAM
and 0.9570 for ResNet-50+CBAM — and the gap is wider on the label that matters
most here (val STEMI AUPRC 0.9513, sensitivity 0.8889, against 0.9701 / 0.9236
for `resnet50_cbam_nat768`). Serving it is a deliberate choice of the
unmodified reference over the better scores; if that was not the intent, swap
with `CARDIOSENTRY_ACTIVE_MODEL=resnet50_cbam_nat768` or by repointing
`models/active.yaml`.

One train/serve difference is deliberate and worth stating in any writeup: the
round-3 training corpus had the printed ID/Age/Sex block painted out, and **the
app does not redact uploaded photographs** — an uploaded sheet reaches the model
as photographed. See `models/README.md`.

## Security & privacy (read before serving)

- Bind `0.0.0.0` only on networks you own; on shared Wi-Fi use
  `CARDIOSENTRY_LOCAL_ONLY=true` or set `CARDIOSENTRY_APP_TOKEN`.
- No port forwarding, no public tunnels: the app has no authentication beyond
  the optional token.
- `captures/` is gitignored **before the first run** — photos may carry
  printed patient identifiers. GPS EXIF is stripped by default (§13.4).
- `.pt` loading uses `torch.load(weights_only=False)` — acceptable only
  because the checkpoints are self-produced and sha-pinned; never accept a
  `.pt` through the API.

## Layout

```
backend/   FastAPI app: api/ (routes) · services/ (ingest, detect, rectify,
           preprocess, quality, registry, runner, storage, explain) · arch/
           (vendored model architectures) · schemas/ (pydantic records)
frontend/  Vite + React 18 + TS PWA (zero client-side image processing)
models/    descriptors + weights
scripts/   verify_model.py · reprocess.py · export_dataset.py · make_cert.sh
tests/     preprocess parity (★) · detection · API smoke · explainability
           (fixtures included)
```

## Known v1 limits (all deliberate — see the plan)

- No waveform digitization, no database (filesystem + JSON is the source of
  truth), no auth beyond the token, no queue (single user, CPU inference
  ~0.5–2 s), no ONNX/TorchScript (fidelity over speed), service worker only
  registers once HTTPS is set up (phase 3, `scripts/make_cert.sh`).
- Scores are **not calibrated** on real photographs; threshold-derived labels
  are provisional by design (§17.5).
- Attribution runs synchronously in the request — there is no job queue, so an
  Integrated Gradients run at high `steps` holds one HTTP request for minutes.
  It also peaks around 2.5 GB RSS, which is why the Riemann sum walks one path
  point at a time rather than batching: fewer labels and fewer steps are the
  knobs that pay.
# cardiosentry-application
# cardiosentry-application
