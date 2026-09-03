# Models directory

A model = **one `.pt` + one YAML descriptor** — the model is data, not code.
Swapping models is adding two files and pointing `active.yaml` (or
`CARDIOSENTRY_ACTIVE_MODEL`) at the new descriptor. Zero application-code
changes.

## Layout

```
models/
├── active.yaml -> efficientnetv2_s_nat768.yaml   # which model the app serves
├── efficientnetv2_s_nat768.yaml           # round-3 PLAIN baseline, 768x1024 (default)
├── efficientnetv2_s_cbam_nat768.yaml      # round-3, same geometry + CBAM
├── resnet50_cbam_nat768.yaml              # round-3, same geometry, other backbone
├── resnet50_hires.yaml                    # round-2 control, 768px square
├── resnet50_dual.yaml                     # round-2 dual-stream 384+768
├── resnet50_v1.yaml                       # round-1, 384px, tensor call style
└── weights/
    ├── efficientnetv2_s_nat768.pt
    ├── resnet50_cbam_nat768.pt
    ├── efficientnetv2_s_cbam_nat768.pt
    ├── resnet50_hires.pt
    ├── resnet50_dual.pt
    └── resnet50_v1.pt
```

## Where the `.pt` files come from

The checkpoints are the training outputs in the research tree; copy them here:

```bash
cp "../prototype-results/efficientnetv2_s@nat768/weights/best.pt" \
   weights/efficientnetv2_s_nat768.pt
cp "../prototype-results/resnet50_cbam@nat768/weights/best.pt" \
   weights/resnet50_cbam_nat768.pt
cp "../prototype-results/efficientnetv2_s_cbam@nat768/weights/best.pt" \
   weights/efficientnetv2_s_cbam_nat768.pt
cp ../prototype-results/hybrid_v1/resnet50_hires/weights/best.pt weights/resnet50_hires.pt
cp ../prototype-results/hybrid_v1/resnet50_dual/weights/best.pt  weights/resnet50_dual.pt
cp ../prototype-results/v1/resnet50/weights/best.pt              weights/resnet50_v1.pt
```

`weights/` is gitignored (the files are 100–500 MB training artifacts). The
descriptors pin each checkpoint with `expected_sha256` and
`expected_protocol_fingerprint`, so a swapped or corrupted `.pt` **refuses to
start** instead of silently invalidating every recorded prediction (§16 #18).

## The descriptor format

See `backend/schemas/model_descriptor.py` (the pydantic schema that validates
each YAML at startup) and `resnet50_cbam_nat768.yaml` for the annotated
example. Key fields:

- `architecture.module` — vendored arch package (`arch.hybrids_v2` /
  `arch.hybrids_v1` / `arch.runs_v1`, under `backend/arch/`); `registry_key` —
  key in its REGISTRY.
- `architecture.call_style` — `view_dict` (round-2: `model({768: t})`) or
  `tensor` (round-1 and round-3: `model(t)`).
- `architecture.views` — per-view input geometry. An int is a square view
  (`[384, 768]` = two square views); a pair is `[height, width]`
  (`[[768, 1024]]`). `view_dict` keys its input dict by a single integer, so
  that call style accepts square views only — the schema refuses the
  combination rather than silently transposing something.
- `labels` — MUST match the checkpoint's head order (`[STEMI, AF, LVH,
  NORMAL]`, F1); `display_names` are the UI strings.
- `thresholds.source` — `checkpoint` reads the frozen per-class thresholds
  from the `.pt` (default; startup refuses if absent); `explicit` takes the
  values from the descriptor.
- `preprocess` — the training input chain parameters; keep them at the
  training values unless you are deliberately running the experiment those
  flags exist for. `input_size` is the one that decides the geometry: `768`
  (square) or `[768, 1024]` (**height, width** — torch order, the same order the
  research code's `config.IMG_SIZE` uses and the opposite of PIL's `(w, h)`).
  Transposing it produces a portrait ECG that still scores and is silently
  wrong.
- `verify` — the startup self-check's inputs for THIS checkpoint.
  `expected_scores` is the path (relative to the app root) of the run's own
  stored score vector for `tests/fixtures/corpus_sheet.jpg`; pinning it here is
  what keeps a model swap from leaving the check comparing the newly active
  model against another run's numbers. `score_tolerance` overrides
  `settings.verify_score_tolerance` (2e-3). Raise it only against a
  measurement: the default is a bound in PROBABILITY space, so what it permits
  depends on where a run's fixture scores sit on the sigmoid — 2e-3 at p=0.999
  allows a ~2.0 logit shift, at p=0.93 only ~0.03.
- `explain` — attribution defaults for this checkpoint.
  `target_layer` is the Grad-CAM feature map as a dotted module path; `null`
  auto-picks the last 4-D feature map, which is right for every checkpoint
  here (`GET /api/models/<id>/layers` lists the candidates and what auto
  chose). `header_box` is the printed ID/Age/Sex rectangle as
  `[x0, y0, x1, y1]` fractions — **no pixels are altered**, it is the region
  every attribution map is scored against. `header_redacted_in_training` says
  whether this run trained with that block painted out; see below for why that
  changes how a header-heavy map should be read.

## The header is not redacted at serving time

The round-3 corpus was built with the printed ID/Age/Sex block painted out
(`hybrids_v2` `REDACT_BOX = [0.10, 0.02, 0.30, 0.18]`), because the record ID
separates STEMI exactly in that synthetic corpus. **The application does not
redact uploaded photographs** — an uploaded sheet reaches the model as
photographed.

That is a deliberate train/serve difference, and it is the kind that belongs in
a writeup rather than in a footnote: the model sees a header band it never saw
printed text in during training. On the checked-in fixture sheet it moves the
score vector by ~1e-4, which is why `scripts/verify_model.py` still reproduces
the prototype's stored numbers well inside its 2e-3 tolerance — but that is one
sheet, and a real photograph carries a real patient header rather than a
synthetic `ECGnnnnnn`.

This is the difference the explainability stage exists to measure rather than
argue about. Every attribution map reports `header_frac` — the share of its
mass inside `explain.header_box` — and `lift`, that share over the box's share
of the sheet *area*. On the checked-in fixture (`corpus_sheet.jpg`, target
layer `attn`) the two round-3 checkpoints do **not** behave the same there:

| Grad-CAM header lift | STEMI | AF | LVH | NORMAL |
|---|---|---|---|---|
| `efficientnetv2_s_nat768` *(active)* | 0.78 | 0.42 | 0.52 | 1.11 |
| `efficientnetv2_s_cbam_nat768` | 1.64 | 0.50 | 1.86 | 1.62 |
| `resnet50_cbam_nat768` | 0.005 | 0.003 | 0.034 | 0.000 |

A lift of 1.0 is uniform attention, and the peak lands outside the box for
every row here. The active plain baseline sits at or below uniform on three of
four labels — better than its CBAM sibling, which puts ~1.6-1.9x uniform weight
there — but nowhere near the two ResNet-50 rows, which are three orders of
magnitude lower. Read that alongside the self-check: painting the training
REDACT_BOX back on moves this checkpoint's AF score by 6.7e-3, against 1.2e-4
for the CBAM run, so the header demonstrably reaches its output. That is one
sheet, so it is a flag rather than a verdict: run the same measurement over
real photographs and report the distribution before quoting this model's STEMI
score anywhere.

## Adding a new model

1. Copy the `.pt` into `weights/`.
2. Write `models/<id>.yaml` (start from `resnet50_cbam_nat768.yaml`);
   fill `expected_sha256` with `sha256sum` of the file, and
   `expected_protocol_fingerprint` from the blob's `protocol_fingerprint`
   (or `fingerprint`, which is what round-1 and round-3 checkpoints store).
3. If the run used an architecture package that is not vendored yet, vendor it
   under `backend/arch/` first — see `backend/arch/README.md`.
4. Point `active.yaml` at it (or set `CARDIOSENTRY_ACTIVE_MODEL=<id>`).
5. Write the run's own stored score for `tests/fixtures/corpus_sheet.jpg` to
   `tests/fixtures/expected_scores_<id>.json` (that sheet is ECG000002, so the
   row is in the run's `predictions/test_predictions.csv`), add it to
   `EXPECTED_SCORES` in `tests/make_fixtures.py`, and point the descriptor's
   `verify.expected_scores` at it. Then restart and run
   `python scripts/verify_model.py --model <id>` — a shape check alone will not
   catch a transposed or wrong geometry, and a matching score vector will. If
   it fails, measure WHY before touching `verify.score_tolerance`: painting the
   training `REDACT_BOX` onto the fixture separates the documented header
   train/serve gap from actual timm/weights drift.
6. Sanity-check the Grad-CAM target with
   `curl localhost:8000/api/models/<id>/layers` — `auto_target_layer` should
   be the deepest feature map the head pools (`attn` on a CBAM hybrid,
   `backbone.layer4` on a plain ResNet-50). Pin `explain.target_layer` in the
   YAML if it is not.
7. Old captures can be re-scored by the new model without re-photographing:
   `python scripts/reprocess.py --all --model <id>`. When the new model's
   geometry differs from the one a capture was stored at, the script rebuilds
   `model_input.png` from `corrected.png` rather than resampling a square into
   a rectangle.
