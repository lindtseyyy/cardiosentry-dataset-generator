---
name: cardiosentry-app-map
description: Use when reading, changing, debugging, or running anything under cardiosentry-application/ (the FastAPI + React ECG-sheet serving app) — routes a task straight to the files that own it, records the invariants and the verification commands, so the tree does not have to be re-explored every time.
whenToUse: Any task touching cardiosentry-application/ — swapping or adding a model, editing the preprocess/detect/rectify/explain pipeline, changing API routes or the frontend, adjusting settings, or running its tests.
---

# CardioSentry application — map

**Root:** `cardiosentry-application/` (paths below are relative to it).
Last verified against the tree: **2026-08-29**. Keep it that way — see
[Maintaining this skill](#maintaining-this-skill), which is **mandatory** on
every change to the app.

## Read these first, not the whole tree

| Question | File | Don't |
|---|---|---|
| What is this app, what are the stages, why is the code like this | `README.md` (§Layout, §The non-negotiables, §Models shipped) | re-derive it from source |
| What each model is, how to swap one, the header caveat | `models/README.md` | read every `models/*.yaml` |
| Where a vendored architecture came from and what was changed | `backend/arch/README.md` | diff against `prototype-code/` |
| The spec the `§n.n` comments cite | `../CARDIOSENTRY_WEB_APP_PLAN.md` (1105 lines) | grep it unless a `§` reference needs resolving |

## Layout

```
backend/
  main.py          lifespan: registry -> load model -> warm-up -> self-check -> QR/URLs -> static mount
  settings.py      ALL config; pydantic-settings, env prefix CARDIOSENTRY_, reads ./.env
  errors.py        ApiError / NotFound (code + status, stable error codes)
  api/             routes only — thin; state.py owns the lifespan singletons
  services/        the actual work (see table below)
  schemas/         pydantic: request/response AND the on-disk record shapes
  arch/            VENDORED training-time nn.Module code, one package per round
models/            <id>.yaml descriptors + active.yaml symlink + weights/ (gitignored)
frontend/          Vite + React 18 + TS PWA; src/{screens,components,api}
scripts/           verify_model.py · reprocess.py · export_dataset.py · make_cert.sh
tests/             conftest pins ONE model; fixtures/ checked in
captures/          runtime data (gitignored) — filesystem + JSON IS the database
```

### services/ — who owns what

| File | Owns | Note |
|---|---|---|
| `ingest.py` | upload validation, EXIF orient, GPS strip, hash, store `original.jpg` | size/pixel bombs gated here |
| `detect.py` | paper-quad detection | classical OpenCV + fallback ladder, no ML |
| `rectify.py` | homography warp -> `corrected.png` on a fixed canvas | `training_canvas` \| `native_aspect` |
| `preprocess.py` | ★ **the canonical training-input chain** -> `model_input.png` + tensor | see tripwire 1 |
| `quality.py` | blur/brightness/contrast/glare/px-per-mm warnings | warn, **never** block |
| `registry.py` | discover + validate `models/*.yaml`, resolve `arch.*`, hash weights | all at startup, fails loud |
| `runner.py` | build arch, `load_state_dict(strict=True)`, thresholds, views, predict | F2/F3 adapters live here |
| `explain.py` | Grad-CAM / Integrated Gradients / Jacobian, header metric, memory plan | 937 lines, the biggest module |
| `storage.py` | capture dirs, atomic JSON, id regexes, **file-serving whitelists** | path-traversal gate |

### API surface

```
GET    /api/health                                  health.py
GET    /api/config                                  config.py   (frontend holds NO model knowledge)
GET    /api/models                 /api/models/active            models.py
GET    /api/models/{id}/layers                      explain.py  (models_router — NOT models.py)
POST   /api/captures                                captures.py  ingest + detect + quality
POST   /api/captures/{id}/rectify                   captures.py  warp + preprocess
POST   /api/captures/{id}/predict                   captures.py  forward
GET    /api/captures  ·  /api/captures/{id}         captures.py
GET    /api/captures/{id}/files/{name}              captures.py  FILE_WHITELIST only
PATCH  /api/captures/{id}/annotation                captures.py  reference labels (§17.6)
DELETE /api/captures/{id}                           captures.py  trash, never unlink
POST   /api/captures/{id}/explain                   explain.py
GET    /api/captures/{id}/explain[/{xid}[/files/{name}]]         explain.py
DELETE /api/captures/{id}/explain/{xid}             explain.py
```

### On-disk capture record

```
captures/YYYY-MM-DD/YYYY-MM-DD_HHMMSS_xxxxxx/
  capture.json  original.jpg  preview.jpg  overlay.jpg  corrected.png  model_input.png
  predictions/run_<ts>_<model_id>.json
  explain/xai_<ts>_<method>_<model_id>/{<method>_<LABEL>_{heatmap,overlay}.png, record.json}
```
Ids and servable names are regex-gated in `storage.py`
(`CAPTURE_ID_RE`, `FILE_WHITELIST`, `EXPLAIN_ID_RE`, `EXPLAIN_FILE_RE`). A new
stage file must be added to the whitelist or it 404s.

## Task → files (change these together)

| Task | Touch |
|---|---|
| **Swap the active model** | `models/active.yaml` symlink → done. Then the coupled docs/tests below. |
| **Add a model** | see [Adding a model](#adding-a-model) — it is a 7-step checklist, follow it |
| Change a threshold/label/geometry for one model | that `models/<id>.yaml` only — **never** `settings.py` |
| Add a descriptor field | `schemas/model_descriptor.py` + every `models/*.yaml` that needs it + `models/README.md` "descriptor format" |
| Change the input chain | `services/preprocess.py` **and** `tests/test_preprocess_parity.py` (it proves parity vs training code to 1e-6) |
| Add/alter an API route | `api/<mod>.py` + `schemas/` + `frontend/src/api/{client,types}.ts` + `tests/test_api_smoke.py` |
| Add a stage output file | `services/storage.py` FILE_WHITELIST + `api/captures.py` + frontend |
| Change explain behaviour | `services/explain.py` + `api/explain.py` + `schemas/explain.py` + `tests/test_explain.py` |
| Add a setting | `settings.py` + `.env.example` (keep them in sync — the example is the docs) |
| Vendor a new arch package | `backend/arch/<pkg>/` + `backend/arch/README.md` provenance table + deviations list |
| Frontend change | `frontend/src/`; rebuild `npm run build` — the backend serves `frontend/dist/` |

## Adding a model

`models/README.md` has the authoritative 7-step version. The parts that are
easy to get wrong:

1. Which **arch package** the run used decides everything, and the run's
   directory name does not say: `efficientnetv2_s@nat768` and
   `efficientnetv2_s_cbam@nat768` are different packages. The banner at the top
   of `prototype-results/<run>/report.txt` names it (`CardioSentry baselines v2
   | <run>`), and `torch.load(best.pt)["model"]` settles it: bare timm keys
   (`conv_stem…classifier`) = `baselines_v2`, `backbone.*`/`attn.*` =
   `hybrids_v2`. Wrong package = `strict=True` failure, not a subtle bug. Not
   vendored yet? Vendor it first (`backend/arch/README.md`).
2. `expected_protocol_fingerprint` — round-1/round-3 blobs store only
   `fingerprint`; the runner falls back to it. Copy from `config_snapshot.json`.
3. `explain.target_layer` — probe, don't guess: `GET /api/models/<id>/layers`,
   or `xai.auto_target_layer(lm)`. `attn` on a CBAM hybrid, `bn2` on a plain
   EfficientNet, `layer4`/`backbone.layer4` on a ResNet.
4. `verify.expected_scores` — the run's own row for `ECG000002` from
   `prototype-results/<run>/predictions/test_predictions.csv`, written to
   `tests/fixtures/expected_scores_<id>.json` **and** added to `EXPECTED_SCORES`
   in `tests/make_fixtures.py` (which regenerates them).
5. If the self-check fails, **measure why before raising `verify.score_tolerance`**
   — see tripwire 3.

## Tripwires (each of these has already cost a debugging session)

1. **`input_size` is `[height, width]`** — torch order, the opposite of PIL's
   `(w, h)`. Transposed, it produces a portrait ECG that still scores and is
   silently wrong. Same for `views`.
2. **The app does not redact uploaded photographs**, but every round-3 model
   trained with the printed ID/Age/Sex block painted out
   (`REDACT_BOX = [0.10, 0.02, 0.30, 0.18]`). That band is a hard STEMI leak in
   the synthetic corpus. It is a deliberate, documented train/serve gap, and it
   is what `explain`'s `header_frac`/`lift` metric exists to measure. Never
   "fix" it silently.
3. **`verify.score_tolerance` is a bound in PROBABILITY space**, so what it
   permits depends on where a run's fixture scores sit on the sigmoid: 2e-3 at
   p=0.999 allows a ~2.0 logit shift, at p=0.93 only ~0.03. An unsaturated run
   needs a bigger number for the *same* logit tolerance. Before raising it,
   paint the training `REDACT_BOX` onto the fixture and re-score — that
   separates the documented header gap from real timm/weights drift.
4. **The test suite pins ONE model on purpose** (`tests/conftest.py`:
   `resnet50_cbam_nat768`), independent of `models/active.yaml`, because
   several assertions are ResNet-50 facts (`backbone.layer3`, 2048 channels).
   Swapping the active model must **not** repoint that pin. `test_api_smoke.py`
   asserting `resnet50_cbam_nat768` is correct and expected.
5. **The model is data, not code.** A swap = two files + a symlink, zero
   application changes. If a swap seems to need Python edits, that is the bug.
6. **`settings.py` preprocess values are FALLBACKS** — the active descriptor
   wins. Same for `verify_expected_scores_json` / `verify_score_tolerance`,
   which the descriptor's `verify` block overrides.
7. **Do not upgrade timm.** The vendored packages depend on `create_model` tag
   semantics; the pin is `timm 1.0.28`.

## Running things

No virtualenv and **no `python` binary** — use system `python3` (torch, timm,
fastapi, cv2 all present). Run from the app root.

```bash
python3 -m pytest -q                     # 43 tests, real CPU model, minutes not seconds
python3 -m pytest -q tests/test_api_smoke.py
python3 scripts/verify_model.py                  # active model vs its own stored vector
python3 scripts/verify_model.py --model <id>     # any descriptor; picks up its verify block
python3 scripts/reprocess.py --all --model <id>  # re-score stored captures, no re-photographing
python3 -m backend.main                          # serve (prints LAN URL + QR)
cd frontend && npm run build                     # backend serves frontend/dist/
```

Boot health, without curl or a server:

```python
from fastapi.testclient import TestClient
from backend.main import app
with TestClient(app) as c:
    print(c.get("/api/health").json())
```
A healthy boot logs `verify_model self-check passed: max |Δ| = …`. The check is
**warn-only** — the app serves anyway, so read the log, don't assume.

Scratch scripts that import `backend.*` need
`sys.path.insert(0, "<app root>")`; put them in the scratchpad, not the repo.

## Maintaining this skill

**This skill is part of the app's surface: any change to
`cardiosentry-application/` must update it in the same turn as the code.** It
exists so the tree is not re-explored every session; a stale map is worse than
none, because it is trusted.

Update when you:

- add/remove/rename a **route**, a **service**, a **schema**, or a **script**
  → the Layout, services table, API surface, or task table
- add a **model**, a **descriptor field**, or a **vendored arch package**
  → the relevant section, and `models/README.md` / `backend/arch/README.md`
- change how the app is **run, tested, or verified** → Running things
- hit a bug that cost real time to diagnose → add it as a **tripwire**, with
  the measurement that proved it
- change the **active model** → nothing here names it (deliberately: the map
  must not go stale on a swap) — confirm that stays true

Then bump `Last verified` at the top to the date you checked the tree, and keep
counts (43 tests, LOC, route list) honest — correct them rather than leaving a
number that is now wrong.
