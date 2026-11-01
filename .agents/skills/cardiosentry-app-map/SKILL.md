---
name: cardiosentry-app-map
description: Use when reading, changing, debugging, or running anything under cardiosentry-application/ (the FastAPI + React ECG-sheet serving app) — routes a task straight to the files that own it, records the invariants and the verification commands, so the tree does not have to be re-explored every time.
whenToUse: Any task touching cardiosentry-application/ — swapping or adding a model, editing the preprocess/detect/rectify/explain pipeline, changing API routes or the frontend, adjusting settings, or running its tests.
---

# CardioSentry application — map

**Root:** `cardiosentry-application/` (paths below are relative to it).
Last verified against the tree: **2026-10-28**. Keep it that way — see
[Maintaining this skill](#maintaining-this-skill), which is **mandatory** on
every change to the app.

## Read these first, not the whole tree

| Question | File | Don't |
|---|---|---|
| What is this app, what are the stages, why is the code like this | `README.md` (§Layout, §The non-negotiables, §Models shipped) | re-derive it from source |
| What each model is, how to swap one, the header caveat | `models/README.md` | read every `models/*.yaml` |
| Where a vendored architecture came from and what was changed | `backend/arch/README.md` | diff against `prototype-code/` |
| The spec the `§n.n` comments cite | `../CARDIOSENTRY_WEB_APP_PLAN.md` (1105 lines; no-database requirement, Q9 and §14 "warn, never block" superseded) | grep it unless a `§` reference needs resolving |

## Layout

```
backend/
  main.py          lifespan: connect/migrate DB -> accounts -> registry -> load model -> warm-up -> load ECG gate -> self-check -> QR/URLs -> static mount
  settings.py      ALL config; pydantic-settings, CARDIOSENTRY_ prefix + DATABASE_URL alias, reads ./.env
  errors.py        ApiError / NotFound (code + status, stable error codes)
  api/             routes only — thin; state.py owns the lifespan singletons
  services/        the actual work (see table below)
  schemas/         pydantic: request/response AND stored record shapes
  arch/            VENDORED training-time nn.Module code, one package per round
models/            <id>.yaml descriptors + active.yaml symlink + gates/<id>.yaml (ECG gates: ocr_lead_labels_v1 default, learned alternative) + weights/ (gitignored)
frontend/          Vite + React 18 + TS PWA; src/{screens,components,api}
scripts/           verify_model.py · reprocess.py · export_dataset.py · import_filesystem.py · make_cert.sh
tests/             conftest pins ONE model; fixtures/ checked in
captures/          runtime PHI images/binary arrays only (gitignored); live records are in PostgreSQL; legacy JSON archived outside the repo
docs/              paper-ready system flowcharts and supporting design notes
```

- **Frontend:** the topbar username opens `#/account` for username/password
  edits in `screens/AccountScreen.tsx`; `components/PasswordInput.tsx` is shared
  with `screens/AuthScreen.tsx`.
- **Frontend typeface/layout:** Roboto only, self-hosted via
  `@fontsource/roboto` in `main.tsx` (no CDN: the app makes no outbound
  calls). `.app` is a flex column; `.app-main` grows so `.app-footer` sits at
  the bottom of the screen.

### services/ — who owns what

| File | Owns | Note |
|---|---|---|
| `db.py` | local-only psycopg pool and versioned `MIGRATIONS` | connects/migrates before model load; readable startup errors without secrets; advisory transaction lock |
| `records.py` | `RecordStore`: capture/prediction/explain JSONB, indexed ownership, live-row reads, locked updates, soft delete and import | run rows + capture run-id lists change transactionally; removes EXIF NUL padding |
| `accounts.py` | PostgreSQL `accounts` / `sessions`, salted PBKDF2-SHA256 (600k), persistent 30-day sessions, permanent `account_id` | creates `admin`/`admin` only if no account has `ADMIN_ACCOUNT_ID`, with a WARNING; missing/null owner → admin account id via `owner_of`. Rename keeps sessions; password change revokes other sessions transactionally, keeping the current one |
| `ingest.py` | upload validation, EXIF orient, GPS strip, hash, store `original.jpg` | size/pixel bombs gated here |
| `ecg_gate.py` | ECG / not-ECG gate on every upload, before detect: `load_gate` reads `models/gates/<id>.yaml` (`CARDIOSENTRY_ECG_GATE`, default `ocr_lead_labels_v1`, empty = off) and dispatches on `kind`; also the `learned` gate: frozen clean12 ConvNeXt-V1 Tiny + RFPN + `gate_head` (own `.pt`, not the diagnostic Tiny weights), threshold `gate_threshold` 0.5 | independent of the active model, not in `/api/models`; reject = `422 NOT_AN_ECG`, folder `rmtree`d, no record; accepted → `record.ecg_gate` (`kind`, `score`, `threshold`, `leads_found`); the learned gate lets EEG printouts through |
| `ecg_gate_ocr.py` | `ocr_lead_labels` gate: resize to 2400 px, keep character-sized ink blobs per `cleanup` pass (max-channel; min-channel + grid-line removal), `tesseract` CLI `--psm 11` (subprocess, `OMP_THREAD_LIMIT=1`, passes in parallel), accept at ≥ `min_leads` (4) of `counted_leads` (aVR, aVL, aVF, V1–V6) | I/II/III reported, never counted (noise); needs the tesseract binary (startup fails without it); ~2–4 s accept, ≤ ~7 s reject; far-away photos are rejected (labels unreadable) |
| `detect.py` | paper-quad detection | classical OpenCV + fallback ladder, no ML |
| `rectify.py` | homography warp -> `corrected.png` on a fixed canvas | `training_canvas` \| `native_aspect` |
| `preprocess.py` | ★ **the canonical training-input chain** -> `model_input.png` + tensor | see tripwire 1 |
| `quality.py` | brightness/contrast/clipping/glare/area/skew/px-per-mm at upload, blur + grid FFT at rectify; `poor_reasons` | "warn" flags never block; "poor" → `422 POOR_IMAGE_QUALITY` when `quality_block_on_poor` (default true): upload folder `rmtree`d, no record; rectify writes/records nothing. Geometry flags only when detection succeeded. UI: `components/QualityWarnings.tsx` on corner + preview screens |
| `registry.py` | discover + validate `models/*.yaml`, resolve `arch.*`, hash weights | all at startup, fails loud; only exception: a non-active descriptor whose `.pt` is missing is skipped with a warning |
| `runner.py` | build arch, `load_state_dict(strict=True)`, thresholds, views, predict | F2/F3 adapters live here |
| `explain.py` | Grad-CAM (the only method; IG/Jacobian removed 2026-10-04), header metric, memory guard | ~550 lines |
| `guidance.py` | four-tier urgency + next steps over a stored run and ephemeral patient answers; symptom/history escalation, conflict/caveat combinations and combination steps; Philippine hotline 911; under-18 Yellow floor + `cautions` | per-model `RULE_SETS`, locked to `SUPPORTED_MODEL_IDS`; import-time rule/combination coverage asserts; answers never stored; UI shows `cautions` prominently and `combination_notes` visibly |
| `storage.py` | image/binary dirs, id regexes, **file-serving whitelists**, trash moves | path-traversal gate; no JSON persistence |

### API surface

```
GET    /api/health                                  health.py
GET    /api/config                                  config.py   (frontend holds NO model knowledge)
POST   /api/auth/register                          auth.py      create account + sign in (201)
POST   /api/auth/login                             auth.py      sign in (200)
POST   /api/auth/logout                            auth.py      revoke session + clear cookie
GET    /api/auth/me                                auth.py      current username; session required
PATCH  /api/auth/account                           auth.py      edit username and/or password; session + current password required
GET    /api/models                 /api/models/active            models.py
GET    /api/models/{id}/layers                      explain.py  (models_router — NOT models.py)
POST   /api/captures                                captures.py  ingest + ECG gate (422 NOT_AN_ECG) + detect + quality gate (422 POOR_IMAGE_QUALITY)
POST   /api/captures/{id}/rectify                   captures.py  warp + preprocess + sharpness gate (422 POOR_IMAGE_QUALITY, nothing written)
POST   /api/captures/{id}/predict                   captures.py  forward
POST   /api/captures/{id}/guidance                  guidance.py  stored positives + optional patient context
GET    /api/captures  ·  /api/captures/{id}         captures.py
GET    /api/captures/{id}/files/{name}              captures.py  FILE_WHITELIST only
PATCH  /api/captures/{id}/annotation                captures.py  reference labels (§17.6)
DELETE /api/captures/{id}                           captures.py  trash, never unlink
POST   /api/captures/{id}/explain                   explain.py
GET    /api/captures/{id}/explain[/{xid}[/files/{name}]]         explain.py
DELETE /api/captures/{id}/explain/{xid}             explain.py
```

`main.py`'s `auth_gate` returns `401 AUTH_REQUIRED` without a valid session on
every `/api/*` path except health, config, login, register and logout. The
optional `CARDIOSENTRY_APP_TOKEN` gate still applies in addition. All capture
routes (including files, guidance and explain) are owner-scoped; another user's
capture returns `404 CAPTURE_NOT_FOUND`. Listing filters by owner before paging;
admin has no access to other users' captures.

Use `GET /api/captures/{id}/explain/{xid}` for the stored explain record.
The former `/explain/{xid}/files/record.json` URL returns `404 UNKNOWN_FILE`;
only PNGs and optional `raw.npz` are file-served.

### Storage

The record source of truth is local PostgreSQL (schema v1): `accounts`,
`sessions`, `captures`, `prediction_runs`, `explain_runs`, `schema_migrations`.
Capture, prediction and explain records use JSONB; files contain images and
optional attribution arrays:

```
captures/YYYY-MM-DD/YYYY-MM-DD_HHMMSS_xxxxxx/
  original.jpg  preview.jpg  overlay.jpg  corrected.png  model_input.png
  explain/xai_<ts>_<method>_<model_id>/
    <method>_<LABEL>_{heatmap,overlay}.png  raw.npz (optional)
```
New records store top-level `owner`: the creator's permanent account id
(username at registration; never changes on rename).
`captures.owner_id = owner_of(record)` falls back to admin for missing/null
owners, even after admin renames itself; imported record JSON is not backfilled.
DELETE sets `trashed_utc` and moves the folder to `captures/_trash/<date>/<id>/`.
Live queries exclude trashed rows.

Legacy `capture.json`, run JSON and explain `record.json` were imported,
verified and archived outside the repo:
`~/.local/share/cardiosentry/legacy-json/legacy-json.tar.gz`, per-file
`legacy-json.sha256`, tarball `legacy-json.tar.gz.sha256`, aggregate `VERIFY.txt`.
The legacy archive is PHI (directories 0700, files 0600), never app-served.
Restore: extract, then `scripts/import_filesystem.py --data-dir <dir>`.
The app never reads legacy JSON. Read/write records through `RecordStore`;
`atomic_write_json` is removed. `api/auth.py`'s `owned_capture` returns the
live owner-scoped record and its image folder.

Ids and servable names are regex-gated in `storage.py`
(`CAPTURE_ID_RE`, `FILE_WHITELIST`, `EXPLAIN_ID_RE`, `EXPLAIN_FILE_RE`). A new
stage file must be added to the whitelist or it 404s.

Protect both the images and volume `postgres-db-cardiosentry-data`.
`docker exec postgres-db-cardiosentry pg_dump -U cardiosentry cardiosentry`
backs up records, not images or the legacy archive; image backups also exclude
the archive. Losing the volume loses accounts and records; the legacy archive
restores only pre-migration data, not new captures or later edits.
The legacy archive is PHI; erasure must also cover it. README §Security & privacy
gives restore commands, permanent-erasure SQL, archive removal and retained-copy
handling. The archive is not encrypted at rest; `age`/`gpg` is a later option only.

## Task → files (change these together)

| Task | Touch |
|---|---|
| **Swap the active model** | `models/active.yaml` symlink → done. Then the coupled docs/tests below. |
| **Add a model** | see [Adding a model](#adding-a-model) — it is a 7-step checklist, follow it |
| Change a threshold/label/geometry for one model | that `models/<id>.yaml` only — **never** `settings.py` |
| Change the ECG gate | switch: `CARDIOSENTRY_ECG_GATE=<id>` only. Tune the OCR rule: `models/gates/ocr_lead_labels_v1.yaml` (`min_leads`, `counted_leads`, `cleanup`). Code: `services/{ecg_gate,ecg_gate_ocr}.py` + `schemas/ecg_gate.py` + `api/captures.py` `create_capture` + `models/README.md` §ECG gate (measurement table) + `tests/test_ecg_gate.py` + `tests/test_api_smoke.py` (`no_paper.jpg` must be rejected, `photo_clean.jpg` accepted); turning it off falsifies the privacy notice |
| Change the quality gate | thresholds: `settings.py` `warn_*` + `.env.example` (warn) / `services/quality.py` `_POOR` (poor); on/off: `CARDIOSENTRY_QUALITY_BLOCK_ON_POOR`. Code: `services/quality.py` (`poor_reasons` text) + `api/captures.py` (`_blocks`, `_poor_quality_error`) + `frontend/src/components/QualityWarnings.tsx` (one line per warn flag) + `tests/test_api_smoke.py` (dark `photo_clean` rejected at upload, blurred one at rectify). Measured: clean fixtures carry warn-level `glare` (white paper ≥ 250) — never move glare into "poor" without real photos; turning blocking off falsifies the privacy notice |
| Add a descriptor field | `schemas/model_descriptor.py` + every `models/*.yaml` that needs it + `models/README.md` "descriptor format" |
| Temporarily suppress model heads | descriptor `output.suppressed_labels`; never remove/reorder checkpoint `labels`. API/config/explain use `served_labels` |
| Change the input chain | `services/preprocess.py` **and** `tests/test_preprocess_parity.py` (it proves parity vs training code to 1e-6) |
| Add/alter an API route | `api/<mod>.py` + `schemas/` + `frontend/src/api/{client,types}.ts` + `tests/test_api_smoke.py` |
| Auth/ownership | `services/{accounts,records}.py` + `api/auth.py` + `main.py` `auth_gate` + `schemas/{auth,capture}.py` + `owned_capture` callers (`api/captures.py`, `api/explain.py`, `api/guidance.py`) + `frontend/src/{App.tsx,api/{client,types}.ts,screens/{AuthScreen,AccountScreen}.tsx,components/PasswordInput.tsx}` + `frontend/src/components/PrivacyNotice.tsx` / `api/config.py` notice version + `tests/{conftest,test_accounts,test_api_smoke}.py` |
| DB schema | `services/db.py` `MIGRATIONS` (append a version, never edit v1) + `services/{records,accounts}.py` + `scripts/import_filesystem.py` + `tests/test_database.py` |
| Add a stage output file | `services/storage.py` FILE_WHITELIST + `api/captures.py` + frontend |
| Change explain behaviour | `services/explain.py` + `api/explain.py` + `schemas/explain.py` + `tests/test_explain.py` |
| Give a new model urgency guidance | `services/guidance.py` `RULE_SETS` entry (exact served labels + suppression) + a `FINDING_RULES` entry per new label + `docs/clinical-guidance.md` tables + `tests/test_guidance.py`; frontend gates on `config.guidance.model_ids`, no UI change |
| Change urgency / next-step behaviour | `services/guidance.py` (`RULE_SETS`, `FINDING_RULES`, `COMBINATIONS`; 911, pediatric floor) + `api/guidance.py` if route handling changes + `schemas/guidance.py` (`cautions`, `combination_notes`) + `frontend/src/api/types.ts` + `frontend/src/components/GuidancePanel.tsx` + `frontend/src/styles.css` (visible/printable cautions and clinician notes) + `docs/clinical-guidance.md` + `tests/test_guidance.py` |
| Add a setting | `settings.py` + `.env.example` (keep them in sync — the example is the docs) |
| Change the Data Privacy Notice | `frontend/src/components/PrivacyNotice.tsx` + bump `PRIVACY_NOTICE_VERSION` in `api/config.py` (re-prompts every browser) — and if the backend's data handling changes, the notice text must change with it |
| Vendor a new arch package | `backend/arch/<pkg>/` + `backend/arch/README.md` provenance table + deviations list |
| Frontend routes / reload persistence | `frontend/src/App.tsx`: hash routes `#/scan/<capture>` (corner step) and `#/scan/<capture>/result/<run>`; on reload they rebuild state from `GET /api/captures/{id}`. Classifying is not a route — a reload there returns to the corner step instead of predicting twice. `ExplainPanel` reuses an existing Grad-CAM for the run instead of regenerating it |
| Printable result report | `ResultScreen.tsx` Print button (`window.print()`), print-only header (capture/analysis times, model, sheet image `corrected.png` or `original.jpg`) and disclaimer footer; `<details data-print-open>` are expanded on `beforeprint`; all print layout is the `@media print` block at the end of `styles.css`. Verify with a fresh headless-Chrome profile — the PWA service worker serves stale CSS otherwise |
| Frontend change | `frontend/src/`; rebuild `npm run build` — the backend serves `frontend/dist/` |
| Change the system flowchart | `docs/system-flowchart/`; verify implemented vs proposed stages against the active descriptor, routes, services, frontend, and any clinician-governed safety rules |

## Adding a model

`models/README.md` has the authoritative 7-step version. The parts that are
easy to get wrong:

1. Which **arch package** the run used decides everything, and the run's
   directory name does not say: `efficientnetv2_s@nat768` and
   `efficientnetv2_s_cbam@nat768` are different packages. The banner at the top
   of `prototype-results/<run>/report.txt` names it (`CardioSentry baselines v2
   | <run>`), and `torch.load(best.pt)["model"]` settles it: bare timm keys
   (`conv_stem…classifier`) = `baselines_v2`, `backbone.*`/`attn.*` =
   `hybrids_v2`, `backbone.*` + `neck.*`/`fuse_fc.*` = a clean12 fusion run
   (`convnext_v1_rfpn_clean12` for `CS_FUSION=rfpn`). Wrong package =
   `strict=True` failure, not a subtle bug. Not vendored yet? Vendor it first
   (`backend/arch/README.md`). A clean12 `inference.pt` (no `optimizer`/
   `model_ema`) already holds the selected weights under `model`.
2. `expected_protocol_fingerprint` — checkpoint generations use
   `protocol_fingerprint`, `fingerprint`, or `config_fingerprint`; the runner
   falls back through those spellings. Copy the matching value from
   `config_snapshot.json`.
3. `explain.target_layer` — probe, don't guess: `GET /api/models/<id>/layers`,
   or `xai.auto_target_layer(lm)`. `attn` on a CBAM hybrid, `bn2` on a plain
   EfficientNet, `layer4`/`backbone.layer4` on a ResNet.
4. `verify.expected_scores` — preferably the run's own row for `ECG000002` from
   `prototype-results/<run>/predictions/test_predictions.csv`, written to
   `tests/fixtures/expected_scores_<id>.json` **and** added to `EXPECTED_SCORES`
   in `tests/make_fixtures.py` (which regenerates them). If the run never
   evaluated that exact fixture, pin a measured serving-path vector and identify
   that provenance honestly in the fixture's `source`.
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
8. **A model score is not clinical confidence.** The independent sigmoid heads
   are uncalibrated on real photographs. The UI may format scores as percentages
   for readability, but must label them **Model score**, never confidence,
   probability, likelihood, or diagnostic certainty.
9. **Frontend color is semantic.** The root tokens in `frontend/src/styles.css`
   define a white/cool-neutral clinical surface with teal for interaction.
   Green, orange, and red are reserved for routine, prompt, and emergency
   urgency; model “Flagged” / “Not flagged” states stay neutral so color never
   implies diagnostic severity. Keep the screening/decision-support disclaimer
   to one visible notice; the urgency card shows the level and rule reason
   without repeating a generic triage headline or context-status sentence. The
   refiner's own “Recommended” / “Applied” state communicates context status.
   Its age-group selector uses a narrow 220px cap while remaining full-width
   on narrower screens; its text is compact, but the control retains a
   touch-friendly height.
   The post-prediction Result screen relies on the persistent top-bar New and
   History controls; these use a restrained teal-neutral highlight for
   discoverability. Do not duplicate navigation in a footer there, and do not
   show the internal model name or capture-image strip on the patient-facing
   Result screen. Stored images remain available from capture details.
10. **The patient capture flow always uses `training_canvas`.** There is no
    geometry selector in the corner editor, and the frontend rectify client
    sends that value explicitly. `native_aspect` remains an internal research
    mode in the backend; do not expose it in the patient UI accidentally.
11. **The preview step is retained but hidden.** `App.tsx` sets
    `SHOW_PREVIEW_STEP = false`, so `PreviewScreen` automatically runs the
    active model after rectification without rendering its image/model review
    controls. Its patient-facing progress copy describes ECG classification,
    not model execution. Flip the flag to restore that UI; do not delete the
    screen.
12. **Grad-CAM starts after the Result page renders.** `ResultScreen` opts its
    `ExplainPanel` into automatic generation; the explanation remains a
    separate asynchronous request, so classification results display without
    waiting for the heatmap. A per-result ref prevents duplicate automatic
    runs under React Strict Mode. The patient Result screen shows only compact
    generation progress and the completed maps—not regenerate controls or
    numbered run chips. Capture details retain those manual/history controls.
13. **The Data Privacy Notice (RA 10173, Philippines) states facts about the
    backend.** `PrivacyNotice.tsx` promises server-side storage with no
    third-party transfer, trash-not-unlink, guidance answers not saved, and the
    stored fields. Change any of those and update the notice + bump
    `PRIVACY_NOTICE_VERSION` (currently `2026-10-28.1`; it also promises that uploads the ECG gate
    or the quality gate rejects are not stored, and that a photo rejected as blurry at rectify stays stored but is not analysed).
    `CaptureScreen` hides the photo buttons until the current version is
    accepted; each upload sends `privacy_notice` → `session.privacy_notice_ack`
    (None = API/script caller). Controller/contact come from
    `CARDIOSENTRY_PRIVACY_CONTROLLER` / `_CONTACT`; unset shows "Not yet configured".
14. **`captures.owner_id = owner_of(record)` uses permanent account ids.**
    Missing/null `record.owner` means admin; imported JSON is never backfilled.
    Compare ownership with `current_account(request).account_id`, never
    `.username`. Account ids stay reserved after rename; admin is not a
    superuser. `owned_capture` requires a live owner-scoped row and image folder.
15. **The session cookie is deliberately not Secure over plain HTTP.**
    `cardiosentry_session` is HttpOnly, SameSite=Lax, lasts 30 days, and is Secure
    only over HTTPS; setting Secure unconditionally breaks LAN/TestClient login.
16. **The shared test `client` is logged in as admin — never log it out.**
    Use function-scoped `anon_client` for auth/logout tests. It shares the
    running app with no admin cookie and no second lifespan/model load.
17. **JSONB cannot store `\u0000`.** Phone EXIF strings can carry NUL padding;
    `RecordStore` strips NULs recursively from string keys and values and uses
    `allow_nan=False`. Never write records around that boundary.
18. **Tests only store app data in `<db>_test` / `<db>_test_unit`.**
    `conftest` rewrites settings at import; the session/per-test fixtures reset
    guarded names through the `postgres` maintenance database. Never bypass
    those guards or open the real app database.
19. **Async `auth_gate` must not block on psycopg.** Its synchronous account
    lookup uses `await run_in_threadpool(...)`; keep database I/O off that
    middleware's event loop.
20. **Never edit an applied migration.** Append a new version to `db.MIGRATIONS`,
    not changes to v1; `schema_migrations` makes existing databases skip it.

## Running things

No virtualenv and **no `python` binary** — use system `python3` (torch, timm,
fastapi, cv2 all present). Run from the app root.

The local PostgreSQL 16 container is `postgres-db-cardiosentry`.
`settings.database_url` is a `SecretStr`; `.env` accepts `DATABASE_URL` or
`CARDIOSENTRY_DATABASE_URL` (the prefixed name wins). The target is
`127.0.0.1:5432/cardiosentry (user cardiosentry)`; remote hosts are rejected and
the URL must stay secret. Runtime dependency: `psycopg[binary,pool]>=3.2`
(installed psycopg 3.3.6 / psycopg_pool 3.3.3). The default ECG gate shells out
to the `tesseract` binary (5.3.4 here; `apt install tesseract-ocr`); startup
fails without it.

```bash
python3 -m pip install --user --break-system-packages 'psycopg[binary,pool]'
docker start postgres-db-cardiosentry            # if stopped
python3 scripts/import_filesystem.py --data-dir <dir>  # legacy import/restore into an empty DB
python3 -m pytest -q                     # 251 cases (121 functions), real CPU model, minutes not seconds
python3 -m pytest -q tests/test_api_smoke.py
python3 scripts/verify_model.py                  # active model vs its own stored vector
python3 scripts/verify_model.py --model <id>     # any descriptor; picks up its verify block
python3 scripts/reprocess.py --all --model <id>  # re-score stored captures, no re-photographing
python3 -m backend.main                          # serve (prints LAN URL + QR)
cd frontend && npm run build                     # backend serves frontend/dist/
```

The legacy import was verified on 2026-10-08. Restore source: the extracted
archive passed as `--data-dir`, before an empty database's first app boot;
README §Security & privacy, **Legacy JSON archive**, has the commands.
The importer is manual, read-only on extracted files and idempotent: database
rows always win. If an extracted legacy tree contains `accounts.json`, import
before startup seeds admin to preserve its credentials; this archive contains
no account file. No legacy data means no import is needed.

Collected cases from `python3 -m pytest -q --collect-only`: accounts 2,
api_smoke 16, database 2, detect 9, ecg_gate 8, explain 19, guidance 183,
model_descriptor 4, preprocess_parity 8 (**251 total**). Unique top-level
`def test_` functions in the same files: 2, 16, 2, 9, 5, 19, 57, 4, 7 respectively (**121 total**);
parameterized cases are not separate functions.

`database_url` recreates `<db>_test` (`cardiosentry_test`) once per session;
`isolated_db` creates/migrates `<db>_test_unit` (`cardiosentry_test_unit`) per
test that uses it and drops it afterwards. The role needs permission to create/
drop test databases via `postgres`. The real app database is never opened.
`client` depends on `database_url`; use `client.app.state.csd.{db,records,accounts}`.
Test images live in `$TMPDIR/cardiosentry_test_captures`, never the real captures.

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
counts (251 cases, 121 functions, LOC, route list) honest — correct them rather than leaving a
number that is now wrong.
