# Synthetic paper-ECG from one CUHK record — 0.05–150 Hz

A single-record smoke test of `image_pipeline/` against the **v3** harmonized release,
run at a **0.05–150 Hz** band-pass instead of the corpus default 0.05–40 Hz.
Nothing in v3 or in the released image corpora was written to; this directory is
self-contained.

## The record

| | |
|---|---|
| `global_record_id` | **CUHK_002942** |
| `source_dataset` | CUHK (Philips Sierra XML, XLI-compressed) |
| `original_record_id` | `A965XXXX-2010-4-15_18-6-52` |
| `document_guid` | `0242eb29-618c-458a-a9c6-6028ba1947e5.xml` |
| `split_group_id` | `CUHK::CUHK_PA965` |
| cohort / site | HF / A, acquired 2010-04-15 18:06:52 |
| age / sex | 59 / Male |
| `validation_status` | VALID |

## The label

```
cardiosentry_label = STEMI+LVH        STEMI=1  AF=0  LVH=1  NORMAL=0
```

One of the 405 STEMI+LVH records in v3 — a combination that **only CUHK supplies**;
v2 could not produce it at all.

Where the two positives come from (`accepted_label_mapping`: `LVH31->LVH; AMI23->STEMI`):

| Philips code | statement text | → |
|---|---|---|
| `AMI23` | injury (ACUTE INFARCT) with reciprocal ST depression, ST > .40 mV V1–V3 | **STEMI** |
| `LVH31` | LVH with ST-T abnormalities — LVH voltage, ST-T negative | **LVH** |
| `JT` | Junctional tachycardia, absent P waves, rate ≥ 100 | not a target class |
| `LRR1` | Long R-R interval, > 140 % of normal | not a target class |
| `AFBVCD` | Left anterior fascicular block + IVCD, QRS axis −45°, QRS > 120 ms | not a target class |

> `JT` (absent P waves, rate ≥ 100) is why this record is **AF-negative** despite having no
> visible P waves — the DXL statement calls it junctional, not fibrillation. That distinction
> is the algorithm's, not a clinician's: **every CUHK label is `ALGORITHM_PHILIPS_DXL`.**

`validation_issues` carries `limb leads recomputed: file's own reconstruction differed by
678 µV` — expected for this source; III/aVR/aVL/aVF are always recomputed from I and II
because the 10-WIRE acquisition makes them arithmetic.

## The image

`build/images/test/ECG000001_a0.jpg` — 1686 × 1311, 506 KB, red grid palette, −1.27° rotation.

Standard 3×4 layout with a continuous lead II rhythm strip, 25 mm/s, 10 mm/mV, 150 DPI.
The sheet prints `ID: ECG000001`, not the record id: staged records are renamed so the
model cannot read the source off the page.

`build/images/test/ECG000001_a0.json` holds the per-lead bounding boxes;
`build/rendered/.../ECG000001-0.png` is the clean pre-augmentation sheet.

## The filter

`ECG_LOWPASS_HZ=150.0` gives `config.BANDPASS = (0.05, 150.0)`; stage 2 applies it as a
2nd-order Butterworth run zero-phase (`sosfiltfilt`), so the ST segment is not shifted in
time. Verified on the staged output:

| band (Hz) | source | staged |
|---|---:|---:|
| 0 – 0.05 | 0.03284 | 0.03377 |
| 0.05 – 40 | 0.96595 | 0.96524 |
| 40 – 100 | 0.00104 | 0.00094 |
| 100 – 150 | 0.00008 | 0.00005 |
| 150 – 200 | 0.00004 | **0.00001** |
| 200 – 250 | 0.00005 | **0.00000** |

Fraction of total Welch power per band. Content above 150 Hz is gone; the 40–150 Hz band a
default build would have removed is retained. Re-filtering the source independently agrees
with the staged signal to **0.5 µV** — one int16 LSB at 1000 ADU/mV, i.e. quantisation only.

Note this is a *signal-domain* claim. At 150 DPI and 25 mm/s the rendered sheet's own
Nyquist is ~73.8 Hz, so the extra band survives in `staged/` but is not resolvable on the
page. That is the point of the run: it isolates the filter change from the render.

## Layout

```
input/record_subset.csv          the one v3 row, absolute signal paths, split pinned to "test"
run.sh                           reproduces everything below
build/image_manifest.csv         stage 1
build/staged/…                   stage 2 — 0.05–150 Hz WFDB, renamed, Age/Sex comments
build/rendered/…                 stage 3 — clean PNG from ecg-image-kit
build/images/test/…              stage 4 — final augmented JPEG + annotation sidecar
build/index.csv                  stage 4 — corpus manifest
```

## Reproducing

```bash
./run.sh
```

Stage 5 reports `ALL CHECKS PASSED` (the source-leakage probe is skipped — one source).

## Two things this run surfaced

1. **`image_pipeline/` needs an interpreter with `wfdb`.** The system `python3` has none;
   only `~/.cache/ecgkit-venv/bin/python` does. Stage 1 runs anywhere (no wfdb import),
   stages 2–5 do not, which makes the failure look like a stage-2 bug.

2. **CUHK is nearly ageless at the source — the cleaning did not cause it.** Traced back to
   the raw Philips XMLs: of 67,432 files, **43,145 (64.0 %) carry `<age defaultage="50">`**,
   which is the cart flagging that no age was entered — every one of them reads exactly
   50.0, a single distinct value. A further 23,436 (34.8 %) carry no age at all. Only
   **851 (1.3 %)** carry a real age (median 67, range 12–96).

   `scan_xml_index.py` reads the `defaultage` *attribute*, not just the value, so
   `build_manifest.py:182` can drop the placeholder rather than propagate 43,145 fake
   fifty-year-olds:

   ```python
   "age": "" if r["age_default_used"] == "1" else r["age_years"],
   ```

   That is the only place age is touched in the CUHK chain. Of the 851, **158 survive
   label-eligibility and GUID dedup, and all 158 reach v3 with their age intact** — nothing
   is lost after dedup. 154 of those also have a decodable sex.

   Consequence for this pipeline: the kit's `--print_header` hard-requires `Age`, and stage 1
   drops any record missing it, so a full v3 image corpus would keep **<1 %** of CUHK. The fix
   is to make age optional on the sheet — there is no upstream age to recover.
   Of the 154 usable: STEMI 63, LVH 51, AF 26, STEMI+LVH 8, STEMI+AF 5, AF+LVH 1.

   `config.SEX_DECODERS` also has no `CUHK` entry. It happens to work only because CUHK
   writes the literal words `Male`/`Female`, which `clean_sex` falls through to; a source
   using integer codes would silently decode wrong rather than fail.
