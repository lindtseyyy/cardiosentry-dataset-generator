# CUHK vs. current CardioSentry dataset — compatibility audit

- **Current dataset**: `/home/lindtseyxvii/Codes/cardiosentry/dataset-combination-code/ecg_dataset_pipeline/data/harmonized/harmonized_ecg_metadata.csv` — 19,375 records
- **CUHK release**: `/home/lindtseyxvii/Codes/cardiosentry/dataset-files/cuhk_dataset_clean/metadata.csv` — 17,969 records
- **Combined**: 37,344 records

Every claim below about signals was measured by opening all 37,344 records (0 failed to open).

**Verdict: drop-in compatible on every signal dimension.** The differences that matter are in the labels and in patient structure, not the waveforms.

---

## 1–6. Signal representation

| | PTBXL | CHAPMAN | STEMI | CUHK |
|---|---|---|---|---|
| **Sampling rate (Hz)** | 500 | 500 | 500 | 500 |
| **Number of leads** | 12 | 12 | 12 | 12 |
| **Samples per record** | 5000 | 5000 | 5000 | 5000 |
| **Duration (s)** | 10.0 | 10.0 | 10.0 | 10.0 |
| **ADC gain (ADU/mV)** | 1000 | 1000 | 1000 | 1000 |
| **Units** | mV | mV | mV | mV |
| **WFDB format** | 16 | 16 | 16 | 16 |
| **Lead names** | I II III AVR AVL AVF V1 V2 V3 V4 V | I II III aVR aVL aVF V1 V2 V3 V4 V | I II III aVR aVL aVF V1 V2 V3 V4 V | I II III aVR aVL aVF V1 V2 V3 V4 V |
| **Records with a NaN sample** | 0 | 0 | 0 | 0 |
| **Records with ≥1 flat lead** | 0 (0.0%) | 55 (0.7%) | 5 (0.3%) | 17 (0.1%) |
| **Median lead amplitude (mV p-p)** | 1.30 | 1.31 | 0.99 | 1.61 |
| **p99 max amplitude (mV p-p)** | 8.1 | 13.8 | 53.5 | 25.2 |

**Sampling rate, lead count and duration are identical across all four sources** — 500 Hz, 12 leads, 5000 samples, 10.000 s. No resampling, padding or cropping is needed to concatenate.

**Amplitude/gain is identical too**: every source declares 1000 ADU/mV in mV units, so `p_signal` is directly comparable and a model normalised on the current data transfers without rescaling. CUHK reaches that gain by an exact ×5 integer rescale from its native 5 µV/LSB — a unit change, not a resample; true quantisation stays 5 µV and is recorded per record as `native_resolution_uv`.

Two representation notes specific to CUHK, both already applied and declared in each `.hea`:

1. Source records are 11 s / 5500 samples; the **first 10 s are kept**. The trailing second is where the recorder stops maintaining the derived-lead relationships.
2. **III, aVR, aVL and aVF are recomputed from I and II.** These are 10-wire acquisitions, so those four leads are arithmetic by construction; the XLI-stored residual is corrupt in 26% of files (strictly bimodal — a decoder artefact, not physiology).

Lead *naming* is the one cosmetic difference: PTB-XL writes `AVR/AVL/AVF`, everything else writes `aVR/aVL/aVF`. The pipeline's `normalize_lead()` already folds these, and lead **order** is identical everywhere.

### Missing / dead leads

No record in any source is missing a lead *structurally* — all are 12×5000. A 'missing' lead shows up as a flat trace instead:

| source | records with ≥1 flat lead (<0.05 mV p-p) | worst case |
|---|---|---|
| PTBXL | 0 (0.0%) | 0 leads flat |
| CHAPMAN | 55 (0.7%) | 5 leads flat |
| STEMI | 5 (0.3%) | 1 leads flat |
| CUHK | 17 (0.1%) | 1 leads flat |

CUHK flags these per record: `validation_status = CHECK` covers flat leads and recorder saturation (~41 mV, the ADC rail). Filter on it if your model is sensitive to dead channels.

---

## 7. Patient overlap

| source | records | patient groups | records/group (median, max) | id provenance |
|---|---|---|---|---|
| PTBXL | 10,142 | 9,284 | 1, 9 | real patient id from the distribution |
| CHAPMAN | 7,791 | 7,791 | 1, 1 | **synthetic — one id per record** |
| STEMI | 1,442 | 1,437 | 1, 2 | real patient id from the source CSV |
| CUHK | 17,969 | 2,725 | 4, 65 | 4-char prefix of the anonymised id |

**Cross-dataset patient-id collisions: 0.** **split_group_id collisions: 0.** Every source namespaces its ids (`PTBXL_P…`, `CHAPMAN_SYNTH_…`, `STEMI::P…`, `CUHK_P…`), so a collision is structurally impossible and this is confirmed empirically.

Person-level overlap between CUHK (Hong Kong, 2005–2019) and PTB-XL (Germany) / Chapman (mainland China) / the ACS source is not plausible, and there is no identifier that could establish it either way — the audit therefore tests for it at the signal level instead (§8).

> ⚠️ **The real risk is inside CUHK, not between datasets.** The current data is effectively one ECG per patient (median 1); CUHK is **serial** — median 6–7 ECGs per patient group, up to 65. Splitting the combined set at random would put the same patient in train and test. See §13.

> Note a pre-existing caveat this audit surfaces: **Chapman's patient ids are synthetic**, one per record. If that distribution contains repeat patients, they are already leaking across the current splits. That is not a CUHK problem, but merging does not fix it either.

---

## 8. Duplicate ECGs

Fingerprint = SHA-1 of the full 12×5000 signal quantised to 1 µV, so a genuinely identical recording collides even if stored with a different format or baseline.

- **Exact duplicate pairs across CUHK ↔ current dataset: 0**
- Duplicate groups within a single source: CHAPMAN=15
- Distinct fingerprints: 37,329 over 37,344 records

Within CUHK, duplication was already removed *before* this release: CUHK ships the same acquisition in several cohort folders when a patient belongs to more than one registry. Deduplicating on the Philips document GUID removed **5,713 duplicate copies** (67,431 files → 61,718 unique acquisitions, 8.5%). Counting raw files would have inflated every number in this report.

---

## 9. Label definitions and provenance

All four sources are resolved by the **same** `common.LabelResolver` against YAML in the same schema, so `ACCEPT` / `EXCLUDE_UNCERTAIN` / `EXCLUDE_RECORD` mean the same thing everywhere. What differs is the *evidence* behind a label.

| source | label evidence | provenance | contributes |
|---|---|---|---|
| PTB-XL | SCP-ECG statements, human overread | **human** | STEMI*, AF, LVH, NORMAL |
| Chapman | SNOMED-CT Dx codes | **human** | AF, LVH, NORMAL |
| 2026 ACS/STEMI | dedicated clinician-assigned `STEMI` column | **human** | STEMI |
| CUHK | Philips DXL algorithm interpretation statements | **algorithm** | STEMI, AF, LVH |

> ⚠️ **This is the single most important incompatibility.** CUHK labels are machine-generated: no clinician signed them. Every CUHK row carries `label_provenance: ALGORITHM_PHILIPS_DXL`, and the full Philips statement codes and text are preserved per record, so the cohort can be audited, re-mapped or down-weighted. `provenance/code_decisions.csv` lists all 709 statement codes with counts, statement text and assigned status — that is the table to hand a cardiologist.

Definitions were deliberately kept identical across sources:

- **AF means fibrillation only.** Atrial *flutter* is `EXCLUDE_UNCERTAIN` in PTB-XL, Chapman and CUHK alike.
- **LVH requires a hypertrophy diagnosis.** Voltage criteria alone (`VCLVH` in PTB-XL, `LVH26`/`LVH10` in CUHK) are excluded in both.
- **STEMI requires an explicit ST-elevation assertion.** Location-only MI codes, old/indeterminate infarct, and bare ST elevation never become STEMI in any source.

The one asymmetry: CUHK applies `EXCLUDE_RECORD` far more aggressively than the current sources do — hedged acute ischaemia, bare ST elevation, LVH/LBBB-attributed elevation and paced rhythms drop the whole record rather than assert `STEMI=0`. That is because CUHK is an ACS-registry population where a wrong negative is likely; PTB-XL's general population does not have that problem. The effect is that CUHK's negatives are *cleaner* than the current data's, not dirtier.

**NORMAL: CUHK contributes none, by design.** Every CUHK cohort is a disease registry, so a STEMI-negative record there is a different acute presentation, never a healthy control. All NORMAL supervision keeps coming from PTB-XL and Chapman.

---

## 10–12. Label counts and co-morbidity

### Per class

| class | current | CUHK | combined | change |
|---|---|---|---|---|
| **STEMI** | 1,442 | 2,612 | **4,054** | ×2.81 |
| **AF** | 2,741 | 8,151 | **10,892** | ×3.97 |
| **LVH** | 2,542 | 9,991 | **12,533** | ×4.93 |
| **NORMAL** | 12,906 | 0 | **12,906** | ×1.00 |
| *total records* | 19,375 | 17,969 | **37,344** | ×1.93 |

### Exact label combinations

Each record appears on exactly one row: this is the *precise* set of classes it carries.

| combination | current | CUHK | combined |
|---|---|---|---|
| NORMAL | 12,906 | 0 | 12,906 |
| LVH | 2,286 | 7,423 | 9,709 |
| AF | 2,485 | 5,814 | 8,299 |
| STEMI | 1,442 | 1,990 | 3,432 |
| AF+LVH | 256 | 2,120 | 2,376 |
| STEMI+LVH | 0 | 405 | 405 |
| STEMI+AF | 0 | 174 | 174 |
| STEMI+AF+LVH | 0 | 43 | 43 |
| **multi-label (≥2)** | 256 (1.3%) | 2,742 (15.3%) | 2,998 (8.0%) |

### Co-occurrence (inclusive)

These count **any** record carrying both classes, so a triple positive is counted in all three pair rows. That is why `STEMI + LVH` here (448) exceeds the `STEMI+LVH` row above (405): the difference is the 43 triples.

| | current | CUHK | combined |
|---|---|---|---|
| **STEMI + LVH** | 0 | 448 | 448 |
| **AF + LVH** | 256 | 2163 | 2419 |
| **STEMI + AF** | 0 | 217 | 217 |
| **STEMI + AF + LVH** (triple) | 0 | 43 | 43 |

> **This is the strongest argument for merging.** The current dataset has **zero** STEMI co-morbidity — its only multi-label combination is AF+LVH. That is a structural artefact: the ACS/STEMI source is declared `contributes_classes: [STEMI]`, so it can never produce an AF or LVH positive, and PTB-XL/Chapman contribute no STEMI. A model trained on it has **never seen a STEMI that co-occurs with AF or LVH** — yet that is exactly the patient who is hardest to read and most dangerous to miss (LVH strain and AF both obscure ST interpretation).

CUHK supplies that supervision for the first time. Treat these counts as the headline benefit of the merge, ahead of the raw record count.

---

## 13. Recommended train / validation / test strategy

The two datasets differ in three ways that all point the same direction: CUHK is **serial** (many ECGs per patient), **algorithmically labelled**, and **single-centre**. Merging it into the training pool is worthwhile; merging it into the evaluation pool is not.

### Recommended split

| pool | composition | why |
|---|---|---|
| **Train** | PTB-XL + Chapman + ACS/STEMI + **CUHK** | CUHK's volume and its unique STEMI+AF / STEMI+LVH cases are most valuable as training signal |
| **Validation** | held-out patient groups from **all** sources, including CUHK | model selection should see the same distribution as training |
| **Test (primary)** | **human-labelled sources only** — PTB-XL, Chapman, ACS/STEMI | a metric computed against algorithm labels measures agreement with Philips DXL, not with clinical truth |
| **Test (secondary)** | a CUHK-only held-out split, reported separately | shows transfer to a different institution, hardware and population — never averaged into the headline number |
| **External** | LUDB (137 records, cardiologist-labelled, Russia) | fully independent institution; keep untouched by training |

### Non-negotiables

1. **Split on `split_group_id`, never on records.** Use `GroupShuffleSplit` / `StratifiedGroupKFold`. CUHK averages 6.6 ECGs per patient group (median 4, max 65), so random splitting leaks heavily. The column is already populated for every source.
2. **Never let one patient group span two pools**, even across datasets — assign the group, then the records follow.
3. **Cap records per patient group** in training (8–10 is a reasonable start). A handful of CUHK patients contribute dozens of near-identical serial ECGs and would otherwise dominate the loss. Cap by sampling, not by dropping patients.
4. **Stratify on the label combination**, not on individual classes — STEMI+AF+LVH is rare enough that naive stratification can empty it from a fold.

### Strongly recommended

5. **Carry `label_provenance` as a feature of the run, not of the model.** Report metrics split by provenance. If a CUHK-trained model scores well on CUHK test and poorly on PTB-XL test, it has learned Philips DXL's decision rule, not the disease.
6. **Consider down-weighting CUHK's loss** (e.g. 0.5–0.7) or pre-training on CUHK and fine-tuning on the human-labelled pool. The second is usually the cleaner design.
7. **Watch the NORMAL balance.** NORMAL is currently 66.6% of the data; after merging it falls to 34.6%. That is healthier for the pathological heads, but it changes every operating threshold you have already tuned — re-tune, do not carry them over.
8. **Keep the `CHECK` records out of the test pool.** Recorder saturation and dead leads are real-world noise worth training on, but they make an evaluation number harder to interpret.

### Before any of this

9. **Get the label mapping reviewed by a cardiologist.** `provenance/code_decisions.csv` — all 709 codes, with counts and the Philips statement text. The ~2,600 STEMI positives rest on the 51 codes in the `MI_ACUTE` bucket.
10. **Decide whether `EXCLUDE_RECORD` is too aggressive.** It drops 14,685 CUHK records, many of them hedged-acute or bare-ST-elevation cases a clinician could adjudicate into real positives. That is the largest pool of recoverable data in the collection.

