# CUHK ECG collection — investigation findings

Written so this does not have to be re-derived. Everything below was measured
from the data in `dataset-files/`, not assumed. Where a number came from a
sample rather than a full pass, it says so.

- **Scope**: `dataset-files/cuhk_dataset/` (STEMI folders + `others/`), plus the
  Lobachevsky University Database (LUDB) that was sitting unintegrated in
  `dataset-files/`.
- **Outcome**: usable. 17,969 labelled 12-lead records released from CUHK and
  137 staged from LUDB, in the project's target representation.
- **Released to**: `dataset-files/cuhk_dataset_clean/` (standalone, 2.2 GB).
  Working area and scripts in `dataset-staging/cuhk/`. The main dataset is
  untouched.
- **See also**: `COMPATIBILITY_AUDIT.md` — CUHK vs. the current dataset.

---

## 1. The headline finding: the waveforms are XLI-compressed

Every CUHK file is **Philips Sierra ECG XML** (`<restingecgdata>`,
`documenttype = PhilipsECG`, version 1.04 / 1.04.01), UTF-16 encoded.

The waveform block declares:

```xml
<parsedwaveforms dataencoding="Base64" compression="XLI"
                 numberofleads="12" samplespersecond="500"
                 durationperchannel="11000" resolution="5" bitspersample="16">
```

`compression="XLI"` is the reason a plain XML read gets you nothing usable —
base64-decoding that block yields compressed bytes, not samples. XLI is
Philips' proprietary scheme (LZW over byte-plane-separated second-order
differences, with the derived limb leads stored as residuals).

**It is solved.** The reverse engineering lives in
[`sixlettervariables/sierra-ecg-tools`](https://github.com/sixlettervariables/sierra-ecg-tools),
and there is a maintained Python port on PyPI:

```bash
pip install sierraecg
```

```python
import sierraecg, numpy as np
ecg = sierraecg.read_file(path)                       # handles UTF-16 + XLI
x = np.array([l.samples for l in ecg.leads])          # (12, 5500), 5 µV/LSB
labels = [l.label for l in ecg.leads]                 # I II III aVR aVL aVF V1..V6
```

**All 67,431 files parsed; 1 failure** (a zero-byte file,
`HF/E128XXXX-...2008-10-14_7-12-32.xml`).

### The collection is completely uniform

Across all 67,431 files, without exception: 500 Hz · 11.000 s · 12 leads ·
5 µV/LSB · 16-bit · XLI · `acquisitiontype = 10-WIRE` · `electrodeplacement =
STD`. Only two things vary, and both are cosmetic:

- chest-lead naming: `V1…V6` (most) vs. European `C1…C6`
- document version: 1.04 (66,450) vs. 1.04.01 (981)

Acquisition dates span **1980-01-05 → 2019-08-13** (the 1980 date is a
mis-set recorder clock; the bulk is 2005–2019).

---

## 2. How the decode was verified

XLI has no checksum, so "it produced numbers" is not evidence. Three
independent checks, all against values the Philips machine itself wrote into
the *same* file:

| check | result |
|---|---|
| decoded heart rate (R-peak detection on lead II) vs. the file's `<heartrate>` | **median difference 1.0 bpm** (n=135 sample) |
| decoded per-lead amplitudes vs. the file's `<leadmeasurement>` R/S values | consistent |
| Einthoven/Goldberger relations (III = II − I, aVR = −(I+II)/2, …) | holds to ≤ 2 LSB (10 µV) in 74% of files — see §3.1 |

---

## 3. Four gotchas — each one will bite if you skip it

### 3.1 The derived limb leads are wrong in 26% of files

Every record is a **10-wire acquisition**: only 8 leads (I, II, V1–V6) are
physically measured. III, aVR, aVL and aVF are arithmetic by construction and
*cannot* carry independent information. XLI stores those four as a residual
against that derivation.

Measured over a 160-file random sample:

- **74%**: residual is zero to within 2 LSB (10 µV) — correct
- **26%**: residual is hundreds of µV, up to 1.4 mV
- **no intermediate cases** — the split is strictly bimodal

A bimodal split like that is a decoder artefact, not physiology (a filtering
difference would give a continuum). The checks in §2 confirm I, II and V1–V6
decode correctly in *both* groups, so only the residual channels are affected.

**Fix**: always recompute III, aVR, aVL, aVF from I and II. For 74% of records
this changes nothing; for the rest it replaces a corrupt residual with the
only values physically consistent with the acquisition. The discrepancy is
kept per record as `max_lead_relation_err_uv`.

### 3.2 The same ECG ships in several cohort folders

A patient in both the STEMI registry and the HF registry has their ECGs in
both folders. Raw file count double-counts badly:

```
67,431 files  ->  61,718 unique acquisitions   (5,713 duplicate copies, 8.5%)
```

**Dedup key**: `<documentname>`, the Philips document GUID — an exact
per-acquisition identity. It agrees with `(patient_id, date, time)` on all but
one record, and no GUID maps to two patients. Do **not** dedup on filename:
the same acquisition has different filenames in different folders.

Which copy you keep does not affect labels (the Philips statements are
identical in every copy), only the `cohort` attribution. The staging keeps the
copy from the most specific registry and preserves every membership in
`cohorts_all`.

### 3.3 Anonymised patient IDs over-group, and it is not optional

IDs are anonymised by keeping the first 4 characters and masking the rest:
`A016XXXX`. Two consequences, in opposite directions:

- **One prefix covers several real patients.** 934 of 3,813 masked IDs carry
  *conflicting sex values* — proof of collision. So "3,897 patients" is really
  3,897 **patient groups**.
- **79 records escaped the mask** and carry a full ID (e.g. `B0514816`), 6 of
  those prefixes also appearing masked elsewhere. Grouping on the full string
  would split one patient across folds.

**Therefore group on the first 4 characters.** That over-groups (costs a
little data) but can never leak a patient across a split. Over-grouping is the
safe error; under-grouping is not.

This matters because these are **serial ECGs**: 17,973 candidate records come
from 2,726 patient groups — median 4 per group, max 65. Random splitting would
leak heavily.

### 3.4 ~1% of records hit the acquisition rail

Records peaking at ~41 mV peak-to-peak. That is
exactly 8192 LSB × 5 µV = 40.96 mV — the ADC rail, i.e. genuine recorder
saturation (lead-off / electrode artefact) present in the source, not a decode
bug. Plus a handful of dead leads (< 0.05 mV peak-to-peak). These are marked
`validation_status = CHECK` rather than dropped — see the release README for
the current count.

One bug of mine surfaced here and is fixed: saturated samples were clipped to
-32768, which is WFDB's reserved "invalid sample" sentinel for format 16, so
they read back as NaN. Clipping now stops at -32767.

Separately, **8,097 of 67,431 files** carry the Philips QA verdict
`Poor ECG, retry if possible`. That verdict is passed through to
`validation_issues` verbatim rather than acted on.

---

## 4. Labels: there is no diagnosis table, only the algorithm

CUHK ships **no** per-record clinical labels. The only per-record labelling is
the Philips DXL algorithm's interpretation statements:

```xml
<unparsedstatement statementnumber="55" code="SR" format="Narrow">
  <lhsstatement>SINUS RHYTHM</lhsstatement>
  <rhsstatement>normal P axis, V-rate ***-***</rhsstatement>
</unparsedstatement>
```

**709 distinct statement codes** across the collection. All 709 are classified
by an ordered rule table over the statement *text*
(`cuhk/config/cuhk_code_rules.yaml`), which generates one explicit YAML entry
per code plus a reviewable table (`cuhk/reports/code_decisions.csv`).

> ⚠️ **These labels are machine-generated.** PTB-XL is human-overread; the 2026
> ACS/STEMI source has a clinician-assigned STEMI column. Every CUHK row
> carries `label_provenance: ALGORITHM_PHILIPS_DXL`, and the full statement
> codes and text are preserved per record so the cohort can be audited,
> re-mapped, or down-weighted.

### The decisions that matter

| bucket | codes | status | why |
|---|---|---|---|
| unambiguous acute injury / acute infarct (`PINJI`, `IMIA`, `AMIA`, `EAMIA`, `STE34`, …) | 51 | **ACCEPT → STEMI** | direct ST-elevation myocardial injury assertion |
| atrial fibrillation (`AFIB`, `AFIB0…8`, `AFIBT`) | 12 | **ACCEPT → AF** | explicit, unhedged rhythm assertion |
| left ventricular hypertrophy (`LVH`, `LVHREP`, `LVH31`, …) | 39 | **ACCEPT → LVH** | LVH named as the finding |
| "INFARCT, **POSSIBLY** ACUTE" / "INFARCT, RECENT" | 38 | **EXCLUDE_RECORD** | `STEMI=0` would be a wrong negative on thousands of records in an ACS cohort; `STEMI=1` would be a guess |
| bare ST elevation, no mechanism (`BSTEI`, `MSTEA`, `STE21`, …) | 23 | **EXCLUDE_RECORD** | same reasoning |
| STE attributed to **LVH or LBBB** (`STELVH`, `IMI81`, …) | 12 | **EXCLUDE_RECORD** | STEMI mimics that also *mask* occlusion — undecidable both ways |
| STE attributed to early repolarisation / normal variant / pericarditis | 12 | EXCLUDE_NOT_TARGET | the source positively asserts a non-ischaemic cause |
| paced rhythm; "no further analysis attempted" | 32 | **EXCLUDE_RECORD** | ST uninterpretable, or the algorithm says it did not finish |
| old / age-indeterminate infarct | 51 | EXCLUDE_UNCERTAIN | not STEMI; kept as a legitimate negative |
| atrial **flutter**; LVH-by-**voltage**-only | 24 | EXCLUDE_UNCERTAIN | mirrors the `AFLT` / `VCLVH` calls already made for PTB-XL |

Two project-wide rules were held to deliberately, so a label means the same
thing across every source: **flutter is not AF**, and **voltage criteria alone
are not LVH**.

**`normal_assertion: null`** — every CUHK cohort is a disease registry. A
STEMI-negative record here is a different acute presentation, never a healthy
control, so **CUHK contributes no NORMAL records at all**.

### Cohort membership is provenance, not a label

The folder name (`NSTEMI XML`, `HF_Additional_*`, …) says which registry a
patient was enrolled in — not what any individual ECG shows. A patient in the
NSTEMI registry has serial ECGs, some of which the algorithm reads as acute
STEMI. Cohort is recorded per record (`cohort`, `cohorts_all`) and used for
nothing else.

### Does the label agree with the waveform?

A consistency check only — the labels come from an algorithm reading the same
signal — but worth doing. Measuring ST amplitude straight off the converted
WFDB signals, with code that never sees a Philips statement (median beat, ST
at R+60…80 ms vs. a PR baseline, max over the 8 independent leads):

```
                        n    median max ST   >=0.1 mV in >=2 leads
STEMI = 1            2,330        0.149 mV                     40%
STEMI = 0 (AF/LVH)  13,606        0.054 mV                     15%
```

Right direction, roughly 3× (15,936 of 17,969 records had enough detectable
beats to measure). Still not a clean separator, and it should not be
read as one — the measurement approximates the J point from the R peak rather
than locating it. It confirms the labels are not scrambled; it does not
replace clinician review.

---

## 5. What the collection actually contains

Seven registries, 20 folders. `others/` was the surprise: it is roughly **3×**
the size of the original STEMI folders.

| cohort | folders | files | unique | dupes removed | patient groups |
|---|---|---|---|---|---|
| STEMI | 3 | 16,755 | 16,755 | 0 | 1,337 |
| HF (heart failure) | 11 | 25,942 | 22,770 | 3,172 | 1,965 |
| NSTEMI | 1 | 7,454 | 6,913 | 541 | 610 |
| MIXED_REDACTED (`Redacted Version 2`: HF, HF MR, ARVD) | 1 | 7,673 | 6,295 | 1,378 | 440 |
| PERICARDITIS (`Peri-inflammation`) | 3 | 7,053 | 6,610 | 443 | 831 |
| HF_AS (`HF AS XML`, aortic stenosis) | 1 | 2,207 | 2,028 | 179 | 125 |
| MYOCARDITIS (`Myoinflammation`) | 1 | 347 | 347 | 0 | 37 |
| **total** | **20** | **67,431** | **61,718** | **5,713** | **3,897** |

---

## 6. Final staged counts

```
67,431 XML files scanned  (+1 unreadable)
 5,713  duplicate copies removed (GUID dedup)
61,718  unique acquisitions
14,685  dropped: carry an EXCLUDE_RECORD label
29,060  dropped: no target label
17,973  CANDIDATES                    (2,726 patient groups)
     4  failed to decode (3 truncated XML, 1 PhilipsEXG doc type)
17,969  RELEASED                      (2,725 patient groups)
```

### Per condition

| class | records | patient groups |
|---|---|---|
| **STEMI** | **2,612** | 957 |
| **AF** | **8,152** | 1,588 |
| **LVH** | **9,995** | 1,912 |
| NORMAL | 0 | — (by design, §4) |

### Multi-label

| combination | records | % of candidates |
|---|---|---|
| LVH only | 7,426 | 41.3% |
| AF only | 5,814 | 32.3% |
| AF + LVH | 2,121 | 11.8% |
| STEMI only | 1,990 | 11.1% |
| STEMI + LVH | 405 | 2.3% |
| STEMI + AF | 174 | 1.0% |
| STEMI + AF + LVH | 43 | 0.2% |
| **single-label** | **15,230** | 84.7% |
| **multi-label (≥2)** | **2,743** | **15.3%** |

### By source cohort

| cohort | records | STEMI | AF | LVH |
|---|---|---|---|---|
| HF | 7,138 | 296 | 4,442 | 3,671 |
| STEMI | 3,604 | 1,758 | 774 | 1,507 |
| NSTEMI | 2,291 | 168 | 843 | 1,570 |
| MIXED_REDACTED | 2,039 | 132 | 837 | 1,436 |
| PERICARDITIS | 1,910 | 205 | 903 | 1,086 |
| HF_AS | 933 | 47 | 330 | 689 |
| MYOCARDITIS | 58 | 6 | 23 | 36 |

The STEMI-cohort row is byte-identical to the STEMI-only run done before
`others/` was added — dedup and the added cohorts did not disturb it.

---

## 7. LUDB — the other staged source

Found unintegrated in `dataset-files/`. 200 records, 200 distinct subjects,
already WFDB at 12 leads / 500 Hz / 5000 samples, so **no transcoding** —
signals are referenced in place.

Its value is not size, it is **provenance**: labels are written by
cardiologists, and it is the only staged source contributing NORMAL records.
The diagnosis vocabulary is closed — 82 distinct lines, all 82 declared, zero
undeclared terms at runtime.

| STEMI | AF | LVH | NORMAL | total |
|---|---|---|---|---|
| 8 | 12 | 106 | 25 | 137 |

10 records dropped for ventricular pacing (same call as CUHK), 53 for no
target label.

---

## 8. Not usable as ECG data

| file | why |
|---|---|
| `cuhk_dataset/Myocarditis registry anon.csv` (574 rows) | Clinical registry + extracted ECG *measurements* (rate, PR, QRS, QT, axes). No waveforms, no key linking rows to any XML. |
| `cuhk_dataset/Valvular heart disease ECG XML anonymized.xlsx` (2,209 × 509) | Table of Philips per-lead *measurements*. The XMLs it was derived from are not in this distribution. |
| `others/HFLAStrain/`, `others/HFLVStrain/` (1,767 `.xls`) | Echocardiographic strain measurements, not ECG. |

All three support tabular analysis; none can contribute a record to a
waveform dataset.

---

## 9. Signal representation produced

| | CUHK native | staged |
|---|---|---|
| leads | 12, `V1..V6` or `C1..C6` | 12, `I II III aVR aVL aVF V1..V6` |
| rate | 500 Hz | 500 Hz (unchanged) |
| length | 11.000 s / 5500 samples | **10.000 s / 5000 samples** — first 10 s kept |
| units | 5 µV/LSB (gain 200) | **gain 1000 ADU/mV** — exact ×5 integer rescale |
| limb leads | residual-decoded | **recomputed from I and II** (§3.1) |

The trailing 1 s is dropped because that is where the recorder stops
maintaining the derived-lead relationships. All changes are declared in each
`.hea` comment block and in `source_extra_json`. Output was verified by
reading it back with `wfdb.rdrecord`: checksums and initial values recompute
exactly, and Einthoven holds to floating-point precision.

---

## 10. Reproducing

```bash
pip install sierraecg numpy pyyaml wfdb

cd dataset-staging/cuhk
python scripts/scan_xml_index.py            # ~2 min, all 67k files, no waveform decode
python scripts/build_label_map.py           # instant
python scripts/build_manifest.py            # ~10 s
python scripts/convert_to_wfdb.py           # ~70 min on 4 cores (XLI decode ~0.8 s/file)
python scripts/validate_labels_vs_signal.py # optional QC

cd ../ludb
python scripts/build_ludb_staging.py        # ~1 s
```

To change a clinical decision, edit `cuhk/config/cuhk_code_rules.yaml` and
re-run from `build_label_map.py`. **No Python edit is needed for any label
change** — that was a design goal, matching the main pipeline's rule that
every clinical decision lives in YAML.

`scan_xml_index.py` takes `--stemi-only` / `--others-only`;
`convert_to_wfdb.py` takes `--limit N` and `--metadata-only` (rebuilds the
metadata CSV from an existing conversion report without re-decoding).

---

## 11. Open decisions left for you

1. **Clinician review of the label mapping.** `cuhk/reports/code_decisions.csv`
   lists all 709 codes with counts, statement text and assigned status. The
   ~2,600 STEMI positives rest on the 51 codes in the `MI_ACUTE` bucket.
2. **How to weight algorithmic vs. human labels** when CUHK is merged with
   PTB-XL and Chapman.
3. **Whether to cap records per patient group.** Median 4, max 65 — a few
   patients could dominate a class.
4. **Whether `EXCLUDE_RECORD` is too aggressive.** It drops 14,685 records,
   many of them hedged-acute or bare-ST-elevation cases that a cardiologist
   could adjudicate into real STEMI positives. That is the single largest
   pool of recoverable data in the collection.
5. **Whether to hold LUDB out entirely** as an external validation set rather
   than folding 137 records into training.
