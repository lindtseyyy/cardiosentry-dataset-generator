# CUHK STEMI registry — staged for CardioSentry

**Status: usable. 3,604 labelled 12-lead records staged, in the project's target
representation, ready to concatenate with the main harmonized dataset.**
Nothing in the main dataset has been modified.

| | |
|---|---|
| Source | `dataset-files/cuhk_dataset/STEMI XML {AB,CE,GZ}` |
| Format | Philips Sierra ECG XML 1.04 (UTF-16), waveform Base64 + **XLI** compression |
| Files | 16,755 XML, ~3.2 GB, 1,341 patients, 2005-09 → 2019-07 |
| Native signal | 12 leads, 500 Hz, 11.000 s, 5 µV/LSB, 16-bit, 10-WIRE acquisition |
| Labels | Philips DXL algorithmic interpretation statements (**not** human-adjudicated) |

---

## 1. Can the XML be parsed and converted?  Yes.

The waveforms are not plain Base64 — they use Philips' proprietary **XLI**
compression (`<parsedwaveforms compression="XLI">`), so a generic XML reader
returns nothing usable. They decode correctly with
[`sierraecg`](https://pypi.org/project/sierraecg/) (the Python port of
`sixlettervariables/sierra-ecg-tools`, the reference reverse-engineering of the
format):

```bash
pip install sierraecg
```

All 16,755 files parsed with **zero errors** and are perfectly uniform:
500 Hz, 11 s, 12 leads, 5 µV/LSB, 16-bit, XLI, `10-WIRE`, `STD` electrode
placement. 13,105 files label the chest leads `V1…V6` and 3,650 use the
European `C1…C6`; the converter normalises them.

### Decoding was verified three ways

1. **Heart rate.** R-peak detection on the decoded lead II vs. the
   `<heartrate>` the Philips machine wrote into the same file: median
   difference **1.0 bpm**.
2. **Per-lead amplitudes** vs. the `<leadmeasurement>` R/S amplitudes in the
   XML — consistent.
3. **Einthoven/Goldberger relations** (III = II − I, aVR = −(I+II)/2, …), which
   must hold exactly on a 10-wire acquisition.

Check 3 exposed the one real problem, and it is handled — see §3.

---

## 2. Labels

CUHK ships no diagnosis table. The only per-record labelling is the Philips
DXL algorithm's own interpretation statements. There are **626 distinct
statement codes**; all 626 are classified and every decision is written down:

- `config/cuhk_code_rules.yaml` — the ordered rule table (the decisions)
- `config/label_map_cuhk.yaml` — generated: one explicit entry per code, in
  **exactly** the schema of the main pipeline's `config/label_mapping.yaml`,
  so it can be pasted under that file's `sources:` key unchanged
- `reports/code_decisions.csv` — all 626 codes, counts, statement text and
  status in one table. **This is the table to hand a clinician.**

Labels are resolved with the main pipeline's own `common.LabelResolver`, not a
reimplementation, so a status means the same thing here as for PTB-XL.

### The decisions that matter

| bucket | codes | status | rationale |
|---|---|---|---|
| unambiguous acute injury / acute infarct (`PINJI`, `IMIA`, `AMIA`, `EAMIA`, `STE34`, …) | 44 | **ACCEPT → STEMI** | direct ST-elevation myocardial injury assertion |
| atrial fibrillation (`AFIB`, `AFIB0…8`, `AFIBT`) | 12 | **ACCEPT → AF** | explicit, unhedged rhythm assertion |
| left ventricular hypertrophy (`LVH`, `LVHREP`, `LVH31`, …) | 30 | **ACCEPT → LVH** | LVH named as the finding |
| "INFARCT, **POSSIBLY** ACUTE" / "INFARCT, RECENT" | 36 | **EXCLUDE_RECORD** | in a STEMI registry, `STEMI=0` here would be a wrong negative on ~2,000 records; `STEMI=1` would be a guess |
| bare ST elevation, no mechanism (`BSTEI`, `MSTEA`, `STE21`, …) | 22 | **EXCLUDE_RECORD** | same reasoning; the main pipeline already refuses to map bare STE to STEMI |
| STE attributed to **LVH or LBBB** (`STELVH`, `IMI81`, …) | 10 | **EXCLUDE_RECORD** | STEMI mimics that also *mask* occlusion — undecidable both ways |
| STE attributed to early repolarisation / normal variant / pericarditis | 11 | EXCLUDE_NOT_TARGET | the source positively asserts a non-ischaemic cause, so `STEMI=0` is *its* assertion |
| paced rhythm, and "no further analysis attempted" | 31 | **EXCLUDE_RECORD** | ST segment uninterpretable, or the algorithm says it did not finish |
| old / age-indeterminate infarct | 46 | EXCLUDE_UNCERTAIN | not STEMI; kept as a legitimate negative, as PTB-XL's location-only MI codes are |
| atrial **flutter**, LVH-by-**voltage**-only | 21 | EXCLUDE_UNCERTAIN | mirrors the `AFLT` and `VCLVH` decisions already made for PTB-XL |

`normal_assertion: null` — this is an ACS-referral cohort, so a STEMI-negative
record is a different acute presentation, never a healthy control. **CUHK
contributes no NORMAL records**, exactly as the 2026 ACS/STEMI source does not.

### Resulting cohort

```
16,755 XML scanned
   3,859  dropped: carry an EXCLUDE_RECORD label
   9,292  dropped: no target label
   3,604  CANDIDATES                      (806 patients)
          -> all 3,604 converted, 0 failures; 60 marked CHECK (see 3.4)

   STEMI  1,758 records / 573 patients
   AF       774 records / 246 patients
   LVH    1,507 records / 383 patients
   NORMAL     0 (by design)
```

For scale: the existing harmonized dataset has 1,442 STEMI positives, so this
source alone would **more than double** the STEMI class.

---

## 3. Four things you must know before merging

### 3.1 The labels are machine-generated

PTB-XL is human-overread; the 2026 ACS/STEMI source has a clinician-assigned
STEMI column. **These labels come from an algorithm.** Every row carries
`label_provenance: ALGORITHM_PHILIPS_DXL` in `source_extra_json`, and the full
Philips statement codes and text are preserved per record, so the cohort can be
audited, re-mapped, or down-weighted later. `reports/code_decisions.csv` is
laid out for clinician review.

### 3.2 Serial ECGs — 4.5 per patient

3,604 records come from only **806 patients** (median 3, max 35 per patient).
Splitting these at random would leak the same patient across train and test.
Every row carries `split_group_id = CUHKSTEMI::CUHKSTEMI_P<patient>`; the main
pipeline's group-aware splitter must be used, and you will probably also want
to cap records per patient when balancing.

### 3.3 The limb leads are recomputed (a decoder issue, resolved)

Every record is a `10-WIRE` acquisition: only 8 leads (I, II, V1–V6) are
physically measured, so III, aVR, aVL and aVF are arithmetic by construction.
XLI stores those four as a residual against that derivation. In **74%** of
files the decoded residual is zero to within 2 LSB (10 µV); in the other
**26%** it is hundreds of µV — with *no intermediate cases*. A bimodal split
like that is a decoder artefact, not physiology, and the independent checks in
§1 confirm I, II and V1–V6 decode correctly in both groups.

The converter therefore always recomputes the four limb leads from I and II.
For 74% of records this changes nothing; for the rest it replaces a corrupt
residual with the only values physically consistent with the acquisition. The
discrepancy is recorded per record as `max_lead_relation_err_uv` in
`reports/conversion_report.csv`.

### 3.4 Signal quality: 60 records marked `CHECK`, not `VALID`

Conversion produced **zero failures** (3,604/3,604). 931 records carry the
expected "limb leads recomputed" note from §3.3 and stay `VALID`. Sixty
records carry a real signal-quality problem and are marked
`validation_status = CHECK` so you can filter or inspect them:

| issue | records | what it is |
|---|---|---|
| amplitude ~41 mV peak-to-peak | 49 | recorder saturation — a hard ceiling at 8192 LSB × 5 µV = 40.96 mV, i.e. the acquisition rail. Lead-off / electrode artefact, present in the source. |
| one lead near-flat (< 0.05 mV) | 11 | a disconnected or dead electrode |
| clipped at int16 | 1 | consequence of the saturation above |

2,049 of the 16,755 source files also carry the Philips QA verdict
`Poor ECG, retry if possible`; where such a record survived labelling, that
verdict is appended verbatim to `validation_issues` rather than acted on.

---

## 4. Does the label agree with the waveform?

The labels come from an algorithm reading the same signal, so this is a
consistency check, not independent evidence — but it is worth doing, and
`scripts/validate_labels_vs_signal.py` does it with code that never sees a
Philips statement: R peaks from a squared-derivative detector on lead II, a
median beat per lead, ST amplitude at R+60…80 ms against a PR baseline at
R−80…−40 ms, maximum over the 8 independent leads.

```
group                    n   median max ST   p25     p75    >=0.1 mV in >=2 leads
STEMI = 1             1590       0.136 mV   0.049   0.298                    36%
STEMI = 0 (AF/LVH)    1656       0.071 mV   0.020   0.136                    21%
```

The STEMI group carries roughly twice the ST elevation of the labelled
negatives, in the expected direction. The separation is not dramatic, and it
should not be read as one: this crude measurement approximates the J point
from the R peak rather than locating it, and STEMI elevation is regional while
this takes a maximum over all leads. It confirms the labels are not scrambled;
it does not substitute for the clinician review in §3.1.

---

## 5. What was produced

```
config/
  cuhk_code_rules.yaml       ordered rule table -- the label decisions
  label_map_cuhk.yaml        GENERATED: 626 explicit code entries, pipeline schema
scripts/
  scan_xml_index.py          pass 1: inventory 16,755 XML (no waveform decode, ~30 s)
  build_label_map.py         rules + inventory -> label_map_cuhk.yaml
  build_manifest.py          resolve labels, pick the cohort
  convert_to_wfdb.py         pass 2: XLI decode -> WFDB
                             (--metadata-only rebuilds the CSV without re-decoding)
  validate_labels_vs_signal.py  measures ST elevation off the converted signals
data/
  index/xml_index.csv        one row per XML: ids, acquisition params, statements
  index/code_inventory.csv   626 codes with frequencies and statement text
  index/candidates.csv       the 3,604 records chosen for conversion
  signals/CUHKSTEMI_*.hea    WFDB: 12 leads, 500 Hz, 5000 samples,
  signals/CUHKSTEMI_*.dat    format 16, gain 1000 ADU/mV, ADC zero 0
  signals/checksums.sha256
  cuhk_stemi_metadata.csv    <- THE DELIVERABLE: harmonized schema, 30 columns,
                                column-for-column identical to
                                data/harmonized/harmonized_ecg_metadata.csv
reports/
  code_decisions.csv         all 626 codes and their status  <- for clinician review
  label_decisions.csv        all 16,755 records and their decision
  excluded_records.csv       every dropped record with the reason
  cohort_summary.txt         counts by class, patient and site
  conversion_report.csv      per record: status, QC metrics, warnings
  label_signal_agreement.csv per record: measured ST elevation vs its label
```

### Signal representation

| | native | staged |
|---|---|---|
| leads | 12, `V1..V6` or `C1..C6` | 12, `I II III aVR aVL aVF V1..V6` |
| rate | 500 Hz | 500 Hz (unchanged) |
| length | 11.000 s / 5500 samples | **10.000 s / 5000 samples** — first 10 s kept |
| units | 5 µV/LSB (gain 200) | **gain 1000 ADU/mV** — exact ×5 integer rescale |

Both changes are declared in each `.hea` comment block and in
`source_extra_json` (`native_resolution_uv`, `trimmed_to_first_n_samples`,
`amplitude_rescale_factor`). The trailing 1 s is dropped because that is where
the recorder stops maintaining the derived-lead relationships.

Output verified by reading it back with `wfdb.rdrecord`: checksums and initial
values recompute exactly, and Einthoven holds to floating-point precision.

---

## 6. Reproducing

```bash
pip install sierraecg numpy pyyaml
cd dataset-staging/cuhk
python scripts/scan_xml_index.py      # ~30 s
python scripts/build_label_map.py     # instant
python scripts/build_manifest.py      # ~2 s
python scripts/convert_to_wfdb.py     # ~40 min on 4 cores (XLI decode is ~0.8 s/file)
python scripts/validate_labels_vs_signal.py   # optional QC, ~3 min
```

To change a clinical decision, edit `config/cuhk_code_rules.yaml` and re-run
from `build_label_map.py`. No Python edit is needed for any label change.

---

## 7. Merging into the main dataset (not done — deliberately)

1. Paste the `cuhk_stemi:` block from `config/label_map_cuhk.yaml` under
   `sources:` in `ecg_dataset_pipeline/config/label_mapping.yaml`.
2. Add a `cuhk_stemi:` entry to `config/paths.yaml` and a `CUHKSTEMISource`
   reader to `scripts/sources.py` — or, simpler, concatenate
   `data/cuhk_stemi_metadata.csv` onto `harmonized_ecg_metadata.csv`, since the
   columns already match exactly and `signal_path` points at real WFDB files.
3. Re-run `create_balanced_dataset.py` / `freeze_splits.py`, making sure the
   splitter groups on `split_group_id` (§3.2).
4. Decide how to weight algorithmic vs. human labels (§3.1).

---

## 8. The rest of `cuhk_dataset/` — not usable as ECG data

| file | verdict |
|---|---|
| `Myocarditis registry anon.csv` (574 rows) | Clinical registry + extracted ECG *measurements* (rate, PR, QRS, QT, axes, RV5/SV1). **No waveforms, no key linking to any XML.** Myocarditis is also not a target class. Not integratable. |
| `Valvular heart disease ECG XML anonymized.xlsx` (2,209 rows × 509 cols) | A table of Philips per-lead *measurements* (`pamp`, `ramp`, … × 12 leads) extracted from ECG XMLs. **The waveforms themselves are not present** — the XMLs it was derived from are not in this distribution. Not integratable as signal data. |

Both could support tabular analysis, but neither can contribute a record to a
waveform dataset.
