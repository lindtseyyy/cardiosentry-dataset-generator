# CardioSentry — ECG dataset preparation pipeline

Builds a **reproducible, auditable** 12-lead ECG dataset for a four-head
(STEMI / AF / LVH / NORMAL) multi-label classifier, from three source
databases, without ever modifying the originals.

Every clinical labelling decision lives in **`config/label_mapping.yaml`**.
No label decision is buried in Python.

---

## 1. Results on the current data

Run end to end on the three datasets under `dataset-files/`:

| Class  | Source  | Candidate | Valid  | Excluded | % kept |
| ------ | ------- | --------: | -----: | -------: | -----: |
| STEMI  | ACS     |     1,442 |  1,442 |        0 | 100.00 |
| AF     | PTB-XL  |     1,514 |  1,341 |      173 |  88.57 |
| AF     | Chapman |     1,408 |  1,400 |        8 |  99.43 |
| LVH    | PTB-XL  |     2,132 |  1,902 |      230 |  89.21 |
| LVH    | Chapman |       643 |    640 |        3 |  99.53 |
| NORMAL | PTB-XL  |     8,064 |  7,145 |      919 |  88.60 |
| NORMAL | Chapman |     5,910 |  5,761 |      149 |  97.48 |

**Combined valid totals — STEMI 1,442 · AF 2,741 · LVH 2,542 · NORMAL 12,906**
(19,375 harmonized records; 256 carry two labels, all `AF+LVH`).

Signal compatibility: **all three sources are natively 12-lead, 500 Hz, 10 s,
5,000 samples/lead, mV units, identical lead order.** No resampling, cropping
or padding is required anywhere, and none is performed.

### Six findings you should know about before writing your methods chapter

1. **Your PTB-XL copy is incomplete.** 2,309 of 21,799 records referenced by
   `ptbxl_database.csv` have no `.dat`/`.hea` on disk, and 30 more are
   truncated (`cannot reshape array of size 32768`). This is the single
   largest source of loss (1,293 candidates). Re-downloading PTB-XL would
   recover roughly 170 AF, 230 LVH and 900 NORMAL records.
2. **`AF` in Chapman means atrial *flutter*, not fibrillation.** Chapman's
   `ConditionNames_SNOMED-CT.csv` maps the acronym `AF` to SNOMED
   `164890007` = *Atrial Flutter* (8,036 records), while atrial fibrillation
   is `164889003` (1,774 records). Mapping by acronym would have injected
   8,036 flutter ECGs into the AF class. Only `164889003` is accepted.
3. **SNOMED `55827005` — 5,384 records dropped entirely.** The code is
   undefined in the shipped condition file. Signal-level testing indicates it
   *is* LVH (§6.1), but because that evidence is behavioural rather than
   documentary, every record carrying it is excluded outright
   (`EXCLUDE_RECORD`) rather than kept under some other label. Keeping them
   would silently assert `LVH=0` on ~370 recordings that most likely do have
   LVH. See §6.1 for the evidence and the one-line change to accept it.
4. **Chapman ships no patient identifier.** Patient-level splitting is
   possible within PTB-XL (9,284 patients / 10,142 ECGs, up to 9 per patient)
   and the ACS set (1,437 patients / 1,442 ECGs), but not within Chapman.
5. **135 Chapman recordings have ≥6 dead leads** (typically all six
   precordials flat) and are rejected as corrupt; 45 more have 1–3 flat leads
   and are kept with a warning.
6. **16 byte-identical signal-file pairs exist inside Chapman.** They are
   reported, not deleted.

---

## 2. Requirements

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`pandas`, `numpy`, `wfdb`, `PyYAML`. Python 3.10+.

---

## 3. Configure your dataset paths

Edit **`config/paths.yaml`**. Paths may be absolute or relative to the
pipeline root. Set `enabled: false` on any source you do not have.

```yaml
ptbxl:
  root: "../../dataset-files/ptb-xl-dataset"     # contains ptbxl_database.csv
chapman:
  root: "../../dataset-files/chapman-dataset"    # contains WFDBRecords/
stemi:
  root: "../../dataset-files/STEMI-dataset"      # contains CSV/train.csv
```

**Nothing is downloaded automatically.** All three databases are gated behind
PhysioNet credentialing or a data-use agreement, so automated download would
breach their access terms. Obtain them yourself and point the config at your
local copies.

---

## 4. Run it

```bash
python scripts/run_pipeline.py                    # steps 1-9, ~10 min
python scripts/run_pipeline.py --near-duplicates  # + near-identical detection
python scripts/run_pipeline.py --workers 8        # parallel signal reads
```

Then, **only after you have reviewed the reports**:

```bash
python scripts/create_balanced_dataset.py --use-recommended --copy-signals
```

---

## 5. What each script does

| Script | Step | Does | Writes |
| --- | --- | --- | --- |
| `inspect_datasets.py` | 1 | Reads every metadata/label file and all ~83k WFDB headers. Reports structure, counts, columns, label vocabularies + frequencies, ids, fs, leads, duration, file availability. Assumes nothing that can be read. | `dataset_inspection_report.csv`, `label_frequency_report.csv`, `dataset_inspection_summary.txt` |
| `extract_records.py` | 2, 3, 11 | Applies `label_mapping.yaml` to every record. Builds the four binary label columns. Flags contradictions. | `label_mapping_audit.csv`, `extracted_candidates.csv`, `excluded_records.csv`, `unmapped_labels_report.csv`, `ambiguity_report.csv` |
| `validate_signals.py` | 4 | Reads every sample of every candidate. Checks leads, order, fs, duration, samples, units, NaN/Inf, flat leads, corruption. **Never modifies a signal.** | `signal_compatibility_report.csv` |
| `check_duplicates.py` | 6 | Duplicate record ids, repeated patient ECGs, SHA-256 file duplicates, optional near-duplicate fingerprints, cross-source collisions. **Deletes nothing.** | `duplicate_patient_report.csv`, `patient_summary.csv` |
| `build_harmonized_dataset.py` | 5, 7, 8, 9 | Joins the above into the maximum valid pool. Class counts. Balancing recommendation. | `harmonized_ecg_metadata.csv`, `harmonization_report.csv`, `final_class_counts.csv`, `balancing_recommendation.csv` |
| `probe_undefined_label.py` | — | Evidence script behind §6.1: measures LVH voltage criteria per label group. Re-run to reproduce that table. | stdout |
| `probe_qrs_duration.py` | — | Evidence script behind §6.1: measures QRS duration to exclude LBBB. | stdout |
| `create_balanced_dataset.py` | 10 | Optional, separate, seeded (42), patient-group aware balancing. | `balanced_ecg_metadata.csv`, `balancing_applied_report.csv` |
| `run_pipeline.py` | 12 | Runs steps 1–9 and prints the summary. Does **not** balance. | `cardiosentry_summary.txt` |
| `export_dataset.py` | 13 | Optional. Materialises a **versioned, self-contained** copy of the records the CSVs reference — signal files, rewritten headers, checksums, provenance. **Copies nothing destructively.** | `data/releases/<version>/` |
| `common.py`, `sources.py` | — | Shared utilities and per-source readers. `sources.py` knows file layout only; it never decides what a label means. | — |

---

## 6. Every label-mapping decision

Full machine-readable form: `config/label_mapping.yaml`, audited into
`reports/label_mapping_audit.csv` (73 declared labels).

### Status vocabulary

| Status | Meaning |
| --- | --- |
| `ACCEPT` | maps to a CardioSentry class |
| `EXCLUDE_UNCERTAIN` | could plausibly be a target class, but the source labelling does not permit asserting it. **Never guessed.** |
| `EXCLUDE_NOT_TARGET` | a real, understood diagnosis that simply is not one of our four classes |
| `EXCLUDE_RECORD` | meaning unresolved **and** keeping the record would risk a wrong negative — the **whole record** is dropped, not just the label |
| `NORMAL_DISQUALIFIER` | blocks the NORMAL label without creating a positive one |
| `NORMAL_NEUTRAL` | benign/technical finding that does not block NORMAL |
| `EXCLUDE_UNKNOWN_CODE` | label present in the data but undeclared in the config (auto-assigned, always reported) |

### STEMI — accepted from exactly one label

**`STEMI` column of the 2026 ACS dataset's `train.csv` → STEMI. 1,442 records.**

This is the only label in any of the three databases that asserts
ST-elevation myocardial infarction directly. Everything else was refused:

| Refused | Why |
| --- | --- |
| `NSTEMI` | non-ST-elevation MI is by definition not STEMI |
| `AMI` | acute MI, unspecified type — covers NSTEMI too; the source records STEMI separately |
| `OMI` | old / prior MI (or "occlusion MI") — neither reading is an acute ST-elevation ECG |
| `UA` | unstable angina — ACS without infarction |
| `CTO`, `LM`, `PLAD`…`DRCA` | angiographic culprit-vessel locations, not ECG diagnoses |
| `PCI`, `Prior_PCI` | treatments / history |
| PTB-XL `IMI`, `ASMI`, `AMI`, `ALMI`, `ILMI`, `LMI`, `PMI`, `IPMI`, `IPLMI` | encode infarct *location*, never ST elevation |
| PTB-XL `INJAS`, `INJAL`, `INJIN`, `INJIL`, `INJLA` | **sub**endocardial injury — the electrophysiological opposite of transmural ST elevation |
| PTB-XL `STE_` | "non-specific ST elevation", a *form* statement. ST elevation alone is not STEMI (early repolarisation, pericarditis, LVH strain, LBBB all produce it) |
| Chapman `164930006` (ST extension), `164917005` (abnormal Q wave), `164865005` (lateral-wall MI) | morphology or location only |

PTB-XL does carry acuity in `infarction_stadium1` (Stadium I–III), so
"MI code + Stadium I" was *available* as a heuristic. It was rejected: that
is an inference from two fields, not a label, and it is exactly the
substitution the project rules forbid. **STEMI therefore comes from one
source only.**

### AF

| Accepted | Source |
| --- | --- |
| `AFIB` — atrial fibrillation | PTB-XL, 1,514 |
| `164889003` — Atrial Fibrillation | Chapman, 1,774 |

Refused: PTB-XL `AFLT` (flutter, 73), `SVTAC`, `PSVT`, `SVARR`; Chapman
`164890007` (flutter, 8,036 — **the acronym trap**), `426761007` (SVT),
`713422000` (atrial tachycardia), `233897008` (AVRT), `427393009` (sinus
irregularity), `195101003` (wandering atrial pacemaker).

Chapman AF is internally consistent: `164889003` never co-occurs with
flutter or with sinus rhythm.

**PTB-XL likelihood caveat.** PTB-XL `scp_codes` values are annotator
likelihoods, but 1,466 of the 1,514 AFIB records carry likelihood `0.0` —
for *rhythm* statements a 0 means "unscored", not "absent". Filtering rhythm
statements on likelihood would silently destroy 97% of the AF class.
`rules.ptbxl_likelihood.apply_to_rhythm_statements` is therefore `false`.
The threshold (`min_diagnostic_likelihood`, default 0) applies to diagnostic
statements only; per-record likelihoods are preserved in
`source_extra_json` so you can re-filter without re-extracting.

### LVH

| Accepted | Source |
| --- | --- |
| `LVH` — left ventricular hypertrophy (`diagnostic=1`, class HYP) | PTB-XL, 2,132 |
| `164873001` — left ventricle hypertrophy | Chapman, 645 |

Refused:

- **PTB-XL `VCLVH`** (875) — "voltage criteria (QRS) for LVH". Verified from
  `scp_statements.csv`: this is a **form** statement (`form=1`,
  `diagnostic=NaN`, no diagnostic class). It records a voltage measurement,
  not an LVH diagnosis, and PTB-XL's own annotators kept it distinct.
- `RVH`, `SEHYP`, `LAO/LAE`, `RAO/RAE`, Chapman `89792004` (RVH),
  `446358003` (RAH) — wrong chamber.
- Chapman `251146004` (low QRS voltage) — the opposite pattern.

### 6.1 SNOMED `55827005` — investigated, then excluded outright

5,384 Chapman records (12%) carry `55827005`, which the shipped
`ConditionNames_SNOMED-CT.csv` does not define. Rather than guess, the code
was tested against the waveforms.

**LVH voltage criteria** (Sokolow-Lyon: S_V1 + max R_V5/V6 ≥ 3.5 mV; Cornell,
sex-adjusted), measured directly from the signals:

| Group | n | SL median | SL+ | Cornell+ | Either+ |
| --- | --: | --: | --: | --: | --: |
| Chapman `164873001` (documented LVH) | 640 | 4.80 mV | 92.8% | 51.7% | 96.2% |
| **Chapman `55827005` (undefined)** | 895 | **3.73 mV** | **63.6%** | **19.4%** | **69.5%** |
| **PTB-XL `LVH` (independent gold standard)** | 897 | **3.59 mV** | **53.8%** | **40.7%** | **72.0%** |
| Chapman sinus-rhythm-only controls | 897 | 2.21 mV | 2.1% | 2.7% | 4.6% |
| PTB-XL `NORM` controls | 893 | 2.18 mV | 3.4% | 3.5% | 6.2% |

Records carrying `55827005` are **indistinguishable from LVH confirmed by a
different institution in a different country**. Two independent control groups
agree at ~2.2 mV / ~3%, cross-validating the measurement.

Two confounders were excluded:

- *"Sick older patients simply have larger voltages."* Comorbidity-matched
  flutter patients carrying neither LVH code: 2.22 mV, 8.8% SL-positive —
  control level.
- *"It is LBBB, which also inflates these voltages."* QRS duration is 70 ms,
  identical to normals (70 ms) and LVH (74 ms), against 148 ms for confirmed
  `CLBBB` (95.4% >120 ms). Not conduction disease.

Median age 68 (`55827005`) vs 74 (`164873001`) vs 52 (controls) is consistent.
The co-occurrence profiles suggest `164873001` is *LVH with strain* (76.7%
co-occur with ST-T abnormality) and `55827005` is LVH by voltage criteria —
two severities of one diagnosis.

**Decision: `EXCLUDE_RECORD` — every record carrying the code is dropped.**

The evidence is behavioural, not documentary: the distribution still does not
define the code and no clinician has signed it off. `EXCLUDE_RECORD` is used
deliberately rather than `EXCLUDE_UNCERTAIN`, because merely ignoring the code
would leave ~370 of those recordings in the cohort under other labels with
`LVH=0` — training the LVH head against the truth on records that most likely
*do* have LVH. Dropping them forfeits data but asserts nothing false.

Cost: AF falls from 3,107 to 2,741 valid records; LVH is essentially unchanged
(2,544 → 2,542); 5,016 records that had no target label anyway leave the pool.

**To accept it as LVH** once a clinician confirms the code, edit
`config/label_mapping.yaml`:

```yaml
"55827005":
  status: ACCEPT          # was EXCLUDE_RECORD
  maps_to: LVH            # was null
```

and re-run `run_pipeline.py`. Chapman LVH becomes ~6,027 and total LVH ~8,159,
making LVH the largest class and your 2,700 target comfortably reachable.

### NORMAL

A record is NORMAL only if **all three** hold (`rules.normal`):

1. the source makes a **positive** normal assertion,
2. it carries no `NORMAL_DISQUALIFIER` and no undefined label,
3. it carries no accepted STEMI / AF / LVH label.

| Source | Normal assertion | Candidates |
| --- | --- | --- |
| PTB-XL | `NORM` ("normal ECG", `diagnostic_class=NORM`) — a cardiologist's positive statement of normality | 9,514 → **8,064** after disqualification |
| Chapman | `426783006` (Sinus Rhythm) **plus no other non-neutral code** | 8,102 SR → **5,910** |
| ACS/STEMI | *none — this source contributes no NORMAL* | 0 |

- Chapman has no "normal ECG" concept at all, so sinus rhythm is the closest
  positive assertion. Per the project rules it is **not** sufficient alone:
  5,890 records are SR-and-nothing-else, plus 20 whose only companions are
  the two `NORMAL_NEUTRAL` codes (`251198002`/`251199005`, clockwise and
  counter-clockwise transition-zone rotation — positional variants).
- The ACS dataset is an ACS-referral cohort: a STEMI-negative record there is
  not a healthy control, it is a *different* acute coronary syndrome. It is
  configured with `normal_assertion: null` and contributes no NORMAL at all.
- PTB-XL `SR`, `SBRAD`, `STACH`, `SARRH` are `NORMAL_NEUTRAL` — rate and
  respiratory-variation observations that do not disqualify an otherwise
  normal ECG. Everything else, including all 43 undefined Chapman codes,
  disqualifies NORMAL.

*"Why is this particular PTB-XL record NORMAL?"* → look it up in
`harmonized_ecg_metadata.csv`: `original_labels_readable` shows every source
statement with its description, `accepted_label_mapping` shows the mapping
applied. If it was rejected, `excluded_records.csv` gives the reason.

---

## 7. Every exclusion rule

| Rule | Effect |
| --- | --- |
| No accepted target label after mapping | → `excluded_records.csv`, reason `EXCLUDE_NOT_TARGET` / `EXCLUDE_UNCERTAIN` / `NO_TARGET_LABEL` |
| Label undeclared in config | → `EXCLUDE_UNKNOWN_CODE`; cannot create a label, **does** block NORMAL |
| Label marked `EXCLUDE_RECORD` (`55827005`) | whole record dropped (5,384) — see §6.1 |
| `test.csv` of the ACS set (1,995 rows) | no diagnostic columns at all → unlabelable, excluded wholesale |
| Missing `.hea` or `.dat`/`.mat` | `INVALID_MISSING_FILE` (1,288) |
| Unreadable / truncated / malformed header | `INVALID_UNREADABLE` (30) |
| NaN, Inf, all-zero, or ≥6 flat leads | `INVALID_SIGNAL` (135) — threshold `--max-flat-leads`, default 6 |
| Not 12 leads, or a target lead missing | `INVALID_SIGNAL` |

Every exclusion is recorded with **source, record id, original label and
reason** in `excluded_records.csv` and `harmonization_report.csv`.

---

## 8. Every preprocessing operation

**None.** Verified across all 19,743 valid records: 12 leads, 500 Hz, 10.0 s,
5,000 samples, mV units, lead order `I, II, III, aVR, aVL, aVF, V1–V6` — in
all three sources. Lead names are compared case-insensitively because PTB-XL
writes `AVR/AVL/AVF` while Chapman and the ACS set write `aVR/aVL/aVF`; this
is a naming difference only, the ordering is identical.

If a future source does not conform, `validate_signals.py` **reports the
incompatibility and the preprocessing that would be required, and applies
nothing.** Resampling, cropping and padding are deliberately not implemented.

The ACS set also ships median-beat files (`ECG_median_data/*.med`). They are
not raw 10 s rhythm strips and are not used; only `ECG_row_data/*.dat` is.

---

## 9. Duplicates and patient-level splitting

| Source | ECGs | Patients | >1 ECG | Max | Patient-level split |
| --- | --: | --: | --: | --: | --- |
| PTB-XL | 10,142 | 9,284 | 664 | 9 | **YES** |
| ACS/STEMI | 1,442 | 1,437 | 5 | 2 | **YES** |
| Chapman | 7,791 | — | — | — | **NO — no identifier** |

`harmonized_ecg_metadata.csv` carries **`split_group_id`** (`SOURCE::patient`).
**Group your train/val/test split on this column.** Records sharing a value
must land in the same fold. For Chapman it degenerates to one group per
record, flagged by `patient_id_is_synthetic = 1`.

Also found: 669 repeated-ECG patient groups, 16 byte-identical Chapman file
pairs, 0 near-identical signal groups beyond those, 253 cross-source record-id
collisions (different datasets reusing the same local number — harmless, the
`global_record_id` prefix disambiguates). **Nothing is deleted.**

**Stated limitation.** Patient identifiers are not comparable across the three
databases: different institutions, unrelated id schemes, no linkage key.
Cross-dataset patient overlap can be neither confirmed nor excluded. It is
*assumed* negligible because the cohorts are geographically and temporally
distinct — an assumption, not a verified fact, and it should be reported as
such.

---

## 10. Multi-label structure (four sigmoid heads)

`STEMI`, `AF`, `LVH`, `NORMAL` are independent binary columns. 256 records
carry two labels (all `AF+LVH`) and are preserved as such.

`NORMAL` is mutually exclusive with the other three **by construction**, so
`STEMI=1, NORMAL=1` cannot occur. Where a source asserted both, the record is
written to **`ambiguity_report.csv`** with flag `NORMAL_VS_PATHOLOGY`; per
`rules.ambiguity.on_conflict: drop_normal` the pathology is kept and NORMAL is
dropped. 3,251 ambiguity rows were logged — including 37 PTB-XL records
labelled both `NORM` and `AFIB`, and 1 labelled both `NORM` and `LVH`.
Set `on_conflict: exclude` to drop such records entirely instead.

---

## 11. Balancing

`balancing_recommendation.csv` — **advice only, nothing performed**:

| Class | Available | Ratio to STEMI | Your preferred | Recommended | Discarded |
| --- | --: | --: | --: | --: | --: |
| STEMI | 1,442 | 1.000 | 1,442 | **1,442** | 0 |
| AF | 2,741 | 1.901 | 2,800 | **2,741** | 0 |
| LVH | 2,542 | 1.763 | 2,700 | **2,542** | 0 |
| NORMAL | 12,906 | 8.950 | 2,800 | **2,800** | 10,106 |

Two preferred targets are unachievable — **AF 2,800 > 2,741** and
**LVH 2,700 > 2,542** — so both are capped at take-all. Raising them requires
either accepting SNOMED `55827005` (§6.1, ~+5,400 LVH) or re-downloading the
missing PTB-XL records (~+170 AF, ~+230 LVH). STEMI stays the smallest class
and no class falls below it. ✔

`create_balanced_dataset.py` is separate and optional:

```bash
python scripts/create_balanced_dataset.py                      # TARGETS dict
python scripts/create_balanced_dataset.py --use-recommended    # from the report
python scripts/create_balanced_dataset.py --target NORMAL=2544 --target AF=2544
python scripts/create_balanced_dataset.py --seed 7 --copy-signals
```

It reads the harmonized CSV read-only, selects **whole patient groups** so no
patient straddles the cohort boundary, fills classes in scarcity order
(STEMI → LVH → AF → NORMAL) so multi-label spill-over counts honestly, and
uses a fixed seed (default 42) so the cohort is reproducible. Targets live in
the `TARGETS` dict at the top of the script and are overridable on the CLI.

With the recommended targets: 9,269 records (1,442 / 2,741 / 2,542 / 2,800 —
9,525 class-slots minus the 256 dual-label records), 8,720 patient groups,
composition PTB-XL 4,555 · Chapman 3,272 · ACS 1,442.

---

## 12. Answering the panel

> **"Exactly where did your STEMI samples come from?"**
> The `STEMI` column of `CSV/train.csv` in the 2026 ACS dataset — 1,442
> records, all valid, from 1,437 patients. It is the only explicit
> ST-elevation label in any of the three databases. `label_mapping_audit.csv`
> lists the 21 sibling columns (AMI, OMI, NSTEMI, UA, culprit-vessel codes…)
> that were refused, each with its reason.

> **"Why did you classify this particular PTB-XL record as normal?"**
> Find its `global_record_id` in `harmonized_ecg_metadata.csv`.
> `original_labels_readable` lists every SCP statement with its description;
> `accepted_label_mapping` shows the mapping. Cross-check the rule in
> `label_mapping_audit.csv`. If it was excluded, `excluded_records.csv` gives
> the reason; if contradictory, it is in `ambiguity_report.csv`.

> **"How did you combine ECGs from different datasets?"**
> We didn't have to convert anything: all three are natively 12-lead, 500 Hz,
> 10 s, 5,000 samples, mV, same lead order — verified sample-by-sample for
> every record in `signal_compatibility_report.csv`. Records keep a
> provenance-preserving `global_record_id` (`PTBXL_…`, `CHAPMAN_…`,
> `STEMI_…`) plus source, original id, patient id and original labels. The id
> is derived from the source record number, so it is stable: `PTBXL_000123` is
> always PTB-XL `ecg_id` 123, whatever the label config says.

---

## 13. Directory layout

```
ecg_dataset_pipeline/
├── config/
│   ├── label_mapping.yaml          every clinical decision, with reasons
│   └── paths.yaml                  your local dataset paths
├── scripts/
│   ├── common.py                   config loading, label resolver, header parser
│   ├── sources.py                  per-source readers (file layout only)
│   ├── inspect_datasets.py         STEP 1
│   ├── extract_records.py          STEPS 2, 3, 11
│   ├── validate_signals.py         STEP 4
│   ├── check_duplicates.py         STEP 6
│   ├── build_harmonized_dataset.py STEPS 5, 7, 8, 9
│   ├── probe_undefined_label.py    evidence for §6.1 (LVH voltage criteria)
│   ├── probe_qrs_duration.py       evidence for §6.1 (LBBB exclusion)
│   ├── create_balanced_dataset.py  STEP 10 (optional, seeded)
│   └── run_pipeline.py             STEP 12
├── reports/
│   ├── dataset_inspection_report.csv      label_mapping_audit.csv
│   ├── dataset_inspection_summary.txt     extracted_candidates.csv
│   ├── label_frequency_report.csv         excluded_records.csv
│   ├── signal_compatibility_report.csv    unmapped_labels_report.csv
│   ├── duplicate_patient_report.csv       ambiguity_report.csv
│   ├── patient_summary.csv                harmonization_report.csv
│   ├── final_class_counts.csv             balancing_recommendation.csv
│   ├── balancing_applied_report.csv       cardiosentry_summary.txt
├── data/
│   ├── harmonized/harmonized_ecg_metadata.csv
│   ├── balanced/balanced_ecg_metadata.csv
│   └── releases/<version>/          self-contained export (see §16)
├── image_pipeline/                 synthetic paper-ECG image generation
│   ├── config.py                   render / augmentation tunables
│   ├── patch_kit.py                the one ecg-image-kit source patch
│   ├── stage1_manifest_from_csv.py CSV -> render manifest
│   ├── stage2_transcode.py         WFDB harmonisation + band-pass
│   ├── stage3_render.py            ecg-image-kit, parallel
│   ├── stage4_augment.py           paper/scan realism
│   ├── stage5_verify.py            integrity + source-leakage probe
│   ├── run_all.sh  install.sh  visualize_annotations.py
│   └── README.md                   read this before running it
├── requirements.txt
└── README.md
```

Signal files are **referenced in place** by default; the source datasets are
opened read-only and never written to. Pass `--copy-signals` to either builder
to materialise copies under `data/*/signals/` named by `global_record_id` —
but prefer `scripts/export_dataset.py` (§16), which does the same thing
versioned, checksummed and verified.

---

## 14. Reproducing

```bash
python scripts/run_pipeline.py --near-duplicates --workers 8
python scripts/create_balanced_dataset.py --use-recommended --seed 42
```

Runtime ≈ 12 min on 4 cores (dominated by reading ~40k signals twice).
Deterministic given the same inputs, config and seed.

---

## 15. Synthetic paper-ECG images

`image_pipeline/` renders the records this pipeline selects into synthetic
12-lead **paper ECG images**, for a direct 2D vision model rather than a
1D signal model. It consumes `data/balanced/balanced_ecg_metadata.csv` (or the
harmonized CSV) and writes a labelled JPEG corpus plus its own `index.csv`.

```bash
cd image_pipeline
bash install.sh          # separate Python 3.11 venv; ecg-image-kit is already cloned
./run_all.sh
```

It never modifies anything under `data/` or `reports/`, and it decides nothing
clinical — the four binary label columns are copied through verbatim. Two
things it *does* decide, and which the panel will ask about: the train/val/test
split (grouped on `split_group_id`, stratified per class, PTB-XL's own
`strat_fold` honoured) and the signal band (0.05–40 Hz, forced by the 150 DPI
render's 73.8 Hz image Nyquist). Both are argued in `image_pipeline/README.md`,
along with the source-leakage problem that STEMI being single-source creates
for an image model.

---

## 16. Versioned dataset releases

The metadata CSVs are *references*: `signal_path` points into your local copies
of three separate source distributions. That is right for the pipeline (nothing
is duplicated, nothing is modified) and wrong the moment you want to archive
the dataset, move it to another machine, or say precisely what "the dataset"
was six months from now.

`scripts/export_dataset.py` materialises it:

```bash
python scripts/export_dataset.py                     # writes the next free vN
python scripts/export_dataset.py --version v1
python scripts/export_dataset.py --version v2 --note "accepted SNOMED 55827005"
python scripts/export_dataset.py --cohort balanced   # balanced records only
python scripts/export_dataset.py --limit 8           # smoke test, 8 per source
```

**Each cohort is a complete, standalone dataset in its own directory.** Copy
out just `balanced/` and it works — its CSV, its manifest, its checksums and
every signal file it references are inside it, and nothing points back at the
original source distributions.

```
data/releases/v1/
├── VERSION.json                     provenance: inputs, config hashes, counts
├── README.md                        generated, human-readable
├── harmonized/
│   ├── harmonized_ecg_metadata.csv  19,743 rows
│   ├── MANIFEST.csv                 per record: paths, SHA-256, cohort flags
│   ├── checksums.sha256             standard `sha256sum -c` format
│   └── signals/<SOURCE>/<global_record_id>.hea + .dat|.mat
└── balanced/
    ├── balanced_ecg_metadata.csv     9,330 rows
    ├── MANIFEST.csv
    ├── checksums.sha256
    └── signals/<SOURCE>/<global_record_id>.hea + .dat|.mat
data/releases/CHANGELOG.md            appended on every export
```

The balanced cohort is a subset of the harmonized one, so a both-cohort export
writes those ~9,330 shared records twice: **≈3.5 GB** rather than 2.4 GB. That
is the price of each directory standing on its own, and usually worth paying.

`--hardlink` gets the disk back without giving up the layout: the shared records
become one set of bytes with two names, so both trees stay complete and
independently readable at ≈2.4 GB total. Read-only archives only — hard links
mean editing one path edits both.

### Renaming a WFDB record is not a file rename

This is the one non-obvious thing the script does, and the reason it exists
rather than a `cp` loop. A WFDB header names its own record on line 1 and names
its signal file on every signal line:

```
JS00001 12 500 5000
JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I
```

Copy that to `CHAPMAN_000001.hea` alongside `CHAPMAN_000001.mat` and
`wfdb.rdrecord("CHAPMAN_000001")` raises `FileNotFoundError`, hunting for a
`JS00001.mat` that is not there. The export rewrites those two tokens and
**nothing else** — gains, baselines, checksums, Chapman's `16+24` offset
format, the `#Age`/`#Sex`/`#Dx` comments and even trailing whitespace survive
byte for byte, because those fields are what make the copied signal bytes
decode to the same millivolts.

Signal files themselves are never decoded, so they cannot be altered. `--verify`
proves it by reading both copies back and comparing sample by sample:
`sample` (200 records, the default), `all`, or `none`.

> `--copy-signals` on the two builders had exactly this bug and produced
> unreadable output. Both now route through the same corrected copy.

### Versioning

`VERSION.json` fingerprints the SHA-256 of both input CSVs **and of
`config/label_mapping.yaml`**, so two releases can be compared for whether they
came from the same clinical decisions — the question that actually matters when
someone asks why the LVH count changed between v1 and v2. `--version` is free
text (`v1`, `v2`, …, or anything else); omitted, it takes the lowest unused
`vN`. Re-exporting over an existing version needs `--force`.

Integrity is re-checkable at any time, without the pipeline:

```bash
python scripts/export_dataset.py --check v1          # walks every cohort
cd data/releases/v1/balanced && sha256sum -c checksums.sha256
```

### Using a release

Paths in each CSV are relative to **that cohort's own directory** (pass
`--path-style absolute` if you would rather they were not), so a cohort stays
valid after you move, rename or archive it:

```python
import pandas as pd, wfdb
df = pd.read_csv("balanced/balanced_ecg_metadata.csv")
rec = wfdb.rdrecord("balanced/" + df.signal_path[0])
```

`image_pipeline/` reads a release directly — it resolves relative signal paths
against the CSV's own directory:

```bash
cd image_pipeline
python stage1_manifest_from_csv.py \
    --input-csv ../data/releases/v1/balanced/balanced_ecg_metadata.csv
```

For 1D model training, `--npy` additionally writes a `signals.npy` per cohort —
one stacked `(N, 5000, 12)` float32 array in mV, with `signals_index.csv` giving
the row order. It is written through a memmap, so the ~4.7 GB never has to fit in RAM.

**Licensing.** A release is a compilation of licence-gated databases, one of
them CC BY-NC-**ND**. Internal use only; do not redistribute. `data/releases/`
is gitignored.
