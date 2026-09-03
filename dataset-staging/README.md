# `dataset-staging/` — datasets prepared, not yet merged

Everything here is **ready to combine** with the main harmonized dataset and
**nothing here touches it**. `dataset-combination-code/ecg_dataset_pipeline/`
and its `data/` are untouched; merging is a separate, deliberate step
(each subfolder's README §"Merging").

Each staged source produces a metadata CSV whose columns are
**column-for-column identical** to
`ecg_dataset_pipeline/data/harmonized/harmonized_ecg_metadata.csv`, with
`signal_path` / `header_path` / `data_path` pointing at real WFDB records, so
merging can be as simple as a concatenation.

Label semantics are not re-invented: every staged source is resolved with the
main pipeline's own `common.LabelResolver` against a YAML written in the same
schema as `config/label_mapping.yaml`, so `ACCEPT` / `EXCLUDE_UNCERTAIN` /
`EXCLUDE_RECORD` mean here what they mean there.

## What is staged

| folder | source | records staged | STEMI | AF | LVH | NORMAL | label provenance |
|---|---|---|---|---|---|---|---|
| [`cuhk/`](cuhk/README.md) | CUHK ECG collection, 67,431 Philips XML across 7 registries | **17,969** (2,725 patient groups) | 2,612 | 8,151 | 9,991 | 0 | **algorithm** (Philips DXL) |
| [`ludb/`](ludb/README.md) | Lobachevsky University Database, 200 WFDB records | **137** (137 subjects) | 8 | 12 | 106 | 25 | **human** (cardiologists) |

For scale, the current harmonized dataset holds 19,375 records with 1,442
STEMI positives — so CUHK alone would more than double the STEMI class.

## Headlines

**CUHK STEMI — yes, it converts.** The waveforms are Philips **XLI**-compressed
(not plain Base64), which is why a naive XML read yields nothing. They decode
with the `sierraecg` package; all 16,755 files parsed with zero errors, and the
decode was cross-validated against the heart rate and per-lead measurements the
Philips machine wrote into the same files. Converted to WFDB at 12 leads /
500 Hz / 10 s / gain 1000 ADU/mV.

Two caveats you need before merging, both detailed in
[`cuhk/README.md`](cuhk/README.md):

1. **The labels are algorithmic**, not clinician-adjudicated. All 626 Philips
   statement codes are explicitly classified in
   `cuhk/reports/code_decisions.csv` — that table is laid out for
   clinician review.
2. **Serial ECGs**: 3,604 records from 806 patients (median 3 each). Splits
   must group on `split_group_id` or the same patient leaks across folds.
3. All 3,604 converted with **zero failures**; 60 are marked
   `validation_status = CHECK` (recorder saturation or a dead lead) so you can
   filter them.

**LUDB — no conversion needed.** It already ships as 12-lead 500 Hz 10 s WFDB,
so signals are referenced in place. It is small, but it is human-labelled and
is the only staged source contributing NORMAL records.

## Not usable (assessed, rejected)

| file | why |
|---|---|
| `cuhk_dataset/Myocarditis registry anon.csv` | Clinical registry plus extracted ECG *measurements*. No waveforms, and no key linking rows to any XML. Myocarditis is not a target class. |
| `cuhk_dataset/Valvular heart disease ECG XML anonymized.xlsx` | 2,209 × 509 table of Philips per-lead *measurements*. The waveforms it was derived from are not in this distribution. |

Both support tabular analysis only; neither can contribute a record to a
waveform dataset.

## Dependency

The CUHK converter needs one package that is not already in the project:

```bash
pip install sierraecg     # Philips Sierra ECG / XLI decoder
```
