# LUDB (Lobachevsky University Database) — staged for CardioSentry

**Status: usable, no transcoding needed. 137 labelled records staged.**
Small, but the only staged source with **cardiologist-written** labels *and*
NORMAL records, which makes it valuable out of proportion to its size — as an
independent external test set in particular.

| | |
|---|---|
| Source | `dataset-files/lobachevsky-university-electrocardiography-database-1.0.1` |
| Format | WFDB — **already** 12 leads, 500 Hz, 5000 samples (10 s) |
| Records | 200, from 200 distinct subjects (no serial ECGs) |
| Labels | manual, by doctors in Nizhny Novgorod (LUDB README); one diagnosis per `#` line in each `.hea` |
| Provenance | `HUMAN_CARDIOLOGIST` |

Because LUDB already ships in the target representation, **no signals are
copied or converted**. `data/ludb_metadata.csv` points `signal_path` /
`header_path` / `data_path` at the original files in `dataset-files/`, exactly
as the harmonized metadata does for PTB-XL and Chapman.

## Labels

The diagnosis vocabulary is **closed**: 82 distinct lines across the whole
database, and `config/label_map_ludb.yaml` declares **all 82** explicitly —
the run reports zero undeclared terms. Same schema as the main pipeline's
`label_mapping.yaml`, so the `ludb:` block can be pasted straight in.

Decisions kept deliberately consistent with the rest of the project:

- `STEMI: <wall>` → **ACCEPT → STEMI** (the only direct ST-elevation assertion
  in LUDB; a cardiologist wrote it)
- `Ischemia: <wall>`, `Undefined ischemia/scar/supp.NSTEMI: <wall>` →
  EXCLUDE_UNCERTAIN — ischaemia is not ST-elevation MI, and the "undefined"
  term is undecidable by construction
- `Scar formation: <wall>` → EXCLUDE_NOT_TARGET (healed infarct)
- `Rhythm: Atrial fibrillation` → **ACCEPT → AF**; `Atrial flutter, typical` →
  EXCLUDE_UNCERTAIN (same call as `AFLT` for PTB-XL and CUHK)
- `Left ventricular hypertrophy` → **ACCEPT → LVH**; `Left ventricular
  overload` → EXCLUDE_UNCERTAIN (strain, not hypertrophy); atrial and
  right-sided hypertrophy → EXCLUDE_NOT_TARGET
- ventricular pacing → **EXCLUDE_RECORD** (ST uninterpretable) — the same call
  made for CUHK. Atrial-only pacing is kept.

### NORMAL

LUDB has no "normal ECG" statement. What it has is a **completed checklist**:
annotators filled in every abnormality category for every record, so a record
with a rhythm and an axis but nothing in any pathology category is a positive
assertion that none were found. The reader emits a synthetic
`LUDB_NO_PATHOLOGICAL_FINDING` token for exactly those records, and that token
is declared in the YAML as this source's `normal_assertion`. Sinus
brady/tachy/arrhythmia and vertical/horizontal axis count as neutral —
matching how the main pipeline lets a PTB-XL `NORM`+`SBRAD` record be NORMAL.

### Resulting cohort

```
200 records
   10  dropped: ventricular pacing (EXCLUDE_RECORD)
   53  dropped: no target label
  137  CANDIDATES

  STEMI     8
  AF       12
  LVH     106
  NORMAL   25
```

## Files

```
config/label_map_ludb.yaml      all 82 diagnosis terms, explicit, pipeline schema
scripts/build_ludb_staging.py   resolve labels -> metadata (no transcoding)
data/ludb_metadata.csv          <- THE DELIVERABLE: harmonized schema, 137 rows
reports/label_decisions.csv     all 200 records, diagnoses, decision
reports/cohort_summary.txt
```

## Reproducing

```bash
cd dataset-staging/ludb
python scripts/build_ludb_staging.py    # ~1 s
```

## Note before merging

At 137 records LUDB cannot move the training set much, and its LVH share
(106/137) is lopsided. Its value is the label quality and the independent
institution: consider holding it out entirely as an external validation set
rather than folding it into training. That is a modelling decision, so nothing
here presumes it.
