# FCMAE morphology pretraining corpus builder

This pipeline creates `dataset-files/fcmae-pretraining-v1` without modifying or
copying the source databases. The release points to the raw signals through
relative symbolic links, so it consumes little additional disk space and keeps
source provenance auditable.

The builder is intentionally separate from the supervised CardioSentry V2
pipeline. It uses labels only for curation into five non-exclusive morphology
families and never emits STEMI/AF/LVH/NORMAL training targets.

## Clinical curation policy

The five morphology flags are `normal_baseline`, `ischemic_stt`,
`atrial_rhythm`, `structural_voltage`, and `conduction_other`. A record may
carry several flags but appears only once in the release.

Exact AF and exact LVH assertions are reserved rather than selected. No source
in this build has an explicit STEMI diagnosis; acute MI, MI, ST elevation, and
ischemia remain named as such and are eligible ischemic/ST-T neighbours. The
locally undocumented Chapman code `55827005` is also reserved because the V2
pipeline's signal audit found strong LVH-like voltage behaviour but could not
document the code from the source distribution.

V2 overlap is removed by canonical source record ID: Chapman `JSxxxxx` and
PTB-XL numeric `ecg_id`. The Challenge-2020 `HRxxxxx` PTB-XL identifiers are
converted to that numeric ID before comparison. The local standalone PTB-XL
metadata is joined only to recover patient IDs and source details; the signal
reference remains the Challenge-2020 record.

Known patients are capped at two selected recordings, and a second recording
is retained only if it adds a morphology family not already represented for
that patient. Sources without distributed patient IDs use explicitly marked
synthetic one-record grouping keys; these are not reported as known patients.

## Build

```bash
.venv/bin/python -m pip install -r dataset-combination-code/fcmae_pretraining_pipeline/requirements.txt
.venv/bin/python dataset-combination-code/fcmae_pretraining_pipeline/build_fcmae_dataset.py
```

Use `--dry-run` to generate only an inventory/selection report in a temporary
directory. A normal build is atomic: the final release directory is renamed
into place only after validation, duplicate hashing, manifests, and symlinks
have all completed. The command refuses to overwrite an existing release.

Source quotas and minimum morphology exposure shares are in `config.json`.
Shares are soft lower bounds, not mutually exclusive class proportions; one
multi-family ECG can advance several family counts.

## Release contents

- `manifest.csv`: one row per selected raw-signal record.
- `records/`: relative symlinks to native waveform/header files.
- `candidate_audit.csv`: every locally inventoried signal candidate and final
  decision.
- `exclusions.csv` and `exclusion_summary.csv`: record-level and aggregate
  exclusion reasons.
- `v2_overlap.csv`: exact Chapman/PTB-XL record IDs found in V2.
- `exact_target_audit.csv`: exact/uncertain target-like records reserved.
- `duplicates.csv`: duplicate IDs and exact raw-file hashes.
- `source_summary.csv`, `morphology_summary.csv`, and
  `morphology_combinations.csv`: final distributions.
- `missing_metadata.csv`: selected records with unavailable patient IDs or
  incomplete source metadata.
- `unmapped_diagnoses.csv`: source codes absent from the available official or
  source-shipped vocabulary, with selected/reserved/excluded frequencies.
- `meeti_audit.json`: local MEETI modality inventory and the reason it could
  not contribute raw signals when applicable.
- `summary.json`, `BUILD_PROVENANCE.json`, and `checksums.sha256`.

The release README records limitations that cannot be repaired from the local
files, especially unavailable patient linkage in Challenge sources and
Chapman, and the absence of raw MEETI waveforms in the downloaded tree.

The symlink release does not relicense any source data. Access, use, and any
redistribution remain governed by each source distribution's license and data
use terms.
