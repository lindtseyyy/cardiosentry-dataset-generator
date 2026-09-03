# External STEMI validation

A held-out test corpus and evaluation harness for the five trained CardioSentry
checkpoints. **Nothing here trains anything.** The weights live in Drive and are
opened read-only.

## The question

The prototype's evaluation report closes on a finding it could not resolve:

> All 1,442 STEMI records come from a single database … the per-source
> diagnostic cannot audit the STEMI head at all — the one head that most needs
> auditing.

Every STEMI positive came from the Chongqing ACS set, and no other source
contributed one. Inside that corpus, two very different models produce the
identical confusion matrix:

| Hypothesis | Predicts on the internal test set | Predicts on a STEMI from elsewhere |
|---|---|---|
| learnt ST elevation | high STEMI scores | **fires** |
| learnt the Chongqing scanner | high STEMI scores | **silent** |

The internal corpus cannot separate the rows. A STEMI positive from a different
database can. That is all this folder is for.

Note that ResNet-50's overall win rests on this head: its per-label STEMI AUPRC
of 0.9560 is what separates it from the field, so the ranking in the prototype
report inherits whatever the answer turns out to be.

## The corpus

**413 sheets**, one canonical image per ECG, `split=test` throughout.

| Cohort | Sheets | STEMI | AF | LVH | NORMAL | What it is for |
|---|---:|---:|---:|---:|---:|---|
| `LUDB` | 200 | **8** | 15 | 108 | 25 | **Gold.** A source seen in no class |
| `PTBXL_MI` | 58 | 58 | 1 | 2 | 0 | **Proxy.** Seen source, never labelled STEMI |
| `PTBXL_NORM` | 155 | 0 | 0 | 0 | 155 | Negative controls from the same source |
| **Total** | **413** | **66** | **16** | **110** | **180** | |

Of which **59** are all-negative hard negatives (see below), and **0** belong to
a patient in the training corpus.

**LUDB** — Lobachevsky University Database, Nizhny Novgorod. Schiller Cardiovit
AT-101, 12-lead, 500 Hz, 10 s. Its cardiologist reports use the literal word
`STEMI: <wall>`, so the label is *asserted*, not inferred. Eight positives is
the entire supply in the database.

**PTBXL_MI** — the 275 hand-inspected records from
[`dgedon/nstemi_prediction`](https://github.com/dgedon/nstemi_prediction),
200 `normal` / 75 `mi`.

> ⚠️ **This is not a new database — it is PTB-XL**, which is already a training
> source. The repo ships label ids only, not signals. Its value is
> one-directional: PTB-XL contributed **zero** STEMI labels to training
> (`label_mapping.yaml` refuses its MI codes because they encode infarct
> *location*, not ST elevation), so a model that fires here cannot be firing on
> the paper. But the label is `mi` + `Stadium I` (acute), and acute MI still
> admits NSTEMI. **It is a proxy. Never merge its figure with LUDB's.**

61 records whose *patient* appears in the balanced training corpus are excluded
— patient-level, because the corpus split on patient.

### The 59 hard negatives

`stage1_manifest_from_csv.py` drops records with no positive label, which is
right for a training corpus and wrong for this one. Those 59 LUDB records are
the sharpest negatives available:

| n | Finding | Why it matters |
|---:|---|---|
| 31 | non-specific repolarisation abnormality | ST-T change without infarction |
| 8 | **early repolarisation syndrome** | **benign ST elevation — the textbook STEMI mimic** |
| 8 | undefined ischemia / scar / supp. NSTEMI | coronary disease that is *not* STEMI |
| 7 | scar formation | old infarct |

They are rescued by a second render pass — see `01_build_metadata.py`.

## Running it

```bash
cd build
python 00_fetch_missing.py     # label CSV + any missing PTB-XL records
python 01_build_metadata.py    # every clinical decision lives in this file
./02_render_images.sh          # ~25 min; drives the EXISTING image pipeline
python 03_merge.py             # -> data/external_testset/{images,index.csv}
python 04_package.py           # -> external_testset.tar + PROVENANCE.json
```

Then upload the tar to Drive and run the evaluation in Colab:

```bash
cd eval && python make_notebook.py    # -> CardioSentry_External_Validation.ipynb
```

Before either, `eval/selftest.py` exercises the whole scoring and reporting path
on simulated predictions — no torch, no weights, a few seconds — and asserts
that the three possible verdicts come out of the three behaviours that should
produce them. Run it after touching `evaluate.py` or `report.py`.

Upload that one notebook to Colab — it carries `evaluate.py` and `report.py`
inside. Or run locally if the weights are reachable:

```bash
python eval/evaluate.py --weights /path/to/cardiosentry_runs_v1
python eval/report.py
```

### Scoring a v2 / hybrid run — `eval/evaluate_run.py`

Everything from `baselines_v2` onward names its run folder `<model>@<dim>` and
picks its input geometry from the `CS_DIM` environment variable, which
`config.py` reads **once, at import**. `evaluate.py` cannot resolve that for
itself, so `evaluate_run.py` is the front end: it finds the run, reads the
geometry off the run's own `result.json`, sets `CS_DIM`, and only then imports
the training code. Scoring — cohorts, per-head metrics, bootstrap CIs — is
`evaluate.py`'s, imported and reused, and `report.py` writes the markdown.

```bash
python eval/evaluate_run.py                        # efficientnetv2_s_cbam@nat768
python eval/evaluate_run.py resnet50_cbam@nat768   # any other run
python eval/evaluate_run.py effnet... resnet...    # several, into one table
python eval/evaluate_run.py resnet50_cbam@nat768 --merge   # add to what is there
python eval/evaluate_run.py --list                 # what is scoreable
```

`--merge` is how a table gets built one model at a time: without it each
invocation replaces `external_metrics.json`, and the comparison the report is
built around loses every row but the newest. It carries forward the runs already
scored into `--out`, and refuses to when the stored geometry, config fingerprint
or corpus differs from this invocation's — a table that puts a `nat768` score
beside an `sq768` one reads as a comparison and is not one. Scoring several runs
in one command needs no flag.

The model is the first positional argument — a folder name under
`prototype-results/`, or a path to a run directory anywhere. `DEFAULT_RUN` at
the top of the file is what runs when nothing is passed. Useful flags:

| flag | for |
|---|---|
| `--code` | training package, default `prototype-code/cardiosentry_hybrids_v2` |
| `--runs-root` | where run folders live, default `prototype-results/` |
| `--limit N` | smoke test on N sheets; output is quarantined in `_smoke/` |
| `--batch`, `--workers`, `--device` | inference only; they do not touch a number |
| `--bootstrap` | resamples for the CIs, default 2000 |
| `--merge` | keep runs already scored into `--out` (see above) |

Results land in `results/<code package>/` — `external_metrics.{json,csv}`,
`predictions/<run>.csv`, `EXTERNAL_VALIDATION_REPORT.md` — so a v2 run never
overwrites the v1 numbers.

Three things it refuses to do, each because doing them would produce a number
that looks fine and means nothing:

- **guess the geometry.** It comes from the run, and the run's config
  fingerprint is checked against this checkout's. A `nat768` checkpoint fed
  768×768 squares scores badly for a reason unrelated to the question.
- **re-tune thresholds.** `best.pt` carries the ones frozen on its original
  validation split, and those are applied unchanged.
- **score two geometries in one invocation.** `config.py` fixes the input size
  at import, so one process scores one geometry.

The header redaction is likewise not optional: the printed record ID is a STEMI
leak, the model was trained never to see it, and un-redacting at test time would
show it a band it has never been given.

## Design decisions worth knowing about

**The renderer is not reimplemented.** `02_render_images.sh` shells out to the
same `image_pipeline/stage{1,2,3,4}.py`, same `config.py`, same
`ecg-image-kit`, same interpreter that built the training corpus. If these
sheets came from a second implementation, a model scoring badly on them would be
evidence of a rendering mismatch and the experiment would answer nothing.

**Sheet numbering is shuffled.** Stage 2 renames records to `ECG%06d` in read
order and the kit *prints that number on the sheet*. Written in source order the
printed ID would be a near-perfect proxy for source and label, rendered straight
into the pixels. Stage 5's leakage probe measured it: **0.562 source accuracy
before shuffling** (chance 0.406), **0.491 after** (chance 0.519) — i.e. after
shuffling, source is not recoverable from the sheets at all.

**Thresholds are never re-tuned.** Each `best.pt` carries thresholds frozen on
its original validation split and those are applied unchanged. Re-tuning here
would be tuning on test, and would hand a model that ranks external positives
barely above chance an operating point that manufactures sensitivity.

**AF, LVH and NORMAL are the control, and they are load-bearing.** They had
multi-source training, so they carry no confound and ride on the identical
sheets. A STEMI collapse means nothing without them:

| STEMI | Control heads | Conclusion |
|---|---|---|
| collapses | hold | **confound confirmed** — failure specific to the confounded head |
| collapses | collapse | **inconclusive** — generic domain shift |
| holds | hold | **transfers** — confound was in the data, model did not use it |

## Reading the results

- **Eight gold positives.** Every LUDB interval is wide. Enough to detect a
  collapse; nowhere near enough to certify a head works. Quote intervals.
- **`mean p(pos)` when sensitivity is zero.** A model can rank external
  positives correctly and still fire on none, because its frozen threshold sits
  above every probability it assigned. Calibration failure ≠ blindness, and the
  fixes differ.
- **Perfect specificity on the mimics is not automatically good news.** A model
  that learnt "Chongqing paper ⇒ STEMI" scores zero false positives on every
  mimic here, because none are on Chongqing paper. Only meaningful alongside
  the true-positive rate.
- **Prevalence differs.** LVH is ~54% in LUDB. AUROC is prevalence-independent;
  AUPRC, F1 and precision are not, and do not compare across cohorts.

## Layout

```
build/
  00_fetch_missing.py     label CSV + missing PTB-XL records
  01_build_metadata.py    EVERY clinical labelling decision
  02_render_images.sh     drives the existing pipeline, two passes
  03_merge.py             stitch passes, undo the render sentinel
  04_package.py           tar + PROVENANCE.json
eval/
  evaluate.py             score checkpoints, frozen thresholds, bootstrap CIs
  evaluate_run.py         score a <model>@<dim> run - resolves geometry, reuses
                          evaluate.py's scoring, then calls report.py
  report.py               metrics JSON -> markdown, computed verdict
  selftest.py             exercise the whole path on simulated predictions
  make_notebook.py        self-contained Colab notebook
data/                     built corpus (gitignored)
results/                  metrics + report
```

## Sources

- LUDB — Kalyakulina et al., *LUDB: A New Open-Access Validation Tool for ECG
  Delineation Algorithms*, IEEE Access 8:186181, 2020.
  <https://physionet.org/content/ludb/1.0.1/>
- Curated PTB-XL selection — <https://github.com/dgedon/nstemi_prediction>,
  from Gustafsson et al., *Deep learning ECG-based prediction of myocardial
  infarction in emergency department patients*, Sci Rep, 2022.
- PTB-XL — <https://physionet.org/content/ptb-xl/1.0.3/>
