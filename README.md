# CardioSentry

Image-space multi-label classifier for printed 12-lead ECG sheets, and the
tooling that produced it.

This repository holds **source and notebooks only**. Trained checkpoints,
rendered image corpora, evaluation dumps and the licence-gated source datasets
are deliberately excluded — see [`.gitignore`](.gitignore), which documents how
to regenerate each one.

| Directory | What it is |
|---|---|
| `dataset-combination-code/` | Dataset pipeline (`ecg-dataset-pipeline/`): harmonises the source datasets and renders the synthetic paper-ECG corpus |
| `dataset-staging/` | Per-source staging: decodes waveforms and harmonises labels |
| `prototype-code/` | Training/evaluation notebooks for the classification experiments |
| `experiments/` | Experiment scripts: external validation, filter-bandwidth and AF × LVH error analysis |
| `system_architecture_image/` | Architecture diagrams |

## The web application (separate repository)

The web application — photograph a real ECG sheet on a phone, rectify it to the
training canvas, and score it with any checkpoint through a preprocessing chain
that byte-for-byte mirrors training — lives in its own repository,
[lindtseyyy/cardiosentry-application](https://github.com/lindtseyyy/cardiosentry-application),
and is ignored by this one. The full design rationale is in
[CARDIOSENTRY_WEB_APP_PLAN.md](CARDIOSENTRY_WEB_APP_PLAN.md).

Check it out **inside** this repository's root: its tests and setup steps read
`../prototype-code/` and `../prototype-results/` by relative path.

| Folder | Branch | Storage |
|---|---|---|
| `cardiosentry-application/` | `main` | PostgreSQL |
| `cardiosentry-application-local/` | `local-json` (a `git worktree` of the same repository) | Local JSON files, for presentation hosting |

```bash
git clone https://github.com/lindtseyyy/cardiosentry-application.git cardiosentry-application
git -C cardiosentry-application worktree add ../cardiosentry-application-local local-json
```

## Not in this repository

| Excluded | Why | How to get it |
|---|---|---|
| `dataset-files/` | Licence-gated (PhysioNet credentialed / DUA); never redistributed | Obtain from source, then point `config/paths.yaml` at it |
| `prototype-results/` | ~4.0 GB of checkpoints and evaluation output | Rerun the notebooks in `prototype-code/` |
| Rendered image corpora | ~1.1 GB, reproducible | Rerun `dataset-combination-code/.../image_pipeline/` |
| `cardiosentry-application/`, `cardiosentry-application-local/` | Separate repository (see above) | Clone it as shown above |
