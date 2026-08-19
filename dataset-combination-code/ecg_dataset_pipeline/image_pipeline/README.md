# image_pipeline — synthetic paper-ECG image generation

Turns the record CSV this repo already produces into a corpus of **synthetic
12-lead paper ECG images**, for training a direct 2D vision model.

```
data/balanced/balanced_ecg_metadata.csv
        │
        │  stage1  adapt: decode age/sex, resolve paths, assign splits
        │  stage2  transcode: one WFDB dialect, band-pass, anonymise
        │  stage3  render:    ecg-image-kit, clean sheets, parallel
        │  stage4  augment:   paper texture, creases, tilt, lighting, JPEG
        │  stage5  verify:    integrity, balance, pixel stats, leakage probe
        ▼
build/images/<split>/<class>/ECG######_r<k>.jpg   +   build/index.csv
```

Extracted from `ECG-diagnosis-prototype/`, which built the same corpus but
resolved labels from the three raw catalogues itself. That part is gone: label
decisions now arrive as data, from `config/label_mapping.yaml` via the dataset
pipeline. Everything downstream of "which records, which labels" is unchanged,
including the reasons behind each render parameter.

**Nothing has been generated yet.** Every stage has been run far enough to
verify it works — stages 1, 2 and 4 on real records, stage 3 on one real sheet
through the actual kit — but no corpus exists in `build/`.

---

## 1. Setup

`../ecg-image-kit/` is **already cloned and already patched** (631 MB, shallow
clone, gitignored). To redo it from scratch:

```bash
git clone --depth 1 https://github.com/alphanumericslab/ecg-image-kit.git ../ecg-image-kit
python patch_kit.py                              # idempotent
```

What is *not* done yet is the interpreter. The kit needs its own — Python 3.11
with NumPy pinned to 1.26, because `imgaug` calls `np.sctypes`, removed in
NumPy 2.0. This is *separate from* the dataset pipeline's `.venv` on purpose,
and it pulls TensorFlow, so budget ~2 GB and 10-15 minutes:

```bash
bash install.sh                                  # builds ~/.cache/ecgkit-venv
```

`install.sh` also pins both `opencv-python` and `opencv-python-headless` below
5.0 for the same ABI reason — `imgaug` pulls the former in transitively and an
unpinned install drags NumPy 2.x back in through the side door.

Overrides: `VENV=` (install location), `ECGKIT_PY=` (interpreter used by the
stages), `ECGKIT_DIR=` (kit checkout), `ECG_IMAGE_BUILD=` (output root).

`patch_kit.py` applies the one required source change to the kit: it insets the
printed header away from the paper corner, where rotation and cropping would
otherwise eat it. Stage 3 refuses to render until it has been applied.

---

## 2. Run it

```bash
./run_all.sh                                     # full build
./run_all.sh 3                                   # resume from stage 3
CSV=../data/harmonized/harmonized_ecg_metadata.csv ./run_all.sh
WORKERS=4 ./run_all.sh
```

Stages are idempotent — 3 and 4 skip work that already exists — so an
interrupted build restarts with the same command. Before resuming, check for
orphaned kit workers left behind by an interrupted run:

```bash
pgrep -af "stage3_render|gen_ecg_images|stage4_augment"
```

Smoke-test a slice first rather than discovering a problem four hours in:

```bash
python stage1_manifest_from_csv.py --limit 40
python stage2_transcode.py
python stage3_render.py --limit 2
python stage4_augment.py
```

At the default 1 render per record the balanced CSV gives 9,306 images; see
§4b for what that costs and why more cores, not a GPU, is the lever.

---

## 3. Input

Default `../data/balanced/balanced_ecg_metadata.csv`, falling back to the
harmonized CSV. Any CSV works if it carries the columns named in
`config.CSV_COLUMNS` — change the *names there*, not the stage code:

| Role | Default column | Used for |
| --- | --- | --- |
| record id | `global_record_id` | provenance; never printed on the sheet |
| source | `source_dataset` | per-source sex decoding, leakage reporting |
| patient / group | `split_group_id` | split grouping — records sharing it stay together |
| labels | `STEMI` `AF` `LVH` `NORMAL` | copied through verbatim to `index.csv` |
| class name | `cardiosentry_label` | directory name only (`AF+LVH` → `AF_LVH`) |
| signal | `signal_path` | WFDB base path, no extension |
| metadata | `age`, `sex` | printed on the sheet |
| split | `split` *(optional)* | honoured if present, otherwise assigned |
| validity | `validation_status` *(optional)* | filtered against `ACCEPT_VALIDATION` |

A **versioned release** from `scripts/export_dataset.py` works directly, and is
the better input if you want the corpus to be reproducible from an archived
snapshot rather than from whatever is on disk today — relative signal paths are
resolved against the CSV's own directory:

```bash
python stage1_manifest_from_csv.py \
    --input-csv ../data/releases/v1/balanced/balanced_ecg_metadata.csv
```

On the current balanced CSV, stage 1 keeps **9,306 of 9,330** records. The 24
losses are all Chapman rows with a missing or zero `age`: the kit's
`--print_header` does a bare dict lookup for `Age` and `Sex` and raises
`KeyError` if either is absent, so a record without both cannot be rendered.

### What stage 1 has to fix up

- **Sex is encoded three different ways, two of them opposite.** PTB-XL uses
  `0 = male`, the Chongqing ACS set uses `1 = male`, Chapman writes the words.
  The mapping is per source in `config.SEX_DECODERS`; a bare integer is
  meaningless without knowing which catalogue it came from.
- **PTB-XL masks any age over 89 as 300.** Printed as 90, not dropped — the
  patient is genuinely over 89.
- **Splits are assigned here, grouped and stratified.** The dataset pipeline
  supplies `split_group_id` but no split. Whole groups move together, and
  stratification is per class, because assigning globally lets PTB-XL's forced
  folds eat the test budget and starves the classes PTB-XL does not contribute
  to. PTB-XL's own `strat_fold` (dug out of `source_extra_json`) is honoured:
  fold 10 → test, fold 9 → val. Stage 1 asserts no group spans two splits.
- **Multi-label records survive as multi-label.** 256 `AF+LVH` records get one
  directory of their own; the four binary columns in `index.csv` remain the
  training truth. `RENDERS_PER_RECORD` takes the largest value among a record's
  positive classes.

---

## 4. Output

```
build/images/<split>/<class>/ECG######_r<k>.jpg
build/index.csv    image_path, record, record_id, original_id, cls,
                   STEMI, AF, LVH, NORMAL, split, source, render_k,
                   patient_id, age, sex, note
```

Train from `index.csv` — the four binary columns, not the directory name.

| Property | Value |
|---|---|
| Layout | 3×4 with continuous lead II rhythm strip |
| Paper speed / gain | 25 mm/s, 10 mm/mV |
| Signal band | 0.05–40 Hz |
| Resolution | 150 DPI → 1686 × 1311 px (1722 × 1347 after margin) |
| Format | JPEG q85, 4:2:0 |
| Grid | grey — major `(0.4,0.4,0.4)`, minor `(0.75,0.75,0.75)`, black trace |
| Printed metadata | anonymised ID, age, sex, inset from the corner |
| Calibration pulse | on every sheet |
| Margin | thin (18 px) |

Each sheet then gets **paper-level environmental noise** — the damage a physical
printout accumulates, not noise on the waveform:

| Effect | Range |
|---|---|
| Paper texture | photographs of real wrinkled paper, as a shading field (every sheet) |
| Creasing | 1–3 hard fold lines with a bright ridge and shadow side, plus broad bowing (80%) |
| Tilt | ±3°, rotated on an expanded canvas filled with the sheet's own paper colour |
| Perspective | slight keystone, as if photographed rather than scanned (70%) |
| Lighting | directional ramp, corner falloff, off-sheet edge shadow |
| Exposure / contrast / white balance | mild, neutral-centred |
| Sensor noise | σ 1.5–4.5 |

All of it is bounded so the trace survives, and every effect is drawn from one
distribution shared by all classes — style must never correlate with class, or
the model learns the renderer instead of the ECG. In the prototype's 48-variant
sweep, paper-to-trace contrast stayed between 230 and 252 of a clean sheet's
250 greyscale levels; stage 5 re-checks that on the built corpus.

---

## 4b. Cost, and why a GPU will not help

**This workload is CPU-only. There is no GPU path to enable.** Measured on the
real kit, this repo's flags, one real staged record:

| Step | Per image, 1 core | What it actually does |
| --- | --- | --- |
| stage 3 render | **3.8 s** | matplotlib Agg rasterisation |
| stage 4 augment | **1.5 s** | NumPy / Pillow array maths |

A `cProfile` of the render is entirely matplotlib: Agg text metrics, tick
generation (the 1 mm grid becomes ~1,000 tick objects per sheet), `Line2D.draw`
and coordinate transforms. Nothing in it is a tensor operation.

The kit imports TensorFlow exactly once, in `HandwrittenText/generate.py`, for
the handwritten-annotation feature this pipeline does not use — and that module
pins itself to CPU with `device_count={'GPU': 0}`. The import still costs ~7 s
per kit invocation because `gen_ecg_image_from_data.py` imports it
unconditionally at module top; that is why `CHUNK_SIZE` is 400 rather than
something smaller.

So the only lever is **core count**, via `-j/--workers`. At 1 render per record
(9,306 images):

| Cores | stage 3 | stage 4 | Total |
| --- | --- | --- | --- |
| 2 | ~5.0 h | ~1.9 h | ~7 h |
| 4 | ~2.5 h | ~1.0 h | ~3.5 h |
| 8 | ~1.2 h | ~0.5 h | ~1.7 h |

Budget ~750 KB per final JPEG — about 7 GB for the full corpus, plus ~3 GB of
intermediate PNGs under `build/rendered/` that stage 4 does not delete.

**On Colab:** a GPU runtime buys nothing here, and Colab's CPU allocation
(commonly 2 vCPUs — check with `!nproc`) is likely *fewer* cores than a laptop.
Renting a many-core CPU box would help; a T4 would not. If the render ever does
need to be faster, the profile says to attack the grid: drawing it as a
`LineCollection` or a pre-rendered tile instead of ~1,000 matplotlib ticks is
where the time is.

---

## 4c. What each record produces

At the default `RENDERS_PER_RECORD = 1`, one record gives you **two files, one
of which is the corpus**:

| File | Where | In `index.csv`? |
| --- | --- | --- |
| clean render `<record>-0.png` (1650×1275) | `build/rendered/<cls>/r0/<chunk>/` | no — intermediate |
| annotation `<record>-0.json` | same directory | no — intermediate |
| **augmented** `<record>_r0.jpg` (1686×1311) | `build/images/<split>/<cls>/` | **yes** |

Stage 4 does not delete the clean renders, so after a build you have both the
pristine sheet and the paper-damaged one for every record — but only the
augmented JPEG is indexed, and only it is what you train on. There is no
"clean" row in `index.csv`.

If you want clean images *in* the corpus, that is a stage 4 change: emit the
untouched render as a second row alongside the augmented one. Nothing upstream
needs to move.

### Raising the render count buys less than it looks

Set `RENDERS_PER_RECORD["STEMI"] = 4` and you get 4 clean PNGs and 4 JPEGs per
STEMI record. But the four *clean* sheets are near-identical: with the flags in
`KIT_RENDER_FLAGS`, resolution, padding, calibration pulse, grid presence, grid
palette, layout and header printing are all forced, and the kit's seed does not
work (see §5). The only thing that varies is the typeface, from an unseeded
draw over the kit's 10 fonts.

**All the real variety is stage 4's** — tilt, keystone, paper texture, creases,
lighting, exposure, noise — and that layer *is* properly seeded, on
`(record, render_k)`, so it is reproducible and genuinely different per `k`.
So multiple renders per record works as augmentation, but you are paying the
3.8 s render cost again for a sheet you could have re-augmented for 1.5 s.
Re-augmenting one clean render at several seeds would be the cheaper design.

---

## 4d. Splits, and why evaluation gets exactly one image per ECG

The split is frozen upstream, in the dataset release itself:

```bash
python ../scripts/freeze_splits.py --release v2
```

That writes a `split` column into `harmonized_ecg_metadata.csv` and
`balanced_ecg_metadata.csv` and never recomputes it. Stage 1 detects the column
and uses it as data; it only assigns a split when the CSV brings none. A model
evaluated today is therefore comparable with one evaluated after any later
rebuild of the image corpus.

**Grouping.** Splits are assigned over whole `split_group_id` groups -- the
patient wherever a patient id exists. Verified on v2: 10,721 real patients, none
spanning two folds, zero group leakage across both cohorts. PTB-XL's own
stratified folds 9 and 10 are honoured as val and test.

**The Chapman limitation, stated plainly.** Chapman/Ningbo ships no patient
identifier, so its 7,791 records are grouped one per record. That is the safest
grouping the data permits, but if the same patient was recorded twice in
Chapman, those two ECGs can land in different folds. This residual leakage
cannot be detected or prevented from the data as distributed, and belongs in
the thesis limitations.

**Variants per split** (`config.VARIANTS_PER_SPLIT`):

| split | records | variants | images |
| --- | ---: | ---: | ---: |
| train | 7,406 | 4 | 29,624 |
| val | 924 | **1** | 924 |
| test | 939 | **1** | 939 |
| total | 9,269 | | **31,487** |

Train may be expanded freely: more photographs of one sheet is honest
augmentation when the model never meets those records at evaluation time.

`val` and `test` are pinned to exactly one image per ECG and must stay there.
Two variants of one recording in a test set silently double that patient's
weight in every metric, so the score starts depending on which ECGs happened to
draw more variants. The pin is enforced twice -- in stage 1 after splits are
resolved, and again in `stage4._variants_for()` -- because the guarantee should
not depend on one file being edited correctly.

## 4e. Grid colour, and where it is applied

Real ECG paper is printed with a coloured grid and a black trace. The kit can
only pick a grid palette per *invocation*, and each invocation pays a ~7 s
TensorFlow import, so per-image colour through the kit would mean one process
per image. Every sheet is therefore rendered once in the kit's grey `bw` style
and **stage 4 recolours the grid per image** with a 256-entry lookup table.

This works because the rendered PNG is pure greyscale with the grid on two
exact levels. Measured on a real render: 23.3% of pixels at 191 (minor grid),
5.1% at 102 (major grid), 1.4% below 60 (trace and text), 66.8% paper. The LUT
interpolates between those control points, so antialiased pixels blend smoothly
instead of banding, and everything below `GRID_TRACE_MAX` stays neutral -- the
recolour never tints the trace the model has to read.

Four palettes, weighted red 0.38 / orange 0.22 / pink 0.20 / grey 0.20, each
with per-image ink, paper and saturation jitter.

**Tuning note.** The minor gridline is 23.3% of the sheet and the major only
5.1%, so the *minor* colour is what the eye reads as the colour of the paper. A
first pass used washed-out minors and every sheet came out looking pink or
sepia regardless of which palette it drew. The minors are now kept saturated
and separated in hue.

**Colour cannot leak the label.** The palette is drawn from the same per-image
RNG as every other effect, seeded on output identity only, never on class.
Simulated over the real 31,487-image corpus:

| class | red | pink | orange | grey |
| --- | ---: | ---: | ---: | ---: |
| AF | 38.5% | 19.6% | 22.3% | 19.6% |
| AF+LVH | 38.6% | 20.1% | 20.2% | 21.1% |
| LVH | 37.5% | 20.3% | 21.8% | 20.3% |
| NORMAL | 38.6% | 19.2% | 22.0% | 20.2% |
| STEMI | 38.2% | 20.2% | 21.3% | 20.3% |
| *target* | *38.0%* | *20.0%* | *22.0%* | *20.0%* |

Every class is within ~1% of target. `index.csv` records the palette per image
as `grid_palette`, so this is auditable on the real corpus, not just in
simulation.

## 4f. The full effect list

Applied in physical order -- ink, then geometry, then the surface, then the
lens, then the sensor:

| effect | detail |
| --- | --- |
| grid colour | red / pink / orange / grey, per-image LUT, ink+paper+saturation jitter |
| paper tilt | ±3°, rotated about centre on a paper-coloured field |
| perspective | keystone, 0.4-1.8% corner displacement, p=0.7 |
| wrinkles | 19 real wrinkled-paper photographs as a multiplicative shading field |
| creases | 1-3 fold lines, derivative-of-Gaussian profile, p=0.62 |
| bowing | 1-2 low-frequency sine terms -- sheets are never flat |
| lighting | directional ramp + corner falloff |
| edge shadow | exponential falloff from one edge, p=0.65 |
| white balance | mild, neutral-centred (σ=1.2%) |
| exposure/contrast | ±6% / ±8% |
| paper level | compounded shading renormalised to a plausible paper luminance |
| defocus | Gaussian 0.3-0.9 px, p=0.55 |
| sensor noise | Gaussian σ=1.5-4.5 (0-255) |
| JPEG | q85, 4:2:0 |

Bounds are deliberately tight. At 150 DPI the ST segment is only a few pixels
tall, so a blur or a noise floor big enough to be obvious is big enough to
erase the feature STEMI depends on. Verified at 1:1 crops: the trace stays
black and crisp under every palette.

**Why `_normalise_paper` exists.** The shading effects each look mild on their
own but they *multiply*, and an unlucky draw compounded ramp x falloff x edge
shadow x exposure into roughly 0.5x, producing sheets that read as dark grey
card. The finished sheet's 95th-percentile luminance is now rescaled into
`PAPER_LEVEL_TARGET`, which bounds the product while preserving each effect's
relative shading.

## 4g. Annotations, and how they survive augmentation

With `--store_config --lead_bbox --lead_name_bbox` the kit writes a JSON beside
every rendered sheet:

```
sampling_frequency, width, height, resolution, x_grid, y_grid
leads[13]  →  lead_name
              lead_bounding_box     tight box around the drawn trace
              text_bounding_box     box around the printed lead label
              start_sample/end_sample   which slice of the 5,000 samples
              plotted_pixels        every (y, x) point of the trace
```

13 entries: the 12 lead panels plus the full-mode lead II rhythm strip.

**Coordinate convention** (verified against a real render, not assumed): every
coordinate is a `[y, x]` pair with y measured DOWN from the top-left of the
image the JSON describes. The kit builds them as `[height - y2, x1]`, i.e. it
has already flipped matplotlib's bottom-up axis for you.

**The problem this solves.** The kit measures those coordinates on the CLEAN
render. Stage 4 then rotates the sheet, warps it and adds a margin, so the
numbers stop describing the image the model actually sees. Overlaying them on
an augmented JPEG put every box visibly off the trace.

Stage 4 now records the exact geometry it applied — rotation angle, the
perspective quad, the margin — and replays the same maths on the coordinates:

1. rotation about the centre (`_rot_points`; PIL turns the image CCW, which in
   a y-down system is `x' = x cos + y sin`, `y' = -x sin + y cos`);
2. the projective map sending the QUAD's source corners to the destination
   rectangle (`_homography` / `_apply_h`);
3. the margin offset.

Every augmented image therefore gets its own `<image>.json`, correct for that
image, and `index.csv` carries `rotation_deg` and `has_annotation`.

**Inspecting it.** `plot_annotations.py` overlays the boxes and the traced
pixel path on any sheet, clean or augmented:

```bash
python plot_annotations.py path/to/ECG000123_a0.jpg
python plot_annotations.py sheet.jpg --lead V2 --zoom --stride 1
python plot_annotations.py sheet.jpg --what boxes -o check.jpg
```

It finds the JSON automatically and warns if the annotation's declared size
does not match the image, which is the signature of a stale annotation.

**What they are worth.** `plotted_pixels` plus the original WFDB signal is a
complete supervised dataset for image→signal digitisation. The boxes give you
lead detection for free, let you crop V1–V4 for a STEMI head or the lead II
strip for AF, and let a Grad-CAM figure say *which lead* the model attended to
rather than which corner of the page. `x_grid = 29.528 px` per 5 mm square at
25 mm/s and 10 mm/mV makes the pixel→(seconds, mV) conversion exact.

**Cost.** ~1.8 MB per record, because `plotted_pixels` stores every trace
point — about 17 GB across the corpus, on top of ~23 GB of images and ~2.4 GB
of intermediate PNGs. Dropping the three kit flags saves it, at the price of
re-rendering for hours if you later decide you wanted them.

## 5. Things that looked reasonable and were wrong

Each of these cost real debugging time in the prototype. Don't redo it.

- **The kit's `--augment`/`--wrinkles` flags are broken, not just ugly.**
  `-t/--temperature` is dead code — `gen_ecg_image_from_data.py` hardcodes a
  forced orange-or-blue cast regardless of what you pass, and
  `--deterministic_temp` is declared in argparse and never read. `-noise 0`
  crashes. `--wrinkles` composites its (excellent) wrinkle photographs with a
  full-strength overlay blend that buries the trace even at its lightest
  setting. `iaa.Affine` fills rotation corners with black and, with no margin,
  clips the printed header off the top. Stage 4 reimplements both effects — same
  source photographs, bounded multiplicative blend instead of the broken one.
- **The kit's `-se/--seed` is dead in batch mode.** `run_single_file()` calls
  `random.seed(args.seed)` only under `if hasattr(args, "st")`, and the batch
  driver sets `args.start_index`, never `args.st`; `get_paper_ecg()` then takes
  a `seed` parameter it never reads. Stage 3 still passes it, for the day
  upstream fixes this, but do not expect it to do anything today. The practical
  consequence is in §4c: a second *clean* render of the same record differs
  only by typeface.
- **`.mat` (Chapman) and `.dat` (PTB-XL/STEMI) are not interchangeable in the
  kit.** `.mat` loads as raw ADC counts, `.dat` as millivolts via `wfdb`; the
  kit reads the ADC gain into a variable and never applies it, so Chapman
  renders 1000× over-scale and crashes on write. Stage 2 pushes everything
  through `wfdb` into one dialect before the kit ever sees it.
- **150 DPI, not 200 or 300, and 0.05–40 Hz, not the diagnostic 0.05–150 Hz.**
  Same arithmetic behind both: at 150 DPI and 25 mm/s the paper carries
  147.6 px/s, so image Nyquist is 73.8 Hz. 150 Hz cannot be represented at any
  DPI this corpus renders at and would only alias against the 1 mm grid.
- **Amplitude is never normalised across sources** — that would erase the
  voltage criteria LVH is defined by. Only filtering and baseline are
  harmonised, and identically for every source, because that harmonisation is
  the main lever against the source/class confound.
- **Records are renamed `ECG000001…` before rendering.** The kit prints
  `ID: <record name>` on every sheet, and the native filenames
  (`00001_hr` / `JS00001` / `00101`) announce the source dataset in plain text.
- **Annotation JSON only applies to the clean render, never the final image.**
  Stage 4's rotation/keystone/shading/margin move every pixel without updating
  the sidecar JSON `--store_config` writes. `visualize_annotations.py` compares
  image dimensions against the JSON's recorded width/height and refuses to draw
  on a mismatch. Point any new annotation tooling at `build/rendered/`, never
  `build/images/`.
- **Never write into `build/rendered/<class>/r<k>/<chunk>/`.** Stage 3's resume
  logic counts `*.png` files per chunk against the `.dat` count in the matching
  staged chunk to decide whether that chunk is done. Anything else writing
  there can silently corrupt the count and make a resumed build skip real work.
  `visualize_annotations.py` detects this and redirects to `build/annotated/`.

---

## 6. The confound worth staying honest about

**STEMI is single-source.** Every STEMI image comes from the Chongqing ACS
dataset; nothing else in this project supplies the label (see the main README
§6 for why every alternative was refused). A vision model can learn "which
hospital scanned this" as a proxy for STEMI without ever finding ST elevation.

Mitigations carried over from the prototype: identical signal conditioning
across sources, class-independent render styling, anonymised record IDs.
The prototype additionally drew a third of its NORMAL class from that same
Chongqing cohort to break the 1:1 source↔class mapping — **this pipeline
cannot**, because the dataset pipeline deliberately contributes no NORMAL from
the ACS set (a STEMI-negative record in an ACS-referral cohort is a *different*
acute coronary syndrome, not a healthy control). That is the more defensible
clinical call, and it makes the confound worse, not better: source and STEMI
are now perfectly separable in the corpus.

`stage5_verify.py` runs a patient-grouped source-leakage probe. If it comes
back well above chance, don't trust the STEMI numbers — and report per-source
metrics regardless of what it says.

---

## 7. Files

```
config.py                     every tunable; read this first for "why is X"
patch_kit.py                  the three ecg-image-kit source patches (header
                              inset, lead-name position, rhythm-strip label)
stage1_manifest_from_csv.py   CSV -> render manifest (the new part)
stage2_transcode.py           WFDB harmonisation, band-pass, anonymisation
stage3_render.py              parallel kit invocation (clean renders only)
stage4_augment.py             paper/scan realism; also standalone: -i DIR -o DIR
stage5_verify.py              integrity + balance + leakage probe
run_all.sh                    driver; ./run_all.sh <n> resumes from stage n
install.sh                    builds the Python 3.11 kit venv
visualize_annotations.py      draws the kit's lead/text boxes on a clean render
plot_annotations.py           overlays boxes + traced pixels on ANY sheet,
                              clean or augmented; --lead V2 --zoom to inspect
```

`stage4_augment.py -i some/dir -o out/dir --renders 3` runs the augmentation
alone on a folder of PNGs, without touching the corpus or its index — the fast
loop for tuning the effects.

---

## 8. Licensing

The Chongqing STEMI dataset is CC BY-NC-**ND** 4.0. The rendered corpus is a
derivative work, so it may be used internally but **not redistributed**.
