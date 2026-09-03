# Photo-realistic ECG rendering — v1 validation set

5 validation images, built from v3 signals, revising the synthetic ECG renderer
against a real smartphone photograph of a printed ECG
(`experiments/actual_ecg_image sample/ecg_sample.jpg`, Xiaomi POCO X6 5G, 4624x2600).

**Nothing was regenerated.** The v2 image corpus and its pipeline are untouched;
every change is either a new file or an environment-gated default.

## What was wrong, measured

Every number below came from measuring the two images, not from judgement.

| property | real photo | old synthetic | now |
|---|---:|---:|---:|
| px per mm on the sheet | 15.7 | 5.91 | 7.6 (`ECG_PHOTO_LONG_EDGE` raises it) |
| 1 mm grid present? | yes | **no — destroyed by stage 4** | yes |
| trace stroke width | 0.19 mm | 0.34 mm | 0.21 mm |
| trace core grey | 1–3 | 3 | 13–18 |
| paper–trace contrast | 186 | 165 | 110–142 |
| paper level (p99) | 207 | 221 | 202–227 |
| paper R−B (warmth) | +7.0 | +15.2 | +7.2 |
| trace R−B | +8.0 | — | +0.3…+16.1 |
| background visible? | yes | **no — 18 px paper border** | yes |

Two findings drove most of the work:

1. **The old stage 4 was erasing the 1 mm grid.** At 150 dpi a 1 mm square is
   6 px; defocus plus JPEG 4:2:0 wiped it, leaving only the 5 mm lines. That is
   why the old sheet read as a coarse digital plot.
2. **"Printed ink is never pure black" was wrong.** The reference's trace core
   reaches grey 1–3 — a thermal stylus lays down an essentially black line. An
   ink value of 32 produced a washed-out trace. What is *not* uniform is the
   density along the stroke, which is now modelled explicitly.

## The chain

The renderer is now ordered as the physical process, which is what makes the
result read as photographed rather than noised:

```
ECG signal (0.05–150 Hz)
  -> PRINT    ecg-image-kit at 300 dpi, 0.40 pt ink, grey grid
  -> PRESS    ink separated into trace + grid layers, then per-layer physics:
              density mottle, fibre grain, diffusion, bleed, Beer-Lambert opacity
  -> SHEET    paper stock: colour, fibre, high-passed wrinkle field, fold lines
  -> CAMERA   pose (roll/yaw/pitch/distance -> true keystone), placement on a
              surface with a cast shadow, illumination, DOWNSAMPLE to the frame,
              lens blur, vignette, chromatic shift, shot+read noise, JPEG
```

**The downsample is the load-bearing step.** Area-averaging a 300 dpi sheet into
a 2600 px frame is what a lens and sensor physically do, and it is what removes
the hard vector edges. The old pipeline rendered at the output resolution and
blurred afterwards, which just produces a blurry vector graphic.

## Parameters changed

### Physical scale — and why the layout was already correct
The kit's geometry was *not* wrong. Paper is 11 x 8.5 in; `row_height =
(8.5 x 0.5 mV) / (5/25.4 in) / (rows+2) = 3.60 mV = 36.0 mm`, against ~33 mm
measured on the reference. 25 mm/s and 10 mm/mV were already physically
consistent with the grid, and the waveform-to-grid ratio was already correct.
**No gain, scale or signal value was altered.** The "too zoomed in" impression
came from resolution and framing, not from scaling — so the fix is resolution
and framing.

| parameter | old | new |
|---|---|---|
| render DPI | 150 (5.91 px/mm) | **300 (11.81 px/mm)**, `ECG_DPI` |
| output long edge | 1686 px | **2600 px** (`ECG_PHOTO_LONG_EDGE`) |
| output px/mm on sheet | 5.91 | **7.6** (3400 -> 10.2, 5000 -> 15.0) |
| trace stroke | 0.75 pt = 0.34 mm | **0.40 pt = 0.21 mm** |
| grid stroke | 0.5 pt | **0.40 pt** |
| paper margin | 18 px paper-coloured border | **sheet spans 0.84–0.99 of frame, real background** |

### Ink (new)
| parameter | value |
|---|---|
| trace ink RGB | `(17, 13, 10)` — warm, cancels 4:2:0 chroma pulling thin strokes blue |
| trace peak alpha | 0.945–0.995, Beer-Lambert `1-exp(-k*amount)` |
| trace density mottle | ±10–24 % at 3–14 mm scale |
| trace fibre grain | ±4–10 % at 0.25 mm |
| ink diffusion | sigma 0.028–0.055 **mm** (resolution-independent) |
| ink bleed | 5–13 % into a 2.4x halo |
| grid peak alpha | 0.30–0.52 (grid is much lighter than the trace) |
| grid mottle | ±8–20 % at 8–30 mm (roller pressure) |
| grid palettes | red 0.44, orange 0.22, pink 0.14, grey 0.20 |

The major gridline needs no emphasis knob: the kit renders major at grey 102 and
minor at 191, so measured against paper the major already carries ~2.5x the ink,
and Beer-Lambert darkens it by exactly that ratio for free.

### Camera (new)
| parameter | value |
|---|---|
| roll | ±2.8° |
| yaw / pitch | ±7.0° / ±5.5° — a true pinhole projection, so the far edge is genuinely shorter |
| viewing distance | 2.6–4.5 sheet-widths |
| sheet fill of frame | 0.84–0.99 |
| frame aspect | 1.45–1.62 |
| illumination ramp | 6–17 %, radial falloff 4–13 %, edge shadow 8–22 % (p=0.55) |
| paper target (p98, pre-camera) | 222–248, landing near the reference's 205 after vignette |
| cast shadow | offset 1–4 mm, blur 1.5–5 mm, strength 0.25–0.55 |
| lens blur | 0.35–0.95 px, at output resolution |
| vignette | 3–10 % |
| chromatic shift | 0–0.28 px |
| sensor noise | read 1.2–3.0, shot 0.55–1.35 x sqrt(signal) |
| JPEG | quality 78–90, 4:2:0 |

### Paper (retuned)
The kit's 19 crease photographs carry large dark blobs as well as fine texture;
composited whole they read as a stained sheet. They are now **high-passed** at
7 mm — the blob scale is subtracted, the fine relief kept — at 1.4–3.6 %
amplitude, against the 16–38 % `stage4_augment.py` used.

## The resolution trade-off

Paper–trace contrast still reads 110–142 against the reference's 186. This is
sampling, not the ink model, and it was measured directly:

| output | px/mm | contrast | size |
|---|---:|---:|---:|
| 2600 px (default) | 7.6 | 126 | ~0.8 MB |
| 3400 px | 10.2 | 139 | ~1.9 MB |
| 4200 px | 12.3 | 122* | ~2.9 MB |
| reference | 15.7 | 186 | 6.9 MB |

A physically-correct 0.19 mm stroke is 1.4 px wide at 7.6 px/mm, and a 1.4 px
stroke cannot hold the density of a 3 px one — the downsample averages it with
paper. **Raising `ECG_PHOTO_LONG_EDGE` is the only honest fix**; thickening the
stroke would buy contrast by making the print wrong. At 37 k records the cost is
~30 GB at 2600 px vs ~70 GB at 3400 px, so the default is deliberate and the
knob is one environment variable. (*the 4200 row was measured before the ink
correction and is not comparable; the 2600/3400 rows are current.)

## The 5 validation images

`validation/` — see `index.csv` for the full per-image effect record.

| file | record | source | label | palette | JPEG q |
|---|---|---|---|---|---|
| ECG000001_a0.jpg | CUHK_002942 | CUHK | **STEMI+LVH** | grey | 90 |
| ECG000002_a0.jpg | CUHK_000477 | CUHK | AF | pink | 89 |
| ECG000003_a0.jpg | PTBXL_000002 | PTBXL | NORMAL | red | 82 |
| ECG000004_a0.jpg | CHAPMAN_000001 | CHAPMAN | AF | orange | 78 |
| ECG000005_a0.jpg | STEMI_000004 | STEMI | STEMI | grey | 89 |

`comparison_real_vs_synthetic.jpg` — real / old / new, full sheets.
`comparison_detail_42mm.jpg` — the same 42 mm of paper from each, which is where
the 1 mm grid loss and the stroke-width difference are obvious.

Every random draw is keyed on `(record, variant)` only, never on the class, so
no visual property can carry label information.

## Files

| file | status |
|---|---|
| `image_pipeline/photo_config.py` | **new** — the print/camera model |
| `image_pipeline/stage4_photo.py` | **new** — replaces stage4_augment for this build |
| `image_pipeline/patch_kit.py` | **modified** — stroke widths now read the environment; unset reproduces stock |
| `image_pipeline/config.py` | **modified** — `DPI` reads `ECG_DPI`; default still 150 |
| `image_pipeline/stage4_augment.py` | untouched — v2 stays reproducible |

## Reproducing

```bash
./run.sh
```

## Not done, on purpose

The corpus was **not** regenerated. `stage4_photo.py` is not wired into
`run_all.sh` and `stage4_augment.py` is still the default, so nothing changes
until the rendering is signed off.

## Known gaps

- Paper–trace contrast is below the reference at the default resolution; see
  the trade-off table.
- The reference's trace carries fine baseline tremor (muscle noise). Ours does
  not, because the *signal* is clean — that is a property of the recording, not
  of the renderer, and faking it would corrupt the waveform.
- The background is procedural. It reads as an out-of-focus surface, which is
  what the reference shows, but it is not a photograph of a real desk.
