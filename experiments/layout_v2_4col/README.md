# v2 renderer + layout fix — 5 validation images

Reverted to the **v2 rendering path**: 150 dpi, the kit's stock stroke widths,
`stage4_augment.py`. The photo-realism model (`stage4_photo.py`, 300 dpi,
`photo_config.py`) is left on disk but is **not used** by this build.

The only change to the v2 look is the sheet **layout**.

## What was wrong with the spacing

Measured on a clean v2 render:

| | before | after |
|---|---:|---:|
| header ink | 16.7 – 26.7 mm | 16.6 – 25.6 mm |
| lead row 1 (label / trace) | 77.6 / 90.2 mm | **42.2 / 55.2 mm** |
| lead row 2 | 114.2 / 125.9 mm | 90.4 / 102.1 mm |
| lead row 3 | 150.1 / 160.8 mm | 137.8 / 149.9 mm |
| rhythm strip | 195.5 / 209.0 mm | 194.6 / 208.8 mm |
| **dead band under the header** | **51 mm** | **17 mm** |
| **row spacing** | 36.0 mm | **47.3 mm** |

The kit divides the sheet into `rows + 2` slots but only ever fills `rows` of
them, and the two spare slots land at the **top**. That left a 51 mm empty band
— a quarter of the sheet — between the printed header and the first lead row,
while the traces were crowded into the lower two thirds. That is what read as
"wrong proportions"; the waveform-to-grid scaling was never wrong.

`patch_kit.py` now replaces the row-height formula with an explicit layout:

```python
_top_reserve = float(os.environ.get('ECG_TOP_RESERVE', 5.5))     # plot units = 55 mm
_bottom_reserve = float(os.environ.get('ECG_BOTTOM_RESERVE', 1.9))
y_base = _bottom_reserve
row_height = (_y_full - _top_reserve - _bottom_reserve) / max(1, rows - 1)
```

`y_base` is the rhythm strip's baseline and the anchor the lead rows step up
from, so it replaces every `row_height/2` the kit used for that purpose (four
sites). 25 mm/s and 10 mm/mV are untouched and still physically consistent with
the grid; only the vertical *distribution of rows* changed, which is why the
traces now have 47 mm of room each instead of 36 mm.

## Leads per row

Still **4 per row** (the clinical 3x4 standard, 2.5 s per lead).

While wiring it I found a latent kit bug worth recording. `extract_leads.py`
hardcodes the per-column time offset to `columns == 4`:

```python
if columns == 4 and key in format_4_by_3[1]:
    shiftedStart = start + int(rate*lead_length_in_seconds)
```

With any other column count every branch is skipped, `shiftedStart` stays at
`start`, and all N panels of a row plot the **same opening segment** of their
lead instead of a continuous 10 s. The sheet still looks plausible, which is
what makes it dangerous. `patch_kit.py` now generalises this to a `format_<N>`
table lookup in `config.yaml` — and patches **both** copies of the block, since
the one stage 3 actually reaches is the second.


## The 5 validation images

`validation/` — same 5 records as the previous experiment, so the layouts are
directly comparable.

| file | record | source | label |
|---|---|---|---|
| ECG000001_a0.jpg | CUHK_002942 | CUHK | **STEMI+LVH** |
| ECG000002_a0.jpg | CUHK_000477 | CUHK | AF |
| ECG000003_a0.jpg | PTBXL_000002 | PTBXL | NORMAL |
| ECG000004_a0.jpg | CHAPMAN_000001 | CHAPMAN | AF |
| ECG000005_a0.jpg | STEMI_000004 | STEMI | STEMI |

150 dpi, 1686x1311, ~590 KB, JPEG q85 — the v2 corpus's own settings.

## Reproducing

```bash
./run.sh
```

## Note on reproducibility

The layout patch changes the sheet for **every** build, including a rebuild of
v2, so a v2 image corpus regenerated now will not match the published one
pixel-for-pixel. Set `ECG_TOP_RESERVE`/`ECG_BOTTOM_RESERVE` if you need to tune
it, but there is no env value that restores the old `rows + 2` behaviour — say
so if you want one added.
