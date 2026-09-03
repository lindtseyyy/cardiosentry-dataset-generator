# v2 renderer, no header, tightened lead rows — 5 validation images

Same v2 path as before (150 dpi, stock stroke widths, `stage4_augment.py`,
4 leads per row, no printed header). One change: the three twelve-lead rows are
closer together, and the rhythm strip is not.

## What changed

The kit used a single `row_height` for **every** vertical gap, so tightening
the lead rows would have dragged the rhythm strip up with them. The layout
patch now takes two independent numbers:

```python
ROW_GAP    = float(os.environ.get("ECG_ROW_GAP",    4.4))   # between lead rows
RHYTHM_GAP = float(os.environ.get("ECG_RHYTHM_GAP", 5.5))   # row 3 -> rhythm strip
```

The lead rows step up from the rhythm strip by `RHYTHM_GAP`, then by `ROW_GAP`
each. Whatever is left over stays blank at the **top** of the sheet, which is
the cheap place for slack now that nothing is printed there.

## Measured, on the clean render

| | before | after |
|---|---:|---:|
| lead row 1 trace | 24.2 mm | 53.8 mm |
| lead row 2 trace | 81.4 mm | 97.9 mm |
| lead row 3 trace | 139.9 mm | 140.9 mm |
| rhythm strip trace | 194.6 mm | 194.6 mm |
| **gap, rows 1→2** | 57.2 mm | **44.1 mm** |
| **gap, rows 2→3** | 58.5 mm | **43.0 mm** |
| **gap, row 3 → rhythm** | 68.9 mm | **67.9 mm** — held |
| blank at top | 10 mm | 41 mm |

Row 3 and the rhythm strip did not move. The slack moved to the top, as
intended.

## The trade-off, stated plainly

44 mm between lead rows and a precordial QRS that routinely reaches 4 mV
(= 40 mm at 10 mm/mV) means V2/V3 will cross into the neighbouring row on
high-voltage records — visible on `ECG000001_a0.jpg`, which is an LVH case.
That is normal on real printouts and is **not** a reason to rescale: the gain
stays 10 mm/mV and the signal is untouched. If it is too tight, raise
`ECG_ROW_GAP`; 4.8 gives ~48 mm and still leaves the rhythm gap alone.

## The 5 validation images

`validation/` — same 5 records as the previous experiments.

| file | record | source | label |
|---|---|---|---|
| ECG000001_a0.jpg | CUHK_002942 | CUHK | **STEMI+LVH** |
| ECG000002_a0.jpg | CUHK_000477 | CUHK | AF |
| ECG000003_a0.jpg | PTBXL_000002 | PTBXL | NORMAL |
| ECG000004_a0.jpg | CHAPMAN_000001 | CHAPMAN | AF |
| ECG000005_a0.jpg | STEMI_000004 | STEMI | STEMI |

150 dpi, 1686x1311, ~590 KB, JPEG q85.

## Reproducing

```bash
./run.sh
```

## Still open

- The corpus has still not been regenerated.
